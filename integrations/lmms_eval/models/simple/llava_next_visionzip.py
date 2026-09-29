"""lmms-eval wrapper: LLaVA-NeXT (llama3-llava-next-8b-hf) with VisionZip token
pruning + Key channel pruning. ``LlavaNextWrapper`` holds the generation loop
shared with the FastV wrapper in ``llava_next_fastv.py``."""
from typing import List, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, LlavaNextConfig

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.llava_next.llava_next_visionzip import (
    LlavaNextVisionZipForConditionalGeneration,
)
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)


def load_llava_next(model_cls, pretrained, stamp, attn_implementation, device_map):
    """Load `model_cls` with `stamp(text_config)` applied. The CLIP tower is
    always eager: VisionZip reads its attention map, which FA2/SDPA do not
    materialize."""
    config = LlavaNextConfig.from_pretrained(pretrained)
    config.vision_config._attn_implementation = "eager"
    stamp(config.text_config)
    if attn_implementation != "eager":
        attn_implementation = {"text_config": attn_implementation, "vision_config": "eager"}
    return model_cls.from_pretrained(
        pretrained, config=config, torch_dtype="bfloat16", device_map=device_map,
        attn_implementation=attn_implementation,
    ).eval()


class LlavaNextWrapper(lmms):
    def __init__(self, model, pretrained, batch_size, device, device_map):
        super().__init__()
        self._model = model
        self._device = torch.device(device)
        self.device_map = device_map
        self.processor = AutoProcessor.from_pretrained(pretrained)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self._config = model.config
        self.batch_size_per_gpu = int(batch_size)

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 4096

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            return -len(self.tokenizer.encode(x[0])), x[0]

        pbar = tqdm(total=len(requests), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        for chunk in re_ords.get_batched(n=self.batch_size, batch_fn=None):
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task, split = task[0], split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]

            contexts = [c.replace("<image>", "").strip() if "<image>" in c else c for c in contexts]

            prompts, flat_images = [], []
            for visuals, context in zip(visual_list, contexts):
                images = [v.convert("RGB") for v in (visuals or []) if isinstance(v, Image.Image)]
                flat_images.extend(images)
                content = [{"type": "image"} for _ in images] + [{"type": "text", "text": context}]
                prompts.append(self.processor.apply_chat_template(
                    [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False))

            # decoder-only batched generation needs left padding
            self.processor.tokenizer.padding_side = "left"
            inputs = self.processor(images=flat_images or None, text=prompts,
                                    return_tensors="pt", padding=True)
            inputs = inputs.to("cuda" if self.device_map == "auto" else self.device)

            gen = {"max_new_tokens": 1024, "temperature": 0.0, "top_p": None, "num_beams": 1, **gen_kwargs}
            do_sample = gen["temperature"] > 0
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                do_sample=do_sample,
                temperature=gen["temperature"] if do_sample else None,
                top_p=gen["top_p"] if do_sample else None,
                num_beams=gen["num_beams"],
                max_new_tokens=gen["max_new_tokens"],
                use_cache=True,
            )
            answers = self.processor.batch_decode(
                [out[len(inp):] for inp, out in zip(inputs.input_ids, cont)],
                skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )
            for ans, ctx in zip(answers, contexts):
                for term in until:
                    if term:
                        ans = ans.split(term)[0]
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (ctx, gen_kwargs), clean_ans)
                pbar.update(1)

        pbar.close()
        return re_ords.get_original(res)


@register_model("llava_next_visionzip")
class LlavaNext_VisionZip(LlavaNextWrapper):
    def __init__(
        self,
        pretrained: str = "llava-hf/llama3-llava-next-8b-hf",
        visionzip_dominant_ratio: float = 0.40,
        visionzip_contextual_ratio: float = 0.05,
        channel_ratio: float = 0.75,
        channel_method: str = "rotatek",
        attn_implementation: str = "flash_attention_2",
        # "triton" routes every method's decode through Triton kernels, as in the
        # paper's latency comparison; accuracy runs keep the default "fa2".
        decode_attention_backend: str = "fa2",
        batch_size: int = 1,
        device: str = "cuda",
        device_map: str = "auto",
    ) -> None:
        def stamp(tc):
            tc.dominant_ratio = visionzip_dominant_ratio
            tc.contextual_ratio = visionzip_contextual_ratio
            tc.channel_ratio = channel_ratio
            tc.channel_method = channel_method
            tc.decode_attention_backend = decode_attention_backend

        model = load_llava_next(LlavaNextVisionZipForConditionalGeneration, pretrained, stamp,
                                attn_implementation, device_map)
        super().__init__(model, pretrained, batch_size, device, device_map)
