# ------------------------------------------------------------------------
# lmms-eval wrapper for LLaVA-NeXT + FastV. Mirrors LlavaNext_VisionZip's
# CLI surface but with FastV-specific knobs (no VisionZip dominant/contextual
# ratios). Uses LlavaNextFastVForConditionalGeneration as the host model.
# ------------------------------------------------------------------------

from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    LlavaNextConfig,
)

from lmms_eval.models.model_utils.llava_next.llava_next_fastv import (
    LlavaNextFastVForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)


@register_model("llava_next_fastv")
class LlavaNext_FastV(lmms):
    """LLaVA-NeXT (HF) with FastV inplace token pruning.

    FastV slices `hidden_states` at decoder layer K based on layer K-1's
    text-to-vision attention scores, so layers >= K cache shorter KV
    (genuine memory reduction). Vision span bounds are detected dynamically
    per forward — no fixed SYS_LENGTH / IMAGE_TOKEN_LENGTH assumptions.
    """

    def __init__(
        self,
        pretrained: str = "llava-hf/llama3-llava-next-8b-hf",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        # NOTE (2026-05-06 v2): 기본 FA2. Per-layer dispatch (LlamaFastVAttention)
        # 이 layer K-1 prefill에서만 eager로 자동 전환. 모델 단에서 attn 클래스를
        # 직접 LlamaFastVAttention으로 박아넣으므로 attn_implementation은 사실상
        # cosmetic이지만, HF가 config 검증에 쓸 수 있으므로 FA2 명시.
        attn_implementation: Optional[str] = "flash_attention_2",
        system_prompt: Optional[str] = None,
        conv_template: Optional[str] = "llava_llama_3",
        # ------ FastV knobs ------
        use_fast_v: bool = True,
        fast_v_agg_layer: int = 2,
        fast_v_keep_ratio: Optional[float] = 0.25,
        fast_v_attention_rank: Optional[int] = None,
        fast_v_inplace: bool = True,
        # ------ Channel pruning / decode backend knobs ------
        channel_ratio: float = 0.0,
        channel_method: str = "think",
        decode_attention_backend: str = "fa2",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # NOTE (2026-05-06 v2): eager 강제하지 않음.
        # LlamaFastVAttention.forward가 output_attentions 파라미터로 per-layer
        # dispatch (eager vs FA2)하므로 wrapper에서 강제할 필요 없음.
        # Layer K-1 prefill에서만 output_attentions=True로 호출되어 eager로 분기,
        # 나머지 모든 layer + decode는 output_attentions=False → FA2로 분기.
        # 원복하려면 아래 블록을 다시
        #     if attn_implementation != "eager":
        #         eval_logger.warning(...); attn_implementation = "eager"
        # 형태로 풀면 됨.
        # (잔여 가드 — 사용자가 명시적으로 "eager"를 지정하면 그대로 존중)

        # Coerce string-form bools that come through CLI arg parsing.
        def _to_bool(v):
            if isinstance(v, bool):
                return v
            return str(v).strip().lower() in ("1", "true", "yes")

        use_fast_v = _to_bool(use_fast_v)
        fast_v_inplace = _to_bool(fast_v_inplace)

        # keep_ratio takes precedence; fall back to attention_rank.
        if fast_v_keep_ratio is not None:
            fast_v_keep_ratio = float(fast_v_keep_ratio)
            fast_v_attention_rank = None
        elif fast_v_attention_rank is not None:
            fast_v_attention_rank = int(fast_v_attention_rank)
        elif use_fast_v:
            raise ValueError(
                "use_fast_v=True requires either fast_v_keep_ratio or "
                "fast_v_attention_rank to be set."
            )

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {
            "torch_dtype": "bfloat16",
            "device_map": self.device_map,
        }
        # NOTE (2026-05-06): VisionZip wrapper와 동일한 패턴으로 변경.
        # eager가 아니면 텍스트만 해당 kernel, vision tower는 eager로 분리 지정.
        # 원복하려면 아래 블록 전체를 지우고 위 dict에 다시
        #     "attn_implementation": attn_implementation,
        # 한 줄을 추가하면 됨.
        if attn_implementation is not None and attn_implementation != "eager":
            model_kwargs["attn_implementation"] = {
                "text_config": attn_implementation,
                "vision_config": "eager",
            }
        elif attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # Stamp FastV + channel-pruning knobs onto text_config so the patched
        # LlamaFastVModel picks them up via getattr in its forward.
        config = LlavaNextConfig.from_pretrained(pretrained)
        # NOTE (2026-05-06): vision tower는 항상 eager로 강제 (VisionZip wrapper와 동일).
        # 원복하려면 아래 한 줄 삭제.
        config.vision_config._attn_implementation = "eager"
        tc = config.text_config
        tc.use_fast_v = use_fast_v
        tc.fast_v_agg_layer = int(fast_v_agg_layer)
        tc.fast_v_keep_ratio = fast_v_keep_ratio
        tc.fast_v_attention_rank = fast_v_attention_rank
        tc.fast_v_inplace = fast_v_inplace
        # Channel pruning (passes through visionzip's kv_cluster machinery).
        # `dominant_ratio` is required by init_visionzip but unused on the
        # FastV path (no vision-encoder pruning); set a benign placeholder.
        tc.channel_ratio = float(channel_ratio)
        tc.channel_method = str(channel_method).strip().lower()
        tc.decode_attention_backend = str(decode_attention_backend).strip().lower()
        tc.dominant_ratio = 0.0
        tc.contextual_ratio = 0.0

        self._model = LlavaNextFastVForConditionalGeneration.from_pretrained(
            pretrained, config=config, **model_kwargs,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(pretrained)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.conv_template = conv_template

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 4096)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP, DistributedType.MULTI_GPU,
            ], "Unsupported distributed type (only DDP/FSDP)."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for LlavaNext_FastV")

    def flatten(self, input):
        return [j for i in input for j in i]

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            return -len(self.tokenizer.encode(x[0])), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]

            contexts = list(contexts)
            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "").strip()

            batched_messages = []
            batched_images = []
            for i, context in enumerate(contexts):
                images = []
                if visual_list[i] is not None:
                    for vis in visual_list[i]:
                        if isinstance(vis, Image.Image):
                            images.append(vis.convert("RGB"))
                batched_images.append(images)

                content = []
                for _ in images:
                    content.append({"type": "image"})
                content.append({"type": "text", "text": context})
                msg = []
                if self.system_prompt:
                    msg.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
                msg.append({"role": "user", "content": content})
                batched_messages.append(msg)

            prompts = [
                self.processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                for m in batched_messages
            ]
            flat_images = [img for imgs in batched_images for img in imgs] or None

            inputs = self.processor(
                images=flat_images,
                text=prompts,
                return_tensors="pt",
                padding=True,
            )
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            default_gen_kwargs = {
                "max_new_tokens": 1024,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )
            for i, ans in enumerate(answers):
                for term in until:
                    if term:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, ctx in zip(answers, contexts):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (ctx, gen_kwargs), clean_ans)
                pbar.update(1)
                print(f"Question: {ctx}", flush=True)
                print(f"Model Raw Response: {ans}", flush=True)
                print(f"Model Clean Response: {clean_ans}", flush=True)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("Multi-round generation not implemented for LlavaNext_FastV")
