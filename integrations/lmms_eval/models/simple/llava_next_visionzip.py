import base64
import re
from io import BytesIO
from typing import List, Optional, Tuple, Union

import numpy as np
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

from lmms_eval.models.model_utils.llava_next.llava_next_visionzip import (
    DEFAULT_CALIBRATION_RESULT_ROOT,
    LlavaNextVisionZipForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)


@register_model("llava_next_visionzip")
class LlavaNext_VisionZip(lmms):
    """LLaVA-NeXT (HF) with VisionZip base-patch token pruning and LLaMA
    channel pruning. Mirrors Qwen2_5_VL_VisionZip's argument surface so the
    same CLI runner works unchanged."""

    def __init__(
        self,
        pretrained: str = "llava-hf/llama3-llava-next-8b-hf",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = "flash_attention_2",
        system_prompt: Optional[str] = None,
        conv_template: Optional[str] = "llava_llama_3",
        visionzip_dominant_ratio: Optional[float] = 0.65,
        visionzip_contextual_ratio: Optional[float] = 0.05,
        channel_ratio: Optional[float] = 0.5,
        channel_method: Optional[str] = "visionk",
        layer_adaptive_channel_budget: bool = False,
        channel_reconstruction: str = "off",
        reconstruction_constant: float = 0.1,
        custom_kernel: bool = True,
        decode_attention_backend: str = "fa2",
        reconstruction_topk: Optional[int] = None,
        mmstar_reconstruct_topk: Optional[int] = None,
        calibration_mode: Optional[str] = "off",
        offline_calibration_tasks: Optional[str] = "channel_importance",
        channel_start: Optional[int] = None,
        channel_end: Optional[int] = None,
        exempt_layer_idx: Optional[int] = None,
        full_channel_first_n_layers: int = 0,
        result_root: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(
                f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}"
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
        # VisionZip token selection requires CLS-attention from the CLIP vision
        # tower. flash_attention_2 / sdpa skip materializing attention weights,
        # so `output_attentions=True` returns None at the vision tower and
        # token pruning silently no-ops (every dominant_ratio gives the same
        # score).
        #
        # Pass attn_implementation as a per-submodule dict: keep the LLM choice
        # as requested but force the vision tower to eager. This works on
        # transformers >= 4.43; on older versions we fall back to a plain
        # string (which would re-introduce the silent-skip on FA2/SDPA).
        if attn_implementation is not None and attn_implementation != "eager":
            model_kwargs["attn_implementation"] = {
                "text_config": attn_implementation,
                "vision_config": "eager",
            }
        elif attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # Load config and stamp VisionZip / channel-pruning knobs onto text_config
        config = LlavaNextConfig.from_pretrained(pretrained)
        # Belt-and-suspenders: also stamp on the config so callers that bypass
        # `attn_implementation` still get the eager vision tower.
        config.vision_config._attn_implementation = "eager"
        tc = config.text_config
        tc.dominant_ratio = visionzip_dominant_ratio
        tc.contextual_ratio = visionzip_contextual_ratio
        tc.channel_ratio = channel_ratio
        tc.channel_method = channel_method
        tc.layer_adaptive_channel_budget = layer_adaptive_channel_budget
        tc.channel_reconstruction = channel_reconstruction
        tc.reconstruction_constant = reconstruction_constant
        tc.decode_attention_backend = decode_attention_backend
        tc.custom_kernel = custom_kernel and channel_method in ("visionk",)
        if reconstruction_topk is None:
            reconstruction_topk = mmstar_reconstruct_topk
        tc.reconstruction_topk = reconstruction_topk
        tc.mmstar_reconstruct_topk = mmstar_reconstruct_topk
        tc.calibration_mode = calibration_mode
        tc.offline_calibration_tasks = offline_calibration_tasks
        tc.channel_start = channel_start
        tc.channel_end = channel_end
        tc.exempt_layer_idx = int(exempt_layer_idx) if exempt_layer_idx is not None else None
        tc.full_channel_first_n_layers = int(full_channel_first_n_layers)
        tc.result_root = str(result_root) if result_root is not None else DEFAULT_CALIBRATION_RESULT_ROOT

        self._model = LlavaNextVisionZipForConditionalGeneration.from_pretrained(
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
        raise NotImplementedError("Loglikelihood is not implemented for LlavaNext_VisionZip")

    def flatten(self, input):
        return [j for i in input for j in i]

    # ------------------------------------------------------------------
    # Generation: use the HF chat template. llama3-llava-next-8b uses the
    # `llava_llama_3` template by convention; its tokenizer already ships
    # with the matching chat template, so we just call
    # `processor.apply_chat_template`.
    # ------------------------------------------------------------------
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

            # Decoder-only batched generation requires LEFT padding: with the
            # default right padding, shorter sequences have pad tokens after the
            # prompt, so generation continues from padding (garbage) and the
            # len(inp)-based trimming below misaligns. Left-pad so every sequence
            # ends at the same position and generation appends on the right.
            self.processor.tokenizer.padding_side = "left"
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
        raise NotImplementedError("Multi-round generation not implemented for LlavaNext_VisionZip")
