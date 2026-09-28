# ------------------------------------------------------------------------
# lmms-eval wrapper for Qwen2.5-VL + FastV. Mirrors the visionzip wrapper's
# CLI surface but with FastV-specific knobs (no VisionZip dominant /
# contextual ratios). Uses Qwen2_5_VLFastVForConditionalGeneration as the
# host model, which patches the LM with FastV inplace token pruning + the
# existing channel-pruning kv_cluster machinery.
# ------------------------------------------------------------------------

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
    Qwen2_5_VLConfig,
)

from lmms_eval.models.model_utils.qwen.qwen2_5vl_visionzip import (
    DEFAULT_CALIBRATION_RESULT_ROOT,
)
from lmms_eval.models.model_utils.qwen.qwen2_5vl_fastv import (
    Qwen2_5_VLFastVForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen2_5_vl_fastv")
class Qwen2_5_VL_FastV(lmms):
    """Qwen2.5-VL (HF) with FastV inplace token pruning.

    FastV slices `hidden_states` at decoder layer K based on layer K-1's
    text-to-vision attention scores, so layers >= K cache shorter KV
    (genuine memory reduction). Vision span bounds are detected dynamically
    per forward from input_ids.
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        # FastV requires attention weights → eager only.
        attn_implementation: Optional[str] = "eager",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        # ------ FastV knobs ------
        use_fast_v: bool = True,
        fast_v_agg_layer: int = 2,
        fast_v_keep_ratio: Optional[float] = 0.25,
        fast_v_attention_rank: Optional[int] = None,
        fast_v_inplace: bool = True,
        # ------ Channel pruning knobs (think / spark / rotatek) ------
        channel_ratio: float = 0.0,
        channel_method: str = "think",
        calibration_mode: Optional[str] = "off",
        offline_calibration_tasks: Optional[str] = "channel_importance",
        result_root: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        if attn_implementation != "eager":
            eval_logger.warning(
                f"FastV requires eager attention; overriding "
                f"attn_implementation={attn_implementation} → 'eager'."
            )
            attn_implementation = "eager"

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
            "attn_implementation": attn_implementation,
        }

        # Stamp FastV + channel-pruning knobs onto config.
        config = Qwen2_5_VLConfig.from_pretrained(pretrained)
        config.use_fast_v = use_fast_v
        config.fast_v_agg_layer = int(fast_v_agg_layer)
        config.fast_v_keep_ratio = fast_v_keep_ratio
        config.fast_v_attention_rank = fast_v_attention_rank
        config.fast_v_inplace = fast_v_inplace
        # `dominant_ratio` / `contextual_ratio` are forced to 0 inside the
        # FastV wrapper anyway (vision-encoder pruning bypassed); set them
        # here too so `_require_calibration_dominant_ratio` doesn't trip.
        config.dominant_ratio = 0.0
        config.contextual_ratio = 0.0
        config.channel_ratio = float(channel_ratio)
        config.channel_method = str(channel_method).strip().lower()
        config.calibration_mode = str(calibration_mode).strip().lower() if calibration_mode is not None else "off"
        config.offline_calibration_tasks = offline_calibration_tasks
        config.result_root = (
            str(result_root) if result_root is not None else DEFAULT_CALIBRATION_RESULT_ROOT
        )
        # Channel-budget ablation knobs (kept for compatibility with the
        # visionzip path; FastV does not exercise them).
        config.channel_start = None
        config.channel_end = None

        self._model = Qwen2_5_VLFastVForConditionalGeneration.from_pretrained(
            pretrained, config=config, **model_kwargs,
        ).eval()

        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals
        self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n") if reasoning_prompt else None

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
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
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2_5_VL_FastV")

    def flatten(self, input):
        return [j for i in input for j in i]

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

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
            elif not isinstance(until, list):
                raise ValueError(f"Expected gen_kwargs['until'] to be str | list, got {type(until)}")
            # Avoid '\n\n' as a Qwen2.5-VL stop token (causes truncation).
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)
            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": self.system_prompt}]
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                if visual_list[i] is not None:
                    for visual in visual_list[i]:
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                            processed_visuals.append({
                                "type": "video",
                                "video": visual,
                                "max_pixels": self.max_pixels,
                                "min_pixels": self.min_pixels,
                            })
                        elif isinstance(visual, Image.Image):
                            base64_image = visual.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            processed_visuals.append({
                                "type": "image",
                                "image": f"data:image/jpeg;base64,{base64_string}",
                                "max_pixels": self.max_pixels,
                                "min_pixels": self.min_pixels,
                            })

                if self.interleave_visuals is False:
                    message.append({
                        "role": "user",
                        "content": processed_visuals + [{"type": "text", "text": context}],
                    })
                else:
                    image_placeholders = re.findall(r"<image \d+>", context)
                    content_parts = []
                    text_parts = re.split(r"<image \d+>", context)
                    if text_parts[0]:
                        content_parts.append({"type": "text", "text": text_parts[0]})
                    for j, placeholder in enumerate(image_placeholders):
                        img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                        if processed_visuals and image_idx < len(processed_visuals):
                            content_parts.append(processed_visuals[image_idx])
                        if j + 1 < len(text_parts) and text_parts[j + 1]:
                            content_parts.append({"type": "text", "text": text_parts[j + 1]})
                    message.append({"role": "user", "content": content_parts})

                batched_messages.append(message)

            texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(batched_messages)
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                indices = np.unique(indices)
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                    indices = np.unique(indices)
                video_inputs[0] = video_inputs[0][indices]
            padding_side = "left" if self.batch_size > 1 else "right"
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                padding_side=padding_side,
                return_tensors="pt",
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
            pad_token_id = self.tokenizer.pad_token_id
            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            try:
                cont = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=current_gen_kwargs["do_sample"],
                    temperature=current_gen_kwargs["temperature"],
                    top_p=current_gen_kwargs["top_p"],
                    num_beams=current_gen_kwargs["num_beams"],
                    max_new_tokens=current_gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )
                generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, cont)]
                answers = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
                )
                for i, ans in enumerate(answers):
                    for term in until:
                        if term:
                            ans = ans.split(term)[0]
                    answers[i] = ans
            except Exception as exc:
                # Wrapper-state bookkeeping bugs (K/V seqlen mismatch, etc.)
                # surface on degenerate samples (e.g., near-empty videos).
                # Skip the batch with empty answers so the eval continues
                # — GPT scorer will assign 0 to empties.
                print(
                    f"[SKIP] generate() failed on batch of {len(contexts)} sample(s): "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                answers = [""] * len(contexts)

            for ans, context in zip(answers, contexts):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)
                pbar.update(1)
                print(f"Question: {context}", flush=True)
                print(f"Model Raw Response: {ans}", flush=True)
                print(f"Model Clean Response: {clean_ans}", flush=True)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("Multi-round generation not implemented for Qwen2_5_VL_FastV")
