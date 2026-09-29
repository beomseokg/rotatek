"""lmms-eval wrapper: Qwen2.5-VL with FastV token pruning + Key channel pruning.

FastV drops the least-attended visual tokens at decoder layer K, scoring them
with layer K-1's attention map, so every layer from K on caches a shorter KV.
Scoring needs attention weights, hence eager attention.
"""
from transformers import Qwen2_5_VLConfig

from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.qwen.qwen2_5vl_fastv import (
    Qwen2_5_VLFastVForConditionalGeneration,
)
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL
from lmms_eval.models.simple.qwen2_5_vl_visionzip import init_qwen_wrapper


@register_model("qwen2_5_vl_fastv")
class Qwen2_5_VL_FastV(Qwen2_5_VL):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        fast_v_agg_layer: int = 2,
        fast_v_keep_ratio: float = 0.40,
        channel_ratio: float = 0.75,
        channel_method: str = "rotatek",
        attn_implementation: str = "eager",
        batch_size: int = 1,
        device: str = "cuda",
        device_map: str = "auto",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
    ) -> None:
        config = Qwen2_5_VLConfig.from_pretrained(pretrained)
        config.fast_v_agg_layer = int(fast_v_agg_layer)
        config.fast_v_keep_ratio = float(fast_v_keep_ratio)
        config.channel_ratio = float(channel_ratio)
        config.channel_method = channel_method
        model = Qwen2_5_VLFastVForConditionalGeneration.from_pretrained(
            pretrained, config=config, torch_dtype="bfloat16", device_map=device_map,
            attn_implementation=attn_implementation,
        ).eval()
        init_qwen_wrapper(self, model, pretrained, batch_size, device, device_map,
                          max_pixels, min_pixels)
