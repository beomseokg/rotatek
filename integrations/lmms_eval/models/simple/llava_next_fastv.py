"""lmms-eval wrapper: LLaVA-NeXT with FastV token pruning + Key channel pruning.

Only layer K-1's prefill runs eager (FastV needs its attention map); every
other layer and all decode steps use FlashAttention-2.
"""
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.llava_next.llava_next_fastv import (
    LlavaNextFastVForConditionalGeneration,
)
from lmms_eval.models.simple.llava_next_visionzip import LlavaNextWrapper, load_llava_next


@register_model("llava_next_fastv")
class LlavaNext_FastV(LlavaNextWrapper):
    def __init__(
        self,
        pretrained: str = "llava-hf/llama3-llava-next-8b-hf",
        fast_v_agg_layer: int = 2,
        fast_v_keep_ratio: float = 0.40,
        channel_ratio: float = 0.75,
        channel_method: str = "rotatek",
        attn_implementation: str = "flash_attention_2",
        batch_size: int = 1,
        device: str = "cuda",
        device_map: str = "auto",
    ) -> None:
        def stamp(tc):
            tc.use_fast_v = True
            tc.fast_v_inplace = True
            tc.fast_v_agg_layer = int(fast_v_agg_layer)
            tc.fast_v_keep_ratio = float(fast_v_keep_ratio)
            tc.fast_v_attention_rank = None
            tc.channel_ratio = float(channel_ratio)
            tc.channel_method = channel_method

        model = load_llava_next(LlavaNextFastVForConditionalGeneration, pretrained, stamp,
                                attn_implementation, device_map)
        super().__init__(model, pretrained, batch_size, device, device_map)
