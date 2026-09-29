"""lmms-eval wrapper: Qwen2.5-VL with VisionZip token pruning + Key channel pruning.

Generation is inherited unchanged from lmms-eval's stock ``qwen2_5_vl`` wrapper;
only model construction differs: the pruning knobs are stamped onto the config
and the patched model class from ``model_utils.qwen`` is loaded.
"""
import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLConfig

from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.qwen.qwen2_5vl_visionzip import (
    Qwen2_5_VLForConditionalGeneration,
)
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL


def init_qwen_wrapper(self, model, pretrained, batch_size, device, device_map,
                      max_pixels, min_pixels):
    """Set the attributes that the stock ``Qwen2_5_VL.generate_until`` reads,
    with the stock defaults (single process, image inputs, no reasoning prompt)."""
    lmms.__init__(self)
    self._model = model
    self._device = torch.device(device)
    self.device_map = device_map
    self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
    self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
    self.max_pixels = max_pixels
    self.min_pixels = min_pixels
    self.max_num_frames = 32
    self.use_custom_video_loader = False
    self.fps = None
    self.max_image_size = None
    self.system_prompt = "You are a helpful assistant."
    self.interleave_visuals = False
    self.reasoning_prompt = None
    self._config = model.config
    self._max_length = 2048
    self.batch_size_per_gpu = int(batch_size)
    self.use_cache = True
    self._rank = 0
    self._world_size = 1


@register_model("qwen2_5_vl_visionzip")
class Qwen2_5_VL_VisionZip(Qwen2_5_VL):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
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
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
    ) -> None:
        config = Qwen2_5_VLConfig.from_pretrained(pretrained)
        config.dominant_ratio = visionzip_dominant_ratio
        config.contextual_ratio = visionzip_contextual_ratio
        # the vision encoder reads the ratio off its own sub-config to decide
        # whether to return the attention map VisionZip scores tokens with
        config.vision_config.dominant_ratio = visionzip_dominant_ratio
        config.channel_ratio = channel_ratio
        config.channel_method = channel_method
        config.decode_attention_backend = decode_attention_backend
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained, config=config, torch_dtype="bfloat16", device_map=device_map,
            attn_implementation=attn_implementation,
        ).eval()
        init_qwen_wrapper(self, model, pretrained, batch_size, device, device_map,
                          max_pixels, min_pixels)
