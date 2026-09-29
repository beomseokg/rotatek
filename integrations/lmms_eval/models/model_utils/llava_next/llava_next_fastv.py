# ------------------------------------------------------------------------
# LlavaNextForConditionalGeneration with FastV (token-level pruning).
#
# Mirrors the visionzip wrapper structure but with a different token
# pruning mechanism: FastV operates *inside* the language model at decoder
# layer K, whereas VisionZip operates *before* the LM at the vision encoder.
# As a result this wrapper is much thinner — it only stamps the dynamic
# vision-span bounds onto the patched LlamaFastVModel before each forward.
# ------------------------------------------------------------------------

import math
import torch
import torch.nn as nn

from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextForConditionalGeneration,
    LlavaNextConfig,
)
from transformers.models.auto import AutoModel

from lmms_eval.models.model_utils.llava_next.llama_fastv import LlamaFastVForCausalLM

__all__ = ["LlavaNextFastVForConditionalGeneration"]


class LlavaNextFastVForConditionalGeneration(LlavaNextForConditionalGeneration):
    """HF LlavaNext wrapper that swaps the language model for the
    FastV-patched LLaMA. The vision tower / projector / pack_image_features
    pipeline is unchanged."""

    def __init__(self, config: LlavaNextConfig):
        # Mirror super().__init__ but swap AutoModelForCausalLM with ours.
        super(LlavaNextForConditionalGeneration, self).__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)

        from transformers.models.llava_next.modeling_llava_next import (
            LlavaNextMultiModalProjector,
        )
        self.multi_modal_projector = LlavaNextMultiModalProjector(config)
        embed_std = 1 / math.sqrt(config.text_config.hidden_size)
        self.image_newline = nn.Parameter(
            torch.randn(config.text_config.hidden_size, dtype=self.dtype) * embed_std
        )

        self.vocab_size = config.text_config.vocab_size
        self.language_model = LlamaFastVForCausalLM(config.text_config)
        if self.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [
                f"language_model.{k}" for k in self.language_model._tied_weights_keys
            ]

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self._padding_side = "left"
        self.post_init()

    # ------------------------------------------------------------------
    # Forward override — only role is to compute & stamp the dynamic vision
    # span bounds onto the LM model before delegating to HF's standard
    # LlavaNext forward (which handles the image-feature merge for us).
    #
    # Bounds are computed from input_ids alone (not requiring access to
    # post-merge inputs_embeds): the placeholder positions in input_ids
    # determine `sys_length` directly, and IMG length is derived inside
    # the LM forward as `seq_full - pre_merge_text_len + num_placeholders`.
    #
    # Signature MUST match HF's exactly for `logits_to_keep` to flow
    # through GenerationMixin (same constraint as the visionzip wrapper).
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        logits_to_keep=0,
        **lm_kwargs,
    ):
        # Stamp bounds at prefill (input_ids contains <image> placeholders).
        # During decode, input_ids has only the new token; bounds set at
        # prefill remain valid since the cache holds the vision span.
        if input_ids is not None and pixel_values is not None:
            image_tok = self.config.image_token_index
            mask = (input_ids == image_tok)
            if mask.any():
                # Batch[0] only — lmms-eval calls with batch_size=1.
                nz = mask[0].nonzero(as_tuple=True)[0]
                lm_model = self.language_model.model
                lm_model.fast_v_sys_length = int(nz[0].item())
                lm_model.fast_v_pre_merge_text_len = int(input_ids.shape[1])
                lm_model.fast_v_num_placeholders = int(mask[0].sum().item())  # per-sample (matches nz=mask[0]); was mask.sum()=batch-total -> broke batch>1
                # ThinK / channel pruning needs prompt_seqlen (text BEFORE the
                # vision span in the merged sequence) and query_seqlen (text
                # AFTER vision). Stamp on text_config so init_channel_pruner picks
                # them up at prefill. Mirrors visionzip wrapper.
                self.config.text_config.prompt_seqlen = int(nz[0].item())
                self.config.text_config.query_seqlen = int(input_ids.shape[1] - nz[-1].item() - 1)

        return super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )
