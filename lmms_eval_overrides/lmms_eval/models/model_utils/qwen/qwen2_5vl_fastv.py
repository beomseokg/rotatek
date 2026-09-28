# ------------------------------------------------------------------------
# Qwen2.5-VL backbone with FastV (token-level pruning) hooks. Mirrors
# `llava_next/llama_fastv.py` but adapted to Qwen2.5-VL's mrope-based LM.
#
# FastV (inplace mode):
#   * At decoder layer K, score each visual token by the last query row of
#     layer K-1's head-averaged attention; keep top `attention_rank` tokens.
#   * Slice `hidden_states` / `position_ids` in place at layer K so layers
#     >= K cache shorter KV — actual KV memory reduction.
#
# Channel pruning (ThinK / SparK / RotateK truncated):
#   * Inherits the existing kv_cluster / DynamicCache machinery used by
#     `Qwen2_5_VLFlashAttention2` in `qwen2_5vl_visionzip.py`. Re-implements
#     ONLY the attention compute kernel as eager so layer K can read
#     attn_weights from layer K-1.
#   * RotateK truncated decode delegates to `methods.rotatek.rotatek_decode_fused`
#     which produces attn_output directly; eager softmax is skipped.
#
# Differences vs. LLaVA-NeXT FastV:
#   * 3D mrope: `position_ids` has shape [3, B, S]. After FastV slice we
#     index along the last dim and recompute (cos, sin) via the model's
#     `rotary_emb` on the kept positions.
#   * Image-token bounds are read directly from `input_ids` (Qwen does NOT
#     expand placeholders post-merge, unlike LLaVA-NeXT anyres). The
#     wrapper stamps `fast_v_sys_length` and `fast_v_image_token_length`
#     onto the LM model before each forward.
#   * `Qwen2_5_VLDecoderLayer` unpacks 4 attention return values; FastV
#     attention returns `(attn_output, attn_weights, past_key_value, None)`.
# ------------------------------------------------------------------------

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig

from lmms_eval.models.model_utils.cache_utils import Cache, DynamicCache
from lmms_eval.models.model_utils.llava_next.llama_visionzip import (
    _stamp_channel_defaults,
)
from lmms_eval.models.model_utils.qwen import qwen2_5vl_visionzip as _vz_mod
from lmms_eval.models.model_utils.qwen.qwen2_5vl_visionzip import (
    Qwen2_5_VLAttention,
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLRotaryEmbedding,
    Qwen2MLP,
    Qwen2RMSNorm,
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)


__all__ = [
    "Qwen2_5_VLFastVAttention",
    "Qwen2_5_VLFastVDecoderLayer",
    "Qwen2_5_VLFastVModel",
    "Qwen2_5_VLFastVForConditionalGeneration",
    "_stamp_fastv_defaults",
]


def _stamp_fastv_defaults(config):
    """Stamp FastV-related fields on the config with safe defaults so
    `from_pretrained` calls without explicit kwargs don't crash."""
    if not hasattr(config, "use_fast_v"):
        config.use_fast_v = False
    if not hasattr(config, "fast_v_agg_layer"):
        config.fast_v_agg_layer = 2
    if not hasattr(config, "fast_v_keep_ratio"):
        config.fast_v_keep_ratio = None
    if not hasattr(config, "fast_v_attention_rank"):
        config.fast_v_attention_rank = None
    if not hasattr(config, "fast_v_inplace"):
        config.fast_v_inplace = True


# ------------------------------------------------------------------------
# Eager Qwen2 attention with VisionZip-style channel-pruning cache.
# Subclass of Qwen2_5_VLAttention: reuses q/k/v/o projections + rotary_emb
# init; replaces forward() to (a) write the channel-pruned cache during
# prefill, (b) reconstruct K from the pruned cache during decode, and (c)
# return attn_weights so layer K-1's output is visible to layer K's FastV
# slice. Returns a 4-tuple to stay compatible with Qwen2_5_VLDecoderLayer
# which unpacks `(attn_output, attn_weights, past_key_value, original_hidden_states)`.
# ------------------------------------------------------------------------
class Qwen2_5_VLFastVAttention(Qwen2_5_VLAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = True,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        from lmms_eval.models.model_utils.kv_pruning_utils import init_visionzip

        bsz, q_len, _ = hidden_states.size()

        channel_method = getattr(self.config, "channel_method", "think")
        channel_ratio = float(getattr(self.config, "channel_ratio", 0.0) or 0.0)

        if q_len > 1:
            init_visionzip(self)

        query_states = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states_no_rope = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            raise ValueError("Qwen2_5_VLFastVAttention requires position_embeddings.")
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states_no_rope, cos, sin, self.rope_scaling["mrope_section"]
        )

        attn_output = None  # set by rotatek-truncated fused decode path
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

            if q_len > 1:  # prefill
                if channel_ratio == 0.0:
                    past_key_value.store_unified(key_states, value_states, self.layer_idx)
                elif channel_method == "think":
                    kv_pruned, kv_prompt, kv_text, mask, value_states_compress = (
                        self.kv_cluster.update_think(
                            key_states, query_states, value_states, attention_mask,
                            num_key_value_groups=self.num_key_value_groups,
                            calibration_channel_importance=getattr(
                                self, "calibration_channel_importance", None,
                            ),
                        )
                    )
                    past_key_value.store_pruned(
                        kv_pruned, kv_prompt, kv_text, mask, value_states_compress,
                        self.layer_idx, cache_kwargs,
                    )
                    past_key_value.think_mask.append(self.kv_cluster.current_think_mask)
                elif channel_method == "spark":
                    kv_pruned, kv_prompt, kv_text, mask, value_states_compress = (
                        self.kv_cluster.update_spark(
                            key_states, query_states, value_states, attention_mask,
                            num_key_value_groups=self.num_key_value_groups,
                        )
                    )
                    past_key_value.store_pruned(
                        kv_pruned, kv_prompt, kv_text, mask, value_states_compress,
                        self.layer_idx, cache_kwargs,
                    )
                    past_key_value.spark_mask.append(self.kv_cluster.current_spark_mask)
                    past_key_value.spark_pruned_mean.append(self.kv_cluster.current_spark_pruned_mean)
                elif channel_method == "rotatek":
                    kv_pruned, kv_prompt, kv_text, mask, value_states_compress = (
                        self.kv_cluster.update_rotatek(
                            key_states, query_states, value_states, attention_mask,
                            num_key_value_groups=self.num_key_value_groups,
                            calibration_rotation_R_partial=getattr(
                                self, "calibration_rotation_R_partial", None,
                            ),
                        )
                    )
                    past_key_value.store_pruned(
                        kv_pruned, kv_prompt, kv_text, mask, value_states_compress,
                        self.layer_idx, cache_kwargs,
                    )
                    past_key_value.rotatek_rotations.append(
                        self.kv_cluster.current_rotatek_R_partial
                    )
                    past_key_value.rotatek_means.append(
                        self.kv_cluster.current_rotatek_delta_mu
                    )
                else:
                    raise NotImplementedError(
                        f"channel_method={channel_method!r} not supported. "
                        "Use 'think', 'spark', 'rotatek', or channel_ratio=0.0."
                    )

                # Prefill compute uses the ORIGINAL full K/V; channel pruning
                # only affects what's stored in cache for decode.
                k_for_compute = key_states
                v_for_compute = value_states

            else:  # decode
                if channel_ratio == 0.0:
                    k_for_compute, v_for_compute = past_key_value.update_unified(
                        key_states, value_states, self.layer_idx,
                    )
                else:
                    text_key_states, value_states_full, key_pruned, key_prompt, mask = (
                        past_key_value.update(
                            key_states, value_states, self.layer_idx, cache_kwargs,
                        )
                    )
                    v_for_compute = value_states_full

                    if channel_method == "think":
                        think_mask = past_key_value.think_mask[self.layer_idx]
                        bsz_r, h_kv, seq_r = key_pruned.shape[:3]
                        recovered = torch.zeros(
                            bsz_r, h_kv, seq_r, self.head_dim,
                            dtype=key_pruned.dtype, device=key_pruned.device,
                        )
                        mask_expanded = think_mask.unsqueeze(2).expand(-1, -1, seq_r, -1)
                        recovered[mask_expanded] = key_pruned.reshape(-1)
                        k_for_compute = torch.cat(
                            [key_prompt, recovered, text_key_states], dim=-2,
                        )

                    elif channel_method == "spark":
                        spark_mask = past_key_value.spark_mask[self.layer_idx]
                        pruned_mean = past_key_value.spark_pruned_mean[self.layer_idx]
                        bsz_r, h_kv, seq_r, _ = key_pruned.shape
                        recovered = pruned_mean.expand(
                            bsz_r, h_kv, seq_r, self.head_dim,
                        ).contiguous()
                        recovered[spark_mask] = key_pruned.reshape(-1)
                        k_for_compute = torch.cat(
                            [key_prompt, recovered, text_key_states], dim=-2,
                        )

                    elif channel_method == "rotatek":
                        if key_pruned.shape[-1] == self.head_dim:
                            raise RuntimeError(
                                "RotateK 'full' storage mode is not exposed in "
                                "Qwen2_5_VLFastVAttention. Set ROTATEK_STORAGE=truncated."
                            )
                        from methods.rotatek import rotatek_decode_fused

                        q_squeezed = query_states.squeeze(2)  # [B, H_q, D]
                        R_partial = past_key_value.rotatek_rotations[self.layer_idx]
                        delta_mu = past_key_value.rotatek_means[self.layer_idx]

                        s_prompt = key_prompt.shape[-2]
                        s_vision = key_pruned.shape[-2]
                        v_prompt = value_states_full[:, :, :s_prompt, :]
                        v_vision = value_states_full[:, :, s_prompt:s_prompt + s_vision, :]
                        v_text = value_states_full[:, :, s_prompt + s_vision:, :]

                        k_full = torch.cat([key_prompt, text_key_states], dim=-2)
                        v_full = torch.cat([v_prompt, v_text], dim=-2)
                        s_full = k_full.shape[-2]
                        mask_full = torch.ones(
                            q_squeezed.shape[0], s_full,
                            device=q_squeezed.device, dtype=torch.uint8,
                        )
                        attn_output, _, _ = rotatek_decode_fused(
                            q_full=q_squeezed,
                            R_partial=R_partial,
                            delta_mu=delta_mu,
                            k_full=k_full,
                            v_full=v_full,
                            mask_full=mask_full,
                            k_sparse=key_pruned,
                            v_sparse=v_vision,
                            num_kv_groups=self.num_key_value_groups,
                        )
                        attn_output = attn_output.unsqueeze(1).reshape(bsz, q_len, -1)
                        k_for_compute = None
                    else:
                        raise NotImplementedError(
                            f"channel_method={channel_method!r} not supported."
                        )
        else:
            k_for_compute = key_states
            v_for_compute = value_states

        if attn_output is None:
            kv_seq_len = k_for_compute.shape[-2]
            k_rep = repeat_kv(k_for_compute, self.num_key_value_groups)
            v_rep = repeat_kv(v_for_compute, self.num_key_value_groups)

            if not output_attentions:
                # FA2 — same speed as SDPA on GPU+bf16 but with looser
                # shape contract: 2D attention_mask, intrinsic causal mask,
                # native GQA (no repeat_kv). Avoids the SDPA strict 4D
                # mask checks that surface unrelated wrapper-state bugs
                # (mask length vs cache length mismatch on edge-case inputs).
                from transformers.modeling_flash_attention_utils import (
                    _flash_attention_forward,
                )
                # FA2 expects [B, S, H, D]; pass un-repeated K/V so FA2's
                # GQA path handles the head broadcast natively.
                attn_output = _flash_attention_forward(
                    query_states.transpose(1, 2),
                    k_for_compute.transpose(1, 2),
                    v_for_compute.transpose(1, 2),
                    attention_mask if (attention_mask is not None and attention_mask.dim() == 2) else None,
                    q_len,
                    is_causal=self.is_causal,
                )
                attn_output = attn_output.contiguous().reshape(bsz, q_len, -1)
                attn_weights = None
            else:
                # Eager — required only for prefill layer K-1, where FastV
                # reads attn_weights to pick the top-k vision tokens kept at K.
                attn_weights = torch.matmul(query_states, k_rep.transpose(2, 3)) / math.sqrt(self.head_dim)
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask[:, :, :, :kv_seq_len]
                if query_states.dtype == torch.float16:
                    attn_weights = torch.where(
                        torch.isinf(attn_weights),
                        torch.zeros_like(attn_weights),
                        attn_weights,
                    )
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                    query_states.dtype
                )
                attn_output = torch.matmul(attn_weights, v_rep)
                attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        else:
            attn_weights = None

        attn_output = self.o_proj(attn_output)
        # 4-tuple: matches Qwen2_5_VLDecoderLayer's unpacking. The 4th
        # element (`original_hidden_states`) is unused on this path.
        return attn_output, attn_weights, past_key_value, None


# ------------------------------------------------------------------------
# Decoder layer that uses FastV attention. We bypass Qwen2_5_VLDecoderLayer's
# attention construction (which goes through QWEN2_5_VL_ATTENTION_CLASSES
# keyed by config._attn_implementation) and substitute our class directly,
# avoiding a wasted Qwen2_5_VLAttention allocation per layer.
# ------------------------------------------------------------------------
class Qwen2_5_VLFastVDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Qwen2_5_VLFastVAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


# ------------------------------------------------------------------------
# Qwen2.5-VL LM with FastV-aware forward.
# Per-forward state (set externally by Qwen2_5_VLFastVForConditionalGeneration):
#   self.fast_v_sys_length         : int  — start of vision span in input_ids
#   self.fast_v_image_token_length : int  — number of visual tokens
#
# Config knobs (read via getattr):
#   use_fast_v            : bool
#   fast_v_agg_layer      : int  — K, must be > 0 in inplace mode
#   fast_v_keep_ratio     : Optional[float]  — preferred (anyres-friendly)
#   fast_v_attention_rank : Optional[int]    — fallback (legacy abs count)
#   fast_v_inplace        : bool — True = slice hidden_states (KV memory ↓)
#   channel_ratio         : float — ThinK channel-pruning ratio (0 disables)
#   channel_method        : str  — "think" / "spark" / "rotatek"
# ------------------------------------------------------------------------
class Qwen2_5_VLFastVModel(Qwen2_5_VLPreTrainedModel):
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)
        # Stamp BOTH visionzip channel-pruning defaults AND FastV defaults,
        # so init_visionzip() inside Qwen2_5_VLFastVAttention sees every
        # field it needs (`prompt_seqlen`, `query_seqlen`, etc.).
        _stamp_channel_defaults(config)
        _stamp_fastv_defaults(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2_5_VLFastVDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self._attn_implementation = "eager"
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Layer-adaptive budget — kept for compatibility; FastV does not use it.
        self.channel_ratio_high = None
        self.channel_ratio_low = None
        self._layer_budget_by_sparsity = {}

        # Per-forward FastV bounds, set by the wrapper.
        self.fast_v_sys_length: Optional[int] = None
        self.fast_v_image_token_length: Optional[int] = None

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _resolve_attention_rank(self, image_token_length: int) -> int:
        keep_ratio = getattr(self.config, "fast_v_keep_ratio", None)
        if keep_ratio is not None:
            return max(1, int(round(float(keep_ratio) * image_token_length)))
        rank = getattr(self.config, "fast_v_attention_rank", None)
        if rank is None:
            raise ValueError(
                "FastV is enabled but neither fast_v_keep_ratio nor "
                "fast_v_attention_rank is set on config."
            )
        return min(int(rank), image_token_length)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # `output_attentions` is set per-layer inside the loop below so only
        # prefill layer K-1 runs eager (the rest go through SDPA / FA-2).
        # Keep this assignment as a no-op so the variable is initialised; the
        # model itself never returns attention tensors.
        output_attentions = True
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        bsz, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device

        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            elif not isinstance(past_key_values, DynamicCache) and len(past_key_values) == 0:
                past_key_values = DynamicCache()

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_len, device=device)

        # Defensive: HF generate sometimes hands us an attention_mask that
        # is shorter than the actual cache length (observed on samples
        # where the unified-cache path stamps `_seen_tokens` differently
        # from how generate tracks state). Pad with ones (= attend) so the
        # downstream 4D causal-mask construction matches the K/V seqlen
        # the layers will see.
        if attention_mask is not None and attention_mask.dim() == 2:
            expected_kv_len = past_seen_tokens + seq_len
            if attention_mask.shape[-1] < expected_kv_len:
                pad_len = expected_kv_len - attention_mask.shape[-1]
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones(
                            (attention_mask.shape[0], pad_len)
                        ),
                    ],
                    dim=-1,
                )

        # mrope: position_ids is [3, B, S]; the wrapper computes it from
        # get_rope_index() at prefill and reuses with rope_deltas at decode.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        is_prefill = seq_len > 1
        use_fast_v_cfg = bool(getattr(self.config, "use_fast_v", False))
        agg_layer = int(getattr(self.config, "fast_v_agg_layer", 0) or 0)
        fastv_inplace = bool(getattr(self.config, "fast_v_inplace", False))

        sys_length = None
        image_token_length = 0
        if (
            use_fast_v_cfg
            and is_prefill
            and self.fast_v_sys_length is not None
            and self.fast_v_image_token_length is not None
        ):
            sys_length = int(self.fast_v_sys_length)
            image_token_length = int(self.fast_v_image_token_length)

        do_fastv_prefill = (
            use_fast_v_cfg
            and is_prefill
            and image_token_length > 0
            and sys_length is not None
            and agg_layer > 0
        )
        if fastv_inplace and use_fast_v_cfg and is_prefill:
            assert agg_layer > 0, "FastV inplace requires fast_v_agg_layer > 0"

        attention_rank = (
            self._resolve_attention_rank(image_token_length) if do_fastv_prefill else 0
        )

        causal_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (bsz, seq_len), inputs_embeds, past_seen_tokens
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        layer_attention_mask = causal_mask
        layer_position_ids = position_ids
        layer_position_embeddings = position_embeddings
        layer_cache_position = cache_position
        last_attn_weights = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            # ---------- FastV inplace slice at boundary K ---------------
            if do_fastv_prefill and fastv_inplace and layer_idx == agg_layer:
                # last_attn_weights: [B, H, Q, K] from layer K-1.
                # Faithful to original FastV: head-mean → last query row →
                # image-column slice → topk.
                avg_heads = last_attn_weights.mean(dim=1)[0]            # [Q, K]
                last_row = avg_heads[-1]                                # [K]
                img_scores = last_row[sys_length: sys_length + image_token_length]
                top_idx = img_scores.topk(attention_rank).indices + sys_length

                keep_indexs = torch.cat([
                    torch.arange(sys_length, device=device),
                    top_idx,
                    torch.arange(sys_length + image_token_length, seq_len, device=device),
                ]).sort().values

                hidden_states = hidden_states[:, keep_indexs, :]
                # mrope: slice the [3, B, S] tensor along the seq dim.
                layer_position_ids = position_ids[:, :, keep_indexs]
                layer_position_embeddings = self.rotary_emb(hidden_states, layer_position_ids)
                # Layers >= K start with empty cache; rebuild causal mask for
                # the shortened sequence with past=0.
                new_seq_len = keep_indexs.shape[0]
                layer_attention_mask = _prepare_4d_causal_attention_mask(
                    None, (bsz, new_seq_len), hidden_states, 0
                )
                layer_cache_position = None  # let the cache write fresh positions

            # Only prefill layer K-1 needs attn_weights (for FastV's top-k
            # decision at layer K). All other layers and every decode step go
            # through SDPA / FlashAttention-2 for speed and bf16 stability.
            need_attn = (
                do_fastv_prefill and fastv_inplace and layer_idx == agg_layer - 1
            )
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_attention_mask,
                position_ids=layer_position_ids,
                past_key_value=past_key_values,
                output_attentions=need_attn,
                use_cache=use_cache,
                cache_position=layer_cache_position,
                position_embeddings=layer_position_embeddings,
            )
            hidden_states = layer_outputs[0]
            if need_attn:
                last_attn_weights = layer_outputs[1]

        hidden_states = self.norm(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, None, None] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=None,
            attentions=None,
        )


# ------------------------------------------------------------------------
# Multimodal wrapper. Inherits from the visionzip
# `Qwen2_5_VLForConditionalGeneration` to reuse: vision tower, LM head,
# rope_deltas plumbing, calibration loaders, and the prefill/decode forward.
# Two tweaks:
#   1. self.model is Qwen2_5_VLFastVModel (achieved by monkey-patching the
#      `Qwen2_5_VLModel` reference in the visionzip module just before
#      super().__init__ runs — single allocation, no double init).
#   2. Per-forward stamping of `fast_v_sys_length` / `fast_v_image_token_length`
#      from input_ids before delegating to the parent forward.
#
# `dominant_ratio` / `contextual_ratio` are forced to 0 so the parent's
# vision-encoder VisionZip pruning path (`select_pixel`) is bypassed —
# FastV does pruning at the LM decoder side instead.
# ------------------------------------------------------------------------
class Qwen2_5_VLFastVForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config: Qwen2_5_VLConfig):
        # Disable visionzip's vision-encoder token pruning (FastV is decoder-side).
        config.dominant_ratio = 0.0
        config.contextual_ratio = 0.0
        _stamp_fastv_defaults(config)
        # FastV needs attn_weights → eager only.
        config._attn_implementation = "eager"

        # Patch the `Qwen2_5_VLModel` symbol the parent's __init__ resolves
        # at call time, so it constructs a Qwen2_5_VLFastVModel instead.
        # This avoids a wasted full-LM allocation that would otherwise
        # peak memory at 2x during init.
        saved_model_cls = _vz_mod.Qwen2_5_VLModel
        _vz_mod.Qwen2_5_VLModel = Qwen2_5_VLFastVModel
        try:
            super().__init__(config)
        finally:
            _vz_mod.Qwen2_5_VLModel = saved_model_cls

    # ------------------------------------------------------------------
    # Forward override — only role is to compute & stamp dynamic vision-span
    # bounds onto the LM model before delegating to the visionzip parent
    # forward (which handles vision encoding, embed merging, mrope deltas,
    # and the LM call). Bounds are computed once at prefill from input_ids;
    # decode forwards have no input_ids → no re-stamping needed.
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ):
        # Reset FastV / channel-pruning per-sample state at every prefill
        # (q_len > 1). This guards against degenerate samples where the
        # video failed to decode entirely (`pixel_values_videos is None`):
        # without this, a stale `fast_v_image_token_length` from the prior
        # sample drives `topk(k)` past the current sequence length and
        # crashes with "selected index k out of range".
        if input_ids is not None and input_ids.shape[-1] > 1:
            self.model.fast_v_sys_length = None
            self.model.fast_v_image_token_length = None
            self.config.prompt_seqlen = 0
            self.config.query_seqlen = 0

        if input_ids is not None and (
            pixel_values is not None or pixel_values_videos is not None
        ):
            visual_token_mask = (
                (input_ids == self.config.image_token_id)
                | (input_ids == self.config.video_token_id)
            )
            if visual_token_mask.dim() == 2:
                visual_token_mask = visual_token_mask[0]
            idxs = torch.nonzero(visual_token_mask, as_tuple=True)[0]
            if idxs.numel() > 0:
                first = int(idxs[0].item())
                last = int(idxs[-1].item())
                self.model.fast_v_sys_length = first
                self.model.fast_v_image_token_length = last - first + 1
                # ThinK / SparK / RotateK read these to slice K into
                # [prompt | vision | text]. Without stamping, prompt cache
                # is empty (kv_prompt = K[:, :, :0, :]) and decode loses
                # access to the question, producing garbage-token loops.
                self.config.prompt_seqlen = first
                self.config.query_seqlen = input_ids.shape[-1] - last

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
        )
