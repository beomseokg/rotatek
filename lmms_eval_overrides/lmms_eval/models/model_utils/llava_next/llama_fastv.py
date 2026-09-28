# ------------------------------------------------------------------------
# LLaMA backbone with FastV (token-level pruning) hooks for LLaVA-NeXT.
# Optionally combined with VisionZip-style channel pruning (ThinK) for
# two-axis KV memory reduction:
#   * FastV    : at decoder layer K, slice hidden_states/position_ids by
#                top-attention vision tokens.  Layers >= K cache shorter KV.
#   * ThinK    : within each attention, prune K-channels per head and store
#                a compact (split prompt / pruned / text) cache.
#
# Faithful port of the original FastV inplace mode:
#   * At decoder layer K, score each visual token by the last query row of
#     layer K-1's head-averaged attention; keep top `attention_rank` tokens.
#   * Slice `hidden_states` / `position_ids` in place at layer K so layers
#     >= K cache shorter KV — actual KV memory reduction.
#
# Differences from the upstream FastV reference (modeling_llama_fastv...):
#   * Transformers 4.49 API (Cache object, position_embeddings kwarg,
#     _prepare_4d_causal_attention_mask).
#   * Vision span (sys_length, image_token_length) is dynamic per forward
#     instead of fixed constants — required for LLaVA-NeXT anyres geometry.
#     Bounds are stamped onto the model instance by
#     `LlavaNextFastVForConditionalGeneration` before each forward.
#   * `use_cache=True` is supported.  Layers < K cache full-length KV;
#     layers >= K cache pruned-length KV; eager attention slices the mask
#     per layer.
#
# Channel pruning integration:
#   * Inherits from `LlamaVisionZipAttention` to reuse all kv_cluster /
#     custom-DynamicCache machinery.  Only the attention-compute kernel is
#     swapped from FA2/Triton to eager (FastV needs attn_weights).
#   * ThinK / Spark / RotateK (truncated mode only) are integrated.
#     RotateK truncated decode uses visionzip's fused Triton kernel
#     `rotatek_decode_fused` which returns attn_output directly — eager
#     compute is skipped at that decode step.  RotateK `full` storage
#     mode is not exposed (set ROTATEK_STORAGE=truncated, the default).
# ------------------------------------------------------------------------

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaPreTrainedModel,
    LlamaForCausalLM,
    apply_rotary_pos_emb,
    repeat_kv,
)

from lmms_eval.models.model_utils.cache_utils import Cache, DynamicCache
from lmms_eval.models.model_utils.llava_next.llama_visionzip import (
    LlamaVisionZipAttention,
    LlamaVisionZipDecoderLayer,
    _stamp_channel_defaults,
    DEFAULT_CALIBRATION_RESULT_ROOT,
)


# ------------------------------------------------------------------------
# Eager attention with VisionZip-style channel-pruning cache. Inherits
# projections / kv_cluster setup from LlamaVisionZipAttention; only the
# attention-compute kernel is replaced (FA2/Triton → eager) so FastV at
# layer K can read attn_weights from layer K-1.
# ------------------------------------------------------------------------
class LlamaFastVAttention(LlamaVisionZipAttention):
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
        key_states = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            raise ValueError("LlamaFastVAttention requires position_embeddings.")
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # ------------------------------------------------------------------
        # Cache management — mirrors LlamaVisionZipAttention but only ThinK
        # is exercised through this eager path.  channel_ratio == 0 falls
        # back to the unified-storage path (no prompt/vision/text split).
        # ------------------------------------------------------------------
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

            attn_output = None  # set by rotatek-truncated fused kernel
            if q_len > 1:  # prefill
                if channel_ratio == 0.0:
                    past_key_value.store_unified(key_states, value_states, self.layer_idx)
                elif channel_method == "think":
                    kv_pruned, kv_prompt, kv_text, mask, value_states_compress = (
                        self.kv_cluster.update_think(
                            key_states, query_states, value_states, attention_mask,
                            num_key_value_groups=self.num_key_value_groups,
                            calibration_channel_importance=getattr(
                                self, "calibration_channel_importance", None
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
                                self, "calibration_rotation_R_partial", None
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

                # Prefill compute uses the ORIGINAL full K/V (channel pruning
                # only affects what's stored in cache for decode).
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
                        # Per-head bool keep mask + zero-fill recovery.
                        think_mask = past_key_value.think_mask[self.layer_idx]   # [B, H_kv, D]
                        bsz_r, h_kv, seq_r = key_pruned.shape[:3]
                        recovered_key_states = torch.zeros(
                            bsz_r, h_kv, seq_r, self.head_dim,
                            dtype=key_pruned.dtype, device=key_pruned.device,
                        )
                        mask_expanded = think_mask.unsqueeze(2).expand(-1, -1, seq_r, -1)
                        recovered_key_states[mask_expanded] = key_pruned.reshape(-1)
                        k_for_compute = torch.cat(
                            [key_prompt, recovered_key_states, text_key_states], dim=-2,
                        )

                    elif channel_method == "spark":
                        # Pre-fill full-D buffer with pruned_mean, scatter
                        # compact K into kept channel slots.
                        spark_mask = past_key_value.spark_mask[self.layer_idx]
                        pruned_mean = past_key_value.spark_pruned_mean[self.layer_idx]
                        bsz_r, h_kv, seq_r, _ = key_pruned.shape
                        recovered_key_states = pruned_mean.expand(
                            bsz_r, h_kv, seq_r, self.head_dim,
                        ).contiguous()
                        recovered_key_states[spark_mask] = key_pruned.reshape(-1)
                        k_for_compute = torch.cat(
                            [key_prompt, recovered_key_states, text_key_states], dim=-2,
                        )

                    elif channel_method == "rotatek":
                        # Truncated-mode only — assumes ROTATEK_STORAGE=truncated
                        # (default). key_pruned is [B, H_kv, S_v, D_keep].
                        # `rotatek_decode_fused` Triton kernel absorbs Q@R
                        # rotation + δμ bias into the inner softmax loop AND
                        # produces attn_output directly, so we skip the
                        # standard eager softmax path below.
                        if key_pruned.shape[-1] == self.head_dim:
                            raise RuntimeError(
                                "RotateK 'full' storage mode is not exposed in "
                                "LlamaFastVAttention. Set ROTATEK_STORAGE=truncated."
                            )
                        from methods.rotatek import rotatek_decode_fused

                        q_squeezed = query_states.squeeze(2)  # [B, H_q, D]
                        R_partial = past_key_value.rotatek_rotations[self.layer_idx]
                        delta_mu = past_key_value.rotatek_means[self.layer_idx]

                        s_prompt = key_prompt.shape[-2]
                        s_vision = key_pruned.shape[-2]
                        v_prompt = value_states_full[:, :, :s_prompt, :]
                        v_vision = value_states_full[:, :, s_prompt: s_prompt + s_vision, :]
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
                        # attn_output: [B, H_q, D] from fused kernel
                        # — reshape to [B, q_len=1, H*D] for o_proj.
                        attn_output = attn_output.unsqueeze(1).reshape(
                            bsz, q_len, self.hidden_size
                        )
                        k_for_compute = None  # unused — fused path already produced attn_output
                    else:
                        raise NotImplementedError(
                            f"channel_method={channel_method!r} not supported."
                        )
        else:
            k_for_compute = key_states
            v_for_compute = value_states
            attn_output = None

        # Per-layer dispatch via the output_attentions flag.
        # LlamaFastVModel.forward sets output_attentions=True only on the
        # layer K-1 prefill (eager path needed for attn_weights extraction);
        # every other layer and all decode steps run with output_attentions=
        # False, which routes to FA2 (faster and bf16-stable).
        # To force eager on every layer (e.g., for debugging), replace
        # `_use_eager_attn = output_attentions` with
        # `_use_eager_attn = bool(getattr(self.config, "use_fast_v", False))`.
        _use_eager_attn = bool(output_attentions)
        if attn_output is None:
            if _use_eager_attn:
                # ===== Eager path (FastV scoring needs attn_weights) =====
                kv_seq_len = k_for_compute.shape[-2]
                k_rep = repeat_kv(k_for_compute, self.num_key_value_groups)
                v_rep = repeat_kv(v_for_compute, self.num_key_value_groups)

                attn_weights = torch.matmul(query_states, k_rep.transpose(2, 3)) * self.scaling
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask[:, :, :, :kv_seq_len]
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                    query_states.dtype
                )
                attn_output = torch.matmul(attn_weights, v_rep)
                attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
            else:
                # ===== FA2 path (FastV scoring inactive — matches VisionZip kernel) =====
                from transformers.modeling_flash_attention_utils import _flash_attention_forward
                dropout_rate = 0.0 if not self.training else self.attention_dropout

                # FA2 expects [B, S, H, D] layout; we transpose from [B, H, S, D].
                # GQA repeat_kv is handled inside FA2, so we skip it here.
                q_fa = query_states.transpose(1, 2)
                k_fa = k_for_compute.transpose(1, 2)
                v_fa = v_for_compute.transpose(1, 2)
                # LlamaFastVModel.forward converts attention_mask via
                # _prepare_4d_causal_attention_mask into a 4D causal mask.
                # FA2 only accepts a 2D padding mask or None; passing a 4D
                # mask through triggers an index out-of-bounds inside
                # _upad_input (CUDA assert in ScatterGatherKernel.cu).
                # Since lmms-eval uses batch_size=1 with no padding, we
                # pass None and rely on is_causal=True for causality.
                fa_attention_mask = (
                    attention_mask
                    if attention_mask is not None and attention_mask.dim() == 2
                    else None
                )
                attn_output = _flash_attention_forward(
                    q_fa, k_fa, v_fa,
                    fa_attention_mask, q_len,
                    dropout=dropout_rate,
                    sliding_window=None,
                    is_causal=self.is_causal,
                    use_top_left_mask=self._flash_attn_uses_top_left_mask,
                )
                attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
                attn_weights = None
        else:
            # rotatek-truncated decode handled by fused kernel; no attn_weights.
            attn_weights = None

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value


class LlamaFastVDecoderLayer(LlamaVisionZipDecoderLayer):
    """Same forward as upstream — just swaps in `LlamaFastVAttention`."""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = LlamaFastVAttention(config=config, layer_idx=layer_idx)


# ------------------------------------------------------------------------
# Model: implements FastV decision logic + channel-pruning-aware cache.
#
# Per-forward state set externally (by LlavaNextFastVForConditionalGeneration):
#   self.fast_v_sys_length         : int   — start of vision span in merged seq
#   self.fast_v_pre_merge_text_len : int   — input_ids length before merge
#   self.fast_v_num_placeholders   : int   — count of <image> placeholders
#
# IMG length is derived inside forward:
#   IMG = inputs_embeds.shape[1] - pre_merge_text_len + num_placeholders
#
# Config knobs (read via getattr):
#   use_fast_v            : bool
#   fast_v_agg_layer      : int   — K, must be > 0 in inplace mode
#   fast_v_keep_ratio     : Optional[float]  — preferred (anyres-friendly)
#   fast_v_attention_rank : Optional[int]    — fallback (legacy abs count)
#   fast_v_inplace        : bool  — True = slice hidden_states (KV memory ↓)
#   channel_ratio         : float — ThinK channel-pruning ratio (0 disables)
#   channel_method        : str   — "think" only (validated). spark/rotatek
#                                   not exercised through this forward.
# ------------------------------------------------------------------------
class LlamaFastVModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        # Stamp BOTH FastV defaults AND visionzip channel-pruning defaults
        # so init_visionzip() inside LlamaFastVAttention has every field it
        # needs (`prompt_seqlen`, `query_seqlen`, `dominant_ratio`, etc.).
        _stamp_channel_defaults(config)
        _stamp_fastv_defaults(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaFastVDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Per-forward dynamic bounds — stamped by the multimodal wrapper at
        # prefill. Persisted across decode steps (the cache already holds the
        # vision span; new tokens append to it).
        self.fast_v_sys_length: Optional[int] = None
        self.fast_v_pre_merge_text_len: Optional[int] = None
        self.fast_v_num_placeholders: Optional[int] = None

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
        # We do NOT force output_attentions at the module level. The per-layer
        # decision is made inside the layer loop below (same pattern as Qwen FastV):
        #   need_attn = do_fastv_prefill and fastv_inplace and layer_idx == agg_layer - 1
        # which is True only on the layer K-1 prefill (eager path needed for
        # attn_weights extraction) and False elsewhere (FA2 dispatch).
        # To force eager on every layer (e.g., for debugging), replace the
        # following line with `output_attentions = True`.
        output_attentions = (
            output_attentions if output_attentions is not None
            else self.config.output_attentions
        )
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
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # ---------------------------------------------------------------
        # FastV bounds + parameters
        # ---------------------------------------------------------------
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
            and self.fast_v_pre_merge_text_len is not None
            and self.fast_v_num_placeholders is not None
        ):
            sys_length = int(self.fast_v_sys_length)
            image_token_length = int(
                seq_len - self.fast_v_pre_merge_text_len + self.fast_v_num_placeholders
            )

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

        # We always build the 4D mask. Only layer K-1 runs eager and that
        # layer needs the 4D mask to apply -inf padding to attn_weights.
        # Layers that take the FA2 path drop the mask to None inside
        # LlamaFastVAttention.forward via the `attention_mask.dim() != 2`
        # check, delegating causality to is_causal=True (functionally
        # equivalent under lmms-eval's batch=1, no-padding regime).
        causal_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (bsz, seq_len), inputs_embeds, past_seen_tokens
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        layer_attention_mask = causal_mask
        layer_position_ids = position_ids
        layer_position_embeddings = position_embeddings
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
                layer_position_ids = keep_indexs.unsqueeze(0)
                # Recompute RoPE for the kept positions.
                layer_position_embeddings = self.rotary_emb(hidden_states, layer_position_ids)
                # Rebuild causal mask for the shortened sequence; cache for
                # layers >= K starts empty (from layer K's cache write).
                new_seq_len = keep_indexs.shape[0]
                layer_attention_mask = _prepare_4d_causal_attention_mask(
                    None, (bsz, new_seq_len), hidden_states, 0
                )

            # Per-layer dispatch (same pattern as Qwen FastV):
            # output_attentions=True is set only on the prefill layer K-1
            # (eager path extracts attn_weights, which the next layer K uses
            # for FastV topk selection); for every other layer and all decode
            # steps it is False, which auto-routes to the FA2 branch inside
            # LlamaFastVAttention.forward.
            # If the caller explicitly passes output_attentions=True to
            # model.forward, every layer falls back to eager; we preserve
            # that path via the `or output_attentions` clause below.
            need_attn = (
                do_fastv_prefill and fastv_inplace and layer_idx == agg_layer - 1
            ) or bool(output_attentions)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_attention_mask,
                position_ids=layer_position_ids,
                past_key_value=past_key_values,
                output_attentions=need_attn,
                use_cache=use_cache,
                cache_position=cache_position if not (do_fastv_prefill and fastv_inplace and layer_idx >= agg_layer) else None,
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


class LlamaFastVForCausalLM(LlamaForCausalLM):
    """LlamaForCausalLM with FastV-patched LlamaModel.

    Checkpoint keys are identical to upstream LLaMA (only forward logic
    changes), so HF `from_pretrained` loads vanilla weights as-is.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LlamaFastVModel(config)
        self.post_init()


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
