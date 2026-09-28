# ------------------------------------------------------------------------
# Modified from transformers 4.49.0 LLaMA implementation.
# Mirrors the VisionZip/channel-pruning hooks used in
# `lmms_eval.models.model_utils.qwen.qwen2_5vl_visionzip` so that the same
# `channel_method`, `calibration_mode`, `custom_kernel`, reconstruction and
# layer-adaptive budget knobs work for LLaMA-backed LLaVA-NeXT models.
# ------------------------------------------------------------------------

import math
import os
import time as _time
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class _WallMark:
    """Wall-clock timestamp with the cuda.Event-compatible `.record()` /
    `.elapsed_time()` API.

    Used when `_decode_profile_mode == "wall"`: each `.record()` calls
    `torch.cuda.synchronize()` then captures `time.perf_counter()`. Lets the
    same instrumentation code path produce wall-clock measurements per stage
    without rewriting every recording site.
    """
    __slots__ = ("value",)

    def __init__(self):
        self.value: Optional[float] = None

    def record(self) -> None:
        torch.cuda.synchronize()
        self.value = _time.perf_counter()

    def elapsed_time(self, other: "_WallMark") -> float:
        if self.value is None or other.value is None:
            return 0.0
        return (other.value - self.value) * 1000.0

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaMLP,
    LlamaRotaryEmbedding,
    LlamaPreTrainedModel,
    LlamaForCausalLM,
    apply_rotary_pos_emb,
    repeat_kv,
)

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
else:
    _flash_attention_forward = None

from lmms_eval.models.model_utils.kv_pruning_utils import init_visionzip
from lmms_eval.models.model_utils.cache_utils import Cache, DynamicCache

# Reuse the same Triton decode kernel + supplementary-matrix helpers that the
# Qwen implementation uses. The kernels are architecture-agnostic — they
# operate on `[B, H, S, D]` GQA tensors, which matches LLaMA-3.
from lmms_eval.models.model_utils.qwen.qwen2_5vl_visionzip import (
    run_custom_decode_kernel,
    run_dense_decode_kernel,
    load_channel_importance_calibration,
    _require_calibration_dominant_ratio,
)
# RotateK rotation-matrix loader lives with the InternVL adapter; reuse it
# (function is model-agnostic — just reads per-layer .pt files and returns
# top-keep_count eigenvectors of accumulated K^T K).
from lmms_eval.models.model_utils.internvl.internvl2_5_visionzip import (
    load_rotation_matrix_calibration,
)

logger = logging.get_logger(__name__)

DEFAULT_CALIBRATION_RESULT_ROOT = (
    "./results/LLaVA_NeXT_Llama3_8B"
)


class LlamaVisionZipAttention(nn.Module):
    """LLaMA flash-attention forward with VisionZip + channel pruning hooks.

    Weights match `transformers.models.llama.modeling_llama.LlamaAttention`;
    only the forward pass is replaced so existing checkpoints load as-is.
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # Profile mode: "cuda" (default) uses torch.cuda.Event for low-overhead
        # GPU-side timing; "wall" syncs the device and reads time.perf_counter()
        # in `.record()`. Wall mode serializes the GPU pipeline (one sync per
        # stage boundary) but reports pure wall-clock per stage with no
        # event-resolution effects.
        _profile_mode = getattr(self.config, "_decode_profile_mode", "cuda")
        _use_wall_clock = _profile_mode == "wall"

        def _new_evt():
            if _use_wall_clock:
                return _WallMark()
            return torch.cuda.Event(enable_timing=True)

        def _elapsed_ms(start_evt, end_evt):
            if start_evt is None or end_evt is None:
                return 0.0
            if isinstance(start_evt, _WallMark):
                if start_evt.value is None or end_evt.value is None:
                    return 0.0
                return (end_evt.value - start_evt.value) * 1000.0
            return start_evt.elapsed_time(end_evt)

        # profiling event placeholders (mirrors qwen2_5vl_visionzip pattern)
        _evt_qkv_proj_rope_start = _evt_qkv_proj_rope_end = None
        _evt_qkv_proj_start = _evt_qkv_proj_end = None
        _evt_rope_start = _evt_rope_end = None
        _evt_cache_update_start = _evt_cache_update_end = None
        _evt_recovery_start = _evt_recovery_end = None
        _evt_cat_start = _evt_cat_end = None
        _evt_custom_kernel_start = _evt_custom_kernel_end = None
        _evt_transpose_start = _evt_transpose_end = None
        _evt_attn_start = _evt_attn_end = None
        _evt_attn_out_start = _evt_attn_out_end = None

        _profile_layers = getattr(self.config, "_decode_profile_layers", {14})
        _profile = (
            getattr(self.config, "_decode_profile", False)
            and self.layer_idx in _profile_layers
            and q_len == 1
        )
        _kernel_envelope_layers = getattr(
            self.config, "_kernel_envelope_profile_layers", _profile_layers
        )
        _kernel_envelope_profile = (
            getattr(self.config, "_kernel_envelope_profile", False)
            and self.layer_idx in _kernel_envelope_layers
            and q_len == 1
        )
        _custom_kernel_event_profile = _profile or _kernel_envelope_profile

        def _record_kernel_envelope_event():
            if (
                not _kernel_envelope_profile
                or _evt_custom_kernel_start is None
                or _evt_custom_kernel_end is None
            ):
                return
            _events_by_layer = getattr(
                self.config, "_kernel_envelope_events_by_layer", None
            )
            if _events_by_layer is None:
                _events_by_layer = {}
                self.config._kernel_envelope_events_by_layer = _events_by_layer
            _events_by_layer.setdefault(self.layer_idx, []).append(
                (_evt_custom_kernel_start, _evt_custom_kernel_end)
            )

        channel_method = getattr(self.config, "channel_method", "think")
        channel_ratio = getattr(self.config, "channel_ratio", 0.0)

        if q_len > 1:
            init_visionzip(self)

        # qkv proj + view/transpose
        if _profile:
            _evt_qkv_proj_start = _new_evt()
            _evt_qkv_proj_rope_start = _new_evt()
            _evt_qkv_proj_rope_end = _new_evt()
            _evt_qkv_proj_start.record()
            _evt_qkv_proj_rope_start.record()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states_no_rope = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if _profile:
            _evt_qkv_proj_end = _new_evt()
            _evt_qkv_proj_end.record()
            _evt_rope_start = _new_evt()
            _evt_rope_start.record()

        if position_embeddings is None:
            raise ValueError(
                "LlamaVisionZipAttention requires position_embeddings; make sure the forked "
                "LlamaModel passes them through."
            )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states_no_rope, cos, sin
        )

        if _profile:
            _evt_rope_end = _new_evt()
            _evt_rope_end.record()
            _evt_qkv_proj_rope_end.record()

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

            if key_states.shape[-2] > 1:  # prefill
                if channel_ratio == 0.0:
                    # Unified-storage path: K/V stored as single tensors per
                    # layer, no prompt/vision/text split. Decode steps then
                    # use `update_unified` to append in-place, avoiding the
                    # 3-way cat the compressed path needs. Mirrors the Qwen
                    # adapter (qwen2_5vl_visionzip).
                    past_key_value.store_unified(
                        key_states, value_states, self.layer_idx,
                    )
                elif channel_method == "think":
                    kv_pruned, kv_prompt, kv_text, mask, value_states_compress = self.kv_cluster.update_think(
                        key_states,
                        query_states,
                        value_states,
                        attention_mask,
                        num_key_value_groups=self.num_key_value_groups,
                        calibration_channel_importance=getattr(self, "calibration_channel_importance", None),
                    )
                    past_key_value.store_pruned(kv_pruned, kv_prompt, kv_text, mask, value_states_compress, self.layer_idx, cache_kwargs)
                    past_key_value.think_mask.append(self.kv_cluster.current_think_mask)
                elif channel_method == "spark":
                    kv_pruned, kv_prompt, kv_text, mask, value_states_compress = self.kv_cluster.update_spark(
                        key_states,
                        query_states,
                        value_states,
                        attention_mask,
                        num_key_value_groups=self.num_key_value_groups,
                    )
                    past_key_value.store_pruned(kv_pruned, kv_prompt, kv_text, mask, value_states_compress, self.layer_idx, cache_kwargs)
                    past_key_value.spark_mask.append(self.kv_cluster.current_spark_mask)
                    past_key_value.spark_pruned_mean.append(self.kv_cluster.current_spark_pruned_mean)
                elif channel_method == "rotatek":
                    kv_pruned, kv_prompt, kv_text, mask, value_states_compress = (
                        self.kv_cluster.update_rotatek(
                            key_states,
                            query_states,
                            value_states,
                            attention_mask,
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
                    # In `truncated` storage mode (ROTATEK_STORAGE=truncated),
                    # kv_pruned is [S_v, D_keep] and decode needs R_partial to
                    # rotate Q. Stash here; in `full` mode this is just an
                    # extra (cheap) reference that nobody reads.
                    past_key_value.rotatek_rotations.append(
                        self.kv_cluster.current_rotatek_R_partial
                    )
                    # δμ = μ @ (I - P) per kv-head [B, H_kv, D]; decode-time
                    # bias on vision logits that recovers the full-mode
                    # Q @ (I - P) @ μ^T shift missing from Q @ P @ K^T.
                    # None in full mode.
                    past_key_value.rotatek_means.append(
                        self.kv_cluster.current_rotatek_delta_mu
                    )
                else:
                    raise ValueError(f"Unsupported channel_method: {channel_method}")

            else:  # decoding
                if channel_ratio == 0.0:
                    # Unified-storage path: K/V already aligned by
                    # `update_unified`; skips the prompt/vision/text cat
                    # the compressed methods need.
                    if _profile:
                        _evt_cache_update_start = _new_evt()
                        _evt_cache_update_end = _new_evt()
                        _evt_cache_update_start.record()

                    key_states, value_states = past_key_value.update_unified(
                        key_states, value_states, self.layer_idx,
                    )

                    if _profile:
                        _evt_cache_update_end.record()

                    use_dense_triton = (
                        getattr(self.config, "decode_attention_backend", "fa2") == "triton"
                    )
                    if use_dense_triton:
                        if _custom_kernel_event_profile:
                            _evt_custom_kernel_start = _new_evt()
                            _evt_custom_kernel_end = _new_evt()
                            _evt_custom_kernel_start.record()
                        run_custom_decode_kernel._profile_kernel = _profile
                        run_custom_decode_kernel._profile_layer_idx = (
                            self.layer_idx if _profile else None
                        )
                        attn_output = run_dense_decode_kernel(
                            query_states=query_states,
                            key_states=key_states,
                            value_states=value_states,
                            num_key_value_groups=self.num_key_value_groups,
                            attention_mask=attention_mask,
                        )
                        if _custom_kernel_event_profile:
                            _evt_custom_kernel_end.record()
                            _record_kernel_envelope_event()

                else:
                    # Split-storage path (channel-pruning methods).
                    if _profile:
                        _evt_cache_update_start = _new_evt()
                        _evt_cache_update_end = _new_evt()
                        _evt_cache_update_start.record()

                    text_key_states, value_states, key_pruned, key_prompt, mask = past_key_value.update(
                        key_states, value_states, self.layer_idx, cache_kwargs
                    )

                    if _profile:
                        _evt_cache_update_end.record()

                    if channel_method == "think":
                        # Paper-faithful ThinK recovery: per-head bool keep mask
                        # broadcast over the vision span, used as a boolean index
                        # into a pre-zeroed buffer (matches InternVL adapter).
                        if _profile:
                            _evt_recovery_start = _new_evt()
                            _evt_recovery_end = _new_evt()
                            _evt_cat_start = _new_evt()
                            _evt_cat_end = _new_evt()
                            _evt_recovery_start.record()

                        think_mask = past_key_value.think_mask[self.layer_idx]   # [B, H_kv, D]
                        bsz_r, h_kv, seq_r = key_pruned.shape[:3]
                        recovered_key_states = torch.zeros(
                            bsz_r, h_kv, seq_r, self.head_dim,
                            dtype=key_pruned.dtype, device=key_pruned.device,
                        )
                        mask_expanded = think_mask.unsqueeze(2).expand(-1, -1, seq_r, -1)
                        recovered_key_states[mask_expanded] = key_pruned.reshape(-1)

                        if _profile:
                            _evt_recovery_end.record()
                            _evt_cat_start.record()

                        key_states = torch.cat([key_prompt, recovered_key_states, text_key_states], dim=-2)

                        if _profile:
                            _evt_cat_end.record()

                        if getattr(self.config, "decode_attention_backend", "fa2") == "triton":
                            if _custom_kernel_event_profile:
                                _evt_custom_kernel_start = _new_evt()
                                _evt_custom_kernel_end = _new_evt()
                                _evt_custom_kernel_start.record()
                            run_custom_decode_kernel._profile_kernel = _profile
                            run_custom_decode_kernel._profile_layer_idx = (
                                self.layer_idx if _profile else None
                            )
                            attn_output = run_dense_decode_kernel(
                                query_states=query_states,
                                key_states=key_states,
                                value_states=value_states,
                                num_key_value_groups=self.num_key_value_groups,
                                attention_mask=attention_mask,
                            )
                            if _custom_kernel_event_profile:
                                _evt_custom_kernel_end.record()
                                _record_kernel_envelope_event()

                    elif channel_method == "spark":
                        if _profile:
                            _evt_recovery_start = _new_evt()
                            _evt_recovery_end = _new_evt()
                            _evt_cat_start = _new_evt()
                            _evt_cat_end = _new_evt()
                            _evt_recovery_start.record()

                        spark_mask = past_key_value.spark_mask[self.layer_idx]
                        pruned_mean = past_key_value.spark_pruned_mean[self.layer_idx]
                        bsz_r, h_kv, seq_r, _ = key_pruned.shape
                        recovered_key_states = pruned_mean.expand(bsz_r, h_kv, seq_r, self.head_dim).contiguous()
                        recovered_key_states[spark_mask] = key_pruned.reshape(-1)

                        if _profile:
                            _evt_recovery_end.record()
                            _evt_cat_start.record()

                        key_states = torch.cat([key_prompt, recovered_key_states, text_key_states], dim=-2)

                        if _profile:
                            _evt_cat_end.record()

                        if getattr(self.config, "decode_attention_backend", "fa2") == "triton":
                            if _custom_kernel_event_profile:
                                _evt_custom_kernel_start = _new_evt()
                                _evt_custom_kernel_end = _new_evt()
                                _evt_custom_kernel_start.record()
                            run_custom_decode_kernel._profile_kernel = _profile
                            run_custom_decode_kernel._profile_layer_idx = (
                                self.layer_idx if _profile else None
                            )
                            attn_output = run_dense_decode_kernel(
                                query_states=query_states,
                                key_states=key_states,
                                value_states=value_states,
                                num_key_value_groups=self.num_key_value_groups,
                                attention_mask=attention_mask,
                            )
                            if _custom_kernel_event_profile:
                                _evt_custom_kernel_end.record()
                                _record_kernel_envelope_event()

                    elif channel_method == "rotatek":
                        # Two storage modes (selected at prefill via ROTATEK_STORAGE):
                        #   - full      : key_pruned is [S_v, D] (rank-D_keep
                        #                 projection in original basis) → standard
                        #                 cat + FA2 path.
                        #   - truncated : key_pruned is [S_v, D_keep] (rotated
                        #                 truncated form) → fused Triton kernel.
                        if key_pruned.shape[-1] == self.head_dim:
                            if _profile:
                                _evt_cat_start = _new_evt()
                                _evt_cat_end = _new_evt()
                                _evt_cat_start.record()
                            key_states = torch.cat(
                                [key_prompt, key_pruned, text_key_states], dim=-2
                            )
                            if _profile:
                                _evt_cat_end.record()
                        else:
                            # truncated mode: fused phase-1 Triton kernel that
                            # absorbs the Q@R rotation + δμ bias into the inner
                            # softmax loop (matches InternVL adapter).
                            from rotatek import rotatek_decode_fused

                            if _custom_kernel_event_profile:
                                _evt_custom_kernel_start = _new_evt()
                                _evt_custom_kernel_end = _new_evt()
                                _evt_custom_kernel_start.record()

                            q_squeezed = query_states.squeeze(2)  # [B, H_q, D]
                            bsz_q, h_q, head_dim_q = q_squeezed.shape

                            R_partial = past_key_value.rotatek_rotations[self.layer_idx]
                            delta_mu = past_key_value.rotatek_means[self.layer_idx]

                            # Split V into vision / non-vision segments.
                            s_prompt = key_prompt.shape[-2]
                            s_vision = key_pruned.shape[-2]
                            v_prompt = value_states[:, :, :s_prompt, :]
                            v_vision = value_states[:, :, s_prompt:s_prompt + s_vision, :]
                            v_text = value_states[:, :, s_prompt + s_vision:, :]

                            k_full = torch.cat([key_prompt, text_key_states], dim=-2)
                            v_full = torch.cat([v_prompt, v_text], dim=-2)
                            s_full = k_full.shape[-2]
                            mask_full = torch.ones(
                                bsz_q, s_full,
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

                            if _custom_kernel_event_profile:
                                _evt_custom_kernel_end.record()
                                _record_kernel_envelope_event()

                    else:
                        raise ValueError(f"Unsupported channel_method for decode: {channel_method}")

        dropout_rate = 0.0 if not self.training else self.attention_dropout

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        _custom_decode_kernel_handled = (
            past_key_value is not None
            and q_len == 1
            and (
                (
                    getattr(self.config, "decode_attention_backend", "fa2") == "triton"
                    and (
                        channel_ratio == 0.0
                        or channel_method in ("think", "spark")
                    )
                )
                or (
                    # RotateK truncated mode: rotatek_decode_fused has
                    # produced attn_output above; skip the FA2 fallback.
                    # Gated on channel_ratio != 0.0 because the unified path
                    # never assigns `key_pruned`.
                    channel_ratio != 0.0
                    and channel_method == "rotatek"
                    and key_pruned.shape[-1] != self.head_dim
                )
            )
        )

        if not _custom_decode_kernel_handled:
            if _profile:
                _evt_transpose_start = _new_evt()
                _evt_transpose_end = _new_evt()
                _evt_attn_start = _new_evt()
                _evt_attn_end = _new_evt()
                _evt_transpose_start.record()

            q_fa = query_states.transpose(1, 2)
            k_fa = key_states.transpose(1, 2)
            v_fa = value_states.transpose(1, 2)

            if _profile:
                _evt_transpose_end.record()
                _evt_attn_start.record()

            attn_output = _flash_attention_forward(
                q_fa,
                k_fa,
                v_fa,
                attention_mask,
                q_len,
                dropout=dropout_rate,
                sliding_window=None,
                is_causal=self.is_causal,
                use_top_left_mask=self._flash_attn_uses_top_left_mask,
            )

            if _profile:
                _evt_attn_end.record()

        if _profile:
            _evt_attn_out_start = _new_evt()
            _evt_attn_out_end = _new_evt()
            _evt_attn_out_start.record()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if _profile:
            _evt_attn_out_end.record()

        # Accumulate per-layer timings. Mirrors qwen2_5vl_visionzip pattern so
        # downstream benchmark scripts can read the same key set.
        if _profile:
            torch.cuda.synchronize()

            _all_timings = getattr(self.config, "_decode_timings_by_layer", None)
            if _all_timings is None:
                _all_timings = {}
                self.config._decode_timings_by_layer = _all_timings

            _timings = _all_timings.get(self.layer_idx)
            if _timings is None:
                _timings = {
                    "qkv_proj_rope_ms": [],
                    "qkv_proj_ms": [],
                    "rope_ms": [],
                    "cache_update_ms": [],
                    "cat_ms": [],
                    "recovery_ms": [],
                    "custom_decode_kernel_ms": [],
                    "transpose_ms": [],
                    "fa2_ms": [],
                    "attn_out_proj_ms": [],
                }
                _all_timings[self.layer_idx] = _timings

            # backward compat: keep layer 0 as _decode_timings
            if self.layer_idx == 0:
                self.config._decode_timings = _timings

            # Promote any pending kernel-internal events into _kernel_timings_by_layer.
            _kernel_timings_by_layer = getattr(
                run_custom_decode_kernel, "_kernel_timings_by_layer", None
            )
            _kernel_timings = (
                _kernel_timings_by_layer.get(self.layer_idx)
                if _kernel_timings_by_layer is not None
                else None
            )
            if _kernel_timings is not None:
                _pending_dense_events = _kernel_timings.get("_pending_dense_events", [])
                if _pending_dense_events:
                    _kernel_timings.setdefault("triton_ms", [])
                    for _event_record in _pending_dense_events:
                        _kernel_timings["triton_ms"].append(
                            _event_record["triton_start"].elapsed_time(
                                _event_record["triton_end"]
                            )
                        )
                    _kernel_timings["_pending_dense_events"] = []

            _timings["qkv_proj_rope_ms"].append(
                _elapsed_ms(_evt_qkv_proj_rope_start, _evt_qkv_proj_rope_end)
            )
            _timings["qkv_proj_ms"].append(
                _elapsed_ms(_evt_qkv_proj_start, _evt_qkv_proj_end)
            )
            _timings["rope_ms"].append(_elapsed_ms(_evt_rope_start, _evt_rope_end))
            _timings["cache_update_ms"].append(
                _elapsed_ms(_evt_cache_update_start, _evt_cache_update_end)
            )
            _timings["recovery_ms"].append(
                _elapsed_ms(_evt_recovery_start, _evt_recovery_end)
            )
            _timings["cat_ms"].append(_elapsed_ms(_evt_cat_start, _evt_cat_end))
            _timings["custom_decode_kernel_ms"].append(
                _elapsed_ms(_evt_custom_kernel_start, _evt_custom_kernel_end)
            )
            _timings["transpose_ms"].append(
                _elapsed_ms(_evt_transpose_start, _evt_transpose_end)
            )
            _timings["fa2_ms"].append(_elapsed_ms(_evt_attn_start, _evt_attn_end))
            _timings["attn_out_proj_ms"].append(
                _elapsed_ms(_evt_attn_out_start, _evt_attn_out_end)
            )

        attn_weights = None
        return attn_output, attn_weights, past_key_value


class LlamaVisionZipDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = LlamaVisionZipAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
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


class LlamaVisionZipModel(LlamaPreTrainedModel):
    """LlamaModel with VisionZip decoder layers + channel-pruning friendly cache.

    Mirrors the Qwen2_5_VLModel hook points: forces a `DynamicCache` from
    `lmms_eval.models.model_utils.cache_utils` (not HF's vanilla one) so that
    `store_pruned` / `update` return the pruned/prompt/text split.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        # Ensure pruning-related defaults are stamped onto the config once.
        _stamp_channel_defaults(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaVisionZipDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        calibration_mode = str(getattr(config, "calibration_mode", "off")).strip().lower()
        calibration_task_spec = getattr(config, "offline_calibration_tasks", "channel_importance")
        valid_calibration_tasks = {"channel_importance", "modality_score", "supplementary_matrix", "attention_shift", "attention_kl", "all"}
        calibration_tasks = {
            item.strip().lower()
            for item in str(calibration_task_spec).split(",")
            if item.strip() and item.strip().lower() in valid_calibration_tasks
        }
        calibration_tasks_requiring_importance = {"supplementary_matrix", "attention_shift", "attention_kl", "all"}
        should_load_calibration_importance = calibration_mode == "use" or (
            calibration_mode == "collect"
            and any(task in calibration_tasks_requiring_importance for task in calibration_tasks)
        )
        if should_load_calibration_importance:
            calibration_dominant_ratio = _require_calibration_dominant_ratio(config)
            result_root = getattr(config, "result_root", None) or DEFAULT_CALIBRATION_RESULT_ROOT
            calibration_importance_dir = os.path.join(
                str(result_root),
                f"mmstar_calibration_dominant_ratio_{calibration_dominant_ratio:.2f}",
                "channel_importance",
            )
            self.calibration_channel_importance = load_channel_importance_calibration(
                calibration_importance_dir,
                num_layers=len(self.layers),
            )
            if not self.calibration_channel_importance:
                raise FileNotFoundError(
                    f"No channel-importance calibration files were loaded from: {calibration_importance_dir}"
                )
            missing_layers = [i for i in range(len(self.layers)) if i not in self.calibration_channel_importance]
            if missing_layers:
                raise ValueError(
                    f"Missing channel-importance calibration for layers: {missing_layers}"
                )
        else:
            self.calibration_channel_importance = None

        # Rotation-matrix calibration (RotateK only). Loads the per-layer
        # top-D_keep eigenvectors of accumulated K^T K (computed offline).
        # Activates when calibration_mode=="use" AND channel_method=="rotatek".
        channel_method = str(getattr(config, "channel_method", "")).strip().lower()
        if calibration_mode == "use" and channel_method == "rotatek":
            calibration_dominant_ratio = _require_calibration_dominant_ratio(config)
            result_root = getattr(config, "result_root", None) or DEFAULT_CALIBRATION_RESULT_ROOT
            rotation_dir = os.path.join(
                str(result_root),
                f"mmstar_calibration_dominant_ratio_{calibration_dominant_ratio:.2f}",
                "rotation_matrix",
            )
            channel_ratio = float(getattr(config, "channel_ratio", 0.0))
            head_dim = int(getattr(config, "hidden_size", 0)) // int(
                getattr(config, "num_attention_heads", 1)
            )
            prune_count = min(head_dim, max(0, int(head_dim * channel_ratio)))
            keep_count = head_dim - prune_count
            if keep_count > 0:
                self.calibration_rotation_R_partial = load_rotation_matrix_calibration(
                    rotation_dir, num_layers=len(self.layers), keep_count=keep_count,
                )
                if not self.calibration_rotation_R_partial:
                    raise FileNotFoundError(
                        f"No rotation-matrix calibration files were loaded from: {rotation_dir}"
                    )
                missing_layers = [
                    i for i in range(len(self.layers))
                    if i not in self.calibration_rotation_R_partial
                ]
                if missing_layers:
                    raise ValueError(
                        f"Missing rotation-matrix calibration for layers: {missing_layers}"
                    )
            else:
                self.calibration_rotation_R_partial = None
        else:
            self.calibration_rotation_R_partial = None

        for layer_idx, layer in enumerate(self.layers):
            layer_importance = None if self.calibration_channel_importance is None else self.calibration_channel_importance[layer_idx]
            layer.calibration_channel_importance = layer_importance
            layer.self_attn.calibration_channel_importance = layer_importance
            layer_R = (
                None if self.calibration_rotation_R_partial is None
                else self.calibration_rotation_R_partial[layer_idx]
            )
            layer.calibration_rotation_R_partial = layer_R
            layer.self_attn.calibration_rotation_R_partial = layer_R

        # Layer-adaptive channel budget: load per-layer sparsity from the
        # attention_shift calibration (matches Qwen's pattern).
        self.channel_ratio_high = None
        self.channel_ratio_low = None
        self._layer_budget_by_sparsity = {}
        if getattr(config, "layer_adaptive_channel_budget", False):
            budget_sparsity = round(float(config.channel_ratio), 3)
            calibration_dominant_ratio = _require_calibration_dominant_ratio(config)
            result_root = getattr(config, "result_root", None) or DEFAULT_CALIBRATION_RESULT_ROOT
            budget_path = os.path.join(
                str(result_root),
                f"mmstar_calibration_dominant_ratio_{calibration_dominant_ratio:.2f}",
                "attention_shift",
                f"layer_budget_sparsity_{budget_sparsity:.3f}.csv",
            )
            if not os.path.exists(budget_path):
                raise FileNotFoundError(
                    f"Layer-adaptive channel budget CSV not found: {budget_path}. "
                    "Run attention_shift calibration first, or disable layer_adaptive_channel_budget."
                )
            with open(budget_path, "r", newline="") as f:
                self._layer_budget_by_sparsity[budget_sparsity] = [
                    float(line.strip()) for line in f if line.strip()
                ]

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Replace HF's cache with the VisionZip-aware one. Matches the swap
        # done in `Qwen2_5_VLForConditionalGeneration.forward` — HF's
        # `generate()` seeds an HF `DynamicCache` which lacks `store_pruned`,
        # so we upgrade it on the first (empty) call. Subsequent decode
        # steps receive our own cache back through `outputs.past_key_values`.
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            elif not isinstance(past_key_values, DynamicCache) and len(past_key_values) == 0:
                past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Build the causal mask. For the VisionZip prefill we only feed FA2 the
        # padding mask (attention_mask) — HF 4.49 `_flash_attention_forward`
        # applies causal internally when `is_causal=True`. So we do NOT call
        # `_prepare_4d_causal_attention_mask_*` here (matches the Qwen fork).
        causal_mask = attention_mask

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # Cache the user-requested (global) channel_ratio on first pass so we
        # can restore / use it as the fallback when layer-adaptive budget is
        # disabled or missing for some layer.
        if self.channel_ratio_high is None and self.channel_ratio_low is None:
            self.channel_ratio_high = self.config.channel_ratio
            self.channel_ratio_low = 0

        for layer_idx, decoder_layer in enumerate(self.layers):
            # Fix the first N decoder layers to full-channel (no pruning).
            # Useful when the initial layers are known to be sensitive.
            first_n_full = int(getattr(self.config, "full_channel_first_n_layers", 0) or 0)
            # Ablation: if a specific layer is exempted, force full-channel
            # (channel_ratio=0) on that layer only — overrides the adaptive
            # budget and the global ratio.
            exempt_layer_idx = getattr(self.config, "exempt_layer_idx", None)
            if first_n_full > 0 and layer_idx < first_n_full:
                self.config.channel_ratio = 0.0
            elif exempt_layer_idx is not None and int(exempt_layer_idx) == layer_idx:
                self.config.channel_ratio = 0.0
            elif self.config.layer_adaptive_channel_budget:
                # Layer-wise adaptive channel budget: override config.channel_ratio
                # per layer using the precomputed attention_shift budget table.
                budget_sparsity = round(float(self.channel_ratio_high), 3)
                layer_budget = self._layer_budget_by_sparsity.get(budget_sparsity)
                if layer_budget is not None and layer_idx < len(layer_budget):
                    self.config.channel_ratio = layer_budget[layer_idx]
                else:
                    self.config.channel_ratio = self.channel_ratio_high
            else:
                self.config.channel_ratio = self.channel_ratio_high

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaVisionZipForCausalLM(LlamaForCausalLM):
    """Drop-in replacement for `LlamaForCausalLM` that uses the VisionZip
    decoder layers. Checkpoint keys are identical so HF loading just works."""

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        # Swap the vanilla LlamaModel with the VisionZip-aware one.
        self.model = LlamaVisionZipModel(config)
        # Re-tie lm_head to the new embedding table.
        self.post_init()


def _stamp_channel_defaults(config):
    """Ensure VisionZip/channel pruning knobs exist on the config.

    Mirrors the defaults the Qwen2_5_VLModel sets in its __init__ so that
    checkpoints loaded via `from_pretrained` without the extra kwargs still
    have sane values.
    """
    if not hasattr(config, "channel_method"):
        # VisionK was retired (moved to legacy/methods/visionk_method.py).
        # Default to ThinK for any missing config — RotateK / SparK / Full
        # require explicit `channel_method=` selection.
        config.channel_method = "think"
    if not hasattr(config, "layer_adaptive_channel_budget"):
        config.layer_adaptive_channel_budget = False
    if not hasattr(config, "channel_reconstruction"):
        config.channel_reconstruction = "off"
    config.channel_reconstruction = str(config.channel_reconstruction).strip().lower()
    if config.channel_reconstruction == "true":
        config.channel_reconstruction = "matrix"
    elif config.channel_reconstruction == "false":
        config.channel_reconstruction = "off"
    _valid_reconstruction = {"off", "constant", "mean", "matrix"}
    if config.channel_reconstruction not in _valid_reconstruction:
        raise ValueError(
            f"Unsupported channel_reconstruction={config.channel_reconstruction}. "
            f"Use one of: {', '.join(sorted(_valid_reconstruction))}"
        )
    if not hasattr(config, "reconstruction_constant"):
        config.reconstruction_constant = 0.1
    if not hasattr(config, "custom_kernel"):
        config.custom_kernel = True
    if not hasattr(config, "decode_attention_backend"):
        config.decode_attention_backend = "fa2"
    if not hasattr(config, "calibration_mode"):
        config.calibration_mode = "off"
    if not hasattr(config, "offline_calibration_tasks"):
        config.offline_calibration_tasks = "channel_importance"
    # Ablation knob: force a single decoder layer to full-channel (channel_ratio=0)
    # while the rest use the configured budget. None = disabled.
    if not hasattr(config, "exempt_layer_idx"):
        config.exempt_layer_idx = None
    # Fix the first N decoder layers to full-channel (channel_ratio=0). 0 = disabled.
    # Applies ON TOP OF exempt_layer_idx; both can be active simultaneously.
    if not hasattr(config, "full_channel_first_n_layers"):
        config.full_channel_first_n_layers = 0
    if not hasattr(config, "channel_ratio"):
        config.channel_ratio = 0.0
    if not hasattr(config, "dominant_ratio"):
        config.dominant_ratio = 0.75
    if not hasattr(config, "contextual_ratio"):
        config.contextual_ratio = 0.0
    if not hasattr(config, "result_root"):
        config.result_root = DEFAULT_CALIBRATION_RESULT_ROOT
    if not hasattr(config, "reconstruction_topk"):
        config.reconstruction_topk = None
    if not hasattr(config, "mmstar_reconstruct_topk"):
        config.mmstar_reconstruct_topk = None
    if not hasattr(config, "channel_start"):
        config.channel_start = None
    if not hasattr(config, "channel_end"):
        config.channel_end = None
