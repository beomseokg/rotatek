"""
VisionZip + channel-pruning port for InternVL2.5-8B.

This module mirrors `qwen/qwen2_5vl_visionzip.py` but targets the
`OpenGVLab/InternVL2_5-8B-Instruct` stack, which is loaded via
`AutoModel.from_pretrained(..., trust_remote_code=True)` and therefore
uses the HF-cached `modeling_internvl_chat.py` / `modeling_internlm2.py`
/ `modeling_intern_vit.py` classes.

Because those classes come from remote code, we do NOT re-implement them
here. Instead we monkey-patch the loaded model in `apply_visionzip(...)`:

- Each `InternLM2FlashAttention2` layer's `forward` is replaced with
  `_patched_internlm2_flash_attention_forward`, which integrates
  VisionZip token pruning + ThinK/VisionK/SparK channel pruning and
  optionally invokes the custom Triton decode kernel.

- The `InternVLChatModel.generate` method is replaced with
  `_patched_internvl_chat_generate`, which runs the vision encoder,
  extracts dominant + contextual visual tokens (VisionZip), records
  `prompt_seqlen`/`query_seqlen` on `config`, and forwards to the
  language model with a VisionZip-aware `DynamicCache`.

Reused as-is from the Qwen port:
- `lmms_eval.models.model_utils.kv_pruning_utils`:
  `init_visionzip`, `recover_cache`, `VisionZipCluster`, calibration.
- `lmms_eval.models.model_utils.cache_utils`:
  `Cache`, `DynamicCache` (with split prompt/pruned/text caches).
- `visual_channel_pruning.kernel.sparse_channel_flash_decoding_triton`:
  `sparse_channel_decode_triton`, `sparse_channel_decode_triton_direct`.
"""

import csv
import math
import os
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
else:
    _flash_attention_forward = None

from lmms_eval.models.model_utils.kv_pruning_utils import init_visionzip, recover_cache
from lmms_eval.models.model_utils.cache_utils import Cache, DynamicCache

# ---------------------------------------------------------------------------
# Custom decode-time Triton kernels (reused verbatim from the Qwen port).
# ---------------------------------------------------------------------------
try:
    from visual_channel_pruning.kernel.sparse_channel_flash_decoding_triton import (
        _next_power_of_2,
        sparse_channel_decode_triton as _sparse_decode_triton,
        sparse_channel_decode_triton_direct as _sparse_decode_triton_direct,
    )
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path

    _project_root = _Path(__file__).resolve().parents[6]
    if str(_project_root) not in _sys.path:
        _sys.path.insert(0, str(_project_root))
    from kernel.sparse_channel_flash_decoding_triton import (
        _next_power_of_2,
        sparse_channel_decode_triton as _sparse_decode_triton,
        sparse_channel_decode_triton_direct as _sparse_decode_triton_direct,
    )


DEFAULT_CALIBRATION_RESULT_ROOT = (
    "./results/InternVL2_5_8B"
)

# Skip matrix-based channel reconstruction in the first N decoder layers.
# Early layers tend to be most sensitive to the approximation, so we leave
# them with the default ("off") recovery path.
RECONSTRUCTION_SKIP_FIRST_N_LAYERS = 0


# ---------------------------------------------------------------------------
# Calibration loading (same CSV format as the Qwen port).
# ---------------------------------------------------------------------------


def _require_calibration_dominant_ratio(config):
    if not hasattr(config, "dominant_ratio"):
        raise ValueError("config.dominant_ratio must be provided")
    try:
        return round(float(config.dominant_ratio), 2)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid dominant_ratio: {getattr(config, 'dominant_ratio', None)}"
        ) from exc


def load_rotation_matrix_calibration(base_dir, num_layers, keep_count, dtype=torch.float32):
    """Load accumulated K^T K per layer, eigh on CPU, return per-layer
    top-keep_count eigenvectors.

    Returns: dict {layer_idx: R_partial of shape [num_heads, D, keep_count]}
    """
    import glob as _glob

    layerwise = {}
    for layer_idx in range(num_layers):
        pt_path = os.path.join(
            base_dir, f"layer_{layer_idx:02d}_mmstar_calibration.pt"
        )
        if not os.path.exists(pt_path):
            continue
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        avg_cov = ckpt["avg_cov"]  # [num_heads, D, D] fp32
        # Symmetrise (paranoia against fp noise) and eigh on CPU.
        avg_cov = 0.5 * (avg_cov + avg_cov.transpose(-1, -2))
        _, eigvecs = torch.linalg.eigh(avg_cov)  # ascending
        R_full = eigvecs.flip(dims=[-1])  # descending: col 0 = largest
        R_partial = R_full[..., :keep_count].to(dtype=dtype).contiguous()
        layerwise[layer_idx] = R_partial  # [num_heads, D, keep_count]
    return layerwise


def load_channel_importance_calibration(base_dir, num_layers, dtype=torch.float32):
    layerwise = {}
    for layer_idx in range(num_layers):
        csv_path = os.path.join(
            base_dir, f"layer_{layer_idx:02d}_mmstar_calibration.csv"
        )
        if not os.path.exists(csv_path):
            continue

        rows = []
        max_head_idx = -1
        max_channel_idx = -1

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                head_idx = int(row["head_idx"])
                channel_idx = int(row["channel_idx"])
                score = float(row["avg_score"])
                rows.append((head_idx, channel_idx, score))
                max_head_idx = max(max_head_idx, head_idx)
                max_channel_idx = max(max_channel_idx, channel_idx)

        if not rows:
            continue

        scores = torch.zeros(
            max_head_idx + 1, max_channel_idx + 1, dtype=dtype
        )
        for head_idx, channel_idx, score in rows:
            scores[head_idx, channel_idx] = score

        layerwise[layer_idx] = scores

    return layerwise


# ---------------------------------------------------------------------------
# Decode-time sparse attention helpers (reused from the Qwen port).
# ---------------------------------------------------------------------------


def _get_decode_kernel_timings():
    layer_idx = getattr(run_custom_decode_kernel, "_profile_layer_idx", None)
    if layer_idx is None:
        layer_idx = -1
    timings_by_layer = getattr(run_custom_decode_kernel, "_kernel_timings_by_layer", None)
    if timings_by_layer is None:
        timings_by_layer = {}
        run_custom_decode_kernel._kernel_timings_by_layer = timings_by_layer
    timings = timings_by_layer.get(layer_idx)
    if timings is None:
        timings = {
            "triton_ms": [],
            "cuda_ms": [],
            "cache_setup_ms": [],
            "index_setup_ms": [],
            "static_setup_ms": [],
            "view_setup_ms": [],
            "q_prepare_ms": [],
            "q_alloc_ms": [],
            "full_prepare_ms": [],
            "pre_backend_ms": [],
            "post_backend_ms": [],
            "inner_pre_backend_ms": [],
            "inner_post_backend_ms": [],
            "_pending_events": [],
            "head_dim_keep": [],
            "pruned_dim": [],
            "seq_sparse": [],
            "seq_prompt": [],
            "seq_text": [],
            "num_splits": [],
            "block_k": [],
            "block_p": [],
        }
        timings_by_layer[layer_idx] = timings
    return timings


def run_dense_decode_kernel(
    query_states,
    key_states,
    value_states,
    num_key_value_groups,
    attention_mask=None,
):
    """Full-channel decode attention via the dedicated Triton split-K
    kernel (`kernel/full_channel_flash_decoding_triton.py`). Used by
    Full / ThinK / SparK after they recover full-D K — no sparse-channel
    logic, no bias shift, no dual-segment k_full/k_sparse plumbing.
    """
    from kernel.full_channel_flash_decoding_triton import full_channel_decode_triton

    q = query_states.squeeze(-2)  # [B, H_q, D]
    return full_channel_decode_triton(
        q=q,
        k=key_states,
        v=value_states,
        num_kv_groups=num_key_value_groups,
    )


def run_custom_decode_kernel(
    query_states,
    key_pruned,
    value_states,
    key_prompt,
    text_key_states,
    mask,
    keep_idx,
    num_key_value_groups,
    head_dim,
    attention_mask=None,
    supplementary_matrix=None,
    channel_reconstruction="off",
):
    """Sparse decode attention for VisionK (reused from the Qwen port)."""
    bsz, num_heads, _, _ = query_states.shape
    _, _, sparse_seq_len, kept_dim = key_pruned.shape

    has_matrix_recon = supplementary_matrix is not None
    pruned_idx_t = None
    w_t_t = None
    recon_bias_t = None
    pruned_dim = 0

    if has_matrix_recon:
        _sup_cache = getattr(run_custom_decode_kernel, "_sup_cache", {})
        _sup_key = (
            supplementary_matrix["weight"].data_ptr(),
            supplementary_matrix["weight"].shape,
            supplementary_matrix["keep_idx"].data_ptr(),
            supplementary_matrix["pruned_idx"].data_ptr(),
            supplementary_matrix["bias"].data_ptr(),
            bsz,
            query_states.device,
            query_states.dtype,
        )
        if _sup_key not in _sup_cache:
            keep_idx_t = supplementary_matrix["keep_idx"].to(
                device=query_states.device, dtype=torch.long,
            ).contiguous()
            if keep_idx_t.dim() == 2:
                keep_idx_t = keep_idx_t.unsqueeze(0).expand(bsz, -1, -1).contiguous()
            elif keep_idx_t.dim() == 3 and keep_idx_t.shape[0] != bsz:
                keep_idx_t = keep_idx_t[:1].expand(bsz, -1, -1).contiguous()

            w_t_t = (
                supplementary_matrix["weight"]
                .to(device=query_states.device, dtype=query_states.dtype)
                .transpose(-2, -1)
                .contiguous()
            )
            pruned_idx_t = supplementary_matrix["pruned_idx"].to(
                device=query_states.device, dtype=torch.long,
            ).contiguous()
            recon_bias_t = supplementary_matrix["bias"].to(
                device=query_states.device, dtype=query_states.dtype,
            ).contiguous()
            if len(_sup_cache) >= 128:
                _sup_cache.clear()
            _sup_cache[_sup_key] = (keep_idx_t, pruned_idx_t, w_t_t, recon_bias_t)
            run_custom_decode_kernel._sup_cache = _sup_cache
        kv_keep_idx, pruned_idx_t, w_t_t, recon_bias_t = _sup_cache[_sup_key]
        pruned_dim = pruned_idx_t.shape[-1]
    else:
        if keep_idx is not None:
            kv_keep_idx = keep_idx
        else:
            keep_mask = mask if mask.dtype == torch.bool else mask.to(dtype=torch.bool)
            kv_keep_idx = torch.argsort(
                keep_mask.to(torch.int32), dim=-1, descending=True, stable=True,
            )[..., :kept_dim].contiguous()
        pruned_idx_t = kv_keep_idx
        w_t_t = kv_keep_idx
        recon_bias_t = kv_keep_idx

    q_full = query_states.squeeze(-2)

    attn_output, _, _ = _sparse_decode_triton_direct(
        q_full=q_full,
        keep_idx=kv_keep_idx,
        k_prompt=key_prompt,
        k_text=text_key_states,
        k_sparse=key_pruned,
        v_all=value_states,
        pruned_idx=pruned_idx_t if has_matrix_recon else None,
        w_t=w_t_t if has_matrix_recon else None,
        recon_bias=recon_bias_t if has_matrix_recon else None,
        num_kv_groups=num_key_value_groups,
    )
    return attn_output


# ---------------------------------------------------------------------------
# Patched InternLM2FlashAttention2 forward.
# ---------------------------------------------------------------------------


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Standard RoPE from LLaMA/InternLM2 (no multi-resolution sections)."""
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _patched_internlm2_flash_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    """
    VisionZip-aware forward for `InternLM2FlashAttention2`.

    Mirrors `Qwen2_5_VLFlashAttention2.forward` but adapted for InternLM2:
    - Q/K/V are extracted from the unified `self.wqkv` with GQA layout
      (num_heads + 2*num_kv_heads groups of head_dim).
    - RoPE is standard (not MRoPE); we call `self.rotary_emb(value, seq_len)`
      and apply it via `_apply_rotary_pos_emb(..., position_ids)`.
    - Output projection is `self.wo` (not `self.o_proj`).
    - `past_key_value` switches between the HF tuple form (dense path) and
      our custom `DynamicCache` (when `past_key_value` is a `DynamicCache`,
      which the patched chat model installs).
    """
    # Lazy import to avoid circular HF-remote-code resolution.
    from einops import rearrange

    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated; use `attention_mask` instead."
        )
        attention_mask = kwargs.pop("padding_mask")

    output_attentions = False
    bsz, q_len, _ = hidden_states.size()

    # ---- RotateK ratio profiler: bracket the whole attention forward and
    # compare against the PCA time exposed on self.kv_cluster. Controlled
    # by ROTATEK_RATIO=N env var (read once at module import in
    # kv_pruning_utils). Zero overhead when disabled.
    from lmms_eval.models.model_utils import kv_pruning_utils as _kvu
    _rk_ratio_on = (
        _kvu._rotatek_ratio_count[0] < _kvu._ROTATEK_RATIO_LIMIT
        and getattr(self.config, "channel_method", "") == "rotatek"
        and q_len > 1
    )
    _rk_fwd_start = None
    if _rk_ratio_on:
        _rk_fwd_start = torch.cuda.Event(enable_timing=True)
        _rk_fwd_start.record()

    # ------------------------------------------------------------------
    # Per-phase latency profiling (gated by `config._decode_profile`).
    # Mirrors the timing harness in Qwen's port so the same consumer
    # (`benchmark_decoding.py` and friends) works unchanged.
    # Events are only created when `_profile` is True -> zero overhead
    # when profiling is disabled.
    # ------------------------------------------------------------------
    def _new_evt():
        return torch.cuda.Event(enable_timing=True)

    def _elapsed_ms(start_evt, end_evt):
        if start_evt is None or end_evt is None:
            return 0.0
        return start_evt.elapsed_time(end_evt)

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
    # Prefill-only stage events (channel-pruning scoring + KV cache write).
    _evt_score_start = _evt_score_end = None
    _evt_kv_write_start = _evt_kv_write_end = None

    _profile_layers = getattr(self.config, "_decode_profile_layers", {14})
    _decode_profile = (
        getattr(self.config, "_decode_profile", False)
        and self.layer_idx in _profile_layers
        and q_len == 1
    )
    _prefill_profile = (
        getattr(self.config, "_prefill_profile", False)
        and q_len > 1
    )
    _profile = _decode_profile or _prefill_profile

    # ---- Q/K/V projection + GQA unpack ----
    if _profile:
        _evt_qkv_proj_rope_start = _new_evt()
        _evt_qkv_proj_start = _new_evt()
        _evt_qkv_proj_rope_start.record()
        _evt_qkv_proj_start.record()

    qkv_states = self.wqkv(hidden_states)
    qkv_states = rearrange(
        qkv_states,
        "b q (h gs d) -> b q h gs d",
        gs=2 + self.num_key_value_groups,
        d=self.head_dim,
    )
    query_states = qkv_states[..., : self.num_key_value_groups, :]
    query_states = rearrange(query_states, "b q h gs d -> b q (h gs) d")
    key_states_no_rope = qkv_states[..., -2, :]
    value_states = qkv_states[..., -1, :]

    # [B, H, S, D]
    query_states = query_states.transpose(1, 2)
    key_states_no_rope = key_states_no_rope.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if _profile:
        _evt_qkv_proj_end = _new_evt()
        _evt_qkv_proj_end.record()

    using_visionzip_cache = isinstance(past_key_value, DynamicCache)

    # kv_seq_len controls RoPE frequency length.
    kv_seq_len = key_states_no_rope.shape[-2]
    if past_key_value is not None and not using_visionzip_cache:
        if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
            kv_seq_len += past_key_value[0].shape[-2]
    elif using_visionzip_cache:
        # Decode step: past_key_value already spans all previous tokens.
        kv_seq_len += past_key_value.get_seq_length(self.layer_idx)

    # Fall back to sequential position_ids if caller didn't provide any.
    if position_ids is None:
        position_ids = torch.arange(
            kv_seq_len - q_len, kv_seq_len, device=hidden_states.device
        ).unsqueeze(0)
        if position_ids.dim() == 2:
            pass
        else:
            position_ids = position_ids.unsqueeze(0)

    if _profile:
        _evt_rope_start = _new_evt()
        _evt_rope_start.record()

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = _apply_rotary_pos_emb(
        query_states, key_states_no_rope, cos, sin, position_ids
    )

    if _profile:
        _evt_rope_end = _new_evt()
        _evt_rope_end.record()
        _evt_qkv_proj_rope_end = _new_evt()
        _evt_qkv_proj_rope_end.record()

    channel_method = getattr(self.config, "channel_method", "think")
    channel_ratio = getattr(self.config, "channel_ratio", 0.0)

    if q_len > 1 and using_visionzip_cache:
        # Prefill phase: initialise the KV cluster and split KV into
        # prompt / pruned-vision / text segments.
        init_visionzip(self)

        # Method-agnostic hook: if we're in rotation_matrix calibration
        # collect mode, accumulate K^T K on this layer's vision keys. The
        # rotation is a property of K itself, independent of the pruning
        # scheme, so any `channel_method` is fine for collection.
        from lmms_eval.models.model_utils.kv_pruning_utils import (
            _maybe_accumulate_rotation_matrix_calibration as _rk_accum_rot,
        )
        _rk_accum_rot(self.kv_cluster, key_states)

        cache_kwargs = {"sin": sin, "cos": cos}
        if channel_ratio == 0.0:
            # No compression: skip the 3-way split storage and the cat-at-
            # decode round-trip. K and V live as a single growing tensor
            # (standard HF-cache layout) under `key_cache_unified`.
            if _profile:
                _evt_score_start = _new_evt(); _evt_score_start.record()
                _evt_score_end = _new_evt(); _evt_score_end.record()
                _evt_kv_write_start = _new_evt(); _evt_kv_write_start.record()
            past_key_value.store_unified(
                key_states, value_states, self.layer_idx,
            )
            if _profile:
                _evt_kv_write_end = _new_evt(); _evt_kv_write_end.record()
        elif channel_method == "think":
            if _profile:
                _evt_score_start = _new_evt(); _evt_score_start.record()
            kv_pruned, kv_prompt, kv_text, mask, value_states_compress = (
                self.kv_cluster.update_think(
                    key_states,
                    query_states,
                    value_states,
                    attention_mask,
                    num_key_value_groups=self.num_key_value_groups,
                    calibration_channel_importance=getattr(
                        self, "calibration_channel_importance", None
                    ),
                )
            )
            if _profile:
                _evt_score_end = _new_evt(); _evt_score_end.record()
                _evt_kv_write_start = _new_evt(); _evt_kv_write_start.record()
            past_key_value.store_pruned(
                kv_pruned, kv_prompt, kv_text, mask, value_states_compress,
                self.layer_idx, cache_kwargs,
            )
            past_key_value.think_mask.append(
                self.kv_cluster.current_think_mask
            )
            if _profile:
                _evt_kv_write_end = _new_evt(); _evt_kv_write_end.record()
        elif channel_method == "spark":
            if _profile:
                _evt_score_start = _new_evt(); _evt_score_start.record()
            kv_pruned, kv_prompt, kv_text, mask, value_states_compress = (
                self.kv_cluster.update_spark(
                    key_states,
                    query_states,
                    value_states,
                    attention_mask,
                    num_key_value_groups=self.num_key_value_groups,
                )
            )
            if _profile:
                _evt_score_end = _new_evt(); _evt_score_end.record()
                _evt_kv_write_start = _new_evt(); _evt_kv_write_start.record()
            past_key_value.store_pruned(
                kv_pruned, kv_prompt, kv_text, mask, value_states_compress,
                self.layer_idx, cache_kwargs,
            )
            past_key_value.spark_mask.append(
                self.kv_cluster.current_spark_mask
            )
            past_key_value.spark_pruned_mean.append(
                self.kv_cluster.current_spark_pruned_mean
            )
            if _profile:
                _evt_kv_write_end = _new_evt(); _evt_kv_write_end.record()
        elif channel_method == "rotatek":
            if _profile:
                _evt_score_start = _new_evt(); _evt_score_start.record()
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
            if _profile:
                _evt_score_end = _new_evt(); _evt_score_end.record()
                _evt_kv_write_start = _new_evt(); _evt_kv_write_start.record()
            past_key_value.store_pruned(
                kv_pruned, kv_prompt, kv_text, mask, value_states_compress,
                self.layer_idx, cache_kwargs,
            )
            # In `truncated` storage mode (ROTATEK_STORAGE=truncated), kv_pruned
            # is [S_v, D_keep] and the decode branch needs R_partial to rotate
            # Q. Stash it; in `full` mode this is just a (cheap) extra
            # reference that nobody reads.
            past_key_value.rotatek_rotations.append(
                self.kv_cluster.current_rotatek_R_partial
            )
            # δμ = μ @ (I - P) per-kv-head [B, H_kv, D]; decode-time bias on
            # vision logits that recovers the full-mode Q @ (I - P) @ μ^T
            # shift missing from Q @ P @ K^T. None in full mode.
            past_key_value.rotatek_means.append(
                self.kv_cluster.current_rotatek_delta_mu
            )
            if _profile:
                _evt_kv_write_end = _new_evt(); _evt_kv_write_end.record()
        else:
            raise ValueError(f"Unsupported channel_method: {channel_method}")

        # Prefill still runs FA2 over the full (non-pruned) K/V for this step.
        key_states_attn = key_states
        value_states_attn = value_states
        attn_output = None

    elif q_len == 1 and using_visionzip_cache:
        # Decode phase: retrieve split caches + incorporate this step's K/V.
        # ratio=0 was stored unified at prefill (no 3-way split, no
        # decode-time cat) — branch separately so the rest of the chain
        # only handles the compressed/split-storage cases.
        is_unified_layer = (
            self.layer_idx < len(past_key_value.is_unified)
            and past_key_value.is_unified[self.layer_idx]
        )

        if not is_unified_layer:
            if _profile:
                _evt_cache_update_start = _new_evt()
                _evt_cache_update_start.record()

            text_key_states, value_states, key_pruned, key_prompt, mask = (
                past_key_value.update(
                    key_states, value_states, self.layer_idx,
                    {"sin": sin, "cos": cos},
                )
            )

            if _profile:
                _evt_cache_update_end = _new_evt()
                _evt_cache_update_end.record()

        if is_unified_layer:
            if _profile:
                _evt_cache_update_start = _new_evt()
                _evt_cache_update_start.record()
            key_states_attn, value_states_attn = past_key_value.update_unified(
                key_states, value_states, self.layer_idx,
            )
            if _profile:
                _evt_cache_update_end = _new_evt()
                _evt_cache_update_end.record()
            # No cat — unified path. Run dense triton directly.
            if getattr(self.config, "decode_attention_backend", "fa2") == "triton":
                if _profile:
                    _evt_custom_kernel_start = _new_evt()
                    _evt_custom_kernel_start.record()
                attn_output = run_dense_decode_kernel(
                    query_states=query_states,
                    key_states=key_states_attn,
                    value_states=value_states_attn,
                    num_key_value_groups=self.num_key_value_groups,
                    attention_mask=attention_mask,
                )
                if _profile:
                    _evt_custom_kernel_end = _new_evt()
                    _evt_custom_kernel_end.record()
            else:
                attn_output = None

        elif channel_method == "think":
            if _profile:
                _evt_recovery_start = _new_evt()
                _evt_recovery_start.record()

            # Paper-faithful ThinK recovery: per-head bool keep mask is
            # broadcast over the vision span and used as a boolean index
            # into a pre-zeroed buffer (matches the original ThinK code).
            think_mask = past_key_value.think_mask[self.layer_idx]   # [B, H_kv, D]
            bsz_r, h_kv, seq_r = key_pruned.shape[:3]
            recovered = torch.zeros(
                bsz_r, h_kv, seq_r, self.head_dim,
                dtype=key_pruned.dtype, device=key_pruned.device,
            )
            mask_expanded = think_mask.unsqueeze(2).expand(-1, -1, seq_r, -1)
            recovered[mask_expanded] = key_pruned.reshape(-1)

            if _profile:
                _evt_recovery_end = _new_evt()
                _evt_recovery_end.record()
                _evt_cat_start = _new_evt()
                _evt_cat_start.record()

            key_states_attn = torch.cat(
                [key_prompt, recovered, text_key_states], dim=-2
            )
            value_states_attn = value_states

            if _profile:
                _evt_cat_end = _new_evt()
                _evt_cat_end.record()

            if getattr(self.config, "decode_attention_backend", "fa2") == "triton":
                if _profile:
                    _evt_custom_kernel_start = _new_evt()
                    _evt_custom_kernel_start.record()
                attn_output = run_dense_decode_kernel(
                    query_states=query_states,
                    key_states=key_states_attn,
                    value_states=value_states_attn,
                    num_key_value_groups=self.num_key_value_groups,
                    attention_mask=attention_mask,
                )
                if _profile:
                    _evt_custom_kernel_end = _new_evt()
                    _evt_custom_kernel_end.record()
            else:
                attn_output = None

        elif channel_method == "spark":
            if _profile:
                _evt_recovery_start = _new_evt()
                _evt_recovery_start.record()

            # SparK (compact-storage) recovery: pre-fill a [B, H, S_v, D]
            # buffer with the per-token pruned-channel mean, then write
            # compact K into the True positions of the per-token bool mask
            # via boolean indexing — mirrors ThinK's recovery primitive.
            spark_mask = past_key_value.spark_mask[self.layer_idx]            # [B, H, S_v, D] bool
            pruned_mean = past_key_value.spark_pruned_mean[self.layer_idx]    # [B, H, S_v, 1]
            bsz_r, h_kv, seq_r, _ = key_pruned.shape
            recovered = pruned_mean.expand(bsz_r, h_kv, seq_r, self.head_dim).contiguous()
            recovered[spark_mask] = key_pruned.reshape(-1)

            if _profile:
                _evt_recovery_end = _new_evt()
                _evt_recovery_end.record()
                _evt_cat_start = _new_evt()
                _evt_cat_start.record()

            key_states_attn = torch.cat(
                [key_prompt, recovered, text_key_states], dim=-2
            )
            value_states_attn = value_states

            if _profile:
                _evt_cat_end = _new_evt()
                _evt_cat_end.record()

            if getattr(self.config, "decode_attention_backend", "fa2") == "triton":
                if _profile:
                    _evt_custom_kernel_start = _new_evt()
                    _evt_custom_kernel_start.record()
                attn_output = run_dense_decode_kernel(
                    query_states=query_states,
                    key_states=key_states_attn,
                    value_states=value_states_attn,
                    num_key_value_groups=self.num_key_value_groups,
                    attention_mask=attention_mask,
                )
                if _profile:
                    _evt_custom_kernel_end = _new_evt()
                    _evt_custom_kernel_end.record()
            else:
                attn_output = None
        elif channel_method == "rotatek":
            # Two storage modes (selected at prefill via ROTATEK_STORAGE):
            #   - full      : key_pruned is [S_v, D] (rank-D_keep projection in
            #                 original basis) → standard FA2 path.
            #   - truncated : key_pruned is [S_v, D_keep] (rotated truncated
            #                 form) → use sparse kernel with Q rotation.
            # Detect mode by comparing key_pruned's last dim against head_dim.
            if key_pruned.shape[-1] == self.head_dim:
                # full mode
                if _profile:
                    _evt_cat_start = _new_evt()
                    _evt_cat_start.record()

                key_states_attn = torch.cat(
                    [key_prompt, key_pruned, text_key_states], dim=-2
                )
                value_states_attn = value_states

                if _profile:
                    _evt_cat_end = _new_evt()
                    _evt_cat_end.record()

                attn_output = None
            else:
                # truncated mode: fused phase-1 Triton kernel that absorbs
                # the Q@R rotation + δμ bias into the inner softmax loop.
                # The fused path consistently outperforms calling the same
                # `sparse_channel_decode_triton` kernel that Full uses with
                # external Q@R/δμ — the dual-D (k_full=D + k_sparse=D_keep)
                # kernel has split-0 straggler + register-pressure overhead
                # that we measured larger than the kernel-launch savings of
                # reusing Full's kernel.
                from methods.rotatek import rotatek_decode_fused

                q_squeezed = query_states.squeeze(2)  # [B, H_q, D]
                bsz_q, h_q, head_dim = q_squeezed.shape

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

                if _profile:
                    _evt_custom_kernel_start = _new_evt()
                    _evt_custom_kernel_start.record()
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
                if _profile:
                    _evt_custom_kernel_end = _new_evt()
                    _evt_custom_kernel_end.record()
                # attn_output shape: [B, 1, H_q, D] — kernel-produced; the FA2
                # fallback below is skipped since attn_output is non-None.
                key_states_attn = None
                value_states_attn = None
        else:
            raise ValueError(
                f"Unsupported channel_method for decode: {channel_method}"
            )

    else:
        # Fallback: native InternLM2 HF-tuple cache path (no VisionZip).
        if past_key_value is not None and isinstance(past_key_value, (tuple, list)):
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        key_states_attn = key_states
        value_states_attn = value_states
        attn_output = None

    # ---- FlashAttention2 fallback path (runs unless the Triton sparse
    # kernel already produced attn_output above). ----
    if attn_output is None:
        if _profile:
            _evt_transpose_start = _new_evt()
            _evt_transpose_start.record()

        q_fa = query_states.transpose(1, 2)
        k_fa = key_states_attn.transpose(1, 2)
        v_fa = value_states_attn.transpose(1, 2)

        if _profile:
            _evt_transpose_end = _new_evt()
            _evt_transpose_end.record()
            _evt_attn_start = _new_evt()
            _evt_attn_start.record()

        attn_output = self._flash_attention_forward(
            q_fa, k_fa, v_fa, attention_mask, q_len
        )

        if _profile:
            _evt_attn_end = _new_evt()
            _evt_attn_end.record()

    if _profile:
        _evt_attn_out_start = _new_evt()
        _evt_attn_out_start.record()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.wo(attn_output)

    if _profile:
        _evt_attn_out_end = _new_evt()
        _evt_attn_out_end.record()

    # ------------------------------------------------------------------
    # Per-phase timings: APPEND EVENT PAIRS ONLY. We deliberately skip
    # `torch.cuda.synchronize()` and `event.elapsed_time()` here because
    # `elapsed_time()` requires events to be complete and would block
    # the stream — doing that per layer per decode step serialises the
    # whole decode (32 layers × N tokens = thousands of full-device
    # syncs for a single generation). Instead we store raw event pairs
    # and let `print_decode_timings()` / `finalize_decode_timings()`
    # do ONE sync + elapsed_time() pass at the end.
    # ------------------------------------------------------------------
    if _profile:
        # Phase-separated buckets so prefill events can't leak into decode
        # rows in `latency/model_breakdown.py` (and vice versa). The
        # original `_decode_event_pairs_by_layer` is preserved as the
        # decode-phase target for backwards compatibility with
        # `_get_decode_kernel_timings()`.
        _is_pf = q_len > 1
        _bucket_attr = "_prefill_attn_events_by_layer" if _is_pf else "_decode_event_pairs_by_layer"
        _all_events = getattr(self.config, _bucket_attr, None)
        if _all_events is None:
            _all_events = {}
            setattr(self.config, _bucket_attr, _all_events)
        _events = _all_events.get(self.layer_idx)
        if _events is None:
            _events = {
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
                # Prefill-only stages (channel-pruning method-specific work).
                "score_ms": [],
                "kv_write_ms": [],
            }
            _all_events[self.layer_idx] = _events

        _events["qkv_proj_rope_ms"].append((_evt_qkv_proj_rope_start, _evt_qkv_proj_rope_end))
        _events["qkv_proj_ms"].append((_evt_qkv_proj_start, _evt_qkv_proj_end))
        _events["rope_ms"].append((_evt_rope_start, _evt_rope_end))
        _events["cache_update_ms"].append((_evt_cache_update_start, _evt_cache_update_end))
        _events["recovery_ms"].append((_evt_recovery_start, _evt_recovery_end))
        _events["cat_ms"].append((_evt_cat_start, _evt_cat_end))
        _events["custom_decode_kernel_ms"].append((_evt_custom_kernel_start, _evt_custom_kernel_end))
        _events["transpose_ms"].append((_evt_transpose_start, _evt_transpose_end))
        _events["fa2_ms"].append((_evt_attn_start, _evt_attn_end))
        _events["attn_out_proj_ms"].append((_evt_attn_out_start, _evt_attn_out_end))
        if _prefill_profile and _is_pf:
            _events["score_ms"].append((_evt_score_start, _evt_score_end))
            _events["kv_write_ms"].append((_evt_kv_write_start, _evt_kv_write_end))

    # RotateK ratio print — measure full attention forward, compare to PCA.
    if _rk_ratio_on and _rk_fwd_start is not None:
        import sys as _sys_for_rk
        _rk_fwd_end = torch.cuda.Event(enable_timing=True)
        _rk_fwd_end.record()
        torch.cuda.synchronize()
        attn_fwd_ms = _rk_fwd_start.elapsed_time(_rk_fwd_end)
        rk_start_evt = getattr(self.kv_cluster, "current_rotatek_evt_start", None)
        rk_end_evt = getattr(self.kv_cluster, "current_rotatek_evt_end", None)
        rk_ms = rk_start_evt.elapsed_time(rk_end_evt) if rk_start_evt is not None else 0.0
        pct = (rk_ms / attn_fwd_ms * 100.0) if attn_fwd_ms > 0 else 0.0
        _sys_for_rk.stderr.write(
            f"[rotatek-ratio L{self.layer_idx:02d} call#{_kvu._rotatek_ratio_count[0]:03d}] "
            f"attn_fwd={attn_fwd_ms:.2f}ms  rotatek={rk_ms:.2f}ms  ratio={pct:.1f}%\n"
        )
        _sys_for_rk.stderr.flush()
        _kvu._rotatek_ratio_count[0] += 1

    # InternLM2 expects (attn_output, attn_weights, past_key_value).
    # When we use the VisionZip DynamicCache we return it so the LLM's
    # outer loop keeps threading the same object on each step.
    if using_visionzip_cache:
        returned_past = past_key_value
    elif use_cache:
        returned_past = (key_states_attn, value_states_attn)
    else:
        returned_past = None
    return attn_output, None, returned_past


# ---------------------------------------------------------------------------
# Patched InternLM2DecoderLayer.forward with `_layer_timings` instrumentation.
# Records attention-module and FFN (feed_forward) timings per decode step,
# mirroring Qwen's `_layer_timings` output (`prefill_attn_ms`, `prefill_ffn_ms`,
# `decode_attn_ms`, `decode_ffn_ms`). Decode-side recording is gated on
# `_cfg._decode_profile_layers` so that attn_total/ffn/layer_total are
# captured on the SAME layer(s) as the inner attention sub-stages — keeping
# attn_total comparable with the sum of its components.
# ---------------------------------------------------------------------------


def _patched_internlm2_decoder_layer_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    **kwargs,
):
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated; use `attention_mask` instead."
        )

    _seq_len = hidden_states.shape[1]
    # `attention.config` is set to `llm_config` in apply_visionzip().
    _cfg = self.attention.config
    _decode_layers = getattr(_cfg, "_decode_profile_layers", {0})
    _layer_profile = (
        (getattr(_cfg, "_decode_profile", False)
         and getattr(self, "layer_idx", 0) in _decode_layers)
        or (getattr(_cfg, "_prefill_profile", False)
            and _seq_len > 1)
    )

    if _layer_profile:
        _evt_layer_start = torch.cuda.Event(enable_timing=True)
        _evt_layer_start.record()

    residual = hidden_states
    hidden_states = self.attention_norm(hidden_states)

    if _layer_profile:
        _evt_attn_start = torch.cuda.Event(enable_timing=True)
        _evt_attn_start.record()

    hidden_states, self_attn_weights, present_key_value = self.attention(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        **kwargs,
    )

    if _layer_profile:
        _evt_attn_end = torch.cuda.Event(enable_timing=True)
        _evt_attn_end.record()

    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.ffn_norm(hidden_states)

    if _layer_profile:
        _evt_ffn_start = torch.cuda.Event(enable_timing=True)
        _evt_ffn_start.record()

    hidden_states = self.feed_forward(hidden_states)

    if _layer_profile:
        _evt_ffn_end = torch.cuda.Event(enable_timing=True)
        _evt_ffn_end.record()

        # Defer elapsed_time computation — store event pairs only and let
        # `finalize_decode_timings()` do a single sync at the end.
        _events = getattr(_cfg, "_layer_event_pairs", None)
        if _events is None:
            _events = {
                "prefill_attn_ms": [],
                "prefill_ffn_ms": [],
                "decode_attn_ms": [],
                "decode_ffn_ms": [],
                "prefill_layer_total_ms": [],
                "decode_layer_total_ms": [],
            }
            _cfg._layer_event_pairs = _events
        _phase = "prefill" if _seq_len > 1 else "decode"
        _events[f"{_phase}_attn_ms"].append((_evt_attn_start, _evt_attn_end))
        _events[f"{_phase}_ffn_ms"].append((_evt_ffn_start, _evt_ffn_end))

    hidden_states = residual + hidden_states

    if _layer_profile:
        _evt_layer_end = torch.cuda.Event(enable_timing=True)
        _evt_layer_end.record()
        _events[f"{_phase}_layer_total_ms"].append((_evt_layer_start, _evt_layer_end))

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)
    return outputs


# ---------------------------------------------------------------------------
# Patched InternLM2Model.forward (routes our DynamicCache through every layer)
# and patched prepare_inputs_for_generation (tolerates the custom cache).
# ---------------------------------------------------------------------------


def _patched_internlm2_model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    """Patched `InternLM2Model.forward` that threads a single VisionZip
    `DynamicCache` through every decoder layer (instead of indexing a tuple).
    """
    from transformers.modeling_outputs import BaseModelOutputWithPast

    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    # VisionZip path: always use our DynamicCache when enabled. We ignore
    # whatever `past_key_values` was handed in on the *first* call (HF may
    # auto-wrap it into `transformers.cache_utils.DynamicCache`, which has
    # the right interface for HF's own generation helpers but is not the
    # sparse cache our patched attention expects). On subsequent calls HF
    # threads our DynamicCache back in, which we pass through unchanged.
    visionzip_enabled = getattr(self.config, "_visionzip_enabled", False)
    if visionzip_enabled and not isinstance(past_key_values, DynamicCache):
        past_key_values = DynamicCache()

    if past_key_values is None:
        past_key_values_length = 0
    elif hasattr(past_key_values, "get_seq_length"):
        past_key_values_length = past_key_values.get_seq_length() or 0
    else:
        past_key_values_length = (
            past_key_values[0][0].shape[2] if len(past_key_values) > 0 else 0
        )

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.tok_embeddings(input_ids)

    if self.config.attn_implementation == "flash_attention_2":
        attention_mask = (
            attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        )
    else:
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length + past_key_values_length),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training and use_cache:
        use_cache = False

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    # Cache the user-requested (global) channel_ratio on first pass so per-layer
    # budget overrides below don't clobber it for subsequent layers. Cache
    # unconditionally — the exempt_layer_idx path also needs the pristine value
    # so subsequent layers still see the correct non-zero global ratio.
    if not hasattr(self, "_channel_ratio_high"):
        self._channel_ratio_high = float(self.config.channel_ratio)

    for _idx, decoder_layer in enumerate(self.layers):
        # Fix the first N decoder layers to full-channel (no pruning).
        first_n_full = int(getattr(self.config, "full_channel_first_n_layers", 0) or 0)
        # Ablation: if a specific layer_idx is set as exempt, force full-channel
        # (channel_ratio=0) on that layer only, overriding any per-layer budget.
        exempt_layer_idx = getattr(self.config, "exempt_layer_idx", None)
        if first_n_full > 0 and _idx < first_n_full:
            self.config.channel_ratio = 0.0
        elif exempt_layer_idx is not None and int(exempt_layer_idx) == _idx:
            self.config.channel_ratio = 0.0
        elif getattr(self.config, "layer_adaptive_channel_budget", False):
            # Layer-wise adaptive channel budget: override config.channel_ratio with the
            # per-layer value from the budget CSV so each layer prunes at a sparsity that
            # exactly matches a supplementary_matrix bucket (avoids runtime/matrix dim
            # mismatches in the custom decode kernel).
            budget_sparsity = round(float(self._channel_ratio_high), 3)
            layer_budget = getattr(self.config, "_layer_budget_by_sparsity", {}).get(budget_sparsity)
            if layer_budget is not None and _idx < len(layer_budget):
                self.config.channel_ratio = float(layer_budget[_idx])
            else:
                self.config.channel_ratio = self._channel_ratio_high
        else:
            self.config.channel_ratio = getattr(self, "_channel_ratio_high", self.config.channel_ratio)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,  # same DynamicCache object for every layer
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = past_key_values if use_cache else None

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def _patched_internlm2_prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    **kwargs,
):
    """Replacement for `InternLM2ForCausalLM.prepare_inputs_for_generation`
    that understands our `DynamicCache`.

    The stock version reads `past_key_values[0][0].shape[2]` to find the
    past length; our cache uses `get_seq_length()` instead.
    """
    if past_key_values is None:
        past_length = 0
    elif hasattr(past_key_values, "get_seq_length"):
        # Covers both our `DynamicCache` and HF's `Cache`/`DynamicCache`.
        past_length = past_key_values.get_seq_length() or 0
    else:
        # Legacy tuple of (k, v) per layer.
        past_length = (
            past_key_values[0][0].shape[2] if len(past_key_values) > 0 else 0
        )
    if past_length > 0 and input_ids.shape[1] > 1:
        # After prefill we only need the most recently generated token.
        input_ids = input_ids[:, -1:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values is not None:
            position_ids = position_ids[:, -input_ids.shape[1]:]

    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs


# ---------------------------------------------------------------------------
# Patched InternVLChatModel.generate (vision-token VisionZip pruning).
# ---------------------------------------------------------------------------


def _extract_vit_with_attn_logits(chat_model, pixel_values):
    """Run the vision encoder and return (vit_embeds_post_mlp, attn_logits, attn_key).

    VisionZip needs:
      - `attn_logits`: [N_tokens] cls-attention importance per visual token
      - `attn_key`:    [H, N_tokens, D] last-layer visual keys for contextual
        token selection via cosine similarity.

    InternViT's `InternAttention.forward` does not expose these natively, so
    we intercept the final encoder layer's attention via a forward hook.
    """
    vision_model = chat_model.vision_model
    last_attn = vision_model.encoder.layers[-1].attn

    captured = {}

    def _hook(module, inputs, output):
        # inputs: (hidden_states,)
        hidden = inputs[0]
        B, N, C = hidden.shape
        qkv = module.qkv(hidden).reshape(
            B, N, 3, module.num_heads, C // module.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        if module.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = module.q_norm(
                q.transpose(1, 2).flatten(-2, -1)
            ).view(B_, N_, H_, D_).transpose(1, 2)
            k = module.k_norm(
                k.transpose(1, 2).flatten(-2, -1)
            ).view(B_, N_, H_, D_).transpose(1, 2)
        # CLS-attention weights: mean over heads of softmax(q_cls @ k^T)
        attn = (q * module.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        cls_attn = attn[:, :, 0, 1:].mean(dim=1)  # [B, N-1]
        captured["attn_logits"] = cls_attn[0]  # [N-1]
        captured["attn_key"] = k[0, :, 1:, :]   # [H, N-1, D]

    handle = last_attn.register_forward_hook(_hook)
    try:
        vit_embeds = chat_model.extract_feature(pixel_values)
    finally:
        handle.remove()

    return vit_embeds, captured.get("attn_logits"), captured.get("attn_key")


def _visionzip_select_tokens(
    attn_logits: torch.Tensor,
    attn_key: torch.Tensor,
    dominant_ratio: float,
    contextual_ratio: float,
):
    """Return (dominant_idx, contextual_idx, contextual_tokens_fn).

    Ported from the Qwen `forward` (see `qwen2_5vl_visionzip.py` around the
    "Contextual Visual Tokens" block). Mirrors the top-k + similarity-merge
    algorithm but without Qwen-specific tensor shapes.
    """
    n_tokens = attn_logits.size(0)
    dominant_num = int(dominant_ratio * n_tokens)
    contextual_num = max(int(contextual_ratio * n_tokens), 1)
    if n_tokens == 0 or dominant_num <= 0 or dominant_num >= n_tokens:
        return None

    _, topk_indices = torch.topk(attn_logits, dominant_num)
    mask = torch.zeros_like(attn_logits, dtype=torch.bool)
    mask[topk_indices] = True
    contextual_mask = ~mask

    metric_filtered = attn_key[:, contextual_mask]
    metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True)

    step = max(1, metric_normalized.shape[1] // contextual_num)
    target_indices = torch.arange(
        0, metric_normalized.shape[1], step, device=metric_normalized.device
    )[:contextual_num]
    target_tokens = metric_normalized[:, target_indices, :]

    arange_ctx = torch.arange(
        metric_normalized.shape[1], device=metric_normalized.device
    )
    isin_target = torch.isin(arange_ctx, target_indices)
    tokens_to_merge = metric_normalized[:, ~isin_target, :]
    similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
    assign_one_hot = torch.zeros(
        tokens_to_merge.shape[0],
        tokens_to_merge.shape[1],
        contextual_num,
        dtype=attn_logits.dtype,
        device=metric_normalized.device,
    )
    assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
    counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)

    return {
        "topk_indices": topk_indices,
        "contextual_mask": contextual_mask,
        "target_indices": target_indices,
        "isin_target": isin_target,
        "assign_one_hot": assign_one_hot,
        "counts": counts,
    }


def _patched_internvl_chat_generate(
    self,
    pixel_values=None,
    input_ids=None,
    attention_mask=None,
    visual_features=None,
    generation_config=None,
    output_hidden_states=None,
    **generate_kwargs,
):
    """Replacement for `InternVLChatModel.generate` with VisionZip pruning.

    Mirrors the stock InternVL `generate` but:
      1. Runs vision encoder with a hook to also capture CLS-attention
         logits + last-layer keys (required for VisionZip contextual
         token selection).
      2. Optionally prunes vision tokens (dominant + contextual merge).
      3. Records `config.prompt_seqlen` and `config.query_seqlen` so the
         patched attention layers know the vision-token span.
      4. Replaces the LLM's past_key_values with our `DynamicCache` so
         ThinK / VisionK / SparK cached kv pruning actually kicks in.
    """
    assert self.img_context_token_id is not None
    config = self.config

    # NOTE: VisionZip token pruning is currently disabled for InternVL2.5.
    # The original Qwen2.5-VL path prunes vision tokens using the ViT's
    # last-layer CLS-attention, but InternVL applies `pixel_shuffle` (4×
    # downsample) + tile concatenation between the ViT output and the LLM
    # input, so the [1024-per-tile] ViT attention scores don't align with
    # the [256-per-tile × num_tiles] vision span inside `input_embeds`.
    # Doing token pruning correctly requires per-tile selection before
    # `pixel_shuffle` (i.e. inside a patched `extract_feature`). That's a
    # future improvement; for now we only do channel pruning on the LLM
    # side, which is the main performance lever anyway.
    if pixel_values is not None:
        vit_embeds = (
            visual_features
            if visual_features is not None
            else self.extract_feature(pixel_values)
        )

        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds_flat = input_embeds.reshape(B * N, C)
        input_ids_flat = input_ids.reshape(B * N)
        selected = (input_ids_flat == self.img_context_token_id)
        assert selected.sum() != 0
        input_embeds_flat[selected] = vit_embeds.reshape(-1, C).to(input_embeds_flat.device)
        input_embeds = input_embeds_flat.reshape(B, N, C)

        # Locate vision-token span (contiguous block of IMG_CONTEXT placeholders).
        img_mask_1d = (input_ids[0] == self.img_context_token_id)
        img_positions = torch.nonzero(img_mask_1d, as_tuple=True)[0]
        # Attention layers read `prompt_seqlen` / `query_seqlen` from the LLM
        # sub-config (that's what their `self.config` references), NOT the
        # top-level InternVLChatConfig. Stamp both so either works.
        llm_config = getattr(config, "llm_config", config)
        if img_positions.numel() > 0:
            first = img_positions[0].item()
            last = img_positions[-1].item()
            prompt_seqlen = first
            # Clamp query_seqlen to >=1 to avoid `kv_states[:, :, -0:, :]`
            # returning the full tensor (Python slicing quirk); at least one
            # query token must exist or there's nothing to generate against.
            query_seqlen = max(1, input_ids.shape[-1] - last - 1)
        else:
            prompt_seqlen = 0
            query_seqlen = max(1, input_ids.shape[-1])
        config.prompt_seqlen = prompt_seqlen
        config.query_seqlen = query_seqlen
        llm_config.prompt_seqlen = prompt_seqlen
        llm_config.query_seqlen = query_seqlen
    else:
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

    # Do NOT pass a custom past_key_values here — HF's `generate()` does
    # `isinstance(cache, transformers.cache_utils.Cache)` and our cache is
    # not a subclass of it, so it would fall through to a tuple-indexing
    # path and KeyError. Instead we rely on the patched
    # `InternLM2Model.forward` to create our DynamicCache on the first
    # call and thread it through subsequent calls via `outputs.past_key_values`.
    outputs = self.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config,
        output_hidden_states=output_hidden_states,
        use_cache=True,
        **generate_kwargs,
    )
    return outputs


# ---------------------------------------------------------------------------
# Public entry point: patch a freshly loaded InternVL2.5 model.
# ---------------------------------------------------------------------------


def _configure_defaults(config):
    """Normalise config attributes consumed by the VisionZip attention path.

    Mirrors the `Qwen2_5_VLModel.__init__` block that seeds default values.
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
            f"Use one of: {sorted(_valid_reconstruction)}"
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
    # Ablation knob: force a single decoder layer to "full-channel" (channel_ratio=0)
    # while all other layers use the global/per-layer budget. None = disabled.
    if not hasattr(config, "exempt_layer_idx"):
        config.exempt_layer_idx = None
    # Fix the first N decoder layers to full-channel (channel_ratio=0). 0 = disabled.
    # Applies ON TOP OF exempt_layer_idx and takes precedence over it.
    if not hasattr(config, "full_channel_first_n_layers"):
        config.full_channel_first_n_layers = 0

    # The VisionK-specific knobs (layer_adaptive_channel_budget,
    # channel_reconstruction, custom_kernel) are no longer consumed by
    # the active code paths; force them to safe defaults.
    config.layer_adaptive_channel_budget = False
    config.channel_reconstruction = "off"
    config.custom_kernel = False
    # RotateK has its own calibration path (rotation_matrix); leave its
    # calibration_mode intact. Other methods force off.
    if config.channel_method != "rotatek":
        config.calibration_mode = "off"

    if config.channel_reconstruction not in ("off", "matrix"):
        config.custom_kernel = False


def _load_layer_adaptive_budget(config):
    budget_sparsity = round(float(config.channel_ratio), 3)
    calibration_dominant_ratio = _require_calibration_dominant_ratio(config)
    result_root = getattr(config, "result_root", DEFAULT_CALIBRATION_RESULT_ROOT)
    budget_path = os.path.join(
        result_root,
        f"mmstar_calibration_dominant_ratio_{calibration_dominant_ratio:.2f}",
        "attention_shift",
        f"layer_budget_sparsity_{budget_sparsity:.3f}.csv",
    )
    with open(budget_path, "r", newline="") as f:
        return {
            budget_sparsity: [float(line.strip()) for line in f if line.strip()]
        }


def _load_calibration_channel_importance(config, num_layers):
    calibration_mode = str(getattr(config, "calibration_mode", "off")).strip().lower()
    task_spec = getattr(config, "offline_calibration_tasks", "channel_importance")
    valid_tasks = {
        "channel_importance",
        "modality_score",
        "supplementary_matrix",
        "attention_shift",
        "attention_kl",
        "all",
    }
    tasks = {
        item.strip().lower()
        for item in str(task_spec).split(",")
        if item.strip() and item.strip().lower() in valid_tasks
    }
    tasks_requiring_importance = {"supplementary_matrix", "attention_shift", "attention_kl", "all"}
    should_load = calibration_mode == "use" or (
        calibration_mode == "collect"
        and any(task in tasks_requiring_importance for task in tasks)
    )
    if not should_load:
        return None

    calibration_dominant_ratio = _require_calibration_dominant_ratio(config)
    result_root = getattr(config, "result_root", DEFAULT_CALIBRATION_RESULT_ROOT)
    calibration_dir = os.path.join(
        result_root,
        f"mmstar_calibration_dominant_ratio_{calibration_dominant_ratio:.2f}",
        "channel_importance",
    )
    loaded = load_channel_importance_calibration(calibration_dir, num_layers=num_layers)
    if not loaded:
        raise FileNotFoundError(
            f"No channel-importance calibration files were loaded from: {calibration_dir}"
        )
    missing = [i for i in range(num_layers) if i not in loaded]
    if missing:
        raise ValueError(f"Missing channel-importance calibration for layers: {missing}")
    return loaded


def _load_calibration_rotation_matrix(config, num_layers):
    """Mirror of `_load_calibration_channel_importance` for RotateK's R.
    Only activates when calibration_mode=="use" AND channel_method=="rotatek".
    Returns dict {layer_idx: R_partial [num_heads, D, D_keep]} or None.
    """
    calibration_mode = str(getattr(config, "calibration_mode", "off")).strip().lower()
    channel_method = str(getattr(config, "channel_method", "")).strip().lower()
    if calibration_mode != "use" or channel_method != "rotatek":
        return None

    channel_ratio = float(getattr(config, "channel_ratio", 0.0))
    head_dim = int(getattr(config, "hidden_size", 0)) // int(
        getattr(config, "num_attention_heads", 1)
    )
    if head_dim <= 0:
        raise RuntimeError(
            "Cannot infer head_dim from config for RotateK calibration load"
        )
    prune_count = min(head_dim, max(0, int(head_dim * channel_ratio)))
    keep_count = head_dim - prune_count
    if keep_count <= 0:
        return None

    calibration_dominant_ratio = _require_calibration_dominant_ratio(config)
    result_root = getattr(config, "result_root", DEFAULT_CALIBRATION_RESULT_ROOT)
    calibration_dir = os.path.join(
        result_root,
        f"mmstar_calibration_dominant_ratio_{calibration_dominant_ratio:.2f}",
        "rotation_matrix",
    )
    loaded = load_rotation_matrix_calibration(
        calibration_dir, num_layers=num_layers, keep_count=keep_count,
    )
    if not loaded:
        raise FileNotFoundError(
            f"No rotation-matrix calibration files were loaded from: {calibration_dir}"
        )
    missing = [i for i in range(num_layers) if i not in loaded]
    if missing:
        raise ValueError(
            f"Missing rotation-matrix calibration for layers: {missing}"
        )
    return loaded


def apply_visionzip(model):
    """
    Install VisionZip + channel-pruning into a loaded InternVL2.5 model.

    Call this exactly once after `AutoModel.from_pretrained(...)`. It:
      1. Copies top-level visionzip attrs from `model.config` onto
         `model.config.llm_config` so the attention layers see them.
      2. Seeds default channel-pruning config values.
      3. Loads optional calibration CSVs and attaches per-layer tensors
         to each attention module.
      4. Monkey-patches each `InternLM2FlashAttention2.forward` and
         `InternVLChatModel.generate`.
    """
    import types

    top_config = model.config
    llm_config = top_config.llm_config

    # Propagate visionzip-relevant config knobs from the top-level config to
    # the LLM config (the attention layers' `self.config` reference).
    for key in (
        "dominant_ratio",
        "contextual_ratio",
        "channel_ratio",
        "channel_method",
        "layer_adaptive_channel_budget",
        "channel_reconstruction",
        "reconstruction_constant",
        "custom_kernel",
        "decode_attention_backend",
        "calibration_mode",
        "offline_calibration_tasks",
        "channel_start",
        "channel_end",
        "exempt_layer_idx",
        "full_channel_first_n_layers",
        "result_root",
        "_decode_profile",
        "_decode_profile_layers",
    ):
        if hasattr(top_config, key):
            setattr(llm_config, key, getattr(top_config, key))

    _configure_defaults(llm_config)

    num_layers = llm_config.num_hidden_layers

    # layer-adaptive budgets (optional).
    llm_config._layer_budget_by_sparsity = {}
    if getattr(llm_config, "layer_adaptive_channel_budget", False):
        llm_config._layer_budget_by_sparsity.update(
            _load_layer_adaptive_budget(llm_config)
        )

    calibration = _load_calibration_channel_importance(llm_config, num_layers)
    rotation_calibration = _load_calibration_rotation_matrix(llm_config, num_layers)

    # Patch each InternLM2FlashAttention2 layer AND the wrapping
    # InternLM2DecoderLayer. The decoder-layer patch adds the
    # attn-module-vs-FFN timing hooks that populate `_layer_timings`.
    lm_model = model.language_model.model  # InternLM2Model
    for layer_idx, decoder_layer in enumerate(lm_model.layers):
        decoder_layer.layer_idx = layer_idx

        attn = decoder_layer.attention
        attn.layer_idx = layer_idx
        attn.config = llm_config  # ensure patched forward sees the knobs
        layer_importance = (
            None if calibration is None else calibration[layer_idx]
        )
        attn.calibration_channel_importance = layer_importance
        attn.calibration_rotation_R_partial = (
            None if rotation_calibration is None else rotation_calibration[layer_idx]
        )
        attn.forward = types.MethodType(
            _patched_internlm2_flash_attention_forward, attn
        )

        decoder_layer.forward = types.MethodType(
            _patched_internlm2_decoder_layer_forward, decoder_layer
        )

    # Patch InternLM2Model.forward and prepare_inputs_for_generation so that
    # our DynamicCache is threaded through every layer and HF generate
    # doesn't try to index it as a tuple.
    lm_model.forward = types.MethodType(_patched_internlm2_model_forward, lm_model)
    model.language_model.prepare_inputs_for_generation = types.MethodType(
        _patched_internlm2_prepare_inputs_for_generation, model.language_model
    )

    # Flag used by patched forwards to distinguish VisionZip-active inference.
    llm_config._visionzip_enabled = True

    # Patch the top-level chat-model generate (vision token pruning).
    model.generate = types.MethodType(_patched_internvl_chat_generate, model)

    return model


def finalize_decode_timings(model) -> dict:
    """Convert collected event pairs into float timings.

    Patched forwards store `(start_event, end_event)` tuples per phase to
    avoid per-layer `torch.cuda.synchronize()`. This function performs ONE
    sync and computes `elapsed_time()` for every pair, populating:

      - `config.llm_config._decode_timings_by_layer`   (attention internals
         per layer, list[float] per phase)
      - `config.llm_config._layer_timings`             (layer-0 attention
         module vs FFN, list[float] per phase)

    Idempotent: safe to call repeatedly. After conversion the raw event
    buffers are cleared so the next run starts fresh.

    Returns the attention-internals dict (for backward compat with earlier
    callers). `_layer_timings` is accessible directly on the config.
    """
    cfg = getattr(model, "config", None)
    llm_cfg = getattr(cfg, "llm_config", cfg)

    attn_event_pairs = getattr(llm_cfg, "_decode_event_pairs_by_layer", None)
    layer_event_pairs = getattr(llm_cfg, "_layer_event_pairs", None)

    if not attn_event_pairs and not layer_event_pairs:
        return getattr(llm_cfg, "_decode_timings_by_layer", None) or {}

    # Single device-wide sync covers both event sets.
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    def _to_ms(start_evt, end_evt):
        if start_evt is None or end_evt is None:
            return 0.0
        return start_evt.elapsed_time(end_evt)

    # ---- Attention-internals per layer ----
    if attn_event_pairs:
        _all_timings = getattr(llm_cfg, "_decode_timings_by_layer", None)
        if _all_timings is None:
            _all_timings = {}
            llm_cfg._decode_timings_by_layer = _all_timings
        for layer_idx, events in attn_event_pairs.items():
            _timings = _all_timings.get(layer_idx)
            if _timings is None:
                _timings = {k: [] for k in events.keys()}
                _all_timings[layer_idx] = _timings
            for phase, pairs in events.items():
                bucket = _timings.setdefault(phase, [])
                for s, e in pairs:
                    bucket.append(_to_ms(s, e))
        if 0 in _all_timings:
            llm_cfg._decode_timings = _all_timings[0]
        llm_cfg._decode_event_pairs_by_layer = None

    # ---- Layer-0 attention-module vs FFN ----
    if layer_event_pairs:
        _layer_timings = getattr(llm_cfg, "_layer_timings", None)
        if _layer_timings is None:
            _layer_timings = {k: [] for k in layer_event_pairs.keys()}
            llm_cfg._layer_timings = _layer_timings
        for key, pairs in layer_event_pairs.items():
            bucket = _layer_timings.setdefault(key, [])
            for s, e in pairs:
                bucket.append(_to_ms(s, e))
        llm_cfg._layer_event_pairs = None

    return getattr(llm_cfg, "_decode_timings_by_layer", None) or {}


def print_decode_timings(model, max_layers: int = 8):
    """Pretty-print the per-phase decode timings captured via `_decode_profile`.

    Averages are in ms (wall-clock per decode step per layer). Call after
    evaluation finishes, e.g.::

        from lmms_eval.models.model_utils.internvl.internvl2_5_visionzip import (
            print_decode_timings,
        )
        print_decode_timings(lm.model)
    """
    timings_by_layer = finalize_decode_timings(model)
    if not timings_by_layer:
        print("[decode_timings] no data collected (is `_decode_profile` on?)")
        return

    phase_order = [
        "qkv_proj_rope_ms",
        "qkv_proj_ms",
        "rope_ms",
        "cache_update_ms",
        "recovery_ms",
        "cat_ms",
        "custom_decode_kernel_ms",
        "transpose_ms",
        "fa2_ms",
        "attn_out_proj_ms",
    ]
    layers = sorted(timings_by_layer.keys())[:max_layers]
    print(f"[decode_timings] averaged over {{n}} decode steps per layer "
          f"(showing layers {layers})")
    header = "layer  " + "  ".join(f"{p[:-3]:>12s}" for p in phase_order)
    print(header)
    print("-" * len(header))
    for li in layers:
        stats = timings_by_layer[li]
        n = len(stats.get(phase_order[0], []))
        cells = []
        for phase in phase_order:
            xs = stats.get(phase, [])
            avg = (sum(xs) / len(xs)) if xs else 0.0
            cells.append(f"{avg:12.3f}")
        print(f"{li:>5d}  " + "  ".join(cells) + f"   (n={n})")


__all__ = [
    "apply_visionzip",
    "load_channel_importance_calibration",
    "load_rotation_matrix_calibration",
    "run_custom_decode_kernel",
    "run_dense_decode_kernel",
    "print_decode_timings",
    "finalize_decode_timings",
]
