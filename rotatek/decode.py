"""Triton fused decode-time attention for RotateK.

RotateK stores K_vision in a rotated truncated form ``[S_v, D_keep]`` and a
per-head rotation matrix ``R_partial`` ``[D, D_keep]``. At each decode step we:

  1. Rotate the new query via ``q_sparse = Q @ R_partial`` (in the adapter,
     fused into the call site).
  2. Compute attention with split-K Flash-Decoding:
       - vision tokens use the truncated D_keep dim (via q_sparse · k_sparse)
       - prompt/text tokens use the full D dim (via q_full · k_full)
       - logits are merged in a single online softmax pass.

Implementation choice
---------------------
This is *currently* a thin wrapper around VisionK's
``sparse_channel_decode_triton`` (the reconstruction-free variant of its
sparse-channel kernel). The two methods differ only in how ``q_sparse`` is
produced — VisionK gathers Q at ``keep_idx``, RotateK matmuls Q with
``R_partial`` — but both consume an externally-prepared ``[B, H_q, D_keep]``
tensor, so the same kernel works for both today.

A dedicated file is kept so that the **q-rotation can be fused into the
phase-1 kernel later** (e.g. read ``q_full`` + ``R_partial`` and produce
``q_sparse`` inside shared memory, eliminating the external matmul launch
and a roundtrip to HBM). When that fusion lands, the inner phase-1 kernels
will be specialised here without disturbing the VisionK code path.
"""
from __future__ import annotations

from typing import Tuple

import torch

from rotatek.kernels.sparse_channel_flash_decoding import (
    sparse_channel_decode_triton as _visionk_sparse_decode,
)
from rotatek.kernels.fused_decode import (
    rotatek_decode_fused_triton as _rotatek_fused,
)


def rotatek_decode_fused(
    q_full: torch.Tensor,
    R_partial: torch.Tensor,
    delta_mu: torch.Tensor | None,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    mask_full: torch.Tensor,
    k_sparse: torch.Tensor,
    v_sparse: torch.Tensor,
    num_kv_groups: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused-kernel RotateK decode — folds Q@R + δμ into the phase-1
    Triton kernel, eliminating two einsum launches at the call site.

    Inputs match the non-fused signature except `q_sparse`/`bias_shift`
    are replaced by the raw `R_partial` and `delta_mu` from prefill.
    """
    return _rotatek_fused(
        q_full=q_full,
        R_partial=R_partial,
        delta_mu=delta_mu,
        k_full=k_full,
        v_full=v_full,
        mask_full=mask_full,
        k_sparse=k_sparse,
        v_sparse=v_sparse,
        num_kv_groups=num_kv_groups,
    )


def rotatek_decode_triton(
    q_full: torch.Tensor,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    mask_full: torch.Tensor,
    q_sparse: torch.Tensor,
    k_sparse: torch.Tensor,
    v_sparse: torch.Tensor,
    mask_sparse: torch.Tensor | None = None,
    sparse_bias_shift: torch.Tensor | None = None,
    num_kv_groups: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split-K decode attention with vision tokens stored in truncated rotated basis.

    Inputs
    ------
    q_full           : [B, H_q, D]                       — full query (original basis)
    k_full           : [B, H_kv, S_full, D]              — non-vision keys (prompt + text)
    v_full           : [B, H_kv, S_full, D]              — non-vision values
    mask_full        : [B, S_full] uint8                 — validity mask for non-vision
    q_sparse         : [B, H_q, D_keep]                  — Q already rotated by R_partial
    k_sparse         : [B, H_kv, S_sparse, D_keep]       — vision keys, rotated truncated
    v_sparse         : [B, H_kv, S_sparse, D]            — vision values (full D)
    mask_sparse      : ignored (vision tokens are all valid)
    sparse_bias_shift: optional [B, H_q] per-head bias on sparse logits
    num_kv_groups    : H_q // H_kv (1 for MHA, >1 for GQA)

    Returns
    -------
    attn_output     : [B, 1, H_q, D]
    logits_full     : None (online softmax, never materialised)
    logits_sparse   : None

    Notes
    -----
    * Q rotation (``q_full @ R_partial → q_sparse``) is the caller's
      responsibility for now. A future fused phase-1 kernel will absorb it.
    * V is **not** rotated/truncated. Decode-time output dim must remain D so
      the model's downstream layers see the expected shape; the projection
      basis only affects QK^T, not V's role in attention output.
    """
    return _visionk_sparse_decode(
        q_full=q_full,
        k_full=k_full,
        v_full=v_full,
        mask_full=mask_full,
        q_sparse=q_sparse,
        k_sparse=k_sparse,
        v_sparse=v_sparse,
        mask_sparse=mask_sparse,
        sparse_bias_shift=sparse_bias_shift,
        num_kv_groups=num_kv_groups,
    )


__all__ = ["rotatek_decode_triton", "rotatek_decode_fused"]
