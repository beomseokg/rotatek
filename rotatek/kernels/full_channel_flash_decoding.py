"""Dense (full-channel) split-K decode attention via Triton.

A stripped-down kernel for the **all-channel** decode case used by:

  * Full     — no compression at all
  * ThinK    — after boolean-indexing recovery to full-D K
  * SparK    — after boolean-indexing recovery to full-D K

These all hand a single `[B, H_kv, S, D]` K/V buffer to attention.
RotateK does not use this kernel; see ``fused_decode.py``.

Public API: ``full_channel_decode_triton(q, k, v, num_kv_groups)``.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import triton
import triton.language as tl


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------
# Phase 1: each (b, h, split) produces an online-softmax partial.
# ---------------------------------------------------------------------------

@triton.jit
def _dense_phase1_kernel(
    q_ptr, k_ptr, v_ptr,
    partial_m_ptr, partial_l_ptr, partial_acc_ptr,
    # Q strides
    stride_qb, stride_qh, stride_qd,
    # K strides
    stride_kb, stride_kh, stride_ks, stride_kd,
    # V strides
    stride_vb, stride_vh, stride_vs, stride_vd,
    # partial-buffer strides
    stride_pm_bhs, stride_pl_bhs,
    stride_pa_bhs, stride_pa_d,
    seq, head_dim,
    tokens_per_split,
    scale,
    NUM_KV_GROUPS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)            # query-head index in [0, H_q)
    pid_s = tl.program_id(2)            # split index
    kv_h = pid_h // NUM_KV_GROUPS

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    # ---- load Q (single token) ----------------------------------------
    q_base = q_ptr + pid_b * stride_qb + pid_h * stride_qh
    q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)

    # ---- online-softmax state -----------------------------------------
    m = tl.full([], -1e30, dtype=tl.float32)
    l = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # ---- this split's slice of K/V ------------------------------------
    seq_start = pid_s * tokens_per_split
    seq_end = tl.minimum(seq_start + tokens_per_split, seq)

    for block_start in range(0, tokens_per_split, BLOCK_N):
        offs_n = seq_start + block_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_end

        kb = k_ptr + pid_b * stride_kb + kv_h * stride_kh
        k = tl.load(
            kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
            mask=n_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)
        logits = tl.sum(k * q[None, :], axis=1) * scale
        logits = tl.where(n_mask, logits, float("-inf"))

        vb = v_ptr + pid_b * stride_vb + kv_h * stride_vh
        v = tl.load(
            vb + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
            mask=n_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)

        block_max = tl.max(logits)
        new_max = tl.maximum(m, block_max)
        correction = tl.exp(m - new_max)
        block_exp = tl.exp(logits - new_max)
        block_exp = tl.where(n_mask, block_exp, 0.0)
        l = l * correction + tl.sum(block_exp)
        acc = acc * correction + tl.sum(block_exp[:, None] * v, axis=0)
        m = new_max

    # ---- store partial outputs ---------------------------------------
    flat_idx = pid_b * (tl.num_programs(1) * NUM_SPLITS) + pid_h * NUM_SPLITS + pid_s
    tl.store(partial_m_ptr + flat_idx * stride_pm_bhs, m)
    tl.store(partial_l_ptr + flat_idx * stride_pl_bhs, l)
    pa_base = partial_acc_ptr + flat_idx * stride_pa_bhs
    tl.store(pa_base + offs_d * stride_pa_d, acc, mask=d_mask)


# ---------------------------------------------------------------------------
# Phase 2: combine NUM_SPLITS partials into the final attention output.
# ---------------------------------------------------------------------------

@triton.jit
def _dense_phase2_kernel(
    partial_m_ptr, partial_l_ptr, partial_acc_ptr,
    out_ptr,
    stride_pm_bhs, stride_pl_bhs,
    stride_pa_bhs, stride_pa_d,
    stride_ob, stride_oh, stride_od,
    num_heads,
    head_dim,
    NUM_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    global_m = tl.full([], -1e30, dtype=tl.float32)
    for s in range(NUM_SPLITS):
        flat = pid_b * (num_heads * NUM_SPLITS) + pid_h * NUM_SPLITS + s
        m_s = tl.load(partial_m_ptr + flat * stride_pm_bhs)
        global_m = tl.maximum(global_m, m_s)

    global_l = tl.zeros([], dtype=tl.float32)
    global_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for s in range(NUM_SPLITS):
        flat = pid_b * (num_heads * NUM_SPLITS) + pid_h * NUM_SPLITS + s
        m_s = tl.load(partial_m_ptr + flat * stride_pm_bhs)
        l_s = tl.load(partial_l_ptr + flat * stride_pl_bhs)
        acc_s = tl.load(
            partial_acc_ptr + flat * stride_pa_bhs + offs_d * stride_pa_d,
            mask=d_mask, other=0.0,
        )
        correction = tl.exp(m_s - global_m)
        global_l += l_s * correction
        global_acc += acc_s * correction

    output = global_acc / (global_l + 1e-6)
    out_base = out_ptr + pid_b * stride_ob + pid_h * stride_oh
    tl.store(out_base + offs_d * stride_od, output.to(out_ptr.dtype.element_ty), mask=d_mask)


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------

def full_channel_decode_triton(
    q: torch.Tensor,                   # [B, H_q, D]
    k: torch.Tensor,                   # [B, H_kv, S, D]
    v: torch.Tensor,                   # [B, H_kv, S, D]
    num_kv_groups: int = 1,
) -> torch.Tensor:
    """Single-token full-channel decode attention via split-K Flash-Decoding Triton.

    Returns: ``[B, 1, H_q, D]`` (matches FA2's output layout so the caller
    can drop it into the same downstream code path).
    """
    if q.dim() != 3:
        raise ValueError("q must be [B, H_q, D]")
    bsz, heads, head_dim = q.shape
    seq = k.shape[-2]
    if k.shape[-1] != head_dim or v.shape[-1] != head_dim:
        raise ValueError("k/v head_dim must match q's")

    scale = 1.0 / math.sqrt(head_dim)
    block_d = _next_power_of_2(head_dim)

    BLOCK_N = 64
    num_splits = max(1, min((seq + BLOCK_N - 1) // BLOCK_N, 64))
    tokens_per_split = (seq + num_splits - 1) // num_splits

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    total_partials = bsz * heads * num_splits
    partial_m = torch.empty(total_partials, device=q.device, dtype=torch.float32)
    partial_l = torch.empty(total_partials, device=q.device, dtype=torch.float32)
    partial_acc = torch.empty(total_partials, head_dim, device=q.device, dtype=torch.float32)
    out = torch.empty((bsz, heads, head_dim), device=q.device, dtype=q.dtype)

    grid_phase1 = (bsz, heads, num_splits)
    _dense_phase1_kernel[grid_phase1](
        q, k, v,
        partial_m, partial_l, partial_acc,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        partial_m.stride(0), partial_l.stride(0),
        partial_acc.stride(0), partial_acc.stride(1),
        seq, head_dim,
        tokens_per_split,
        scale,
        NUM_KV_GROUPS=num_kv_groups,
        NUM_SPLITS=num_splits,
        BLOCK_N=BLOCK_N,
        BLOCK_D=block_d,
    )

    grid_phase2 = (bsz, heads)
    _dense_phase2_kernel[grid_phase2](
        partial_m, partial_l, partial_acc,
        out,
        partial_m.stride(0), partial_l.stride(0),
        partial_acc.stride(0), partial_acc.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        heads,
        head_dim,
        NUM_SPLITS=num_splits,
        BLOCK_D=block_d,
    )

    # [B, H, D] → [B, 1, H, D] to match FA2's output shape.
    return out.unsqueeze(1)


__all__ = ["full_channel_decode_triton"]
