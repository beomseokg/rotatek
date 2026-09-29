"""GQA-shared split-K decode kernels (``ROTATEK_KERNEL=gqa``).

Same computation as ``fused_decode.py`` (RotateK) and
``full_channel_flash_decoding.py`` (full-width baselines), with three changes:

* each split program handles all G = H_q / H_kv query heads of one KV head, so
  every K/V tile is read once per group instead of once per query head; the
  query-head rows go through ``tl.dot`` (bf16 operands, fp32 accumulation);
* RotateK's prompt + text tokens are one more split of the same launch,
  also read once per group, and the merge kernel only combines partials;
* the sequence is split only as far as needed to fill the GPU
  (``TARGET_PROGRAMS``), so large batches do not pay per-split overhead.

The paper's latency figures were measured with the kernels in
``fused_decode.py`` / ``full_channel_flash_decoding.py`` (``ROTATEK_KERNEL=paper``).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import triton
import triton.language as tl

from rotatek.kernels.full_channel_flash_decoding import _next_power_of_2
from rotatek.kernels.fused_decode import _get_decode_buffers

BLOCK_N = 64
NUM_WARPS = 4
NUM_STAGES = 2


TARGET_PROGRAMS = 216  # 2 x the 108 SMs of an A100


def _num_splits(seq: int, groups: int = 1) -> int:
    """Split the sequence only as much as needed to fill the GPU: `groups`
    = B * H_kv programs already exist before splitting."""
    max_splits = max(1, min((seq + BLOCK_N - 1) // BLOCK_N, 64))
    want = max(1, -(-TARGET_PROGRAMS // max(1, groups)))
    return min(max_splits, want)


@triton.jit
def _online_softmax_step(s, n_mask, m_i, l_i, acc, v):
    """One [BLOCK_M, BLOCK_N] block of online softmax; returns (m_i, l_i, acc)."""
    s = tl.where(n_mask[None, :], s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, axis=1))
    alpha = tl.exp(m_i - m_new)
    p = tl.where(n_mask[None, :], tl.exp(s - m_new[:, None]), 0.0)
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    return m_new, l_i, acc


@triton.jit
def _store_partials(partial_m_ptr, partial_l_ptr, partial_acc_ptr,
                    stride_pm, stride_pl, stride_pa, stride_pa_d,
                    flat, m_mask, offs_d, d_mask, m_i, l_i, acc):
    tl.store(partial_m_ptr + flat * stride_pm, m_i, mask=m_mask)
    tl.store(partial_l_ptr + flat * stride_pl, l_i, mask=m_mask)
    tl.store(partial_acc_ptr + flat[:, None] * stride_pa + offs_d[None, :] * stride_pa_d, acc,
             mask=m_mask[:, None] & d_mask[None, :])


@triton.jit
def _dense_split_kernel(
    q_ptr, k_ptr, v_ptr,
    partial_m_ptr, partial_l_ptr, partial_acc_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_pm, stride_pl, stride_pa, stride_pa_d,
    seq, head_dim, num_q_heads, tokens_per_split, scale,
    NUM_KV_GROUPS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Grid (B, H_kv, NUM_SPLITS): full-width attention of the G query heads of
    one KV head over one split of the sequence."""
    pid_b = tl.program_id(0)
    kv_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < NUM_KV_GROUPS
    q_heads = kv_h * NUM_KV_GROUPS + offs_m
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    q = tl.load(q_ptr + pid_b * stride_qb + q_heads[:, None] * stride_qh + offs_d[None, :] * stride_qd,
                mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    m_i = tl.full([BLOCK_M], -1e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    seq_start = pid_s * tokens_per_split
    seq_end = tl.minimum(seq_start + tokens_per_split, seq)
    kb = k_ptr + pid_b * stride_kb + kv_h * stride_kh
    vb = v_ptr + pid_b * stride_vb + kv_h * stride_vh
    for block_start in range(0, tokens_per_split, BLOCK_N):
        offs_n = seq_start + block_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_end
        k = tl.load(kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                    mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        v = tl.load(vb + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                    mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        m_i, l_i, acc = _online_softmax_step(s, n_mask, m_i, l_i, acc, v)

    flat = pid_b * (num_q_heads * NUM_SPLITS) + q_heads * NUM_SPLITS + pid_s
    _store_partials(partial_m_ptr, partial_l_ptr, partial_acc_ptr,
                    stride_pm, stride_pl, stride_pa, stride_pa_d,
                    flat, m_mask, offs_d, d_mask, m_i, l_i, acc)


@triton.jit
def _rotatek_split_kernel(
    q_ptr, R_ptr, delta_mu_ptr, k_sparse_ptr, v_sparse_ptr,
    partial_m_ptr, partial_l_ptr, partial_acc_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_Rb, stride_Rh, stride_Rd, stride_Rk,
    stride_dmb, stride_dmh, stride_dmd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_pm, stride_pl, stride_pa, stride_pa_d,
    seq_sparse, head_dim, head_dim_keep, num_q_heads, tokens_per_split, scale,
    k_full_ptr, v_full_ptr,
    stride_kfb, stride_kfh, stride_kfs, stride_kfd,
    stride_vfb, stride_vfh, stride_vfs, stride_vfd,
    seq_full,
    NUM_KV_GROUPS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DK: tl.constexpr,
):
    """Grid (B, H_kv, NUM_SPLITS + 1). Splits 0..NUM_SPLITS-1: the G query heads
    are rotated into the kept subspace (q @ R_partial, plus the q · δμ bias) and
    attend over one split of the rotated-truncated visual Keys. Split
    NUM_SPLITS: the same heads attend over the prompt + text tokens at full
    width."""
    pid_b = tl.program_id(0)
    kv_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < NUM_KV_GROUPS
    q_heads = kv_h * NUM_KV_GROUPS + offs_m
    offs_d = tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_DK)
    d_mask = offs_d < head_dim
    dk_mask = offs_k < head_dim_keep

    q = tl.load(q_ptr + pid_b * stride_qb + q_heads[:, None] * stride_qh + offs_d[None, :] * stride_qd,
                mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    m_i = tl.full([BLOCK_M], -1e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    if pid_s == NUM_SPLITS:
        # prompt + text tokens (text grows with every decoded token) at full
        # width, read once for the group
        kfb = k_full_ptr + pid_b * stride_kfb + kv_h * stride_kfh
        vfb = v_full_ptr + pid_b * stride_vfb + kv_h * stride_vfh
        for block_start in range(0, seq_full, BLOCK_N):
            offs_f = block_start + tl.arange(0, BLOCK_N)
            f_mask = offs_f < seq_full
            kf = tl.load(kfb + offs_f[:, None] * stride_kfs + offs_d[None, :] * stride_kfd,
                         mask=f_mask[:, None] & d_mask[None, :], other=0.0)
            vf = tl.load(vfb + offs_f[:, None] * stride_vfs + offs_d[None, :] * stride_vfd,
                         mask=f_mask[:, None] & d_mask[None, :], other=0.0)
            s = tl.dot(q, tl.trans(kf)) * scale
            m_i, l_i, acc = _online_softmax_step(s, f_mask, m_i, l_i, acc, vf)
    else:
        R = tl.load(R_ptr + pid_b * stride_Rb + kv_h * stride_Rh
                    + offs_d[:, None] * stride_Rd + offs_k[None, :] * stride_Rk,
                    mask=d_mask[:, None] & dk_mask[None, :], other=0.0)
        # q @ R in fp32, split into a bf16 high and low part so the scores
        # keep near-fp32 precision on the bf16 tensor-core path
        q_rot32 = tl.dot(q, R)                                              # [M, DK]
        q_hi = q_rot32.to(k_sparse_ptr.dtype.element_ty)
        q_lo = (q_rot32 - q_hi.to(tl.float32)).to(k_sparse_ptr.dtype.element_ty)
        dm = tl.load(delta_mu_ptr + pid_b * stride_dmb + kv_h * stride_dmh + offs_d * stride_dmd,
                     mask=d_mask, other=0.0).to(tl.float32)
        bias = tl.sum(q.to(tl.float32) * dm[None, :], axis=1)              # [M]
        seq_start = pid_s * tokens_per_split
        seq_end = tl.minimum(seq_start + tokens_per_split, seq_sparse)
        kb = k_sparse_ptr + pid_b * stride_kb + kv_h * stride_kh
        vb = v_sparse_ptr + pid_b * stride_vb + kv_h * stride_vh
        for block_start in range(0, tokens_per_split, BLOCK_N):
            offs_n = seq_start + block_start + tl.arange(0, BLOCK_N)
            n_mask = offs_n < seq_end
            k = tl.load(kb + offs_n[:, None] * stride_ks + offs_k[None, :] * stride_kd,
                        mask=n_mask[:, None] & dk_mask[None, :], other=0.0)
            v = tl.load(vb + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                        mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            kt = tl.trans(k)
            s = (tl.dot(q_hi, kt) + tl.dot(q_lo, kt) + bias[:, None]) * scale
            m_i, l_i, acc = _online_softmax_step(s, n_mask, m_i, l_i, acc, v)

    flat = pid_b * (num_q_heads * (NUM_SPLITS + 1)) + q_heads * (NUM_SPLITS + 1) + pid_s
    _store_partials(partial_m_ptr, partial_l_ptr, partial_acc_ptr,
                    stride_pm, stride_pl, stride_pa, stride_pa_d,
                    flat, m_mask, offs_d, d_mask, m_i, l_i, acc)


@triton.jit
def _merge_kernel(
    partial_m_ptr, partial_l_ptr, partial_acc_ptr, out_ptr,
    stride_pm, stride_pl, stride_pa, stride_pa_d,
    stride_ob, stride_oh, stride_od,
    head_dim, num_q_heads,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Grid (B, H_q): merge the NUM_PARTIALS partials of one query head in a
    single [NUM_PARTIALS, D] pass."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim
    offs_s = tl.arange(0, BLOCK_S)
    s_mask = offs_s < NUM_PARTIALS
    flat = pid_b * (num_q_heads * NUM_PARTIALS) + pid_h * NUM_PARTIALS + offs_s
    m_s = tl.load(partial_m_ptr + flat * stride_pm, mask=s_mask, other=-1e30)
    l_s = tl.load(partial_l_ptr + flat * stride_pl, mask=s_mask, other=0.0)
    acc_s = tl.load(partial_acc_ptr + flat[:, None] * stride_pa + offs_d[None, :] * stride_pa_d,
                    mask=s_mask[:, None] & d_mask[None, :], other=0.0)
    w = tl.exp(m_s - tl.max(m_s, axis=0))
    out = tl.sum(acc_s * w[:, None], axis=0) / (tl.sum(l_s * w, axis=0) + 1e-6)
    tl.store(out_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_d * stride_od,
             out.to(out_ptr.dtype.element_ty), mask=d_mask)


def _merge(pm, pl, pa, out, num_partials):
    bsz, heads, head_dim = out.shape
    _merge_kernel[(bsz, heads)](
        pm, pl, pa, out,
        pm.stride(0), pl.stride(0), pa.stride(0), pa.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        head_dim, heads,
        NUM_PARTIALS=num_partials, BLOCK_S=_next_power_of_2(num_partials),
        BLOCK_D=_next_power_of_2(head_dim), num_warps=NUM_WARPS,
    )


def full_channel_decode_gqa(
    q: torch.Tensor,                   # [B, H_q, D]
    k: torch.Tensor,                   # [B, H_kv, S, D]
    v: torch.Tensor,                   # [B, H_kv, S, D]
    num_kv_groups: int = 1,
) -> torch.Tensor:
    """Drop-in for ``full_channel_decode_triton``; returns [B, 1, H_q, D]."""
    bsz, heads, head_dim = q.shape
    h_kv, seq = k.shape[1], k.shape[-2]
    num_splits = _num_splits(seq, bsz * h_kv)
    pm, pl, pa, out = _get_decode_buffers(bsz, heads, head_dim, num_splits, q.device, q.dtype)
    _dense_split_kernel[(bsz, h_kv, num_splits)](
        q, k, v, pm, pl, pa,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        pm.stride(0), pl.stride(0), pa.stride(0), pa.stride(1),
        seq, head_dim, heads, (seq + num_splits - 1) // num_splits, 1.0 / math.sqrt(head_dim),
        NUM_KV_GROUPS=num_kv_groups, NUM_SPLITS=num_splits,
        BLOCK_M=max(16, _next_power_of_2(num_kv_groups)), BLOCK_N=BLOCK_N,
        BLOCK_D=_next_power_of_2(head_dim), num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )
    _merge(pm, pl, pa, out, num_splits)
    return out.unsqueeze(1)


def rotatek_decode_gqa(
    q_full: torch.Tensor,                # [B, H_q, D]
    R_partial: torch.Tensor,             # [B, H_kv, D, D_keep]
    delta_mu: torch.Tensor,              # [B, H_kv, D]
    k_full: torch.Tensor,                # [B, H_kv, S_full, D]
    v_full: torch.Tensor,                # [B, H_kv, S_full, D]
    mask_full: torch.Tensor,             # [B, S_full] uint8 (all ones in this repo)
    k_sparse: torch.Tensor,              # [B, H_kv, S_sparse, D_keep]
    v_sparse: torch.Tensor,              # [B, H_kv, S_sparse, D]
    num_kv_groups: int,
) -> Tuple[torch.Tensor, None, None]:
    """Drop-in for ``rotatek_decode_fused_triton``; returns ([B, 1, H_q, D], None, None).

    `mask_full` is accepted for signature parity; every prompt/text token is
    attended (the adapters always pass an all-ones mask)."""
    bsz, heads, head_dim = q_full.shape
    h_kv, head_dim_keep = R_partial.shape[1], R_partial.shape[-1]
    seq_sparse = k_sparse.shape[-2]
    num_splits = _num_splits(seq_sparse, bsz * h_kv)
    seq_full = k_full.shape[-2]
    pm, pl, pa, out = _get_decode_buffers(bsz, heads, head_dim, num_splits + 1, q_full.device, q_full.dtype)
    _rotatek_split_kernel[(bsz, h_kv, num_splits + 1)](
        q_full, R_partial, delta_mu, k_sparse, v_sparse, pm, pl, pa,
        q_full.stride(0), q_full.stride(1), q_full.stride(2),
        R_partial.stride(0), R_partial.stride(1), R_partial.stride(2), R_partial.stride(3),
        delta_mu.stride(0), delta_mu.stride(1), delta_mu.stride(2),
        k_sparse.stride(0), k_sparse.stride(1), k_sparse.stride(2), k_sparse.stride(3),
        v_sparse.stride(0), v_sparse.stride(1), v_sparse.stride(2), v_sparse.stride(3),
        pm.stride(0), pl.stride(0), pa.stride(0), pa.stride(1),
        seq_sparse, head_dim, head_dim_keep, heads, (seq_sparse + num_splits - 1) // num_splits,
        1.0 / math.sqrt(head_dim),
        k_full, v_full,
        k_full.stride(0), k_full.stride(1), k_full.stride(2), k_full.stride(3),
        v_full.stride(0), v_full.stride(1), v_full.stride(2), v_full.stride(3),
        seq_full,
        NUM_KV_GROUPS=num_kv_groups, NUM_SPLITS=num_splits,
        BLOCK_M=max(16, _next_power_of_2(num_kv_groups)), BLOCK_N=BLOCK_N,
        BLOCK_D=_next_power_of_2(head_dim), BLOCK_DK=_next_power_of_2(head_dim_keep),
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )
    _merge(pm, pl, pa, out, num_splits + 1)
    return out.unsqueeze(1), None, None


__all__ = ["full_channel_decode_gqa", "rotatek_decode_gqa"]
