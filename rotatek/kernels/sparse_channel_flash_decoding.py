"""Fused Triton kernel for channel-sparse decode attention with online softmax.

Split-K (Flash-Decoding style) parallelism:
  Phase 1 — Each CTA handles a portion of the KV sequence, writes partial (m, l, acc).
  Phase 2 — Merge CTA combines partials into the final output.

Dense (prompt/text) tokens are handled by split 0 only (typically short).
Sparse (vision) tokens are evenly distributed across all splits.
"""

from __future__ import annotations

import math
import os
from typing import Tuple

import torch
import triton
import triton.language as tl


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


# ---------------------------------------------------------------------------
# Fused q_sparse preparation: gather + optional matrix reconstruction
# Replaces ~6 PyTorch kernel launches with 1 Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _prepare_q_kernel(
    q_full_ptr,
    keep_idx_ptr,
    q_sparse_out_ptr,
    bias_shift_out_ptr,
    pruned_idx_ptr,
    w_t_ptr,
    recon_bias_ptr,
    stride_qf_b, stride_qf_h, stride_qf_d,
    stride_ki_b, stride_ki_h, stride_ki_d,
    stride_qs_b, stride_qs_h, stride_qs_d,
    stride_wt_h, stride_wt_p, stride_wt_k,
    head_dim,
    head_dim_keep,
    pruned_dim,
    HAS_MATRIX_RECON: tl.constexpr,
    NUM_KV_GROUPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    kv_h = pid_h // NUM_KV_GROUPS

    offs_d = tl.arange(0, BLOCK_D)
    dk_mask = offs_d < head_dim_keep

    ki_base = keep_idx_ptr + pid_b * stride_ki_b + kv_h * stride_ki_h
    keep_idx = tl.load(ki_base + offs_d * stride_ki_d, mask=dk_mask, other=0).to(tl.int64)

    qf_base = q_full_ptr + pid_b * stride_qf_b + pid_h * stride_qf_h
    q_sparse = tl.load(qf_base + keep_idx * stride_qf_d, mask=dk_mask, other=0.0).to(tl.float32)

    bias_val = 0.0

    if HAS_MATRIX_RECON:
        offs_p = tl.arange(0, BLOCK_P)
        p_mask = offs_p < pruned_dim

        pi_base = pruned_idx_ptr + kv_h * pruned_dim
        pruned_idx = tl.load(pi_base + offs_p, mask=p_mask, other=0).to(tl.int64)
        q_pruned = tl.load(qf_base + pruned_idx * stride_qf_d, mask=p_mask, other=0.0).to(tl.float32)

        wt_base = w_t_ptr + kv_h * stride_wt_h
        w_t = tl.load(
            wt_base + offs_p[:, None] * stride_wt_p + offs_d[None, :] * stride_wt_k,
            mask=p_mask[:, None] & dk_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        q_recon = tl.sum(q_pruned[:, None] * w_t, axis=0)
        q_sparse = q_sparse + q_recon

        rb_base = recon_bias_ptr + kv_h * pruned_dim
        recon_bias = tl.load(rb_base + offs_p, mask=p_mask, other=0.0).to(tl.float32)
        bias_val = tl.sum(q_pruned * recon_bias)

    qs_base = q_sparse_out_ptr + pid_b * stride_qs_b + pid_h * stride_qs_h
    tl.store(qs_base + offs_d * stride_qs_d, q_sparse.to(q_sparse_out_ptr.dtype.element_ty), mask=dk_mask)

    if HAS_MATRIX_RECON:
        bs_idx = pid_b * tl.num_programs(1) + pid_h
        tl.store(bias_shift_out_ptr + bs_idx, bias_val.to(bias_shift_out_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Phase 1: each (batch, head, split) computes a partial online-softmax result
# ---------------------------------------------------------------------------

@triton.jit
def _splitk_phase1_kernel(
    # ---- pointers ----
    q_full_ptr, q_sparse_ptr,
    k_full_ptr, k_sparse_ptr,
    v_full_ptr, v_sparse_ptr,
    mask_full_ptr,
    bias_shift_ptr,
    partial_m_ptr, partial_l_ptr, partial_acc_ptr,
    # ---- Q strides ----
    stride_qf_b, stride_qf_h, stride_qf_d,
    stride_qs_b, stride_qs_h, stride_qs_d,
    # ---- K strides ----
    stride_kf_b, stride_kf_h, stride_kf_s, stride_kf_d,
    stride_ks_b, stride_ks_h, stride_ks_s, stride_ks_d,
    # ---- V strides ----
    stride_vf_b, stride_vf_h, stride_vf_s, stride_vf_d,
    stride_vs_b, stride_vs_h, stride_vs_s, stride_vs_d,
    # ---- mask strides ----
    stride_mf_b, stride_mf_s,
    # ---- bias_shift strides ----
    stride_bs_b, stride_bs_h,
    # ---- partial buffer strides ----
    stride_pm_bhs,  # partial_m is [B*H*num_splits] flat
    stride_pl_bhs,
    stride_pa_bhs, stride_pa_d,  # partial_acc is [B*H*num_splits, D]
    # ---- dimensions ----
    seq_full, seq_sparse, head_dim, head_dim_keep,
    sparse_per_split,
    scale,
    HAS_BIAS_SHIFT: tl.constexpr,
    NUM_KV_GROUPS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)  # query head index
    pid_s = tl.program_id(2)
    kv_h = pid_h // NUM_KV_GROUPS  # KV head index for GQA

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim
    dk_mask = offs_d < head_dim_keep

    # --- load bias shift ---
    if HAS_BIAS_SHIFT:
        bias_val = tl.load(bias_shift_ptr + pid_b * stride_bs_b + pid_h * stride_bs_h).to(tl.float32)
    else:
        bias_val = 0.0

    # --- load queries ---
    qf_base = q_full_ptr + pid_b * stride_qf_b + pid_h * stride_qf_h
    q_full = tl.load(qf_base + offs_d * stride_qf_d, mask=d_mask, other=0.0).to(tl.float32)

    qs_base = q_sparse_ptr + pid_b * stride_qs_b + pid_h * stride_qs_h
    q_sparse = tl.load(qs_base + offs_d * stride_qs_d, mask=dk_mask, other=0.0).to(tl.float32)

    # --- online softmax state ---
    m = tl.full([], -1e30, dtype=tl.float32)
    l = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # ---- split 0: process all dense (full-channel) tokens ----
    if pid_s == 0:
        for block_start in range(0, seq_full, BLOCK_N):
            offs_n = block_start + tl.arange(0, BLOCK_N)
            n_mask = offs_n < seq_full

            kf_base = k_full_ptr + pid_b * stride_kf_b + kv_h * stride_kf_h
            k = tl.load(
                kf_base + offs_n[:, None] * stride_kf_s + offs_d[None, :] * stride_kf_d,
                mask=n_mask[:, None] & d_mask[None, :], other=0.0,
            ).to(tl.float32)
            logits = tl.sum(k * q_full[None, :], axis=1) * scale

            valid = tl.load(mask_full_ptr + pid_b * stride_mf_b + offs_n * stride_mf_s, mask=n_mask, other=0)
            logits = tl.where((valid != 0) & n_mask, logits, float("-inf"))

            vf_base = v_full_ptr + pid_b * stride_vf_b + kv_h * stride_vf_h
            v = tl.load(
                vf_base + offs_n[:, None] * stride_vf_s + offs_d[None, :] * stride_vf_d,
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

    # ---- all splits: process their chunk of sparse tokens ----
    sparse_start = pid_s * sparse_per_split
    sparse_end = tl.minimum(sparse_start + sparse_per_split, seq_sparse)

    for block_start in range(0, sparse_per_split, BLOCK_N):
        offs_n = sparse_start + block_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < sparse_end

        ks_base = k_sparse_ptr + pid_b * stride_ks_b + kv_h * stride_ks_h
        k = tl.load(
            ks_base + offs_n[:, None] * stride_ks_s + offs_d[None, :] * stride_ks_d,
            mask=n_mask[:, None] & dk_mask[None, :], other=0.0,
        ).to(tl.float32)
        logits = tl.sum(k * q_sparse[None, :], axis=1) * scale + bias_val * scale

        logits = tl.where(n_mask, logits, float("-inf"))

        vs_base = v_sparse_ptr + pid_b * stride_vs_b + kv_h * stride_vs_h
        v = tl.load(
            vs_base + offs_n[:, None] * stride_vs_s + offs_d[None, :] * stride_vs_d,
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

    # ---- store partial results ----
    flat_idx = pid_b * (tl.num_programs(1) * NUM_SPLITS) + pid_h * NUM_SPLITS + pid_s
    tl.store(partial_m_ptr + flat_idx * stride_pm_bhs, m)
    tl.store(partial_l_ptr + flat_idx * stride_pl_bhs, l)
    pa_base = partial_acc_ptr + flat_idx * stride_pa_bhs
    tl.store(pa_base + offs_d * stride_pa_d, acc, mask=d_mask)


# ---------------------------------------------------------------------------
# Phase 1 GQA-reuse variant: one CTA processes two query heads for one KV head.
# This reduces repeated K/V loads for GQA models while keeping register pressure
# much lower than processing the full query group in a single Triton program.
# ---------------------------------------------------------------------------

@triton.jit
def _splitk_phase1_gqa2_kernel(
    # ---- pointers ----
    q_full_ptr, q_sparse_ptr,
    k_full_ptr, k_sparse_ptr,
    v_full_ptr, v_sparse_ptr,
    mask_full_ptr,
    bias_shift_ptr,
    partial_m_ptr, partial_l_ptr, partial_acc_ptr,
    # ---- Q strides ----
    stride_qf_b, stride_qf_h, stride_qf_d,
    stride_qs_b, stride_qs_h, stride_qs_d,
    # ---- K strides ----
    stride_kf_b, stride_kf_h, stride_kf_s, stride_kf_d,
    stride_ks_b, stride_ks_h, stride_ks_s, stride_ks_d,
    # ---- V strides ----
    stride_vf_b, stride_vf_h, stride_vf_s, stride_vf_d,
    stride_vs_b, stride_vs_h, stride_vs_s, stride_vs_d,
    # ---- mask strides ----
    stride_mf_b, stride_mf_s,
    # ---- bias_shift strides ----
    stride_bs_b, stride_bs_h,
    # ---- partial buffer strides ----
    stride_pm_bhs,
    stride_pl_bhs,
    stride_pa_bhs, stride_pa_d,
    # ---- dimensions ----
    seq_full, seq_sparse, head_dim, head_dim_keep,
    sparse_per_split,
    scale,
    HAS_BIAS_SHIFT: tl.constexpr,
    NUM_KV_GROUPS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_kv = tl.program_id(1)
    pid_gs = tl.program_id(2)
    pid_gt = pid_gs // NUM_SPLITS  # query-head tile within this KV group, tile size 2
    pid_s = pid_gs - pid_gt * NUM_SPLITS

    qh0 = pid_kv * NUM_KV_GROUPS + pid_gt * 2
    qh1 = qh0 + 1
    q0_valid = qh0 < NUM_Q_HEADS
    q1_valid = (pid_gt * 2 + 1 < NUM_KV_GROUPS) & (qh1 < NUM_Q_HEADS)

    offs_d = tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)
    d_mask = offs_d < head_dim
    dk_mask = offs_k < head_dim_keep

    # --- load bias shift ---
    if HAS_BIAS_SHIFT:
        bias0 = tl.load(bias_shift_ptr + pid_b * stride_bs_b + qh0 * stride_bs_h, mask=q0_valid, other=0.0).to(tl.float32)
        bias1 = tl.load(bias_shift_ptr + pid_b * stride_bs_b + qh1 * stride_bs_h, mask=q1_valid, other=0.0).to(tl.float32)
    else:
        bias0 = 0.0
        bias1 = 0.0

    # --- load two queries sharing the same KV head ---
    qf_base0 = q_full_ptr + pid_b * stride_qf_b + qh0 * stride_qf_h
    qf_base1 = q_full_ptr + pid_b * stride_qf_b + qh1 * stride_qf_h
    q_full0 = tl.load(qf_base0 + offs_d * stride_qf_d, mask=q0_valid & d_mask, other=0.0).to(tl.float32)
    q_full1 = tl.load(qf_base1 + offs_d * stride_qf_d, mask=q1_valid & d_mask, other=0.0).to(tl.float32)

    qs_base0 = q_sparse_ptr + pid_b * stride_qs_b + qh0 * stride_qs_h
    qs_base1 = q_sparse_ptr + pid_b * stride_qs_b + qh1 * stride_qs_h
    q_sparse0 = tl.load(qs_base0 + offs_k * stride_qs_d, mask=q0_valid & dk_mask, other=0.0).to(tl.float32)
    q_sparse1 = tl.load(qs_base1 + offs_k * stride_qs_d, mask=q1_valid & dk_mask, other=0.0).to(tl.float32)

    # --- online softmax state for two query heads ---
    m0 = tl.full([], -1e30, dtype=tl.float32)
    l0 = tl.zeros([], dtype=tl.float32)
    acc0 = tl.zeros([BLOCK_D], dtype=tl.float32)
    m1 = tl.full([], -1e30, dtype=tl.float32)
    l1 = tl.zeros([], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_D], dtype=tl.float32)

    # ---- split 0: process all dense (full-channel) tokens ----
    if pid_s == 0:
        kf_base = k_full_ptr + pid_b * stride_kf_b + pid_kv * stride_kf_h
        vf_base = v_full_ptr + pid_b * stride_vf_b + pid_kv * stride_vf_h
        for block_start in range(0, seq_full, BLOCK_N):
            offs_n = block_start + tl.arange(0, BLOCK_N)
            n_mask = offs_n < seq_full

            k = tl.load(
                kf_base + offs_n[:, None] * stride_kf_s + offs_d[None, :] * stride_kf_d,
                mask=n_mask[:, None] & d_mask[None, :], other=0.0,
            ).to(tl.float32)
            v = tl.load(
                vf_base + offs_n[:, None] * stride_vf_s + offs_d[None, :] * stride_vf_d,
                mask=n_mask[:, None] & d_mask[None, :], other=0.0,
            ).to(tl.float32)
            valid = tl.load(mask_full_ptr + pid_b * stride_mf_b + offs_n * stride_mf_s, mask=n_mask, other=0)
            valid_mask = (valid != 0) & n_mask

            logits0 = tl.sum(k * q_full0[None, :], axis=1) * scale
            logits0 = tl.where(valid_mask & q0_valid, logits0, float("-inf"))
            block_max0 = tl.max(logits0)
            new_max0 = tl.maximum(m0, block_max0)
            correction0 = tl.exp(m0 - new_max0)
            block_exp0 = tl.exp(logits0 - new_max0)
            block_exp0 = tl.where(valid_mask & q0_valid, block_exp0, 0.0)
            l0 = l0 * correction0 + tl.sum(block_exp0)
            acc0 = acc0 * correction0 + tl.sum(block_exp0[:, None] * v, axis=0)
            m0 = new_max0

            logits1 = tl.sum(k * q_full1[None, :], axis=1) * scale
            logits1 = tl.where(valid_mask & q1_valid, logits1, float("-inf"))
            block_max1 = tl.max(logits1)
            new_max1 = tl.maximum(m1, block_max1)
            correction1 = tl.exp(m1 - new_max1)
            block_exp1 = tl.exp(logits1 - new_max1)
            block_exp1 = tl.where(valid_mask & q1_valid, block_exp1, 0.0)
            l1 = l1 * correction1 + tl.sum(block_exp1)
            acc1 = acc1 * correction1 + tl.sum(block_exp1[:, None] * v, axis=0)
            m1 = new_max1

    # ---- all splits: process their chunk of sparse tokens ----
    sparse_start = pid_s * sparse_per_split
    sparse_end = tl.minimum(sparse_start + sparse_per_split, seq_sparse)
    ks_base = k_sparse_ptr + pid_b * stride_ks_b + pid_kv * stride_ks_h
    vs_base = v_sparse_ptr + pid_b * stride_vs_b + pid_kv * stride_vs_h

    for block_start in range(0, sparse_per_split, BLOCK_N):
        offs_n = sparse_start + block_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < sparse_end

        k = tl.load(
            ks_base + offs_n[:, None] * stride_ks_s + offs_k[None, :] * stride_ks_d,
            mask=n_mask[:, None] & dk_mask[None, :], other=0.0,
        ).to(tl.float32)
        v = tl.load(
            vs_base + offs_n[:, None] * stride_vs_s + offs_d[None, :] * stride_vs_d,
            mask=n_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)
        valid_mask = n_mask

        logits0 = tl.sum(k * q_sparse0[None, :], axis=1) * scale + bias0 * scale
        logits0 = tl.where(valid_mask & q0_valid, logits0, float("-inf"))
        block_max0 = tl.max(logits0)
        new_max0 = tl.maximum(m0, block_max0)
        correction0 = tl.exp(m0 - new_max0)
        block_exp0 = tl.exp(logits0 - new_max0)
        block_exp0 = tl.where(valid_mask & q0_valid, block_exp0, 0.0)
        l0 = l0 * correction0 + tl.sum(block_exp0)
        acc0 = acc0 * correction0 + tl.sum(block_exp0[:, None] * v, axis=0)
        m0 = new_max0

        logits1 = tl.sum(k * q_sparse1[None, :], axis=1) * scale + bias1 * scale
        logits1 = tl.where(valid_mask & q1_valid, logits1, float("-inf"))
        block_max1 = tl.max(logits1)
        new_max1 = tl.maximum(m1, block_max1)
        correction1 = tl.exp(m1 - new_max1)
        block_exp1 = tl.exp(logits1 - new_max1)
        block_exp1 = tl.where(valid_mask & q1_valid, block_exp1, 0.0)
        l1 = l1 * correction1 + tl.sum(block_exp1)
        acc1 = acc1 * correction1 + tl.sum(block_exp1[:, None] * v, axis=0)
        m1 = new_max1

    # ---- store partial results for the two query heads ----
    flat0 = pid_b * (NUM_Q_HEADS * NUM_SPLITS) + qh0 * NUM_SPLITS + pid_s
    flat1 = pid_b * (NUM_Q_HEADS * NUM_SPLITS) + qh1 * NUM_SPLITS + pid_s
    tl.store(partial_m_ptr + flat0 * stride_pm_bhs, m0, mask=q0_valid)
    tl.store(partial_l_ptr + flat0 * stride_pl_bhs, l0, mask=q0_valid)
    tl.store(partial_m_ptr + flat1 * stride_pm_bhs, m1, mask=q1_valid)
    tl.store(partial_l_ptr + flat1 * stride_pl_bhs, l1, mask=q1_valid)

    pa_base0 = partial_acc_ptr + flat0 * stride_pa_bhs
    pa_base1 = partial_acc_ptr + flat1 * stride_pa_bhs
    tl.store(pa_base0 + offs_d * stride_pa_d, acc0, mask=q0_valid & d_mask)
    tl.store(pa_base1 + offs_d * stride_pa_d, acc1, mask=q1_valid & d_mask)


# ---------------------------------------------------------------------------
# Phase 1 GQA-reuse MMA variant: one CTA processes one KV head and the whole
# query-head group as a small Q block. This mirrors the vLLM-style idea more
# closely than the scalar GQA2 path: Q/K and P/V are computed with tl.dot.
# ---------------------------------------------------------------------------

@triton.jit
def _splitk_phase1_gqa_mma_kernel(
    # ---- pointers ----
    q_full_ptr, q_sparse_ptr,
    k_full_ptr, k_sparse_ptr,
    v_full_ptr, v_sparse_ptr,
    mask_full_ptr,
    bias_shift_ptr,
    partial_m_ptr, partial_l_ptr, partial_acc_ptr,
    # ---- Q strides ----
    stride_qf_b, stride_qf_h, stride_qf_d,
    stride_qs_b, stride_qs_h, stride_qs_d,
    # ---- K strides ----
    stride_kf_b, stride_kf_h, stride_kf_s, stride_kf_d,
    stride_ks_b, stride_ks_h, stride_ks_s, stride_ks_d,
    # ---- V strides ----
    stride_vf_b, stride_vf_h, stride_vf_s, stride_vf_d,
    stride_vs_b, stride_vs_h, stride_vs_s, stride_vs_d,
    # ---- mask strides ----
    stride_mf_b, stride_mf_s,
    # ---- bias_shift strides ----
    stride_bs_b, stride_bs_h,
    # ---- partial buffer strides ----
    stride_pm_bhs,
    stride_pl_bhs,
    stride_pa_bhs, stride_pa_d,
    # ---- dimensions ----
    seq_full, seq_sparse, head_dim, head_dim_keep,
    sparse_per_split,
    scale,
    HAS_BIAS_SHIFT: tl.constexpr,
    NUM_KV_GROUPS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_kv = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)

    q_heads = pid_kv * NUM_KV_GROUPS + offs_m
    q_mask = (offs_m < NUM_KV_GROUPS) & (q_heads < NUM_Q_HEADS)
    d_mask = offs_d < head_dim
    k_mask = offs_k < head_dim_keep

    qf_base = q_full_ptr + pid_b * stride_qf_b + q_heads[:, None] * stride_qf_h
    q_full = tl.load(
        qf_base + offs_d[None, :] * stride_qf_d,
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    qs_base = q_sparse_ptr + pid_b * stride_qs_b + q_heads[:, None] * stride_qs_h
    q_sparse = tl.load(
        qs_base + offs_k[None, :] * stride_qs_d,
        mask=q_mask[:, None] & k_mask[None, :],
        other=0.0,
    )

    if HAS_BIAS_SHIFT:
        bias = tl.load(
            bias_shift_ptr + pid_b * stride_bs_b + q_heads * stride_bs_h,
            mask=q_mask,
            other=0.0,
        ).to(tl.float32)
    else:
        bias = tl.zeros([BLOCK_M], dtype=tl.float32)

    m = tl.full([BLOCK_M], -1e30, dtype=tl.float32)
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # ---- split 0: process all dense/full-channel tokens ----
    if pid_s == 0:
        kf_base = k_full_ptr + pid_b * stride_kf_b + pid_kv * stride_kf_h
        vf_base = v_full_ptr + pid_b * stride_vf_b + pid_kv * stride_vf_h
        for block_start in range(0, seq_full, BLOCK_N):
            n = block_start + offs_n
            n_mask = n < seq_full
            valid = tl.load(mask_full_ptr + pid_b * stride_mf_b + n * stride_mf_s, mask=n_mask, other=0)
            valid_mask = (valid != 0) & n_mask

            k_tile = tl.load(
                kf_base + offs_d[:, None] * stride_kf_d + n[None, :] * stride_kf_s,
                mask=d_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            v_tile = tl.load(
                vf_base + n[:, None] * stride_vf_s + offs_d[None, :] * stride_vf_d,
                mask=n_mask[:, None] & d_mask[None, :],
                other=0.0,
            )

            scores = tl.dot(q_full, k_tile) * scale
            scores = tl.where(q_mask[:, None] & valid_mask[None, :], scores, float("-inf"))
            new_m = tl.maximum(m, tl.max(scores, axis=1))
            alpha = tl.exp(m - new_m)
            p = tl.exp(scores - new_m[:, None])
            p = tl.where(q_mask[:, None] & valid_mask[None, :], p, 0.0)
            l = l * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
            m = new_m

    # ---- all splits: process their sparse-token chunk ----
    sparse_start = pid_s * sparse_per_split
    sparse_end = tl.minimum(sparse_start + sparse_per_split, seq_sparse)
    ks_base = k_sparse_ptr + pid_b * stride_ks_b + pid_kv * stride_ks_h
    vs_base = v_sparse_ptr + pid_b * stride_vs_b + pid_kv * stride_vs_h

    for block_start in range(0, sparse_per_split, BLOCK_N):
        n = sparse_start + block_start + offs_n
        n_mask = n < sparse_end
        valid_mask = n_mask

        k_tile = tl.load(
            ks_base + offs_k[:, None] * stride_ks_d + n[None, :] * stride_ks_s,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        v_tile = tl.load(
            vs_base + n[:, None] * stride_vs_s + offs_d[None, :] * stride_vs_d,
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        )

        scores = tl.dot(q_sparse, k_tile) * scale + bias[:, None] * scale
        scores = tl.where(q_mask[:, None] & valid_mask[None, :], scores, float("-inf"))
        new_m = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.exp(m - new_m)
        p = tl.exp(scores - new_m[:, None])
        p = tl.where(q_mask[:, None] & valid_mask[None, :], p, 0.0)
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
        m = new_m

    flat = pid_b * (NUM_Q_HEADS * NUM_SPLITS) + q_heads * NUM_SPLITS + pid_s
    tl.store(partial_m_ptr + flat * stride_pm_bhs, m, mask=q_mask)
    tl.store(partial_l_ptr + flat * stride_pl_bhs, l, mask=q_mask)
    tl.store(
        partial_acc_ptr + flat[:, None] * stride_pa_bhs + offs_d[None, :] * stride_pa_d,
        acc,
        mask=q_mask[:, None] & d_mask[None, :],
    )


# ---------------------------------------------------------------------------
# Fast VisionK phase 1: no q_sparse buffer, no prompt/text concat, sparse tokens all valid.
# This path is for channel_reconstruction="off" and Qwen-style GQA.
# ---------------------------------------------------------------------------

@triton.jit
def _splitk_phase1_gqa_mma_direct_kernel(
    q_full_ptr, keep_idx_ptr,
    pruned_idx_ptr, w_t_ptr, recon_bias_ptr,
    k_prompt_ptr, k_text_ptr, k_sparse_ptr,
    v_all_ptr,
    partial_m_ptr, partial_l_ptr, partial_acc_ptr,
    # ---- Q / keep-index strides ----
    stride_qf_b, stride_qf_h, stride_qf_d,
    stride_ki_b, stride_ki_h, stride_ki_d,
    stride_wt_h, stride_wt_p, stride_wt_k,
    # ---- K strides ----
    stride_kp_b, stride_kp_h, stride_kp_s, stride_kp_d,
    stride_kt_b, stride_kt_h, stride_kt_s, stride_kt_d,
    stride_ks_b, stride_ks_h, stride_ks_s, stride_ks_d,
    # ---- V-cache strides ----
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    # ---- partial buffer strides ----
    stride_pm_bhs,
    stride_pl_bhs,
    stride_pa_bhs, stride_pa_d,
    # ---- dimensions ----
    seq_prompt, seq_text, seq_sparse, head_dim, head_dim_keep, pruned_dim,
    sparse_per_split,
    scale,
    HAS_MATRIX_RECON: tl.constexpr,
    NUM_KV_GROUPS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_P: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_kv = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)
    offs_p = tl.arange(0, BLOCK_P)

    q_heads = pid_kv * NUM_KV_GROUPS + offs_m
    q_mask = (offs_m < NUM_KV_GROUPS) & (q_heads < NUM_Q_HEADS)
    d_mask = offs_d < head_dim
    k_mask = offs_k < head_dim_keep

    qf_base = q_full_ptr + pid_b * stride_qf_b + q_heads[:, None] * stride_qf_h
    q_full = tl.load(
        qf_base + offs_d[None, :] * stride_qf_d,
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    keep_base = keep_idx_ptr + pid_b * stride_ki_b + pid_kv * stride_ki_h
    keep_idx = tl.load(keep_base + offs_k * stride_ki_d, mask=k_mask, other=0).to(tl.int64)
    q_sparse = tl.load(
        qf_base + keep_idx[None, :] * stride_qf_d,
        mask=q_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    sparse_bias = tl.zeros([BLOCK_M], dtype=tl.float32)

    if HAS_MATRIX_RECON:
        p_mask = offs_p < pruned_dim
        pruned_base = pruned_idx_ptr + pid_kv * pruned_dim
        pruned_idx = tl.load(pruned_base + offs_p, mask=p_mask, other=0).to(tl.int64)
        q_pruned = tl.load(
            qf_base + pruned_idx[None, :] * stride_qf_d,
            mask=q_mask[:, None] & p_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        wt_base = w_t_ptr + pid_kv * stride_wt_h
        w_t = tl.load(
            wt_base + offs_p[:, None] * stride_wt_p + offs_k[None, :] * stride_wt_k,
            mask=p_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        q_sparse = q_sparse + tl.dot(q_pruned, w_t)
        rb_base = recon_bias_ptr + pid_kv * pruned_dim
        recon_bias = tl.load(rb_base + offs_p, mask=p_mask, other=0.0).to(tl.float32)
        sparse_bias = tl.sum(q_pruned * recon_bias[None, :], axis=1)

    m = tl.full([BLOCK_M], -1e30, dtype=tl.float32)
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # split 0 owns the dense prompt/text segments.
    if pid_s == 0:
        kp_base = k_prompt_ptr + pid_b * stride_kp_b + pid_kv * stride_kp_h
        kt_base = k_text_ptr + pid_b * stride_kt_b + pid_kv * stride_kt_h
        v_base = v_all_ptr + pid_b * stride_v_b + pid_kv * stride_v_h

        for block_start in range(0, seq_prompt, BLOCK_N):
            n = block_start + offs_n
            n_mask = n < seq_prompt
            k_tile = tl.load(
                kp_base + offs_d[:, None] * stride_kp_d + n[None, :] * stride_kp_s,
                mask=d_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            v_tile = tl.load(
                v_base + n[:, None] * stride_v_s + offs_d[None, :] * stride_v_d,
                mask=n_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            scores = tl.dot(q_full, k_tile) * scale
            scores = tl.where(q_mask[:, None] & n_mask[None, :], scores, float("-inf"))
            new_m = tl.maximum(m, tl.max(scores, axis=1))
            if USE_EXP2:
                alpha = tl.exp2(m - new_m)
                p = tl.exp2(scores - new_m[:, None])
            else:
                alpha = tl.exp(m - new_m)
                p = tl.exp(scores - new_m[:, None])
            p = tl.where(q_mask[:, None] & n_mask[None, :], p, 0.0)
            l = l * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
            m = new_m

        text_v_start = seq_prompt + seq_sparse
        for block_start in range(0, seq_text, BLOCK_N):
            n = block_start + offs_n
            n_mask = n < seq_text
            k_tile = tl.load(
                kt_base + offs_d[:, None] * stride_kt_d + n[None, :] * stride_kt_s,
                mask=d_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            v_tile = tl.load(
                v_base + (text_v_start + n)[:, None] * stride_v_s + offs_d[None, :] * stride_v_d,
                mask=n_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            scores = tl.dot(q_full, k_tile) * scale
            scores = tl.where(q_mask[:, None] & n_mask[None, :], scores, float("-inf"))
            new_m = tl.maximum(m, tl.max(scores, axis=1))
            if USE_EXP2:
                alpha = tl.exp2(m - new_m)
                p = tl.exp2(scores - new_m[:, None])
            else:
                alpha = tl.exp(m - new_m)
                p = tl.exp(scores - new_m[:, None])
            p = tl.where(q_mask[:, None] & n_mask[None, :], p, 0.0)
            l = l * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
            m = new_m

    sparse_start = pid_s * sparse_per_split
    sparse_end = tl.minimum(sparse_start + sparse_per_split, seq_sparse)
    ks_base = k_sparse_ptr + pid_b * stride_ks_b + pid_kv * stride_ks_h
    v_base = v_all_ptr + pid_b * stride_v_b + pid_kv * stride_v_h

    for block_start in range(0, sparse_per_split, BLOCK_N):
        n = sparse_start + block_start + offs_n
        n_mask = n < sparse_end
        k_tile = tl.load(
            ks_base + offs_k[:, None] * stride_ks_d + n[None, :] * stride_ks_s,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        v_tile = tl.load(
            v_base + (seq_prompt + n)[:, None] * stride_v_s + offs_d[None, :] * stride_v_d,
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        scores = tl.dot(q_sparse.to(k_tile.dtype), k_tile) * scale + sparse_bias[:, None] * scale
        scores = tl.where(q_mask[:, None] & n_mask[None, :], scores, float("-inf"))
        new_m = tl.maximum(m, tl.max(scores, axis=1))
        if USE_EXP2:
            alpha = tl.exp2(m - new_m)
            p = tl.exp2(scores - new_m[:, None])
        else:
            alpha = tl.exp(m - new_m)
            p = tl.exp(scores - new_m[:, None])
        p = tl.where(q_mask[:, None] & n_mask[None, :], p, 0.0)
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
        m = new_m

    flat = pid_b * (NUM_Q_HEADS * NUM_SPLITS) + q_heads * NUM_SPLITS + pid_s
    tl.store(partial_m_ptr + flat * stride_pm_bhs, m, mask=q_mask)
    tl.store(partial_l_ptr + flat * stride_pl_bhs, l, mask=q_mask)
    tl.store(
        partial_acc_ptr + flat[:, None] * stride_pa_bhs + offs_d[None, :] * stride_pa_d,
        acc,
        mask=q_mask[:, None] & d_mask[None, :],
    )


# ---------------------------------------------------------------------------
# Phase 2: merge partial results across splits
# ---------------------------------------------------------------------------

@triton.jit
def _splitk_merge_kernel(
    partial_m_ptr, partial_l_ptr, partial_acc_ptr,
    out_ptr,
    stride_pm_bhs, stride_pl_bhs,
    stride_pa_bhs, stride_pa_d,
    stride_ob, stride_oh, stride_od,
    num_heads,
    head_dim,
    NUM_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    # find global max across splits
    global_m = tl.full([], -1e30, dtype=tl.float32)
    for s in range(NUM_SPLITS):
        flat_idx = pid_b * (num_heads * NUM_SPLITS) + pid_h * NUM_SPLITS + s
        m_s = tl.load(partial_m_ptr + flat_idx * stride_pm_bhs)
        global_m = tl.maximum(global_m, m_s)

    # merge
    global_l = tl.zeros([], dtype=tl.float32)
    global_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for s in range(NUM_SPLITS):
        flat_idx = pid_b * (num_heads * NUM_SPLITS) + pid_h * NUM_SPLITS + s
        m_s = tl.load(partial_m_ptr + flat_idx * stride_pm_bhs)
        l_s = tl.load(partial_l_ptr + flat_idx * stride_pl_bhs)
        acc_s = tl.load(
            partial_acc_ptr + flat_idx * stride_pa_bhs + offs_d * stride_pa_d,
            mask=d_mask, other=0.0,
        )
        if USE_EXP2:
            correction = tl.exp2(m_s - global_m)
        else:
            correction = tl.exp(m_s - global_m)
        global_l += l_s * correction
        global_acc += acc_s * correction

    output = global_acc / (global_l + 1e-6)

    out_base = out_ptr + pid_b * stride_ob + pid_h * stride_oh
    tl.store(out_base + offs_d * stride_od, output.to(out_ptr.dtype.element_ty), mask=d_mask)


# ---------------------------------------------------------------------------
# Diagnostic component kernels for direct-path microbenchmarks.
# These are not production kernels; they intentionally isolate broad cost
# buckets using the same split/grid/shape conventions as phase1.
# ---------------------------------------------------------------------------

@triton.jit
def _direct_qk_dot_component_kernel(
    q_full_ptr, keep_idx_ptr, k_sparse_ptr, scratch_ptr,
    stride_qf_b, stride_qf_h, stride_qf_d,
    stride_ki_b, stride_ki_h, stride_ki_d,
    stride_ks_b, stride_ks_h, stride_ks_s, stride_ks_d,
    stride_s_bhs,
    seq_sparse, head_dim_keep,
    sparse_per_split,
    scale,
    NUM_KV_GROUPS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_kv = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    q_heads = pid_kv * NUM_KV_GROUPS + offs_m
    q_mask = (offs_m < NUM_KV_GROUPS) & (q_heads < NUM_Q_HEADS)
    k_mask = offs_k < head_dim_keep

    qf_base = q_full_ptr + pid_b * stride_qf_b + q_heads[:, None] * stride_qf_h
    keep_base = keep_idx_ptr + pid_b * stride_ki_b + pid_kv * stride_ki_h
    keep_idx = tl.load(keep_base + offs_k * stride_ki_d, mask=k_mask, other=0).to(tl.int64)
    q_sparse = tl.load(
        qf_base + keep_idx[None, :] * stride_qf_d,
        mask=q_mask[:, None] & k_mask[None, :],
        other=0.0,
    )

    sparse_start = pid_s * sparse_per_split
    sparse_end = tl.minimum(sparse_start + sparse_per_split, seq_sparse)
    ks_base = k_sparse_ptr + pid_b * stride_ks_b + pid_kv * stride_ks_h
    m = tl.full([BLOCK_M], -1e30, dtype=tl.float32)

    for block_start in range(0, sparse_per_split, BLOCK_N):
        n = sparse_start + block_start + offs_n
        n_mask = n < sparse_end
        k_tile = tl.load(
            ks_base + offs_k[:, None] * stride_ks_d + n[None, :] * stride_ks_s,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        scores = tl.dot(q_sparse.to(k_tile.dtype), k_tile) * scale
        scores = tl.where(q_mask[:, None] & n_mask[None, :], scores, float("-inf"))
        m = tl.maximum(m, tl.max(scores, axis=1))

    flat = pid_b * (NUM_Q_HEADS * NUM_SPLITS) + q_heads * NUM_SPLITS + pid_s
    tl.store(scratch_ptr + flat * stride_s_bhs, m, mask=q_mask)


@triton.jit
def _direct_qk_accum_component_kernel(
    q_full_ptr, keep_idx_ptr, k_sparse_ptr, scratch_ptr,
    stride_qf_b, stride_qf_h, stride_qf_d,
    stride_ki_b, stride_ki_h, stride_ki_d,
    stride_ks_b, stride_ks_h, stride_ks_s, stride_ks_d,
    stride_s_bhs,
    seq_sparse, head_dim_keep,
    sparse_per_split,
    scale,
    NUM_KV_GROUPS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_kv = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    q_heads = pid_kv * NUM_KV_GROUPS + offs_m
    q_mask = (offs_m < NUM_KV_GROUPS) & (q_heads < NUM_Q_HEADS)
    k_mask = offs_k < head_dim_keep

    qf_base = q_full_ptr + pid_b * stride_qf_b + q_heads[:, None] * stride_qf_h
    keep_base = keep_idx_ptr + pid_b * stride_ki_b + pid_kv * stride_ki_h
    keep_idx = tl.load(keep_base + offs_k * stride_ki_d, mask=k_mask, other=0).to(tl.int64)
    q_sparse = tl.load(
        qf_base + keep_idx[None, :] * stride_qf_d,
        mask=q_mask[:, None] & k_mask[None, :],
        other=0.0,
    )

    sparse_start = pid_s * sparse_per_split
    sparse_end = tl.minimum(sparse_start + sparse_per_split, seq_sparse)
    ks_base = k_sparse_ptr + pid_b * stride_ks_b + pid_kv * stride_ks_h
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for block_start in range(0, sparse_per_split, BLOCK_N):
        n = sparse_start + block_start + offs_n
        n_mask = n < sparse_end
        k_tile = tl.load(
            ks_base + offs_k[:, None] * stride_ks_d + n[None, :] * stride_ks_s,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        scores = tl.dot(q_sparse.to(k_tile.dtype), k_tile) * scale
        scores = tl.where(q_mask[:, None] & n_mask[None, :], scores, 0.0)
        acc += tl.sum(scores, axis=1)

    flat = pid_b * (NUM_Q_HEADS * NUM_SPLITS) + q_heads * NUM_SPLITS + pid_s
    tl.store(scratch_ptr + flat * stride_s_bhs, acc, mask=q_mask)


@triton.jit
def _direct_qk_softmax_component_kernel(
    q_full_ptr, keep_idx_ptr, k_sparse_ptr, scratch_m_ptr, scratch_l_ptr,
    stride_qf_b, stride_qf_h, stride_qf_d,
    stride_ki_b, stride_ki_h, stride_ki_d,
    stride_ks_b, stride_ks_h, stride_ks_s, stride_ks_d,
    stride_sm_bhs, stride_sl_bhs,
    seq_sparse, head_dim_keep,
    sparse_per_split,
    scale,
    NUM_KV_GROUPS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_kv = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    q_heads = pid_kv * NUM_KV_GROUPS + offs_m
    q_mask = (offs_m < NUM_KV_GROUPS) & (q_heads < NUM_Q_HEADS)
    k_mask = offs_k < head_dim_keep

    qf_base = q_full_ptr + pid_b * stride_qf_b + q_heads[:, None] * stride_qf_h
    keep_base = keep_idx_ptr + pid_b * stride_ki_b + pid_kv * stride_ki_h
    keep_idx = tl.load(keep_base + offs_k * stride_ki_d, mask=k_mask, other=0).to(tl.int64)
    q_sparse = tl.load(
        qf_base + keep_idx[None, :] * stride_qf_d,
        mask=q_mask[:, None] & k_mask[None, :],
        other=0.0,
    )

    sparse_start = pid_s * sparse_per_split
    sparse_end = tl.minimum(sparse_start + sparse_per_split, seq_sparse)
    ks_base = k_sparse_ptr + pid_b * stride_ks_b + pid_kv * stride_ks_h
    m = tl.full([BLOCK_M], -1e30, dtype=tl.float32)
    l = tl.zeros([BLOCK_M], dtype=tl.float32)

    for block_start in range(0, sparse_per_split, BLOCK_N):
        n = sparse_start + block_start + offs_n
        n_mask = n < sparse_end
        k_tile = tl.load(
            ks_base + offs_k[:, None] * stride_ks_d + n[None, :] * stride_ks_s,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        scores = tl.dot(q_sparse.to(k_tile.dtype), k_tile) * scale
        scores = tl.where(q_mask[:, None] & n_mask[None, :], scores, float("-inf"))
        new_m = tl.maximum(m, tl.max(scores, axis=1))
        if USE_EXP2:
            alpha = tl.exp2(m - new_m)
            p = tl.exp2(scores - new_m[:, None])
        else:
            alpha = tl.exp(m - new_m)
            p = tl.exp(scores - new_m[:, None])
        p = tl.where(q_mask[:, None] & n_mask[None, :], p, 0.0)
        l = l * alpha + tl.sum(p, axis=1)
        m = new_m

    flat = pid_b * (NUM_Q_HEADS * NUM_SPLITS) + q_heads * NUM_SPLITS + pid_s
    tl.store(scratch_m_ptr + flat * stride_sm_bhs, m, mask=q_mask)
    tl.store(scratch_l_ptr + flat * stride_sl_bhs, l, mask=q_mask)


@triton.jit
def _direct_v_component_kernel(
    v_all_ptr, scratch_acc_ptr,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_sa_bhs, stride_sa_d,
    seq_prompt, seq_sparse, head_dim,
    sparse_per_split,
    NUM_KV_GROUPS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_kv = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_heads = pid_kv * NUM_KV_GROUPS + offs_m
    q_mask = (offs_m < NUM_KV_GROUPS) & (q_heads < NUM_Q_HEADS)
    d_mask = offs_d < head_dim

    sparse_start = pid_s * sparse_per_split
    sparse_end = tl.minimum(sparse_start + sparse_per_split, seq_sparse)
    v_base = v_all_ptr + pid_b * stride_v_b + pid_kv * stride_v_h
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for block_start in range(0, sparse_per_split, BLOCK_N):
        n = sparse_start + block_start + offs_n
        n_mask = n < sparse_end
        v_tile = tl.load(
            v_base + (seq_prompt + n)[:, None] * stride_v_s + offs_d[None, :] * stride_v_d,
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        p = tl.where(q_mask[:, None] & n_mask[None, :], 1.0 / BLOCK_N, 0.0)
        acc += tl.dot(p.to(v_tile.dtype), v_tile)

    flat = pid_b * (NUM_Q_HEADS * NUM_SPLITS) + q_heads * NUM_SPLITS + pid_s
    tl.store(
        scratch_acc_ptr + flat[:, None] * stride_sa_bhs + offs_d[None, :] * stride_sa_d,
        acc,
        mask=q_mask[:, None] & d_mask[None, :],
    )


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def _get_decode_buffers(owner, key, bsz, heads, head_dim, num_splits, device, dtype):
    buf_cache = getattr(owner, "_buf_cache", {})
    bufs = buf_cache.get(key)
    total_partials = bsz * heads * num_splits
    if bufs is None:
        bufs = (
            torch.empty(total_partials, device=device, dtype=torch.float32),
            torch.empty(total_partials, device=device, dtype=torch.float32),
            torch.empty(total_partials, head_dim, device=device, dtype=torch.float32),
            torch.empty((bsz, heads, head_dim), device=device, dtype=dtype),
        )
        # Keep a tiny shape cache; decode usually has one active shape.
        if len(buf_cache) >= 4:
            buf_cache.clear()
        buf_cache[key] = bufs
        owner._buf_cache = buf_cache
    return bufs


def sparse_channel_decode_triton_direct(
    q_full: torch.Tensor,
    keep_idx: torch.Tensor,
    k_prompt: torch.Tensor,
    k_text: torch.Tensor,
    k_sparse: torch.Tensor,
    v_all: torch.Tensor,
    pruned_idx: torch.Tensor | None = None,
    w_t: torch.Tensor | None = None,
    recon_bias: torch.Tensor | None = None,
    num_kv_groups: int = 1,
    return_phase_timings: bool = False,
    triton_num_stages: int = 2,
    softmax_exp2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fast VisionK decode path, with optional matrix reconstruction fused.

    Avoids q_sparse materialization, prompt/text concatenation, all-valid masks,
    and per-step partial/output buffer allocation.
    """
    bsz, heads, head_dim = q_full.shape
    head_dim_keep = k_sparse.shape[-1]
    seq_prompt = k_prompt.shape[-2]
    seq_text = k_text.shape[-2]
    seq_sparse = k_sparse.shape[-2]

    scale = 1.0 / math.sqrt(head_dim)
    if softmax_exp2:
        scale *= 1.4426950408889634  # log2(e), so exp(x) == exp2(x * log2(e)).
    block_d = _next_power_of_2(head_dim)
    block_k = _next_power_of_2(head_dim_keep)
    has_matrix_recon = pruned_idx is not None and w_t is not None and recon_bias is not None
    pruned_dim = pruned_idx.shape[-1] if has_matrix_recon else 0
    block_p = _next_power_of_2(pruned_dim) if pruned_dim > 0 else 1
    BLOCK_N = int(os.environ.get("VISIONK_BLOCK_N", "64"))
    num_splits_cap = int(os.environ.get("VISIONK_NUM_SPLITS_CAP", "64"))
    num_splits = max(1, min((seq_sparse + BLOCK_N - 1) // BLOCK_N, num_splits_cap))
    sparse_per_split = (seq_sparse + num_splits - 1) // num_splits

    q_full = q_full.contiguous()
    key = (
        str(q_full.device), q_full.dtype, bsz, heads, head_dim,
        head_dim_keep, pruned_dim, num_splits,
    )
    partial_m, partial_l, partial_acc, out = _get_decode_buffers(
        sparse_channel_decode_triton_direct,
        key,
        bsz, heads, head_dim, num_splits,
        q_full.device, q_full.dtype,
    )

    kv_heads = k_sparse.shape[1]
    grid1 = (bsz, kv_heads, num_splits)
    dummy = keep_idx
    if not has_matrix_recon:
        pruned_idx = dummy
        w_t = dummy
        recon_bias = dummy

    if return_phase_timings:
        phase1_start = torch.cuda.Event(enable_timing=True)
        phase1_end = torch.cuda.Event(enable_timing=True)
        merge_start = torch.cuda.Event(enable_timing=True)
        merge_end = torch.cuda.Event(enable_timing=True)
        phase1_start.record()

    _splitk_phase1_gqa_mma_direct_kernel[grid1](
        q_full, keep_idx,
        pruned_idx, w_t, recon_bias,
        k_prompt, k_text, k_sparse,
        v_all,
        partial_m, partial_l, partial_acc,
        q_full.stride(0), q_full.stride(1), q_full.stride(2),
        keep_idx.stride(0), keep_idx.stride(1), keep_idx.stride(2),
        w_t.stride(0) if has_matrix_recon else 0,
        w_t.stride(1) if has_matrix_recon else 0,
        w_t.stride(2) if has_matrix_recon else 0,
        k_prompt.stride(0), k_prompt.stride(1), k_prompt.stride(2), k_prompt.stride(3),
        k_text.stride(0), k_text.stride(1), k_text.stride(2), k_text.stride(3),
        k_sparse.stride(0), k_sparse.stride(1), k_sparse.stride(2), k_sparse.stride(3),
        v_all.stride(0), v_all.stride(1), v_all.stride(2), v_all.stride(3),
        partial_m.stride(0), partial_l.stride(0),
        partial_acc.stride(0), partial_acc.stride(1),
        seq_prompt, seq_text, seq_sparse, head_dim, head_dim_keep, pruned_dim,
        sparse_per_split,
        scale,
        HAS_MATRIX_RECON=has_matrix_recon,
        NUM_KV_GROUPS=num_kv_groups,
        NUM_Q_HEADS=heads,
        NUM_SPLITS=num_splits,
        BLOCK_M=max(16, _next_power_of_2(num_kv_groups)),
        BLOCK_N=BLOCK_N,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        BLOCK_P=block_p,
        USE_EXP2=softmax_exp2,
        num_warps=4,
        num_stages=triton_num_stages,
    )

    if return_phase_timings:
        phase1_end.record()
        merge_start.record()

    grid2 = (bsz, heads)
    _splitk_merge_kernel[grid2](
        partial_m, partial_l, partial_acc,
        out,
        partial_m.stride(0), partial_l.stride(0),
        partial_acc.stride(0), partial_acc.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        heads,
        head_dim,
        NUM_SPLITS=num_splits,
        BLOCK_D=block_d,
        USE_EXP2=softmax_exp2,
        num_warps=4,
        num_stages=1,
    )
    if return_phase_timings:
        merge_end.record()
        torch.cuda.synchronize()
        timings = {
            "phase1_ms": phase1_start.elapsed_time(phase1_end),
            "merge_ms": merge_start.elapsed_time(merge_end),
        }
        timings["total_ms"] = timings["phase1_ms"] + timings["merge_ms"]
        return out.unsqueeze(1), None, None, timings
    return out.unsqueeze(1), None, None


def sparse_channel_decode_triton_direct_component(
    q_full: torch.Tensor,
    keep_idx: torch.Tensor,
    k_sparse: torch.Tensor,
    v_all: torch.Tensor,
    seq_prompt: int,
    num_kv_groups: int = 1,
    component: str = "qk_dot",
    triton_num_stages: int = 2,
    softmax_exp2: bool = False,
) -> torch.Tensor:
    """Diagnostic component microbench for the direct decode path.

    Components intentionally do not reproduce the full attention result:
      - qk_accum: K load + q gather + QK dot + simple score-sum store.
      - qk_dot: K load + q gather + QK dot + max-reduction store.
      - qk_softmax: qk_dot plus online softmax m/l update, no V path.
      - v: V load + dummy p@V accumulation + partial-acc store, no QK path.
    """
    bsz, heads, head_dim = q_full.shape
    head_dim_keep = k_sparse.shape[-1]
    seq_sparse = k_sparse.shape[-2]

    scale = 1.0 / math.sqrt(head_dim)
    if softmax_exp2:
        scale *= 1.4426950408889634
    block_d = _next_power_of_2(head_dim)
    block_k = _next_power_of_2(head_dim_keep)
    BLOCK_N = int(os.environ.get("VISIONK_BLOCK_N", "64"))
    num_splits_cap = int(os.environ.get("VISIONK_NUM_SPLITS_CAP", "64"))
    num_splits = max(1, min((seq_sparse + BLOCK_N - 1) // BLOCK_N, num_splits_cap))
    sparse_per_split = (seq_sparse + num_splits - 1) // num_splits

    q_full = q_full.contiguous()
    key = (
        "component", component, str(q_full.device), q_full.dtype, bsz, heads,
        head_dim, head_dim_keep, num_splits,
    )
    scratch_m, scratch_l, scratch_acc, _ = _get_decode_buffers(
        sparse_channel_decode_triton_direct_component,
        key,
        bsz,
        heads,
        head_dim,
        num_splits,
        q_full.device,
        q_full.dtype,
    )

    kv_heads = k_sparse.shape[1]
    grid = (bsz, kv_heads, num_splits)
    block_m = max(16, _next_power_of_2(num_kv_groups))

    if component == "qk_dot":
        _direct_qk_dot_component_kernel[grid](
            q_full, keep_idx, k_sparse, scratch_m,
            q_full.stride(0), q_full.stride(1), q_full.stride(2),
            keep_idx.stride(0), keep_idx.stride(1), keep_idx.stride(2),
            k_sparse.stride(0), k_sparse.stride(1), k_sparse.stride(2), k_sparse.stride(3),
            scratch_m.stride(0),
            seq_sparse, head_dim_keep,
            sparse_per_split,
            scale,
            NUM_KV_GROUPS=num_kv_groups,
            NUM_Q_HEADS=heads,
            NUM_SPLITS=num_splits,
            BLOCK_M=block_m,
            BLOCK_N=BLOCK_N,
            BLOCK_K=block_k,
            USE_EXP2=softmax_exp2,
            num_warps=4,
            num_stages=triton_num_stages,
        )
        return scratch_m

    if component == "qk_accum":
        _direct_qk_accum_component_kernel[grid](
            q_full, keep_idx, k_sparse, scratch_m,
            q_full.stride(0), q_full.stride(1), q_full.stride(2),
            keep_idx.stride(0), keep_idx.stride(1), keep_idx.stride(2),
            k_sparse.stride(0), k_sparse.stride(1), k_sparse.stride(2), k_sparse.stride(3),
            scratch_m.stride(0),
            seq_sparse, head_dim_keep,
            sparse_per_split,
            scale,
            NUM_KV_GROUPS=num_kv_groups,
            NUM_Q_HEADS=heads,
            NUM_SPLITS=num_splits,
            BLOCK_M=block_m,
            BLOCK_N=BLOCK_N,
            BLOCK_K=block_k,
            USE_EXP2=softmax_exp2,
            num_warps=4,
            num_stages=triton_num_stages,
        )
        return scratch_m

    if component == "qk_softmax":
        _direct_qk_softmax_component_kernel[grid](
            q_full, keep_idx, k_sparse, scratch_m, scratch_l,
            q_full.stride(0), q_full.stride(1), q_full.stride(2),
            keep_idx.stride(0), keep_idx.stride(1), keep_idx.stride(2),
            k_sparse.stride(0), k_sparse.stride(1), k_sparse.stride(2), k_sparse.stride(3),
            scratch_m.stride(0), scratch_l.stride(0),
            seq_sparse, head_dim_keep,
            sparse_per_split,
            scale,
            NUM_KV_GROUPS=num_kv_groups,
            NUM_Q_HEADS=heads,
            NUM_SPLITS=num_splits,
            BLOCK_M=block_m,
            BLOCK_N=BLOCK_N,
            BLOCK_K=block_k,
            USE_EXP2=softmax_exp2,
            num_warps=4,
            num_stages=triton_num_stages,
        )
        return scratch_l

    if component == "v":
        _direct_v_component_kernel[grid](
            v_all, scratch_acc,
            v_all.stride(0), v_all.stride(1), v_all.stride(2), v_all.stride(3),
            scratch_acc.stride(0), scratch_acc.stride(1),
            seq_prompt, seq_sparse, head_dim,
            sparse_per_split,
            NUM_KV_GROUPS=num_kv_groups,
            NUM_Q_HEADS=heads,
            NUM_SPLITS=num_splits,
            BLOCK_M=block_m,
            BLOCK_N=BLOCK_N,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=triton_num_stages,
        )
        return scratch_acc

    raise ValueError(f"unknown direct component: {component}")


def sparse_channel_decode_triton(
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
    """Fused split-K decode attention — no logit/prob materialization to HBM.

    Inputs
    ------
    q_full           : [B, H_q, D]
    k_full           : [B, H_kv, S_full, D]       (H_kv may differ from H_q for GQA)
    v_full           : [B, H_kv, S_full, D]
    mask_full        : [B, S_full]  uint8
    q_sparse         : [B, H_q, D_keep]
    k_sparse         : [B, H_kv, S_sparse, D_keep]
    v_sparse         : [B, H_kv, S_sparse, D]
    mask_sparse      : ignored; sparse VisionK tokens are all valid
    sparse_bias_shift: [B, H_q] or None — per-head bias added to sparse logits
    num_kv_groups    : int — H_q // H_kv for GQA (1 = MHA)

    Returns
    -------
    attn_output  : [B, 1, H_q, D]
    logits_full  : None  (not materialized)
    logits_sparse: None  (not materialized)
    """
    if q_full.dim() != 3 or q_sparse.dim() != 3:
        raise ValueError("q_full and q_sparse must be [B, H_q, D]-style tensors")
    if q_full.shape[:2] != q_sparse.shape[:2]:
        raise ValueError("q_full and q_sparse must agree on [B, H_q]")
    if mask_full.dtype != torch.uint8:
        raise ValueError("mask_full must be a uint8 validity mask")

    bsz, heads, head_dim = q_full.shape
    head_dim_keep = q_sparse.shape[-1]
    seq_full = k_full.shape[-2]
    seq_sparse = k_sparse.shape[-2]

    scale = 1.0 / math.sqrt(head_dim)
    block_d = _next_power_of_2(head_dim)
    BLOCK_N = 64

    # choose num_splits for GPU utilization
    num_splits = max(1, min((seq_sparse + BLOCK_N - 1) // BLOCK_N, 64))

    has_bias = sparse_bias_shift is not None
    if has_bias:
        sparse_bias_shift = sparse_bias_shift.contiguous()
        bs_stride_b, bs_stride_h = sparse_bias_shift.stride(0), sparse_bias_shift.stride(1)
    else:
        sparse_bias_shift = q_full
        bs_stride_b, bs_stride_h = 0, 0

    q_full = q_full.contiguous()
    q_sparse = q_sparse.contiguous()
    k_full = k_full.contiguous()
    v_full = v_full.contiguous()
    # k_sparse is produced by prefill gather and v_sparse is a vision-token slice.
    # Kernels consume explicit strides, so avoid copying either tensor here.

    sparse_per_split = (seq_sparse + num_splits - 1) // num_splits

    # allocate partial buffers: [B * H * num_splits] for m/l, [B * H * num_splits, D] for acc
    total_partials = bsz * heads * num_splits
    partial_m = torch.empty(total_partials, device=q_full.device, dtype=torch.float32)
    partial_l = torch.empty(total_partials, device=q_full.device, dtype=torch.float32)
    partial_acc = torch.empty(total_partials, head_dim, device=q_full.device, dtype=torch.float32)

    out = torch.empty((bsz, heads, head_dim), device=q_full.device, dtype=q_full.dtype)

    # Phase 1: split-K computation
    # GQA models use the Q-block/MMA kernel by default: one CTA owns a KV head
    # and processes its query-head group together, reusing K/V tiles. The older
    # scalar GQA2 path remains opt-in for debugging/comparison.
    use_gqa_reuse = num_kv_groups > 1 and os.environ.get("VISIONK_TRITON_GQA_REUSE", "0") != "0"
    use_gqa_mma = (
        num_kv_groups > 1
        and not use_gqa_reuse
        and os.environ.get("VISIONK_TRITON_GQA_MMA", "1") != "0"
    )
    if use_gqa_mma:
        kv_heads = k_sparse.shape[1]
        grid1 = (bsz, kv_heads, num_splits)
        _splitk_phase1_gqa_mma_kernel[grid1](
            q_full, q_sparse,
            k_full, k_sparse,
            v_full, v_sparse,
            mask_full,
            sparse_bias_shift,
            partial_m, partial_l, partial_acc,
            q_full.stride(0), q_full.stride(1), q_full.stride(2),
            q_sparse.stride(0), q_sparse.stride(1), q_sparse.stride(2),
            k_full.stride(0), k_full.stride(1), k_full.stride(2), k_full.stride(3),
            k_sparse.stride(0), k_sparse.stride(1), k_sparse.stride(2), k_sparse.stride(3),
            v_full.stride(0), v_full.stride(1), v_full.stride(2), v_full.stride(3),
            v_sparse.stride(0), v_sparse.stride(1), v_sparse.stride(2), v_sparse.stride(3),
            mask_full.stride(0), mask_full.stride(1),
            bs_stride_b, bs_stride_h,
            partial_m.stride(0), partial_l.stride(0),
            partial_acc.stride(0), partial_acc.stride(1),
            seq_full, seq_sparse, head_dim, head_dim_keep,
            sparse_per_split,
            scale,
            HAS_BIAS_SHIFT=has_bias,
            NUM_KV_GROUPS=num_kv_groups,
            NUM_Q_HEADS=heads,
            NUM_SPLITS=num_splits,
            BLOCK_M=max(16, _next_power_of_2(num_kv_groups)),
            BLOCK_N=BLOCK_N,
            BLOCK_D=block_d,
            BLOCK_K=_next_power_of_2(head_dim_keep),
            num_warps=4,
            num_stages=2,
        )
    elif use_gqa_reuse:
        # GQA-reuse path: each CTA handles two query heads sharing one KV head.
        # This cuts repeated K/V loads for Qwen-style GQA without the register
        # pressure of processing the whole query group in one program.
        kv_heads = k_sparse.shape[1]
        group_tiles = triton.cdiv(num_kv_groups, 2)
        grid1 = (bsz, kv_heads, group_tiles * num_splits)
        _splitk_phase1_gqa2_kernel[grid1](
            q_full, q_sparse,
            k_full, k_sparse,
            v_full, v_sparse,
            mask_full,
            sparse_bias_shift,
            partial_m, partial_l, partial_acc,
            q_full.stride(0), q_full.stride(1), q_full.stride(2),
            q_sparse.stride(0), q_sparse.stride(1), q_sparse.stride(2),
            k_full.stride(0), k_full.stride(1), k_full.stride(2), k_full.stride(3),
            k_sparse.stride(0), k_sparse.stride(1), k_sparse.stride(2), k_sparse.stride(3),
            v_full.stride(0), v_full.stride(1), v_full.stride(2), v_full.stride(3),
            v_sparse.stride(0), v_sparse.stride(1), v_sparse.stride(2), v_sparse.stride(3),
            mask_full.stride(0), mask_full.stride(1),
            bs_stride_b, bs_stride_h,
            partial_m.stride(0), partial_l.stride(0),
            partial_acc.stride(0), partial_acc.stride(1),
            seq_full, seq_sparse, head_dim, head_dim_keep,
            sparse_per_split,
            scale,
            HAS_BIAS_SHIFT=has_bias,
            NUM_KV_GROUPS=num_kv_groups,
            NUM_Q_HEADS=heads,
            NUM_SPLITS=num_splits,
            BLOCK_N=BLOCK_N,
            BLOCK_D=block_d,
            BLOCK_K=_next_power_of_2(head_dim_keep),
            num_warps=4,
            num_stages=2,
        )
    else:
        grid1 = (bsz, heads, num_splits)
        _splitk_phase1_kernel[grid1](
            q_full, q_sparse,
            k_full, k_sparse,
            v_full, v_sparse,
            mask_full,
            sparse_bias_shift,
            partial_m, partial_l, partial_acc,
            q_full.stride(0), q_full.stride(1), q_full.stride(2),
            q_sparse.stride(0), q_sparse.stride(1), q_sparse.stride(2),
            k_full.stride(0), k_full.stride(1), k_full.stride(2), k_full.stride(3),
            k_sparse.stride(0), k_sparse.stride(1), k_sparse.stride(2), k_sparse.stride(3),
            v_full.stride(0), v_full.stride(1), v_full.stride(2), v_full.stride(3),
            v_sparse.stride(0), v_sparse.stride(1), v_sparse.stride(2), v_sparse.stride(3),
            mask_full.stride(0), mask_full.stride(1),
            bs_stride_b, bs_stride_h,
            partial_m.stride(0), partial_l.stride(0),
            partial_acc.stride(0), partial_acc.stride(1),
            seq_full, seq_sparse, head_dim, head_dim_keep,
            sparse_per_split,
            scale,
            HAS_BIAS_SHIFT=has_bias,
            NUM_KV_GROUPS=num_kv_groups,
            NUM_SPLITS=num_splits,
            BLOCK_N=BLOCK_N,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=2,
        )

    # Phase 2: merge partials
    grid2 = (bsz, heads)
    _splitk_merge_kernel[grid2](
        partial_m, partial_l, partial_acc,
        out,
        partial_m.stride(0), partial_l.stride(0),
        partial_acc.stride(0), partial_acc.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        heads,
        head_dim,
        NUM_SPLITS=num_splits,
        BLOCK_D=block_d,
        USE_EXP2=False,
        num_warps=4,
        num_stages=1,
    )

    del partial_m, partial_l, partial_acc
    return out.unsqueeze(1), None, None


__all__ = [
    "sparse_channel_decode_triton",
    "sparse_channel_decode_triton_direct",
    "sparse_channel_decode_triton_direct_component",
    "_prepare_q_kernel",
    "_next_power_of_2",
]
