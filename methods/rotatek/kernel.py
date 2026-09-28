"""Three-kernel split-K decode for RotateK: prelude → (sparse + full) → merge.

Why three kernels (and not the previous "one phase-1 with split-0 branch")
==========================================================================

The earlier design fused the full-D (prompt + text) and sparse (vision in
truncated D_keep) paths into a single phase-1 kernel, gating the full-D
loop on `if pid_s == 0`. This kept the launch count low but had a serious
hidden cost: Triton's compiler must allocate registers for the union of
both branches, so ALL split programs (including pid_s = 1..N-1, which only
run the sparse path) carried the q_full[BLOCK_D=128] + extra control-flow
register footprint. Higher per-program register count → lower SM
occupancy → memory-latency hiding broken → ~4× off from theoretical
bandwidth.

Comparing measured numbers at S=12K (image=1k×1, max_num=48 on A100):

    method   triton_attn (ms)   bandwidth efficiency
    dense       0.158            ~20% of HBM peak (typical Triton)
    rotatek     0.323            ~6%  of HBM peak  (~3× worse than dense)

That gap was entirely on the rotatek kernel: it reads LESS memory than
dense (truncated K) yet ran longer. Splitting the kernels lets the sparse
path stay register-lean.

Layout
------
1. `_rotatek_prelude_kernel` — grid (B, H_q). Computes
       q_sparse[b, h, :] = q_full[b, h, :] @ R_partial[b, kv_h, :, :]
       bias_val[b, h]    = q_full[b, h, :] · δμ[b, kv_h, :]   (if HAS_BIAS)
   Writes both into HBM scratch buffers. R is loaded once per (b, h)
   instead of NUM_SPLITS times.

2. `_rotatek_sparse_kernel` — grid (B, H_q, NUM_SPLITS). Each program
   processes its slice of vision tokens against q_sparse / bias_val.
   No q_full in the inner loop → register-lean → high occupancy.

3. `_rotatek_full_kernel` — grid (B, H_q). Each program iterates the
   prompt + text tokens at full D against q_full. Tiny task (~80 tokens
   total), but kept separate from sparse so its register burden doesn't
   bleed.

4. `_splitk_merge_kernel` (imported, unchanged) — grid (B, H_q). Merges
   NUM_SPLITS sparse partials + 1 full partial = (NUM_SPLITS + 1) partials
   into the final output.

Public entry-point: `rotatek_decode_fused_triton(...)`. Same signature.
"""
from __future__ import annotations

import math
from typing import Tuple

from typing import Dict

import torch
import triton
import triton.language as tl

from kernel.sparse_channel_flash_decoding_triton import (
    _splitk_merge_kernel,
    _next_power_of_2,
)


# ---------------------------------------------------------------------------
# Module-level scratch-buffer cache, keyed by (shape, device, dtype).
#
# The four scratch tensors (partial_m / partial_l / partial_acc / out) are
# fully overwritten every call by the kernels — they're transient working
# memory, not state. Allocating them fresh on each call costs ~5-10μs each
# of Python+allocator overhead (so ~20-40μs/call total). Caching them in a
# module-level dict turns that into a single dict-lookup (~1μs).
#
# Memory cost is negligible (~1MB at typical decode shapes) and PyTorch's
# caching allocator already keeps the underlying blocks live across the
# session anyway — the only difference is that we hold the Tensor objects
# explicitly so we don't pay the per-call construction overhead.
# ---------------------------------------------------------------------------

_DECODE_BUFFER_CACHE: Dict[tuple, tuple] = {}


def _get_decode_buffers(bsz: int, heads: int, head_dim: int,
                         num_splits: int, device: torch.device,
                         out_dtype: torch.dtype) -> tuple:
    """Return cached (partial_m, partial_l, partial_acc, out) for this
    decode shape. Allocates on first miss; subsequent calls are O(1)
    dict lookups."""
    key = (bsz, heads, head_dim, num_splits, str(device), out_dtype)
    bufs = _DECODE_BUFFER_CACHE.get(key)
    if bufs is None:
        total_partials = bsz * heads * num_splits
        partial_m = torch.empty(total_partials, device=device, dtype=torch.float32)
        partial_l = torch.empty(total_partials, device=device, dtype=torch.float32)
        partial_acc = torch.empty(
            total_partials, head_dim, device=device, dtype=torch.float32,
        )
        out = torch.empty(
            (bsz, heads, head_dim), device=device, dtype=out_dtype,
        )
        bufs = (partial_m, partial_l, partial_acc, out)
        _DECODE_BUFFER_CACHE[key] = bufs
    return bufs


# ---------------------------------------------------------------------------
# Prelude: per-(b, h) compute of q_sparse and bias_val.
# ---------------------------------------------------------------------------

@triton.jit
def _rotatek_prelude_kernel(
    q_full_ptr,
    R_ptr,                      # [B, H_kv, D, D_keep]
    delta_mu_ptr,               # [B, H_kv, D] (read only when HAS_BIAS=1)
    q_sparse_out_ptr,           # [B, H_q, D_keep] fp32
    bias_out_ptr,               # [B, H_q]        fp32
    # Q strides
    stride_qf_b, stride_qf_h, stride_qf_d,
    # R strides
    stride_R_b, stride_R_h, stride_R_d, stride_R_k,
    # δμ strides
    stride_dm_b, stride_dm_h, stride_dm_d,
    # q_sparse output strides
    stride_qs_b, stride_qs_h, stride_qs_k,
    # bias_val output strides
    stride_bv_b, stride_bv_h,
    head_dim, head_dim_keep,
    HAS_BIAS: tl.constexpr,
    NUM_KV_GROUPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    kv_h = pid_h // NUM_KV_GROUPS

    offs_d = tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_DK)
    d_mask = offs_d < head_dim
    dk_mask = offs_k < head_dim_keep

    qf_base = q_full_ptr + pid_b * stride_qf_b + pid_h * stride_qf_h
    q = tl.load(qf_base + offs_d * stride_qf_d, mask=d_mask, other=0.0).to(tl.float32)

    R_base = R_ptr + pid_b * stride_R_b + kv_h * stride_R_h
    R_tile = tl.load(
        R_base + offs_d[:, None] * stride_R_d + offs_k[None, :] * stride_R_k,
        mask=d_mask[:, None] & dk_mask[None, :], other=0.0,
    ).to(tl.float32)
    q_sparse = tl.sum(q[:, None] * R_tile, axis=0)
    q_sparse = tl.where(dk_mask, q_sparse, 0.0)

    qs_base = q_sparse_out_ptr + pid_b * stride_qs_b + pid_h * stride_qs_h
    tl.store(qs_base + offs_k * stride_qs_k, q_sparse, mask=dk_mask)

    if HAS_BIAS:
        dm_base = delta_mu_ptr + pid_b * stride_dm_b + kv_h * stride_dm_h
        dm = tl.load(
            dm_base + offs_d * stride_dm_d, mask=d_mask, other=0.0,
        ).to(tl.float32)
        bias_val = tl.sum(q * dm, axis=0)
    else:
        bias_val = tl.zeros([], dtype=tl.float32)
    tl.store(bias_out_ptr + pid_b * stride_bv_b + pid_h * stride_bv_h, bias_val)


# ---------------------------------------------------------------------------
# Sparse path: split-K online softmax over vision tokens (D_keep dim).
# Inlines Q@R + Q·δμ at the top so we don't need a separate prelude
# kernel. The full-D path (prompt+text) is in a SEPARATE kernel
# (`_rotatek_combined_kernel`) so its register tile [BLOCK_N, BLOCK_D]
# doesn't bleed into this kernel's compile-time register count.
#
# We tried inlining the full path into split-0 here (with `if pid_s == 0`)
# but the compiler can't statically eliminate the branch, so the K/V
# [BLOCK_N, BLOCK_D] tile registers were allocated for ALL splits. That
# pushed register count 111→141, dropping occupancy from 25% to 19% and
# adding ~35μs to the sparse kernel — net loss. Keeping full path out is
# the correct trade-off.
# ---------------------------------------------------------------------------

@triton.jit
def _rotatek_sparse_kernel(
    q_full_ptr,                 # [B, H_q, D]
    R_ptr,                      # [B, H_kv, D, D_keep]
    delta_mu_ptr,               # [B, H_kv, D] (read only when HAS_BIAS=1)
    k_sparse_ptr,
    v_sparse_ptr,
    partial_m_ptr, partial_l_ptr, partial_acc_ptr,
    # Q strides
    stride_qf_b, stride_qf_h, stride_qf_d,
    # R strides
    stride_R_b, stride_R_h, stride_R_d, stride_R_k,
    # δμ strides
    stride_dm_b, stride_dm_h, stride_dm_d,
    # K strides
    stride_ks_b, stride_ks_h, stride_ks_s, stride_ks_d,
    # V strides
    stride_vs_b, stride_vs_h, stride_vs_s, stride_vs_d,
    # partial-buffer strides
    stride_pm_bhs,
    stride_pl_bhs,
    stride_pa_bhs, stride_pa_d,
    # dims
    seq_sparse, head_dim, head_dim_keep,
    sparse_per_split,
    scale,
    HAS_BIAS: tl.constexpr,
    NUM_KV_GROUPS: tl.constexpr,
    NUM_SPLITS_SPARSE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)          # 0 .. NUM_SPLITS_SPARSE - 1
    kv_h = pid_h // NUM_KV_GROUPS

    offs_d = tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_DK)
    d_mask = offs_d < head_dim
    dk_mask = offs_k < head_dim_keep

    # ---- Inline Q@R + Q·δμ (was a separate prelude kernel) -----------
    qf_base = q_full_ptr + pid_b * stride_qf_b + pid_h * stride_qf_h
    q_full = tl.load(qf_base + offs_d * stride_qf_d, mask=d_mask, other=0.0).to(tl.float32)

    R_base = R_ptr + pid_b * stride_R_b + kv_h * stride_R_h
    R_tile = tl.load(
        R_base + offs_d[:, None] * stride_R_d + offs_k[None, :] * stride_R_k,
        mask=d_mask[:, None] & dk_mask[None, :], other=0.0,
    ).to(tl.float32)
    q_sparse = tl.sum(q_full[:, None] * R_tile, axis=0)
    q_sparse = tl.where(dk_mask, q_sparse, 0.0)

    if HAS_BIAS:
        dm_base = delta_mu_ptr + pid_b * stride_dm_b + kv_h * stride_dm_h
        dm = tl.load(dm_base + offs_d * stride_dm_d, mask=d_mask, other=0.0).to(tl.float32)
        bias_val = tl.sum(q_full * dm, axis=0)
    else:
        bias_val = tl.zeros([], dtype=tl.float32)

    # ---- online-softmax state ----------------------------------------
    m = tl.full([], -1e30, dtype=tl.float32)
    l = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # ---- sparse (vision, D_keep) tokens ------------------------------
    sparse_start = pid_s * sparse_per_split
    sparse_end = tl.minimum(sparse_start + sparse_per_split, seq_sparse)
    for block_start in range(0, sparse_per_split, BLOCK_N):
        offs_n = sparse_start + block_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < sparse_end

        ks_base = k_sparse_ptr + pid_b * stride_ks_b + kv_h * stride_ks_h
        k = tl.load(
            ks_base + offs_n[:, None] * stride_ks_s + offs_k[None, :] * stride_ks_d,
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

    flat_idx = pid_b * (tl.num_programs(1) * NUM_SPLITS_SPARSE) + pid_h * NUM_SPLITS_SPARSE + pid_s
    tl.store(partial_m_ptr + flat_idx * stride_pm_bhs, m)
    tl.store(partial_l_ptr + flat_idx * stride_pl_bhs, l)
    pa_base = partial_acc_ptr + flat_idx * stride_pa_bhs
    tl.store(pa_base + offs_d * stride_pa_d, acc, mask=d_mask)


# ---------------------------------------------------------------------------
# Combined full + merge kernel. Grid (B, H_q). Each program:
#   (a) does the prompt + text full-D path online-softmax into register state
#   (b) iterates the NUM_SPLITS_SPARSE sparse partials from HBM, merging
#       each into the running (m, l, acc) via online-softmax update
#   (c) writes the final output directly
#
# This kernel runs after sparse, on a small (B, H_q) grid. It IS launch-
# overhead-bound (~50μs) but the work is unavoidable: we have to read the
# sparse partials and produce final output. Combining full path + merge
# into the same launch saves vs (separate full + separate merge) because
# both small kernels have similar launch overhead — better to have one of
# them than two.
# ---------------------------------------------------------------------------

@triton.jit
def _rotatek_combined_kernel(
    q_full_ptr,
    k_full_ptr,
    v_full_ptr,
    mask_full_ptr,
    partial_m_ptr, partial_l_ptr, partial_acc_ptr,
    out_ptr,
    stride_qf_b, stride_qf_h, stride_qf_d,
    stride_kf_b, stride_kf_h, stride_kf_s, stride_kf_d,
    stride_vf_b, stride_vf_h, stride_vf_s, stride_vf_d,
    stride_mf_b, stride_mf_s,
    stride_pm_bhs, stride_pl_bhs,
    stride_pa_bhs, stride_pa_d,
    stride_ob, stride_oh, stride_od,
    seq_full, head_dim, scale,
    NUM_KV_GROUPS: tl.constexpr,
    NUM_SPLITS_SPARSE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    kv_h = pid_h // NUM_KV_GROUPS

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    qf_base = q_full_ptr + pid_b * stride_qf_b + pid_h * stride_qf_h
    q_full = tl.load(qf_base + offs_d * stride_qf_d, mask=d_mask, other=0.0).to(tl.float32)

    m = tl.full([], -1e30, dtype=tl.float32)
    l = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # ---- (a) Full-D path: prompt + text tokens -----------------------
    for block_start in range(0, seq_full, BLOCK_N):
        offs_n = block_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_full

        kf_base = k_full_ptr + pid_b * stride_kf_b + kv_h * stride_kf_h
        k = tl.load(
            kf_base + offs_n[:, None] * stride_kf_s + offs_d[None, :] * stride_kf_d,
            mask=n_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)
        logits = tl.sum(k * q_full[None, :], axis=1) * scale

        valid = tl.load(
            mask_full_ptr + pid_b * stride_mf_b + offs_n * stride_mf_s,
            mask=n_mask, other=0,
        )
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

    # ---- (b) Merge sparse partials -----------------------------------
    n_heads = tl.num_programs(1)
    for s in range(NUM_SPLITS_SPARSE):
        flat = pid_b * (n_heads * NUM_SPLITS_SPARSE) + pid_h * NUM_SPLITS_SPARSE + s
        m_s = tl.load(partial_m_ptr + flat * stride_pm_bhs)
        l_s = tl.load(partial_l_ptr + flat * stride_pl_bhs)
        acc_s = tl.load(
            partial_acc_ptr + flat * stride_pa_bhs + offs_d * stride_pa_d,
            mask=d_mask, other=0.0,
        )
        new_max = tl.maximum(m, m_s)
        corr = tl.exp(m - new_max)
        corr_s = tl.exp(m_s - new_max)
        l = l * corr + l_s * corr_s
        acc = acc * corr + acc_s * corr_s
        m = new_max

    # ---- (c) Final output --------------------------------------------
    output = acc / (l + 1e-6)
    out_base = out_ptr + pid_b * stride_ob + pid_h * stride_oh
    tl.store(out_base + offs_d * stride_od,
             output.to(out_ptr.dtype.element_ty), mask=d_mask)


# ---------------------------------------------------------------------------
# Launcher: sparse split-K  →  combined full + merge.   (2 kernels total)
# ---------------------------------------------------------------------------

def rotatek_decode_fused_triton(
    q_full: torch.Tensor,                # [B, H_q, D]
    R_partial: torch.Tensor,             # [B, H_kv, D, D_keep]
    delta_mu: torch.Tensor | None,       # [B, H_kv, D] or None
    k_full: torch.Tensor,                # [B, H_kv, S_full, D]
    v_full: torch.Tensor,                # [B, H_kv, S_full, D]
    mask_full: torch.Tensor,             # [B, S_full] uint8
    k_sparse: torch.Tensor,              # [B, H_kv, S_sparse, D_keep]
    v_sparse: torch.Tensor,              # [B, H_kv, S_sparse, D]
    num_kv_groups: int,
) -> Tuple[torch.Tensor, None, None]:
    """RotateK decode in 2 kernels — matching the dense kernel's launch
    count after our split-kernel + Q@R-inline + full-merge-fuse refactors.

    Returns
    -------
    out          : [B, 1, H_q, D]
    logits_full  : None
    logits_sparse: None
    """
    if q_full.dim() != 3:
        raise ValueError("q_full must be [B, H_q, D]")
    if R_partial.dim() != 4:
        raise ValueError("R_partial must be [B, H_kv, D, D_keep]")
    if mask_full.dtype != torch.uint8:
        raise ValueError("mask_full must be uint8")

    bsz, heads, head_dim = q_full.shape
    head_dim_keep = R_partial.shape[-1]
    seq_full = k_full.shape[-2]
    seq_sparse = k_sparse.shape[-2]
    h_kv = R_partial.shape[1]
    if heads // num_kv_groups != h_kv:
        raise ValueError(
            f"R_partial H_kv={h_kv} but H_q={heads}, num_kv_groups={num_kv_groups}"
        )

    scale = 1.0 / math.sqrt(head_dim)
    block_d = _next_power_of_2(head_dim)
    block_dk = _next_power_of_2(head_dim_keep)
    BLOCK_N = 64

    # Match dense's split policy: tile-bound, capped at 64.
    num_splits_sparse = max(1, min((seq_sparse + BLOCK_N - 1) // BLOCK_N, 64))
    sparse_per_split = (seq_sparse + num_splits_sparse - 1) // num_splits_sparse

    has_bias = delta_mu is not None
    if has_bias:
        if not delta_mu.is_contiguous():
            delta_mu = delta_mu.contiguous()
        dm_strides = (delta_mu.stride(0), delta_mu.stride(1), delta_mu.stride(2))
    else:
        delta_mu = q_full   # placeholder; never read because HAS_BIAS=False
        dm_strides = (0, 0, 0)

    if not q_full.is_contiguous():
        q_full = q_full.contiguous()
    if not R_partial.is_contiguous():
        R_partial = R_partial.contiguous()

    # ---- Get scratch buffers (module-level cache, no per-call alloc) --
    partial_m, partial_l, partial_acc, out = _get_decode_buffers(
        bsz, heads, head_dim, num_splits_sparse,
        q_full.device, q_full.dtype,
    )

    # ---- Kernel 1: sparse split-K with inlined Q@R + Q·δμ -------------
    _rotatek_sparse_kernel[(bsz, heads, num_splits_sparse)](
        q_full, R_partial, delta_mu,
        k_sparse, v_sparse,
        partial_m, partial_l, partial_acc,
        # Q strides
        q_full.stride(0), q_full.stride(1), q_full.stride(2),
        # R strides
        R_partial.stride(0), R_partial.stride(1),
        R_partial.stride(2), R_partial.stride(3),
        # δμ strides
        dm_strides[0], dm_strides[1], dm_strides[2],
        # K / V strides
        k_sparse.stride(0), k_sparse.stride(1), k_sparse.stride(2), k_sparse.stride(3),
        v_sparse.stride(0), v_sparse.stride(1), v_sparse.stride(2), v_sparse.stride(3),
        # partial strides
        partial_m.stride(0), partial_l.stride(0),
        partial_acc.stride(0), partial_acc.stride(1),
        # dims
        seq_sparse, head_dim, head_dim_keep,
        sparse_per_split,
        scale,
        HAS_BIAS=has_bias,
        NUM_KV_GROUPS=num_kv_groups,
        NUM_SPLITS_SPARSE=num_splits_sparse,
        BLOCK_N=BLOCK_N,
        BLOCK_D=block_d,
        BLOCK_DK=block_dk,
        num_warps=1,
        num_stages=3,
    )

    # ---- Kernel 2: combined full path + merge of sparse partials ------
    _rotatek_combined_kernel[(bsz, heads)](
        q_full, k_full, v_full, mask_full,
        partial_m, partial_l, partial_acc, out,
        q_full.stride(0), q_full.stride(1), q_full.stride(2),
        k_full.stride(0), k_full.stride(1), k_full.stride(2), k_full.stride(3),
        v_full.stride(0), v_full.stride(1), v_full.stride(2), v_full.stride(3),
        mask_full.stride(0), mask_full.stride(1),
        partial_m.stride(0), partial_l.stride(0),
        partial_acc.stride(0), partial_acc.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        seq_full, head_dim, scale,
        NUM_KV_GROUPS=num_kv_groups,
        NUM_SPLITS_SPARSE=num_splits_sparse,
        BLOCK_N=BLOCK_N,
        BLOCK_D=block_d,
        num_warps=1,
        num_stages=2,
    )
    return out.unsqueeze(1), None, None


__all__ = ["rotatek_decode_fused_triton"]
