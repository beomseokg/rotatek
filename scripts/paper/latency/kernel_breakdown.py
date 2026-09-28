"""Holistic latency benchmark for key-channel pruning methods.

Compares ThinK / SparK / RotateK / no-compression across a multi-axis grid
(S, B, sparsity, K_sample) at both prefill and decode, with per-stage
breakdowns so speedups can be attributed to either the *method* (algorithmic)
or *engineering* (torch.compile, sparse kernel, subsample-PCA) layers.

Sweeps included
---------------
  Sweep 1 — S sweep at B=1, D_keep=32 (main scaling story)
  Sweep 2 — Batch size effect at S=8000, eager + compiled
  Sweep 3 — Channel sparsity effect at B=1, S=8000
  Sweep 4 — B × S grid with all methods torch.compile'd + naive full-channel

For each method we time the production prefill+decode path, mirroring the
adapter logic in `lmms_eval.models.model_utils.{internvl,qwen}`.

Decode-path breakdown:

  Naive full-channel (no compression)
    FA2 (Triton split-K) over [S_total, D]   — upper bound

  ThinK (per-head topk channels, full-D recovery)
    1. scatter recovery : zero-fill K_pruned [S_v, D_keep] → [S_v, D]
    2. cat              : [K_prompt, K_recovered, K_text] over seq dim
    3. attention        : Triton split-K dense decode over [S_total, D]

  SparK (per-token topk channels, mean-fill recovery)
    1. mean-fill scatter : per-token expand+scatter to full D
    2. cat               : [K_prompt, K_recovered, K_text]
    3. attention         : Triton split-K dense decode over [S_total, D]

  RotateK truncated (no recovery, sparse split-K)
    1. Q rotation       : einsum Q @ R_partial  → q_sparse [B, H_q, D_keep]
    2. δμ bias          : (Q · δμ_hq).sum(-1)   per (B, H_q)
    3. cat (non-vision) : [K_prompt, K_text] only
    4. sparse kernel    : split-K decode — vision in D_keep, rest in D

Prefill paths: scoring + topk + gather (ThinK), per-token topk (SparK),
PCA via cov + power iteration + projection (RotateK).

Run:
  python -m latency.kernel_breakdown
"""
from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers.modeling_flash_attention_utils import _flash_attention_forward

from rotatek import rotatek_decode_triton
from rotatek.rotation import power_iteration_gpu, _power_iter_loop
from rotatek.baselines.spark import per_token_channel_prune
from rotatek.kernels.sparse_channel_flash_decoding import (
    sparse_channel_decode_triton as _sparse_decode_triton,
)

# torch.compile-captured versions of the prefill compute pieces. Each method
# gets its own compiled variant so the bench can compare eager vs graph-
# captured *for every method* — required for fair attribution between
# "method choice" and "engineering optimization" effects.
_power_iter_loop_graph = torch.compile(_power_iter_loop, mode="reduce-overhead")


# InternLM2.5-8B
H_Q = 32
H_KV = 8
HEAD_DIM = 128
HIDDEN = H_Q * HEAD_DIM
NUM_KV_GROUPS = H_Q // H_KV
NUM_LAYERS = 32

K_KEEP = 32  # 75% sparsity
POWER_ITERS = 5

# Realistic non-vision context (typical mid-generation state)
S_PROMPT = 50
S_TEXT = 20


def _bench(fn, *args, reps=50, warmup=10, **kwargs):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    one_call_ms = (time.perf_counter() - t0) * 1000
    if one_call_ms > 100:
        reps, warmup = 5, 2
    elif one_call_ms > 20:
        reps, warmup = 10, 3

    for _ in range(warmup):
        fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / reps


# ============================================================================
# ThinK pipeline
# ============================================================================

def _think_score_and_gather(K_vision, q_recent, k_keep=K_KEEP):
    """ThinK prefill (paper-faithful): build a per-head bool *keep mask*
    from the q²·k² channel scores. Returns the compact gathered K alongside
    the bool mask (the reference stores the mask, not long indices)."""
    q_score = (q_recent ** 2).mean(dim=-2)
    B, H_q, D = q_score.shape
    H_kv = K_vision.shape[1]
    q_score = q_score.view(B, H_kv, H_q // H_kv, D).mean(dim=2)
    k_score = (K_vision ** 2).mean(dim=-2)
    score = q_score * k_score                                   # [B, H_kv, D]
    _, keep_idx = torch.topk(score, k_keep, dim=-1)             # [B, H_kv, D_keep]
    keep_mask = torch.zeros_like(score, dtype=torch.bool)
    keep_mask.scatter_(-1, keep_idx, True)                      # [B, H_kv, D] bool
    keep_idx_sorted, _ = keep_idx.sort(dim=-1)                  # ascending channel idx so
    keep_idx_exp = keep_idx_sorted.unsqueeze(2).expand(-1, -1, K_vision.shape[2], -1)
    K_pruned = torch.gather(K_vision, dim=-1, index=keep_idx_exp)  # matches mask iteration order
    return K_pruned, keep_mask


_think_score_and_gather_compiled = torch.compile(_think_score_and_gather, mode="reduce-overhead")


def _think_scatter(K_pruned, keep_mask):
    """ThinK decode-time recovery (paper-faithful): boolean indexing.
    `recovered[mask.expand(...)] = kept_keys` mirrors the reference impl."""
    B, H_kv, S, _ = K_pruned.shape
    recovered = torch.zeros(
        B, H_kv, S, HEAD_DIM, dtype=K_pruned.dtype, device=K_pruned.device,
    )
    mask_exp = keep_mask.unsqueeze(2).expand(-1, -1, S, -1)
    recovered[mask_exp] = K_pruned.reshape(-1)
    return recovered


def _cat_full(K_prompt, K_vision, K_text):
    return torch.cat([K_prompt, K_vision, K_text], dim=-2)


def _fa2_full(Q, K, V):
    """Production flash_attn_2 dense decode (transformers wrapper).

    Q: [B, H_q, 1, D]; K/V: [B, H_kv, S, D]. Has native GQA so no
    repeat_interleave is needed. Used as the *external* baseline.
    """
    q_fa = Q.transpose(1, 2)  # [B, 1, H_q, D]
    k_fa = K.transpose(1, 2)  # [B, S, H_kv, D]
    v_fa = V.transpose(1, 2)
    return _flash_attention_forward(q_fa, k_fa, v_fa, None, 1, is_causal=False)


def _fa_prefill(Q, K, V):
    """Prefill-time causal FA: Q/K/V all at q_len = S_total.

    This is the work *every* method does at prefill (regardless of compression);
    the dominant cost since attention is O(S²·D). Compression methods add their
    overhead on top of this. Production uses flash_attn_2 via transformers'
    `_flash_attention_forward`, which has native GQA support.
    """
    q_fa = Q.transpose(1, 2)  # [B, S, H_q, D]
    k_fa = K.transpose(1, 2)  # [B, S, H_kv, D]
    v_fa = V.transpose(1, 2)
    return _flash_attention_forward(
        q_fa, k_fa, v_fa, None, q_fa.shape[1], is_causal=True,
    )


def _fa2_full_triton(Q, K, V, num_kv_groups=NUM_KV_GROUPS):
    """Triton-based dense decode using the same kernel infra as RotateK.

    Mirrors `run_dense_decode_kernel` from the production adapter: routes
    all K/V through the sparse path with the kernel's full-D tiles (no
    truncation, no bias). This keeps the kernel-launch / split-K /
    GQA-MMA characteristics identical to RotateK so the comparison
    isolates *method* effects from *kernel-engineering* effects.
    """
    bsz = Q.shape[0]
    Q_in = Q.squeeze(2)  # [B, H_q, D]
    full_len = K.shape[-2]
    empty_k = K[:, :, :0, :].contiguous()
    empty_v = V[:, :, :0, :].contiguous()
    empty_mask = torch.ones((bsz, 0), dtype=torch.uint8, device=Q.device)
    mask_split = torch.ones((bsz, full_len), dtype=torch.uint8, device=Q.device)
    out, _, _ = _sparse_decode_triton(
        q_full=Q_in, k_full=empty_k, v_full=empty_v, mask_full=empty_mask,
        q_sparse=Q_in, k_sparse=K, v_sparse=V, mask_sparse=mask_split,
        sparse_bias_shift=None, num_kv_groups=num_kv_groups,
    )
    return out


# ============================================================================
# SparK: per-token channel pruning with mean fill at decode
# ============================================================================

def _spark_score_and_gather(K_vision, q_recent, ratio):
    """SparK prefill: per-token Top-K channel selection + pruned-mean stash.

    Costs more than ThinK at prefill because the topk is *per token* (S
    independent topks), not per head.
    """
    return per_token_channel_prune(q_recent, K_vision, ratio=ratio)


_spark_score_and_gather_compiled = torch.compile(_spark_score_and_gather, mode="reduce-overhead")


def _spark_recover(K_compact, keep_mask, pruned_mean):
    """SparK decode recovery (compact-storage variant): pre-fill with the
    per-token pruned-channel mean, then write compact K into the True
    positions of the per-token bool mask via boolean indexing — same
    primitive as ThinK's recovery, with per-token granularity.
    """
    B, H, S, _ = K_compact.shape
    recovered = pruned_mean.expand(B, H, S, HEAD_DIM).contiguous()
    recovered[keep_mask] = K_compact.reshape(-1)
    return recovered


# ============================================================================
# RotateK truncated pipeline
# ============================================================================

def _rotatek_prefill_truncated_impl(K_vision, k, K_sample, power_iters, power_iter_fn):
    """Shared body for eager and graph-captured prefill variants.

    `power_iter_fn` is either eager `_power_iter_loop` or its
    `torch.compile`-wrapped counterpart `_power_iter_loop_graph`.
    """
    S = K_vision.shape[-2]
    if K_sample is not None and S > K_sample:
        idx = torch.randperm(S, device=K_vision.device)[:K_sample]
        K_for_cov = K_vision.index_select(-2, idx)
    else:
        K_for_cov = K_vision
    S_cov = K_for_cov.shape[-2]
    gram = torch.einsum("bhsd,bhse->bhde", K_for_cov, K_for_cov).float()
    mu_sub_sum = K_for_cov.sum(dim=-2, dtype=torch.float32, keepdim=True)
    mean_sub = mu_sub_sum / S_cov
    mu_sub = mean_sub.squeeze(-2)
    cov = gram - S_cov * torch.einsum("bhd,bhe->bhde", mu_sub, mu_sub)
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    g_rng = torch.Generator(device=cov.device).manual_seed(0)
    V = torch.randn(
        *cov.shape[:-2], cov.shape[-1], k,
        device=cov.device, dtype=cov.dtype, generator=g_rng,
    )
    R_partial_fp32 = power_iter_fn(cov, V, power_iters)
    R_partial = R_partial_fp32.to(K_vision.dtype)
    K_pruned = torch.matmul(K_vision, R_partial)
    full_mu_sum = K_vision.sum(dim=-2, dtype=torch.float32, keepdim=True)
    full_mean = full_mu_sum / S
    R_fp32 = R_partial.float()
    mu_proj = torch.matmul(
        torch.matmul(full_mean, R_fp32), R_fp32.transpose(-2, -1),
    )
    delta_mu = (full_mean - mu_proj).squeeze(-2).to(K_vision.dtype)
    return K_pruned, R_partial, delta_mu


def _rotatek_prefill_truncated_compiled(K_vision, k=K_KEEP, K_sample=None, power_iters=POWER_ITERS):
    """torch.compile reduce-overhead path: power-iter loop is graph-captured."""
    return _rotatek_prefill_truncated_impl(
        K_vision, k, K_sample, power_iters, _power_iter_loop_graph,
    )


def _rotatek_prefill_truncated(K_vision, k=K_KEEP, K_sample=None, power_iters=POWER_ITERS):
    """Uncentered formulation: cov = K^T K - S·μμ^T.

    Avoids materialising a fp32 copy of K_vision and a fp32 centered tensor
    (4× memory writes at long S). The bf16 K^T K uses tensor-core fp32
    accumulators on Hopper/Ampere, so we keep precision through the matmul
    and only the final D×D cast pays bf16-rounding cost.

    When K_sample is set and S > K_sample, cov is estimated from a uniform
    random subset (Halko-Martinsson-Tropp). δμ-bias still uses the full
    sequence mean to keep the decode-time correction exact.
    """
    S = K_vision.shape[-2]
    if K_sample is not None and S > K_sample:
        idx = torch.randperm(S, device=K_vision.device)[:K_sample]
        K_for_cov = K_vision.index_select(-2, idx)
    else:
        K_for_cov = K_vision
    S_cov = K_for_cov.shape[-2]
    # K^T K via bf16 matmul on the (possibly subsampled) tensor.
    gram = torch.einsum("bhsd,bhse->bhde", K_for_cov, K_for_cov).float()
    # Subsample mean for the cov identity.
    mu_sub_sum = K_for_cov.sum(dim=-2, dtype=torch.float32, keepdim=True)
    mean_sub = mu_sub_sum / S_cov
    mu_sub = mean_sub.squeeze(-2)
    cov = gram - S_cov * torch.einsum("bhd,bhe->bhde", mu_sub, mu_sub)
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    R_partial_fp32 = power_iteration_gpu(cov, k=k, num_iters=power_iters, seed=0)
    R_partial = R_partial_fp32.to(K_vision.dtype)
    K_pruned = torch.matmul(K_vision, R_partial)
    # Full-sequence mean for the δμ-bias (must match decode-time).
    full_mu_sum = K_vision.sum(dim=-2, dtype=torch.float32, keepdim=True)
    full_mean = full_mu_sum / S
    R_fp32 = R_partial.float()
    mu_proj = torch.matmul(
        torch.matmul(full_mean, R_fp32), R_fp32.transpose(-2, -1),
    )
    delta_mu = (full_mean - mu_proj).squeeze(-2).to(K_vision.dtype)
    return K_pruned, R_partial, delta_mu


def _rotatek_q_rotate(Q, R_partial, num_kv_groups=NUM_KV_GROUPS):
    R_hq = R_partial.repeat_interleave(num_kv_groups, dim=1)
    return torch.einsum("bhd,bhdk->bhk", Q.squeeze(2), R_hq)


def _rotatek_bias(Q, delta_mu, num_kv_groups=NUM_KV_GROUPS):
    delta_mu_hq = delta_mu.repeat_interleave(num_kv_groups, dim=1)
    return (Q.squeeze(2) * delta_mu_hq).sum(dim=-1)


def _cat_non_vision(K_prompt, K_text):
    return torch.cat([K_prompt, K_text], dim=-2)


def _rotatek_kernel(q_full, k_full, v_full, mask_full, q_sparse, k_sparse, v_sparse,
                    bias, num_kv_groups=NUM_KV_GROUPS):
    out, _, _ = rotatek_decode_triton(
        q_full=q_full, k_full=k_full, v_full=v_full, mask_full=mask_full,
        q_sparse=q_sparse, k_sparse=k_sparse, v_sparse=v_sparse,
        sparse_bias_shift=bias, num_kv_groups=num_kv_groups,
    )
    return out


# ============================================================================
# Per-layer benchmarks
# ============================================================================

def measure_layer(S_vision: int, B: int = 1, k_keep: int = K_KEEP, dtype=torch.bfloat16):
    device = "cuda"
    S_total = S_PROMPT + S_vision + S_TEXT

    # ----- Base prefill cost: causal FA over the full S_total sequence -----
    # This is the dominant prefill work that EVERY method pays; compression
    # overhead (scoring/PCA/etc.) is on top of it. Measured here so the
    # "full-channel prefill = 0" misframing doesn't show up in the totals.
    Q_prefill = torch.randn(B, H_Q, S_total, HEAD_DIM, device=device, dtype=dtype) * 0.01
    K_prefill = torch.randn(B, H_KV, S_total, HEAD_DIM, device=device, dtype=dtype) * 0.01
    V_prefill = torch.randn(B, H_KV, S_total, HEAD_DIM, device=device, dtype=dtype) * 0.01
    pf_base_attn_ms = _bench(_fa_prefill, Q_prefill, K_prefill, V_prefill)
    del Q_prefill, K_prefill, V_prefill
    torch.cuda.empty_cache()

    # Prefill K (full vision sequence, used by both methods to pre-compute caches)
    K_vision = torch.randn(B, H_KV, S_vision, HEAD_DIM, device=device, dtype=dtype) * 0.1
    q_recent = torch.randn(B, H_Q, min(32, S_vision), HEAD_DIM, device=device, dtype=dtype) * 0.01

    # SparK uses a GQA-grouped query (mean over groups) for its scoring.
    G = H_Q // H_KV
    q_recent_grouped = q_recent.view(B, H_KV, G, q_recent.shape[2], HEAD_DIM).mean(2)
    spark_ratio = (HEAD_DIM - k_keep) / HEAD_DIM

    # ----- Prefill (one-time cost per image) -----
    # Eager (no torch.compile)
    pf_think_ms = _bench(_think_score_and_gather, K_vision, q_recent, k_keep=k_keep)
    pf_spark_ms = _bench(
        _spark_score_and_gather, K_vision, q_recent_grouped, spark_ratio,
    )
    pf_rotatek_ms = _bench(_rotatek_prefill_truncated, K_vision, k=k_keep)
    pf_rotatek_1024_ms = _bench(
        _rotatek_prefill_truncated, K_vision, k=k_keep, K_sample=1024,
    )
    pf_rotatek_512_ms = _bench(
        _rotatek_prefill_truncated, K_vision, k=k_keep, K_sample=512,
    )
    pf_rotatek_256_ms = _bench(
        _rotatek_prefill_truncated, K_vision, k=k_keep, K_sample=256,
    )
    # torch.compile reduce-overhead — applied to ALL methods for fair
    # attribution between "method choice" and "graph-capture engineering".
    # If ThinK/SparK don't graph-capture cleanly (topk/sort have CPU sync),
    # the same speedup that helps RotateK won't transfer — that itself is
    # a finding worth reporting.
    pf_think_compile_ms = _bench(
        _think_score_and_gather_compiled, K_vision, q_recent, k_keep=k_keep,
    )
    pf_spark_compile_ms = _bench(
        _spark_score_and_gather_compiled, K_vision, q_recent_grouped, spark_ratio,
    )
    pf_rotatek_compile_ms = _bench(
        _rotatek_prefill_truncated_compiled, K_vision, k=k_keep,
    )
    pf_rotatek_compile_512_ms = _bench(
        _rotatek_prefill_truncated_compiled, K_vision, k=k_keep, K_sample=512,
    )

    # Free prefill K_vision before allocating decode tensors at long S to keep
    # peak memory in check.
    del K_vision, q_recent, q_recent_grouped
    torch.cuda.empty_cache()

    # Decode-time tensors
    Q = torch.randn(B, H_Q, 1, HEAD_DIM, device=device, dtype=dtype) * 0.01
    K_prompt = torch.randn(B, H_KV, S_PROMPT, HEAD_DIM, device=device, dtype=dtype) * 0.01
    K_text = torch.randn(B, H_KV, S_TEXT, HEAD_DIM, device=device, dtype=dtype) * 0.01
    V_prompt = torch.randn(B, H_KV, S_PROMPT, HEAD_DIM, device=device, dtype=dtype) * 0.01
    V_vision = torch.randn(B, H_KV, S_vision, HEAD_DIM, device=device, dtype=dtype) * 0.01
    V_text = torch.randn(B, H_KV, S_TEXT, HEAD_DIM, device=device, dtype=dtype) * 0.01
    V_full = torch.cat([V_prompt, V_vision, V_text], dim=-2)
    mask_full = torch.ones(B, S_PROMPT + S_TEXT, device=device, dtype=torch.uint8)

    # ThinK decode artifacts
    K_pruned_think = torch.randn(B, H_KV, S_vision, k_keep, device=device, dtype=dtype) * 0.1
    # Per-head bool keep mask (paper-faithful): exactly k_keep distinct True
    # channels per head, picked via topk of random scores so the mask has no
    # duplicates and `mask.sum(-1) == k_keep` exactly per row.
    rand_think_scores = torch.randn(B, H_KV, HEAD_DIM, device=device)
    think_keep_idx = rand_think_scores.topk(k_keep, dim=-1).indices
    think_mask = torch.zeros(B, H_KV, HEAD_DIM, device=device, dtype=torch.bool)
    think_mask.scatter_(-1, think_keep_idx, True)

    # SparK decode artifacts (compact-storage variant: K_compact + per-token
    # bool keep_mask + per-token pruned_mean — ThinK-parity primitives).
    K_compact_spark = torch.randn(B, H_KV, S_vision, k_keep, device=device, dtype=dtype) * 0.1
    rand_scores_spark = torch.randn(B, H_KV, S_vision, HEAD_DIM, device=device)
    keep_idx_for_mask = rand_scores_spark.argsort(dim=-1, descending=True)[..., :k_keep]
    keep_mask_spark = torch.zeros(B, H_KV, S_vision, HEAD_DIM, device=device, dtype=torch.bool)
    keep_mask_spark.scatter_(-1, keep_idx_for_mask, True)                 # [B, H, S, D] bool
    pruned_mean_spark = torch.randn(B, H_KV, S_vision, 1, device=device, dtype=dtype) * 0.01

    # RotateK decode artifacts
    K_pruned_rot = torch.randn(B, H_KV, S_vision, k_keep, device=device, dtype=dtype) * 0.1
    R_partial = torch.randn(B, H_KV, HEAD_DIM, k_keep, device=device, dtype=dtype) * 0.1
    delta_mu = torch.randn(B, H_KV, HEAD_DIM, device=device, dtype=dtype) * 0.01

    # ----- No-compression baseline (full K cache, full FA2 at decode) -----
    # Prefill = base FA (dense attention over full S; same cost ALL methods pay).
    # Compression methods add their scoring/PCA work on top.
    pf_full_ms = pf_base_attn_ms  # full-channel = base FA only, no compression
    K_vision_full = torch.randn(B, H_KV, S_vision, HEAD_DIM, device=device, dtype=dtype) * 0.1
    K_full_uncompressed = torch.cat([K_prompt, K_vision_full, K_text], dim=-2)
    de_full_attn_ms = _bench(_fa2_full_triton, Q, K_full_uncompressed, V_full)
    de_full_total_ms = de_full_attn_ms
    del K_vision_full, K_full_uncompressed

    # ----- ThinK decode stages -----
    # ATTENTION uses the SAME Triton kernel infra as RotateK, so kernel-level
    # engineering effects are factored out. Comparison isolates the *method*
    # cost (scatter/cat for ThinK vs Q rotation+bias for RotateK).
    de_think_scatter_ms = _bench(_think_scatter, K_pruned_think, think_mask)
    K_recovered = _think_scatter(K_pruned_think, think_mask)
    de_think_cat_ms = _bench(_cat_full, K_prompt, K_recovered, K_text)
    K_full_recovered = _cat_full(K_prompt, K_recovered, K_text)
    de_think_attn_ms = _bench(_fa2_full_triton, Q, K_full_recovered, V_full)
    de_think_total_ms = de_think_scatter_ms + de_think_cat_ms + de_think_attn_ms

    # External reference: flash_attn_2 (no Triton overhead) — gives the
    # *upper bound* of what ThinK/SparK could ever achieve given a hand-tuned
    # production library. Used to attribute kernel vs method differences.
    de_think_attn_fa2_ms = _bench(_fa2_full, Q, K_full_recovered, V_full)

    # ----- SparK decode stages (compact-storage: bool-indexing into mean-filled buffer) -----
    de_spark_scatter_ms = _bench(
        _spark_recover, K_compact_spark, keep_mask_spark, pruned_mean_spark,
    )
    K_recovered_spark = _spark_recover(K_compact_spark, keep_mask_spark, pruned_mean_spark)
    de_spark_cat_ms = _bench(_cat_full, K_prompt, K_recovered_spark, K_text)
    K_full_recovered_spark = _cat_full(K_prompt, K_recovered_spark, K_text)
    de_spark_attn_ms = _bench(_fa2_full_triton, Q, K_full_recovered_spark, V_full)
    de_spark_total_ms = de_spark_scatter_ms + de_spark_cat_ms + de_spark_attn_ms
    de_spark_attn_fa2_ms = _bench(_fa2_full, Q, K_full_recovered_spark, V_full)
    del K_recovered_spark, K_full_recovered_spark

    # ----- RotateK truncated decode stages -----
    de_rk_qrot_ms = _bench(_rotatek_q_rotate, Q, R_partial)
    de_rk_bias_ms = _bench(_rotatek_bias, Q, delta_mu)
    de_rk_cat_ms = _bench(_cat_non_vision, K_prompt, K_text)

    # Pre-compute kernel inputs once (q_sparse, bias, k_full)
    q_sparse = _rotatek_q_rotate(Q, R_partial)
    bias = _rotatek_bias(Q, delta_mu)
    k_full = _cat_non_vision(K_prompt, K_text)
    v_full_kernel = torch.cat([V_prompt, V_text], dim=-2)
    Q_kernel = Q.squeeze(2)

    de_rk_kernel_ms = _bench(
        _rotatek_kernel, Q_kernel, k_full, v_full_kernel, mask_full,
        q_sparse, K_pruned_rot, V_vision, bias,
    )
    de_rk_total_ms = de_rk_qrot_ms + de_rk_bias_ms + de_rk_cat_ms + de_rk_kernel_ms

    return {
        "B": B, "S": S_vision, "S_total": S_total, "D_keep": k_keep,
        # Base prefill cost shared by all methods (dense FA at q_len = S_total).
        "pf_base_attn": pf_base_attn_ms,
        # Prefill — compression overhead, eager (added on top of pf_base_attn)
        "pf_think": pf_think_ms,
        "pf_spark": pf_spark_ms,
        "pf_rotatek": pf_rotatek_ms,
        "pf_rotatek_1024": pf_rotatek_1024_ms,
        "pf_rotatek_512": pf_rotatek_512_ms,
        "pf_rotatek_256": pf_rotatek_256_ms,
        # Prefill — compression overhead, torch.compile (reduce-overhead)
        "pf_think_compile": pf_think_compile_ms,
        "pf_spark_compile": pf_spark_compile_ms,
        "pf_rotatek_compile": pf_rotatek_compile_ms,
        "pf_rotatek_compile_512": pf_rotatek_compile_512_ms,
        # Total prefill = base FA + per-method compression overhead.
        "pf_full_total": pf_base_attn_ms,
        "pf_think_total": pf_base_attn_ms + pf_think_compile_ms,
        "pf_spark_total": pf_base_attn_ms + pf_spark_compile_ms,
        "pf_rotatek_total": pf_base_attn_ms + pf_rotatek_compile_512_ms,
        # No-compression decode baseline
        "pf_full": pf_full_ms,
        "de_full_attn": de_full_attn_ms,
        "de_full_total": de_full_total_ms,
        # ThinK decode breakdown (Triton attention — fair vs RotateK)
        "de_think_scatter": de_think_scatter_ms,
        "de_think_cat": de_think_cat_ms,
        "de_think_attn": de_think_attn_ms,
        "de_think_total": de_think_total_ms,
        "de_think_attn_fa2": de_think_attn_fa2_ms,  # external ref
        # SparK decode breakdown (Triton attention — fair vs RotateK)
        "de_spark_scatter": de_spark_scatter_ms,
        "de_spark_cat": de_spark_cat_ms,
        "de_spark_attn": de_spark_attn_ms,
        "de_spark_total": de_spark_total_ms,
        "de_spark_attn_fa2": de_spark_attn_fa2_ms,  # external ref
        # RotateK decode breakdown
        "de_rk_qrot": de_rk_qrot_ms,
        "de_rk_bias": de_rk_bias_ms,
        "de_rk_cat": de_rk_cat_ms,
        "de_rk_kernel": de_rk_kernel_ms,
        "de_rk_total": de_rk_total_ms,
    }


def _safe_measure(*args, **kwargs):
    try:
        r = measure_layer(*args, **kwargs)
        return r
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def _print_grid(rows, value_fn, header_left="config", header_top="S", row_label_fn=None,
                col_label_fn=lambda r: r["S"], fmt="{:8.3f}", title=""):
    """Pivot rows by (config, S). value_fn(row) returns the cell value (or None)."""
    if title:
        print(f"\n{title}")
        print("-" * max(60, len(title)))
    if not rows:
        print("(no data)")
        return
    cfgs = sorted({row_label_fn(r) for r in rows if r is not None})
    cols = sorted({col_label_fn(r) for r in rows if r is not None})
    by_key = {(row_label_fn(r), col_label_fn(r)): r for r in rows if r is not None}
    print(f"{header_left:>14} | " + "  ".join(f"{header_top}={c:>6}" for c in cols))
    print("-" * (14 + 4 + len(cols) * 12))
    for cfg in cfgs:
        cells = []
        for c in cols:
            r = by_key.get((cfg, c))
            cells.append(fmt.format(value_fn(r)) if r is not None else "    OOM ")
        print(f"{str(cfg):>14} | " + "  ".join(cells))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    print(f"InternLM2.5-8B simulation, D={HEAD_DIM}")
    print(f"Non-vision context: S_prompt={S_PROMPT}, S_text={S_TEXT}")
    print()

    # ============================================================
    # Sweep 1: S only (B=1, D_keep=32 = 75% sparsity)
    # ============================================================
    print("=" * 110)
    print("SWEEP 1 — S sweep at B=1, D_keep=32 (75% sparsity) — main scaling story")
    print("=" * 110)
    S_values = [4000, 8000, 16000, 32000, 64000, 128000]
    rows = []
    for S in S_values:
        rows.append(_safe_measure(S, B=1, k_keep=32))
    rows = [r for r in rows if r is not None]

    # ============================================================
    # FAIRNESS table: torch.compile applied to ALL methods
    # ============================================================
    print("=" * 130)
    print("PREFILL per-layer (ms) — fair comparison: torch.compile applied to ALL methods")
    print("=" * 130)
    print(
        f"{'S_vis':>6}  | "
        f"{'TK eager':>9}  {'TK comp':>9}  {'TK gain':>9}  | "
        f"{'SK eager':>9}  {'SK comp':>9}  {'SK gain':>9}  | "
        f"{'RK eager':>9}  {'RK comp':>9}  {'RK gain':>9}"
    )
    print("-" * 130)
    for r in rows:
        tk_gain = r["pf_think"] / r["pf_think_compile"]
        sk_gain = r["pf_spark"] / r["pf_spark_compile"]
        rk_gain = r["pf_rotatek"] / r["pf_rotatek_compile"]
        print(
            f"{r['S']:>6}  | "
            f"{r['pf_think']:>8.3f}  {r['pf_think_compile']:>8.3f}  {tk_gain:>8.2f}×  | "
            f"{r['pf_spark']:>8.3f}  {r['pf_spark_compile']:>8.3f}  {sk_gain:>8.2f}×  | "
            f"{r['pf_rotatek']:>8.3f}  {r['pf_rotatek_compile']:>8.3f}  {rk_gain:>8.2f}×"
        )

    print()
    print(
        f"{'S_vis':>6}  | "
        f"{'TK comp':>9}  {'SK comp':>9}  {'RK comp':>9}  {'RK c+512':>10}  | "
        f"{'best/TK':>9}"
    )
    print("-" * 130)
    for r in rows:
        best = min(
            r["pf_rotatek_compile"], r["pf_rotatek_compile_512"],
        )
        ratio = best / r["pf_think_compile"]
        print(
            f"{r['S']:>6}  | "
            f"{r['pf_think_compile']:>8.3f}  {r['pf_spark_compile']:>8.3f}  "
            f"{r['pf_rotatek_compile']:>8.3f}  {r['pf_rotatek_compile_512']:>9.3f}  | "
            f"{ratio:>8.2f}×"
        )

    # ============================================================
    # Prefill (one-time per image, per layer) — full breakdown
    # ============================================================
    print()
    print("=" * 150)
    print("PREFILL per-layer (ms) — one-time KV-compression cost")
    print("Eager / subsample / torch.compile (reduce-overhead) variants of RotateK")
    print("=" * 150)
    print(
        f"{'S_vis':>6}  | "
        f"{'ThinK':>8}  {'SparK':>8}  | "
        f"{'RK eager':>9}  {'RK K=1024':>10}  {'RK K=512':>9}  {'RK K=256':>9}  | "
        f"{'RK comp':>8}  {'RK comp+512':>11}  | "
        f"{'best/TK':>9}"
    )
    print("-" * 150)
    for r in rows:
        best = min(
            r["pf_rotatek"], r["pf_rotatek_512"], r["pf_rotatek_256"],
            r["pf_rotatek_compile"], r["pf_rotatek_compile_512"],
        )
        best_ratio = best / r["pf_think"]
        print(
            f"{r['S']:>6}  | "
            f"{r['pf_think']:>7.3f}  {r['pf_spark']:>7.3f}  | "
            f"{r['pf_rotatek']:>8.3f}  {r['pf_rotatek_1024']:>9.3f}  "
            f"{r['pf_rotatek_512']:>8.3f}  {r['pf_rotatek_256']:>8.3f}  | "
            f"{r['pf_rotatek_compile']:>7.3f}  {r['pf_rotatek_compile_512']:>10.3f}  | "
            f"{best_ratio:>8.2f}×"
        )

    # ============================================================
    # Decode breakdown
    # ============================================================
    print()
    print("=" * 130)
    print("DECODE per-layer per-step (ms) — totals across methods (per-stage in printout below)")
    print("=" * 130)
    print(f"{'S_vis':>6}  {'ThinK total':>12}  {'SparK total':>12}  {'RotateK total':>14}  "
          f"{'SK / ThinK':>11}  {'RK / ThinK':>11}  {'RK / SK':>9}")
    print("-" * 130)
    for r in rows:
        sk_v_th = r["de_spark_total"] / r["de_think_total"]
        rk_v_th = r["de_rk_total"] / r["de_think_total"]
        rk_v_sk = r["de_rk_total"] / r["de_spark_total"]
        print(
            f"{r['S']:>6}  "
            f"{r['de_think_total']:>11.3f}  {r['de_spark_total']:>11.3f}  "
            f"{r['de_rk_total']:>13.3f}  "
            f"{sk_v_th:>10.2f}×  {rk_v_th:>10.2f}×  {rk_v_sk:>8.2f}×"
        )

    print()
    print(
        f"{'S_vis':>6}  | "
        f"{'TK scat':>8}  {'TK cat':>8}  {'TK FA2':>8}  | "
        f"{'SK scat':>8}  {'SK cat':>8}  {'SK FA2':>8}  | "
        f"{'RK Q@R':>7}  {'RK δμ':>7}  {'RK cat':>7}  {'RK kern':>8}"
    )
    print("-" * 130)
    for r in rows:
        print(
            f"{r['S']:>6}  | "
            f"{r['de_think_scatter']:>7.3f}  {r['de_think_cat']:>7.3f}  {r['de_think_attn']:>7.3f}  | "
            f"{r['de_spark_scatter']:>7.3f}  {r['de_spark_cat']:>7.3f}  {r['de_spark_attn']:>7.3f}  | "
            f"{r['de_rk_qrot']:>6.3f}  {r['de_rk_bias']:>6.3f}  "
            f"{r['de_rk_cat']:>6.3f}  {r['de_rk_kernel']:>7.3f}"
        )

    # ============================================================
    # Summary speedup vs each baseline
    # ============================================================
    print()
    print("=" * 100)
    print("Summary: RotateK vs ThinK / SparK (per decode step, per layer)")
    print("=" * 100)
    print(f"{'S_vis':>6}  {'ThinK':>9}  {'SparK':>9}  {'RotateK':>9}  "
          f"{'RK vs ThinK':>12}  {'RK vs SparK':>12}")
    print("-" * 100)
    for r in rows:
        sp_th = r["de_think_total"] / r["de_rk_total"]
        sp_sk = r["de_spark_total"] / r["de_rk_total"]
        print(
            f"{r['S']:>6}  "
            f"{r['de_think_total']:>8.3f}  "
            f"{r['de_spark_total']:>8.3f}  "
            f"{r['de_rk_total']:>8.3f}  "
            f"{sp_th:>11.2f}×  {sp_sk:>11.2f}×"
        )

    # ============================================================
    # Full-sample wall-clock projection
    # ============================================================
    print()
    print("=" * 80)
    print(f"Full-sample wall-clock (×{NUM_LAYERS} layers, N_decode = 30 tokens)")
    print("=" * 80)
    print(f"{'S_vis':>6}  {'method':>10}  {'prefill':>10}  {'decode':>10}  {'total':>10}")
    print("-" * 80)
    N_DECODE = 30
    for r in rows:
        for label, pf, de in [
            ("ThinK", r["pf_think"], r["de_think_total"]),
            ("SparK", r["pf_spark"], r["de_spark_total"]),
            ("RotateK", r["pf_rotatek"], r["de_rk_total"]),
        ]:
            pf_t = pf * NUM_LAYERS
            de_t = de * NUM_LAYERS * N_DECODE
            print(
                f"{r['S']:>6}  {label:>10}  "
                f"{pf_t:>9.1f}  {de_t:>9.1f}  {pf_t + de_t:>9.1f}"
            )

    # ============================================================
    # Sweep 2: Batch size (S=8k, D_keep=32)
    # ============================================================
    print()
    print("=" * 110)
    print("SWEEP 2 — Batch size effect at S=8000, D_keep=32")
    print("=" * 110)
    B_values = [1, 2, 4, 8]
    rows_b = []
    for B in B_values:
        r = _safe_measure(8000, B=B, k_keep=32)
        rows_b.append(r)
    rows_b = [r for r in rows_b if r is not None]
    print("Eager prefill:")
    print(f"{'B':>3}  | "
          f"{'TK eager':>9}  {'SK eager':>9}  {'RK eager':>9}  | "
          f"{'RK / TK':>9}  {'SK / TK':>9}")
    print("-" * 80)
    for r in rows_b:
        rk_v_tk = r["pf_rotatek"] / r["pf_think"]
        sk_v_tk = r["pf_spark"] / r["pf_think"]
        print(
            f"{r['B']:>3}  | "
            f"{r['pf_think']:>8.3f}  {r['pf_spark']:>8.3f}  {r['pf_rotatek']:>8.3f}  | "
            f"{rk_v_tk:>8.2f}×  {sk_v_tk:>8.2f}×"
        )

    print()
    print("With torch.compile (reduce-overhead) on all methods:")
    print(f"{'B':>3}  | "
          f"{'TK comp':>9}  {'SK comp':>9}  {'RK comp':>9}  {'RK c+512':>10}  | "
          f"{'RKc+512/TKc':>13}")
    print("-" * 90)
    for r in rows_b:
        ratio = r["pf_rotatek_compile_512"] / r["pf_think_compile"]
        print(
            f"{r['B']:>3}  | "
            f"{r['pf_think_compile']:>8.3f}  {r['pf_spark_compile']:>8.3f}  "
            f"{r['pf_rotatek_compile']:>8.3f}  {r['pf_rotatek_compile_512']:>9.3f}  | "
            f"{ratio:>12.2f}×"
        )

    print()
    print(f"Decode (per step) — {'B':>2}  {'TK dec':>8}  {'SK dec':>8}  {'RK dec':>8}  {'RK vs TK':>10}  {'RK vs SK':>10}")
    print("-" * 80)
    for r in rows_b:
        sp_th = r["de_think_total"] / r["de_rk_total"]
        sp_sk = r["de_spark_total"] / r["de_rk_total"]
        print(
            f"                   {r['B']:>2}  "
            f"{r['de_think_total']:>7.3f}  {r['de_spark_total']:>7.3f}  {r['de_rk_total']:>7.3f}  "
            f"{sp_th:>9.2f}×  {sp_sk:>9.2f}×"
        )

    # ============================================================
    # Sweep 3: Channel sparsity (B=1, S=8k)
    # ============================================================
    print()
    print("=" * 110)
    print("SWEEP 3 — Channel sparsity at B=1, S=8000 (sparsity = 1 - D_keep/128)")
    print("=" * 110)
    D_keep_values = [16, 32, 48, 64, 96]
    rows_d = []
    for k in D_keep_values:
        r = _safe_measure(8000, B=1, k_keep=k)
        rows_d.append(r)
    rows_d = [r for r in rows_d if r is not None]
    print(f"{'D_keep':>7}  {'sparsity':>9}  | "
          f"{'TK pf':>8}  {'SK pf':>8}  {'RK pf':>8}  | "
          f"{'TK dec':>8}  {'SK dec':>8}  {'RK dec':>8}  {'RK vs TK':>10}  {'RK vs SK':>10}")
    print("-" * 110)
    for r in rows_d:
        sp_th = r["de_think_total"] / r["de_rk_total"]
        sp_sk = r["de_spark_total"] / r["de_rk_total"]
        sparsity = 1 - r["D_keep"] / HEAD_DIM
        print(
            f"{r['D_keep']:>7}  {sparsity:>8.3f}  | "
            f"{r['pf_think']:>7.3f}  {r['pf_spark']:>7.3f}  {r['pf_rotatek']:>7.3f}  | "
            f"{r['de_think_total']:>7.3f}  {r['de_spark_total']:>7.3f}  {r['de_rk_total']:>7.3f}  "
            f"{sp_th:>9.2f}×  {sp_sk:>9.2f}×"
        )

    # ============================================================
    # Sweep 4: B × S grid with all methods compiled + full-channel baseline
    # ============================================================
    print()
    print("=" * 130)
    print("SWEEP 4 — B × S grid at D_keep=32, ALL METHODS COMPILED + naive full-channel baseline")
    print("=" * 130)
    grid_B = [1, 2, 4, 8]
    grid_S = [4000, 16000, 64000, 128000]
    grid_rows = []
    for B in grid_B:
        for S in grid_S:
            r = _safe_measure(S, B=B, k_keep=32)
            if r is not None:
                r["config"] = f"B={B}"
            grid_rows.append(r)

    # ----- Per-method PREFILL grids (compiled where applicable) -----
    print()
    print("█" * 60)
    print("█  PREFILL  (per layer, ms) — fair: torch.compile on all methods")
    print("█" * 60)
    _print_grid(
        grid_rows,
        value_fn=lambda r: r["pf_full_total"],
        header_left="B",
        row_label_fn=lambda r: r["config"],
        col_label_fn=lambda r: r["S"],
        fmt="{:7.3f}ms",
        title="full-channel = base FA only (shared by all methods at prefill)",
    )
    _print_grid(
        grid_rows,
        value_fn=lambda r: r["pf_think_total"],
        header_left="B",
        row_label_fn=lambda r: r["config"],
        col_label_fn=lambda r: r["S"],
        fmt="{:7.3f}ms",
        title="ThinK = base FA + scoring/topk/gather (compiled)",
    )
    _print_grid(
        grid_rows,
        value_fn=lambda r: r["pf_spark_total"],
        header_left="B",
        row_label_fn=lambda r: r["config"],
        col_label_fn=lambda r: r["S"],
        fmt="{:7.3f}ms",
        title="SparK = base FA + per-token topk (compiled)",
    )
    _print_grid(
        grid_rows,
        value_fn=lambda r: r["pf_rotatek_total"],
        header_left="B",
        row_label_fn=lambda r: r["config"],
        col_label_fn=lambda r: r["S"],
        fmt="{:7.3f}ms",
        title="RotateK = base FA + PCA/projection (compiled, K_sample=512)",
    )

    # ----- Per-method DECODE grids (per layer per step, ms) -----
    print()
    print("█" * 60)
    print("█  DECODE  (per layer per step, ms)")
    print("█" * 60)
    _print_grid(
        grid_rows,
        value_fn=lambda r: r["de_full_total"],
        header_left="B",
        row_label_fn=lambda r: r["config"],
        col_label_fn=lambda r: r["S"],
        fmt="{:7.3f}ms",
        title="full-channel (no compression, dense Triton FA)",
    )
    _print_grid(
        grid_rows,
        value_fn=lambda r: r["de_think_total"],
        header_left="B",
        row_label_fn=lambda r: r["config"],
        col_label_fn=lambda r: r["S"],
        fmt="{:7.3f}ms",
        title="ThinK (scatter + cat + dense Triton FA)",
    )
    _print_grid(
        grid_rows,
        value_fn=lambda r: r["de_spark_total"],
        header_left="B",
        row_label_fn=lambda r: r["config"],
        col_label_fn=lambda r: r["S"],
        fmt="{:7.3f}ms",
        title="SparK (mean-fill scatter + cat + dense Triton FA)",
    )
    _print_grid(
        grid_rows,
        value_fn=lambda r: r["de_rk_total"],
        header_left="B",
        row_label_fn=lambda r: r["config"],
        col_label_fn=lambda r: r["S"],
        fmt="{:7.3f}ms",
        title="RotateK (Q@R + δμ + cat + sparse split-K)",
    )

    # ----- Per-image WALL-CLOCK (32 layers × (prefill + 128 decode tokens)) -----
    N_DECODE_FIXED = 128
    print()
    print("█" * 80)
    print(f"█  TOTAL WALL-CLOCK per image (×{NUM_LAYERS} layers, N_decode = {N_DECODE_FIXED})")
    print("█" * 80)

    def _total(pf_ms, de_ms):
        return NUM_LAYERS * (pf_ms + N_DECODE_FIXED * de_ms)

    _print_grid(
        grid_rows,
        value_fn=lambda r: _total(r["pf_full_total"], r["de_full_total"]),
        header_left="B",
        row_label_fn=lambda r: r["config"],
        col_label_fn=lambda r: r["S"],
        fmt="{:7.1f}ms",
        title="full-channel (no compression)",
    )
    _print_grid(
        grid_rows,
        value_fn=lambda r: _total(r["pf_think_total"], r["de_think_total"]),
        header_left="B",
        row_label_fn=lambda r: r["config"],
        col_label_fn=lambda r: r["S"],
        fmt="{:7.1f}ms",
        title="ThinK",
    )
    _print_grid(
        grid_rows,
        value_fn=lambda r: _total(r["pf_spark_total"], r["de_spark_total"]),
        header_left="B",
        row_label_fn=lambda r: r["config"],
        col_label_fn=lambda r: r["S"],
        fmt="{:7.1f}ms",
        title="SparK",
    )
    _print_grid(
        grid_rows,
        value_fn=lambda r: _total(r["pf_rotatek_total"], r["de_rk_total"]),
        header_left="B",
        row_label_fn=lambda r: r["config"],
        col_label_fn=lambda r: r["S"],
        fmt="{:7.1f}ms",
        title="RotateK",
    )

    # ----- Speedup vs full-channel baseline (>1 = compression faster) -----
    print()
    print("█" * 80)
    print("█  TOTAL SPEEDUP — full-channel time / method time   (>1 = method faster)")
    print("█" * 80)
    _print_grid(
        grid_rows,
        value_fn=lambda r: _total(r["pf_full_total"], r["de_full_total"]) / _total(r["pf_think_total"], r["de_think_total"]),
        header_left="B",
        row_label_fn=lambda r: r["config"],
        col_label_fn=lambda r: r["S"],
        fmt="{:7.2f}×",
        title="ThinK vs full-channel",
    )
    _print_grid(
        grid_rows,
        value_fn=lambda r: _total(r["pf_full_total"], r["de_full_total"]) / _total(r["pf_spark_total"], r["de_spark_total"]),
        header_left="B",
        row_label_fn=lambda r: r["config"],
        col_label_fn=lambda r: r["S"],
        fmt="{:7.2f}×",
        title="SparK vs full-channel",
    )
    _print_grid(
        grid_rows,
        value_fn=lambda r: _total(r["pf_full_total"], r["de_full_total"]) / _total(r["pf_rotatek_total"], r["de_rk_total"]),
        header_left="B",
        row_label_fn=lambda r: r["config"],
        col_label_fn=lambda r: r["S"],
        fmt="{:7.2f}×",
        title="RotateK vs full-channel",
    )


if __name__ == "__main__":
    main()
