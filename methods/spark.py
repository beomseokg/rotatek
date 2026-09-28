# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
#

import torch


def generate_key_mean_fill(keys, mask):
    """
    Mean-based filling for key cache using tensor operations.
    Strategy: Use mean of pruned (masked-out) dimensions to fill missing positions.

    Args:
        keys: (bsz, num_heads, seq_len, head_dim)
        mask: boolean mask where True = kept, False = pruned
    """
    bsz, num_heads, seq_len, head_dim = keys.shape
    recovered_keys = keys.clone()
    missing_mask = ~mask
    pruned_mask = ~mask

    token_pruned_counts = pruned_mask.sum(dim=3, keepdim=True)  # (bsz, num_heads, seq_len, 1)
    token_pruned_means = (keys * pruned_mask).sum(dim=3, keepdim=True) / (token_pruned_counts + 1e-8)  # (bsz, num_heads, seq_len, 1)

    token_fill = token_pruned_means.expand(-1, -1, -1, head_dim)  # (bsz, num_heads, seq_len, head_dim)

    return torch.where(missing_mask, token_fill, recovered_keys).to(keys.dtype)

def generate_key_no_fill(keys, mask):
    """
    No filling for key cache — simply zero out pruned dimensions.

    Args:
        keys: (bsz, num_heads, seq_len, head_dim)
        mask: boolean mask where True = kept, False = pruned
    """
    return keys * mask

def compute_spark_channel_scores(queries, keys):
    """Compute per-head channel importance scores using SparK's scoring method.

    SparK uses L2 norm of Q across sequence × element-wise K^2 averaged across tokens.
    Returns [B, H, D] scores compatible with key_pruner_query_driven.

    Args:
        queries: (bsz, num_heads, seq_len, head_dim)
        keys:    (bsz, num_heads, seq_len, head_dim) — vision tokens only
    """
    q_norm = torch.norm(queries, dim=-2, p=2)       # [B, H, D]
    k_norm = torch.pow(keys, 2).mean(dim=2)          # [B, H, D]
    return q_norm * k_norm                            # [B, H, D]


def per_token_channel_prune(queries, keys, ratio):
    """Per-token channel pruning with SparK scoring (compact-storage variant).

    ThinK-parity storage shape: keeps a per-token *bool* keep mask alongside
    the compact-D kept channels and a per-token pruned-channel mean. Decode
    pre-fills a full-D buffer with `pruned_mean` and writes compact K into
    the True positions (boolean indexing — same primitive as ThinK).

    Args:
        queries: (bsz, num_heads, seq_len, head_dim)
        keys:    (bsz, num_heads, seq_len, head_dim) — vision tokens only
        ratio:   fraction of channels to prune (e.g. 0.75 = prune 75%)

    Returns:
        K_compact:   (bsz, num_heads, seq_len, D_keep) — gathered kept channels
        keep_mask:   (bsz, num_heads, seq_len, head_dim) bool — True = kept
        pruned_mean: (bsz, num_heads, seq_len, 1) — mean of pruned channels per
                     token (used as the fill value at non-kept positions).
    """
    bsz, num_heads, seq_len, head_dim = keys.shape
    prune_count = int(head_dim * ratio)
    keep_count = head_dim - prune_count

    # Per-token scoring: ||Q||_2 * K^2  (SparK reference).
    q_norm = torch.norm(queries, dim=-2, p=2).unsqueeze(-2)             # [B, H, 1, D]
    scores = torch.pow(keys, 2) * q_norm                                 # [B, H, S, D]

    sorted_indices = torch.argsort(scores, dim=-1, descending=True)      # [B, H, S, D]
    keep_indices = sorted_indices[..., :keep_count]                      # [B, H, S, D_keep]
    pruned_indices = sorted_indices[..., keep_count:]                    # [B, H, S, D_pruned]

    # Sort keep_indices ascending so the gathered K_compact lays out
    # channels in original-channel-index order — matches what bool-indexing
    # assignment iterates over at decode time.
    keep_indices_sorted = keep_indices.sort(dim=-1).values

    K_compact = torch.gather(keys, dim=-1, index=keep_indices_sorted)    # [B, H, S, D_keep]

    keep_mask = torch.zeros_like(keys, dtype=torch.bool)
    keep_mask.scatter_(-1, keep_indices_sorted, True)                    # [B, H, S, D] bool

    # Per-token mean over the PRUNED channels (used to fill non-kept slots).
    pruned_vals = torch.gather(keys, dim=-1, index=pruned_indices)       # [B, H, S, D_pruned]
    pruned_mean = pruned_vals.mean(dim=-1, keepdim=True)                 # [B, H, S, 1]

    return K_compact, keep_mask, pruned_mean


# Optional torch.compile path for SparK's per-token prune. Mirrors
# RotateK's `ROTATEK_COMPILE` env-var pattern. `reduce-overhead` mode
# captures the kernel chain as a CUDA graph (big win at B=1) but is
# sensitive to shape changes; default OFF for safety.
import os as _os_for_spark
_SPARK_COMPILE_MODE = _os_for_spark.environ.get("SPARK_COMPILE", "").strip().lower()
if _SPARK_COMPILE_MODE in ("1", "default", "true", "reduce", "reduce-overhead"):
    # Use default inductor mode only — `reduce-overhead` aliases the
    # function's outputs with CUDA-graph buffers that get overwritten
    # on the next call, silently corrupting `keep_mask` / `pruned_mean`
    # appended to the cluster cache.
    per_token_channel_prune = torch.compile(per_token_channel_prune, dynamic=True)


def dynamic_score_selection_norm(queries, keys, key_channel_compression_ratio=0, recovery=True):
    bsz, num_heads, seq_len, head_dim = keys.shape

    q_norm = torch.norm(queries, dim=-2, p=2).unsqueeze(-2)
    k_norm = torch.pow(keys, 2)
    sorted_indices = torch.argsort(k_norm * q_norm, dim=-1, descending=True)

    mask = torch.ones_like(keys, dtype=torch.bool)
    mask.scatter_(-1, sorted_indices[..., -int(key_channel_compression_ratio * head_dim):], False)

    if recovery:
        return generate_key_mean_fill(keys, mask)
    else:
        return generate_key_no_fill(keys, mask)
