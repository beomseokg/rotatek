# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
#
"""SparK channel-pruning math (per-token channel selection).

Scores each visual token's channels by ||Q||_2 · K² and keeps the top
D_keep per token; the pruned channels are filled with their per-token
mean at decode (see ``kv_pruning_utils.ChannelPruner.update_spark``).
"""
import os

import torch


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


# SPARK_COMPILE=default wraps the prune in torch.compile (the latency sweeps
# set it). Off by default.
_SPARK_COMPILE_MODE = os.environ.get("SPARK_COMPILE", "").strip().lower()
if _SPARK_COMPILE_MODE in ("1", "default", "true", "reduce", "reduce-overhead"):
    # Use default inductor mode only — `reduce-overhead` aliases the
    # function's outputs with CUDA-graph buffers that get overwritten
    # on the next call, silently corrupting `keep_mask` / `pruned_mean`
    # appended to the KV cache.
    per_token_channel_prune = torch.compile(per_token_channel_prune, dynamic=True)
