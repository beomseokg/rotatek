"""ThinK channel-pruning math.

Per-head channel scoring + compact gather, with an optional
torch.compile path gated by the ``THINK_COMPILE`` env var. The
per-layer orchestration lives in
``lmms_eval.models.model_utils.kv_pruning_utils``.
"""
from __future__ import annotations

import os
from typing import Tuple

import torch


def think_score_and_compact(
    keys_vision: torch.Tensor,
    queries_recent: torch.Tensor,
    num_kv_heads: int,
    prune_count: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ThinK score + topk + gather + bool-mask chain.

    Pure-tensor chain without ``.item()`` or Python branches: takes vision
    K ``[B, H_kv, S_v, D]`` and recent Q ``[B, H_q, Q, D]`` (already sliced
    to the query window) and returns:

    * ``K_compact`` ``[B, H_kv, S_v, D_keep]`` — kept-channel slice of K
    * ``keep_mask`` ``[B, H_kv, D]`` bool — True at kept channels (per-head)

    GQA is handled internally by collapsing ``H_q -> H_kv`` via group-mean
    on Q² before the channel score.
    """
    B = keys_vision.shape[0]
    head_dim = keys_vision.shape[-1]
    keep_count = head_dim - prune_count

    # ---- Channel scoring (per-head). GQA: collapse H_q -> H_kv first. ----
    queries_norm = torch.pow(queries_recent, 2).mean(dim=2)         # [B, H_q, D]
    H_q = queries_norm.shape[1]
    queries_norm = queries_norm.view(
        B, num_kv_heads, H_q // num_kv_heads, head_dim,
    ).mean(dim=2)                                                    # [B, H_kv, D]
    keys_norm = torch.pow(keys_vision, 2).mean(dim=2)               # [B, H_kv, D]
    channel_scores = queries_norm * keys_norm                        # [B, H_kv, D]

    # ---- Top-D_keep selection -> bool mask + sorted-ascending indices. ----
    pruned_idx = torch.topk(
        channel_scores, prune_count, dim=-1, largest=False, sorted=False,
    ).indices                                                        # [B, H_kv, D_pruned]
    pruned_mask = torch.zeros_like(channel_scores, dtype=torch.bool)
    pruned_mask.scatter_(-1, pruned_idx, True)
    keep_mask = ~pruned_mask                                         # [B, H_kv, D]

    keep_idx = torch.topk(
        keep_mask.int(), keep_count, dim=-1, largest=True, sorted=True,
    ).indices.sort(dim=-1).values                                    # [B, H_kv, D_keep]

    # ---- Gather kept channels. ----
    keep_idx_exp = keep_idx.unsqueeze(2).expand(-1, -1, keys_vision.shape[2], -1)
    K_compact = torch.gather(keys_vision, dim=-1, index=keep_idx_exp)
    return K_compact, keep_mask


# THINK_COMPILE=default wraps the scoring in torch.compile (the latency
# sweeps set it), as SPARK_COMPILE / ROTATEK_COMPILE do for the others.
_THINK_COMPILE_MODE = os.environ.get("THINK_COMPILE", "").strip().lower()
if _THINK_COMPILE_MODE in ("1", "default", "true", "reduce", "reduce-overhead"):
    # Default inductor mode only — `reduce-overhead` would alias the
    # returned `keep_mask` / `K_compact` with CUDA-graph buffers that
    # get overwritten on the next call.
    think_score_and_compact = torch.compile(think_score_and_compact, dynamic=True)
