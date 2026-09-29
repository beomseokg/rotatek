"""Prefill-time Key channel pruning for the visual span: ThinK, SparK, RotateK.

Each attention layer owns a ``ChannelPruner`` (created by ``init_channel_pruner``
at every prefill). The adapter hands it the layer's full Keys; the pruner
slices out the visual span ``[prompt_seqlen : total - query_seqlen]``, prunes
its channels, and returns

    (pruned visual Keys, prompt Keys, text Keys, keep mask, Values)

while leaving the per-method decode state on itself (``current_*`` attributes)
for the adapter's decode branch.
"""
import os

import torch

# ROTATEK_SOLVER: "power_iter" (default) = Cholesky-QR subspace iteration on GPU;
# "eigh" = exact eigendecomposition (the solver ablation).
_ROTATEK_SOLVER = os.environ.get("ROTATEK_SOLVER", "power_iter").strip().lower()
# ROTATEK_QUERY_AWARE=0 ablates the query weighting of the covariance.
_ROTATEK_QUERY_AWARE = bool(int(os.environ.get("ROTATEK_QUERY_AWARE", "1")))

QUERY_WINDOW = 32  # recent text queries used to score channels (all methods)


def _split_spans(pruner, key_states):
    """(prompt Keys, visual Keys, text Keys) of a [B, H_kv, S, D] tensor."""
    total_seq_len = key_states.shape[-2]
    prompt_seqlen = max(0, min(pruner.prompt_seqlen, total_seq_len))
    query_seqlen = max(0, min(pruner.query_seqlen, max(total_seq_len - prompt_seqlen, 0)))
    vision_end = total_seq_len - query_seqlen if query_seqlen > 0 else total_seq_len
    kv_prompt = key_states[:, :, :prompt_seqlen, :]
    keys_vision = key_states[:, :, prompt_seqlen:vision_end, :]
    kv_text = key_states[:, :, -query_seqlen:, :] if query_seqlen > 0 else key_states[:, :, :0, :]
    return kv_prompt, keys_vision, kv_text


class ChannelPruner:
    def __init__(self, ratio, prompt_seqlen, query_seqlen):
        self.ratio = ratio                  # fraction of channels to prune
        self.prompt_seqlen = prompt_seqlen  # tokens before the visual span
        self.query_seqlen = query_seqlen    # tokens after it

    def update_think(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        """ThinK: one channel subset per head, scored by mean(Q²)·mean(K²).

        Stores ``current_think_mask`` [B, H_kv, D] (True = kept); decode
        zero-fills the pruned channels, as in the ThinK reference.
        """
        from rotatek.baselines.think import think_score_and_compact

        assert key_states.shape[-2] == query_states.shape[-2]
        kv_prompt, keys_vision, kv_text = _split_spans(self, key_states)
        num_kv_heads, head_dim = key_states.shape[1], key_states.shape[-1]
        prune_count = min(head_dim, max(0, int(head_dim * self.ratio)))
        query_window = min(QUERY_WINDOW, query_states.shape[2])
        K_compact, keep_mask = think_score_and_compact(
            keys_vision, query_states[..., -query_window:, :], num_kv_heads, prune_count,
        )
        self.current_think_mask = keep_mask
        return K_compact, kv_prompt, kv_text, keep_mask, value_states

    def update_spark(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        """Per-token channel pruning ("spark").

        Stores compact K [B, H, S, D_keep] + per-token keep mask
        [B, H, S, D] + per-token pruned_mean [B, H, S, 1], the mean of the
        token's pruned Key values. Decode pre-fills a [B, H, S, D] buffer with
        `pruned_mean` and scatters compact K into the kept slots.
        """
        from rotatek.baselines.spark import per_token_channel_prune

        bsz, num_kv_heads, _, head_dim = key_states.shape
        kv_prompt, keys_vision, kv_text = _split_spans(self, key_states)

        # group queries from H_q to H_kv for scoring
        query_window = min(QUERY_WINDOW, query_states.shape[2])
        queries_grouped = query_states[..., -query_window:, :].view(
            bsz, num_kv_heads, num_key_value_groups, query_window, head_dim,
        ).mean(2)  # [B, H_kv, query_window, D]

        K_compact, keep_mask, pruned_mean = per_token_channel_prune(
            queries_grouped, keys_vision, ratio=self.ratio,
        )
        self.current_spark_mask = keep_mask
        self.current_spark_pruned_mean = pruned_mean

        # per-head mask slot of the return tuple; not read on the SparK decode path
        dummy_mask = torch.ones(bsz, num_kv_heads, head_dim, dtype=torch.bool, device=key_states.device)
        return K_compact, kv_prompt, kv_text, dummy_mask, value_states

    def update_rotatek(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        """RotateK: rotate the visual Keys into the top-D_keep eigenbasis of their
        (query-weighted) covariance and keep only those D_keep channels.

        Stored per (batch, kv-head):
            K_vision @ R_partial      [S_v, D_keep]  (returned as the pruned Keys)
            R_partial                 [D, D_keep]    -> current_rotatek_R_partial
            δμ = μ (I - R Rᵀ)         [D]            -> current_rotatek_delta_mu

        K is rotated uncentered, so Q R (K R)ᵀ misses the constant Q·μ(I - P)
        that the rank-D_keep approximation of the centered Keys would keep; the
        decode kernel adds q·δμ back to every visual logit.
        """
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_kv_heads, _, head_dim = key_states.shape
        kv_prompt, keys_vision, kv_text = _split_spans(self, key_states)
        keep_count = head_dim - min(head_dim, max(0, int(head_dim * self.ratio)))

        # Covariance of the centered Keys. Center in bf16 and let the matmul
        # accumulate in fp32 rather than materializing an fp32 copy of K.
        mean_fp32 = keys_vision.float().mean(dim=-2, keepdim=True)          # [B, H_kv, 1, D]
        keys_centered = keys_vision - mean_fp32.to(keys_vision.dtype)
        cov = torch.einsum("bhsd,bhse->bhde", keys_centered, keys_centered).float()
        cov = 0.5 * (cov + cov.transpose(-1, -2))

        if _ROTATEK_QUERY_AWARE:
            # Weighting K's channels by ||q|| commutes through the covariance:
            #   sum_s (K[s,d] q[d]) (K[s,e] q[e]) = q[d] q[e] · cov[d,e]
            # so it costs O(D²) on the [D, D] matrix instead of a pass over K.
            query_window = min(QUERY_WINDOW, query_states.shape[2])
            queries_recent = query_states[..., -query_window:, :]           # [B, H_q, W, D]
            num_q_heads = queries_recent.shape[1]
            if num_q_heads != num_kv_heads:  # GQA: group-mean H_q -> H_kv
                queries_recent = queries_recent.view(
                    bsz, num_kv_heads, num_key_value_groups, query_window, head_dim,
                ).mean(dim=2)
            q_norm = queries_recent.float().norm(dim=-2, p=2)               # [B, H_kv, D]
            # Floor near-zero channels at a fraction of the median so the
            # weighted covariance keeps full rank (else Cholesky can fail).
            q_floor = 1e-3 * q_norm.median(dim=-1, keepdim=True).values
            q_norm = q_norm.clamp(min=q_floor)
            cov = cov * (q_norm.unsqueeze(-1) * q_norm.unsqueeze(-2))
            cov = 0.5 * (cov + cov.transpose(-1, -2))
            # Ridge scaled to the diagonal: negligible against the leading
            # eigenvalues but keeps the matrix strictly PSD.
            ridge = 1e-4 * cov.diagonal(dim1=-2, dim2=-1).abs().mean(
                dim=-1, keepdim=True
            ).clamp(min=1e-6).unsqueeze(-1)
            cov = cov + ridge * torch.eye(head_dim, device=cov.device, dtype=cov.dtype)

        if _ROTATEK_SOLVER == "power_iter":
            from rotatek.rotation import power_iteration_gpu
            R_partial = power_iteration_gpu(cov, k=keep_count, num_iters=5, seed=0).to(keys_vision.dtype)
        else:
            # Exact eigh. For 128x128 matrices LAPACK on CPU beats cuSOLVER,
            # whose launch/sync overhead dominates at this size.
            _, eigvecs = torch.linalg.eigh(cov.cpu())
            eigvecs = eigvecs.to(keys_vision.device, non_blocking=True)
            R_partial = eigvecs.flip(dims=[-1])[..., :keep_count].to(keys_vision.dtype)

        kept_kv_states = torch.matmul(keys_vision, R_partial)
        R_fp32 = R_partial.float()
        mu_proj = torch.matmul(torch.matmul(mean_fp32, R_fp32), R_fp32.transpose(-2, -1))
        self.current_rotatek_R_partial = R_partial
        self.current_rotatek_delta_mu = (mean_fp32 - mu_proj).squeeze(-2).to(keys_vision.dtype)

        keep_mask = torch.zeros(bsz, num_kv_heads, head_dim, dtype=torch.bool, device=keys_vision.device)
        keep_mask[..., :keep_count] = True
        return kept_kv_states, kv_prompt, kv_text, keep_mask, value_states


def init_channel_pruner(self):
    """Attach a fresh pruner to attention module `self` for this prefill."""
    self.channel_pruner = ChannelPruner(
        ratio=self.config.channel_ratio,
        prompt_seqlen=self.config.prompt_seqlen,
        query_seqlen=self.config.query_seqlen,
    )
