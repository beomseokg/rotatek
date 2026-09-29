

import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import os
import sys
import csv

layer_idx = 0

# --- RotateK profiling: stage-level timing for the first few prefills. ----
# Off by default; enable via env var (number of *layer calls* to profile, e.g.
# ROTATEK_PROFILE=96 = 3 prefills of a 32-layer model). When disabled the
# instrumentation short-circuits with ~zero overhead.
_ROTATEK_PROFILE_LIMIT = int(os.environ.get("ROTATEK_PROFILE", "0"))
_rotatek_profile_count = [0]  # mutable holder so nested closures can mutate it

# --- RotateK ratio profiling: compares RotateK PCA time against the total
# prefill attention-forward wall-clock for the same layer call. The adapter
# (internvl2_5_visionzip) records events at the entry/exit of the attention
# forward when this is on and reads `current_rotatek_elapsed_ms` from the
# cluster. Enable via `ROTATEK_RATIO=N` env var (number of prefill layer
# calls to measure).
_ROTATEK_RATIO_LIMIT = int(os.environ.get("ROTATEK_RATIO", "0"))
_rotatek_ratio_count = [0]

# Online solver selection:
#   "power_iter" (default): GPU power iteration + QR, 5 iters. Fastest;
#                           ~0.3pp accuracy drop vs exact eigh. Scales well
#                           with batch (B=32 nearly free).
#   "randomized":           Halko-Martinsson-Tropp top-k sketch. Middle.
#   "eigh":                 Exact CPU eigh via LAPACK. Slowest (A100 CPU can
#                           be fast enough on small matrices) but bit-exact.
_ROTATEK_SOLVER = os.environ.get("ROTATEK_SOLVER", "power_iter").strip().lower()

# Storage mode for RotateK's projected keys:
#   "truncated" (default):  Store K_pruned = K @ R_partial in rotated basis +
#                           cache R_partial. At decode, rotate Q to D_keep dim
#                           and run sparse attention (VisionK kernel). Yields
#                           D_keep/D memory savings AND decode compute savings.
#   "full":                 Store K_approx = K @ R @ R^T + μ in original basis.
#                           Full-D storage, no decode-time Q rotation, plug-in
#                           replacement for ThinK/VisionK in standard FA2 path.
#                           No memory savings, no decode compute savings.
#                           Useful for accuracy-equivalence sanity checks.
_ROTATEK_STORAGE = os.environ.get("ROTATEK_STORAGE", "truncated").strip().lower()

# Subsample size for online PCA covariance estimation. When the vision
# sequence is longer than this, only K_sample uniformly random tokens are
# used to estimate the covariance — eigenvectors are unchanged in
# expectation by Halko-Martinsson-Tropp 2011. Storage and δμ-bias still
# use the full vision sequence, so only the *basis* is approximated.
#   "full" (default = old behaviour): use all S vision tokens
#   integer (e.g. "512")            : subsample K_sample tokens for cov only
def _parse_k_sample(val: str):
    s = val.strip().lower()
    if s in {"", "full", "0", "none"}:
        return None
    try:
        return max(1, int(s))
    except ValueError:
        return None

_ROTATEK_PCA_K_SAMPLE = _parse_k_sample(os.environ.get("ROTATEK_PCA_K_SAMPLE", "full"))

# NOTE: per-model defaults live in each model_utils package (e.g.
# `model_utils.qwen.qwen2_5vl_visionzip.DEFAULT_CALIBRATION_RESULT_ROOT` and
# `model_utils.internvl.internvl2_5_visionzip.DEFAULT_CALIBRATION_RESULT_ROOT`).
# The model adapter is responsible for stamping `config.result_root` before
# `init_visionzip` runs; this module no longer hardcodes a fallback so a
# missing setting fails loudly instead of silently writing to the wrong dir.
CALIBRATION_CHANNEL_IMPORTANCE_TARGET_SAMPLES = 1500
CALIBRATION_CHANNEL_IMPORTANCE_SAVE_INTERVAL = 100
calibration_channel_importance_accumulator = None
calibration_channel_importance_seen_sample_counts = None
calibration_channel_importance_last_saved_steps = None

CALIBRATION_MODALITY_SCORE_TARGET_SAMPLES = 1500
CALIBRATION_MODALITY_SCORE_SAVE_INTERVAL = 100
calibration_modality_score_accumulator = None
calibration_modality_score_seen_sample_counts = None
calibration_modality_score_last_saved_steps = None

SUPPLEMENTARY_MATRIX_SPARSITIES = (0.250, 0.375, 0.500, 0.625, 0.75, 0.875)

# RotateK offline calibration: accumulate K^T K over calibration samples per
# (layer, head); save R (or equivalently the covariance) so runtime can skip
# the expensive per-image eigh.
CALIBRATION_ROTATION_MATRIX_TARGET_SAMPLES = 1500
CALIBRATION_ROTATION_MATRIX_SAVE_INTERVAL = 100
calibration_rotation_matrix_accumulator = None  # [num_layers, num_heads, D, D] fp32 CPU
calibration_rotation_matrix_seen_sample_counts = None
calibration_rotation_matrix_last_saved_steps = None
SUPPLEMENTARY_MATRIX_TARGET_SAMPLES = 1500
SUPPLEMENTARY_MATRIX_SAVE_INTERVAL = 100
SUPPLEMENTARY_MATRIX_RIDGE_LAMBDA = 1e-4
# "k_mse"       → original objective: min ||K_pruned - X β||²   (channel-wise K reconstruction)
# "qk_weighted" → min tr(G_Q (K_pruned - X β)^T (K_pruned - X β))
#                 = min ||ΔA_{prune-induced}||²_F   (directly preserves Q·K^T)
SUPPLEMENTARY_MATRIX_OBJECTIVE = "qk_weighted"
supplementary_matrix_gram_accumulator = {}
supplementary_matrix_cross_accumulator = {}
supplementary_matrix_keep_indices = {}
supplementary_matrix_pruned_indices = {}
supplementary_matrix_target_sq_accumulator = {}
supplementary_matrix_row_count_accumulator = {}
supplementary_matrix_q_gram_accumulator = {}         # per-layer (H_kv, D, D) Σ Q^T Q
supplementary_matrix_q_position_count_accumulator = {}  # per-layer scalar: total Q positions summed into q_gram
supplementary_matrix_seen_sample_counts = None
supplementary_matrix_last_saved_steps = None
supplementary_matrix_file_cache = {}

ATTENTION_SHIFT_SAVE_INTERVAL = 100
attention_shift_before_accumulator = {}
attention_shift_after_pruning_accumulator = {}
attention_shift_after_reconstruction_accumulator = {}
attention_shift_seen_sample_counts = None

# Cross-modal attention pattern divergence (KL between full and pruned
# text-to-vision attention distributions). Two normalization conventions:
#   - "_v"    : vision-only renormalized (pure shape / ranking shift)
#   - "_full" : full-sequence normalized, KL evaluated on vision portion
#               (preserves both mass and shape info)
attention_kl_v_pruning_accumulator = {}
attention_kl_v_reconstruction_accumulator = {}
attention_kl_full_pruning_accumulator = {}
attention_kl_full_reconstruction_accumulator = {}
attention_kl_seen_sample_counts = None
attention_kl_last_saved_steps = None
ATTENTION_KL_SAVE_INTERVAL = 100
attention_shift_last_saved_steps = None

# Flip to True to print per-layer key-channel effective-rank metrics during
# prefill and block with input() after the final layer. Debug aid; leave
# False for normal runs.
DEBUG_PRINT_KEY_EFFECTIVE_RANK = False

# Flip to True to dump per-layer singular-value decay curves (one CSV per
# layer). Each row is (head_idx, rank, sv_k_only, sv_qk_contrib). Saved only
# on the first prefill call per layer in the run to avoid I/O on every sample.
# Debug aid. Set `DEBUG_SV_CURVES_DIR` to an absolute path to override the
# default location (`{result_root}/singular_values/`).
DEBUG_SAVE_SV_CURVES = False
DEBUG_SV_CURVES_DIR = True
_sv_curves_saved_layers: set = set()


def _get_current_cluster():
    return globals().get("current_kv_cluster")


def _get_calibration_result_root():
    current_cluster = _get_current_cluster()
    result_root = getattr(current_cluster, "result_root", None)
    if result_root is None:
        raise ValueError(
            "kv_cluster.result_root is not set. The model adapter "
            "(e.g. simple/qwen2_5_vl_visionzip.py or simple/internvl2_5_visionzip.py) "
            "must inject config.result_root before init_visionzip runs."
        )
    return str(result_root)


def _get_calibration_dominant_ratio():
    current_cluster = _get_current_cluster()
    dominant_ratio = getattr(current_cluster, "dominant_ratio", None)
    if dominant_ratio is None:
        raise ValueError("dominant_ratio must be set in config before using calibration paths")
    try:
        return round(float(dominant_ratio), 2)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid dominant_ratio: {dominant_ratio}") from exc


def _get_calibration_root_dir():
    return os.path.join(
        _get_calibration_result_root(),
        f"mmstar_calibration_dominant_ratio_{_get_calibration_dominant_ratio():.2f}",
    )










def _get_calibration_rotation_matrix_dir():
    return os.path.join(_get_calibration_root_dir(), "rotation_matrix")
















def _maybe_accumulate_rotation_matrix_calibration(cluster, key_states):
    """Method-agnostic entry point used by the adapter during prefill.

    Runs only when calibration_mode=="collect" and the task list includes
    "rotation_matrix". Slices vision keys out of `key_states` using the
    cluster's prompt/query seqlens and delegates to the main accumulator.
    """
    if cluster is None or key_states is None:
        return
    if not _is_collect_calibration_mode_enabled():
        return
    if not _calibration_task_enabled("rotation_matrix"):
        return

    bsz, num_kv_heads, total_seq_len, head_dim = key_states.shape
    prompt_seqlen = max(0, min(getattr(cluster, "prompt_seqlen", 0), total_seq_len))
    query_seqlen = max(
        0,
        min(getattr(cluster, "query_seqlen", 0), max(total_seq_len - prompt_seqlen, 0)),
    )
    vision_end = total_seq_len - query_seqlen if query_seqlen > 0 else total_seq_len
    if vision_end <= prompt_seqlen:
        return
    keys_vision = key_states[:, :, prompt_seqlen:vision_end, :]
    _accumulate_and_maybe_save_rotation_matrix_calibration(keys_vision)


def _accumulate_and_maybe_save_rotation_matrix_calibration(keys_vision):
    """Accumulate (K-μ)^T (K-μ) over calibration samples (per layer, per
    head) and periodically save the running covariance to disk. At load
    time we run eigh on the saved covariance to get R — doing it offline
    avoids the per-prefill eigh cost entirely at inference.

    Per-sample centering (subtract mean-over-tokens before covariance)
    removes the attention-invariant uniform-shift direction from the
    principal subspace, so the top-K basis captures *variance* of K
    rather than wasting a rank slot on the (often large) mean direction.

    keys_vision: [B, H_kv, S_v, D]
    """
    global layer_idx
    global calibration_rotation_matrix_accumulator
    global calibration_rotation_matrix_seen_sample_counts
    global calibration_rotation_matrix_last_saved_steps

    if keys_vision is None or keys_vision.dim() != 4 or keys_vision.shape[-2] == 0:
        return

    # Per-sample, per-head mean over tokens (S_v axis).
    keys_fp32 = keys_vision.float()
    mean_per_head = keys_fp32.mean(dim=-2, keepdim=True)  # [B, H_kv, 1, D]
    keys_centered = keys_fp32 - mean_per_head

    # (K-μ)^T (K-μ) per (batch, head). fp32 for numerical stability.
    cov_batch = torch.einsum(
        "bhsd,bhse->bhde", keys_centered, keys_centered,
    ).cpu()  # [B, H_kv, D, D]

    num_layers = getattr(globals().get("current_kv_cluster"), "num_layers", None)
    if num_layers is None:
        raise RuntimeError(
            "current_kv_cluster.num_layers must be set before using kv pruning utilities"
        )
    batch_size, num_heads, head_dim, _ = cov_batch.shape
    expected_shape = (num_layers, num_heads, head_dim, head_dim)
    if (
        calibration_rotation_matrix_accumulator is None
        or tuple(calibration_rotation_matrix_accumulator.shape) != expected_shape
    ):
        calibration_rotation_matrix_accumulator = torch.zeros(
            expected_shape, dtype=torch.float32, device="cpu",
        )
        calibration_rotation_matrix_seen_sample_counts = torch.zeros(
            num_layers, dtype=torch.long, device="cpu",
        )
        calibration_rotation_matrix_last_saved_steps = torch.full(
            (num_layers,), -1, dtype=torch.long, device="cpu",
        )

    layer_slot = layer_idx % num_layers
    calibration_rotation_matrix_accumulator[layer_slot] += cov_batch.sum(dim=0)
    calibration_rotation_matrix_seen_sample_counts[layer_slot] += batch_size

    seen_samples = int(calibration_rotation_matrix_seen_sample_counts[layer_slot].item())
    current_save_step = (
        min(seen_samples, CALIBRATION_ROTATION_MATRIX_TARGET_SAMPLES)
        // CALIBRATION_ROTATION_MATRIX_SAVE_INTERVAL
    )
    last_saved_step = int(calibration_rotation_matrix_last_saved_steps[layer_slot].item())
    if (
        seen_samples < CALIBRATION_ROTATION_MATRIX_SAVE_INTERVAL
        or current_save_step <= last_saved_step
    ):
        return

    out_dir = _get_calibration_rotation_matrix_dir()
    os.makedirs(out_dir, exist_ok=True)
    # Save the accumulated (un-averaged) covariance and sample count. The
    # loader will average and run eigh. We save the cov rather than the
    # eigenvectors so partial-progress checkpoints remain meaningful: eigh of
    # partial cov gives partial R; the final checkpoint yields the final R.
    avg_cov = (calibration_rotation_matrix_accumulator[layer_slot] / seen_samples).contiguous()
    out_path = os.path.join(out_dir, f"layer_{layer_slot:02d}_mmstar_calibration.pt")
    torch.save(
        {
            "avg_cov": avg_cov,  # [num_heads, D, D]
            "samples": seen_samples,
            "layer_slot": layer_slot,
        },
        out_path,
    )
    calibration_rotation_matrix_last_saved_steps[layer_slot] = current_save_step






















# ---------------------------------------------------------------------------
# Cross-modal attention pattern divergence (KL) calibration.
#
# For each text query `q` (recent window) and each per-layer per-sparsity
# pruning scheme, compute the KL divergence between the FULL and PRUNED
# attention distributions over the VISION span (renormalized). This
# captures how the *ranking* of vision tokens changes under pruning —
# a shape metric, complementary to the attention-mass shift (scalar).
#
#   importance_l(s) = E_{q ∈ text}[ KL( A_l^full(q, ·)|vision
#                                      || A_l^pruned(q, ·)|vision ) ]
#
# Output CSV format mirrors attention_shift:
#   per-layer file  — (sparsity, head_idx, kl_pruning, kl_reconstruction, samples)
#   summary file    — (layer_idx, sparsity, head_average, kl_pruning, kl_reconstruction, samples)
# ---------------------------------------------------------------------------














def _get_sorted_mask_indices(mask, selected_dim=None):
    """Return ascending channel indices where `mask` is True.

    `selected_dim` MUST be supplied by the caller when known Python-side —
    passing it avoids `mask.sum().item()` (which forces a CUDA host sync
    that drains the entire stream and can cost 100s of μs per layer in a
    real generation). Falls back to the .item() count only when the caller
    doesn't know the row's True count up front.
    """
    if selected_dim is None:
        selected_dim = int(mask[0, 0].sum().item())
    if selected_dim == 0:
        return torch.empty(mask.shape[:-1] + (0,), dtype=torch.long, device=mask.device)
    return torch.topk(mask.int(), selected_dim, dim=-1, largest=True, sorted=True).indices.sort(dim=-1).values














def key_pruner_query_driven(
    kv_states,
    q_states,
    prompt_seqlen=0,
    query_seqlen=0,
    query_window_size=32,
    ratio=0.3,
    num_key_value_groups=1,
    calibration_channel_importance=None,
    return_mask=False,
):
    # Q,K,V: (B, H, L, d)
    global layer_idx

    _, num_heads, seqlen, head_dim = kv_states.shape
    vision_end = seqlen - query_seqlen if query_seqlen > 0 else seqlen
    vision_seqlen = vision_end - prompt_seqlen
    keys_vision = kv_states[:, :, prompt_seqlen:vision_end, :]

    # Fast-path for ratio == 0 (used by the bench's "Full" baseline). Skips
    # channel scoring, gather, and the .item() syncs in the main path so
    # the baseline reflects pure FA2 + cache cost (no hidden VisionZip
    # overhead). Disabled when calibration is collecting, since calibration
    # accumulators need the channel scores from the slow path.
    if ratio == 0.0 and not _is_collect_calibration_mode_enabled():
        bsz_fast = kv_states.shape[0]
        keep_mask_full = torch.ones(
            bsz_fast, num_heads, head_dim,
            dtype=torch.bool, device=kv_states.device,
        )
        if return_mask:
            last_meta_fast = keep_mask_full
        else:
            last_meta_fast = (
                torch.arange(head_dim, device=kv_states.device)
                .view(1, 1, head_dim)
                .expand(bsz_fast, num_heads, head_dim)
                .contiguous()
            )
        layer_idx += 1
        del q_states
        return (
            keys_vision,                                       # no copy — view of kv_states
            kv_states[:, :, :prompt_seqlen, :],
            kv_states[:, :, -query_seqlen:, :],
            keep_mask_full,
            last_meta_fast,
        )

    # ---- Compile-friendly fast-path for the common case (no calibration). ----
    # Routes through `rotatek.baselines.think.think_score_and_compact` which is
    # optionally torch.compile'd via `THINK_COMPILE` env var. Calibration
    # / debug paths fall through to the eager slow path below.
    if (
        calibration_channel_importance is None
        and not _is_collect_calibration_mode_enabled()
        and not DEBUG_PRINT_KEY_EFFECTIVE_RANK
        and not DEBUG_SAVE_SV_CURVES
    ):
        from rotatek.baselines.think import think_score_and_compact
        prune_count = min(head_dim, max(0, int(head_dim * ratio)))
        query_window = min(32, q_states.shape[2])
        queries_recent = q_states[..., -query_window:, :]
        K_compact, keep_mask = think_score_and_compact(
            keys_vision, queries_recent, num_heads, prune_count,
        )
        if return_mask:
            last_meta = keep_mask
        else:
            last_meta = _get_sorted_mask_indices(
                keep_mask, selected_dim=head_dim - prune_count,
            )
        layer_idx += 1
        del q_states
        return (
            K_compact,
            kv_states[:, :, :prompt_seqlen, :],
            kv_states[:, :, -query_seqlen:, :],
            keep_mask,
            last_meta,
        )

    if calibration_channel_importance is None:
        # ThinK's implementation of query-driven importance:
        # score = mean(Q^2) * mean(K^2) per channel, averaged over recent queries and all vision tokens
        # Q is grouped from H_q to H_kv before scoring (mathematically equivalent to
        # scoring at H_q then averaging, since K is shared within each GQA group)

        query_window = min(32, q_states.shape[2])
        queries_recent = q_states[..., -query_window:, :]
        queries_norm = torch.pow(queries_recent, 2).mean(dim=2)  # [B, H_q, D]

        # GQA: group queries from H_q to H_kv heads
        bsz_q = queries_norm.shape[0]
        num_q_heads = queries_norm.shape[1]
        queries_norm = queries_norm.view(
            bsz_q, num_heads, num_q_heads // num_heads, queries_norm.shape[-1],
        ).mean(dim=2)  # [B, H_kv, D]

        keys_norm = torch.pow(keys_vision, 2).mean(dim=2)  # [B, H_kv, D]
        channel_scores = queries_norm * keys_norm  # [B, H_kv, D]
    else:
        # Our implementation of MMStar's channel importance (completed during offline calibration):
        # directly use the provided importance scores, which are already averaged over recent queries and vision tokens, and apply grouping if needed

        channel_scores = calibration_channel_importance.to(device=kv_states.device, dtype=keys_vision.dtype)
        if channel_scores.dim() != 2:
            raise ValueError(
                f"Expected calibration_channel_importance to have shape (num_heads, head_dim), got {tuple(channel_scores.shape)}"
            )
        if channel_scores.shape[-1] != head_dim:
            raise ValueError(
                f"MMStar importance head_dim mismatch: expected {head_dim}, got {channel_scores.shape[-1]}"
            )

        # num_heads is now H_kv (no repeat_kv). Accept scores at H_kv or H_q.
        num_q_heads = q_states.shape[1]
        if channel_scores.shape[0] == num_heads:
            pass  # already H_kv, OK
        elif channel_scores.shape[0] == num_q_heads:
            # collapse H_q → H_kv by averaging within groups
            channel_scores = channel_scores.view(
                num_heads, num_q_heads // num_heads, head_dim,
            ).mean(dim=1)
        else:
            raise ValueError(
                f"calibration importance num_heads mismatch: expected {num_heads} (H_kv) or {num_q_heads} (H_q), got {channel_scores.shape[0]}"
            )

        channel_scores = channel_scores.unsqueeze(0).expand(kv_states.shape[0], -1, -1)

    prune_count = min(head_dim, max(0, int(head_dim * ratio)))
    pruned_mask = torch.zeros_like(channel_scores, dtype=torch.bool)
    if prune_count > 0:
        pruned_idx = torch.topk(channel_scores, prune_count, dim=-1, largest=False, sorted=False).indices
        pruned_mask.scatter_(-1, pruned_idx, True)

    keep_mask = ~pruned_mask
    # kept/pruned counts are deterministic from prune_count — avoid the
    # `.item()` host sync that would drain the CUDA stream.
    kept_dim = head_dim - prune_count
    pruned_dim = prune_count

    keep_idx = _get_sorted_mask_indices(keep_mask, selected_dim=kept_dim)
    pruned_idx = _get_sorted_mask_indices(pruned_mask, selected_dim=pruned_dim)

    kept_kv_states = (
        torch.gather(
            keys_vision,
            dim=-1,
            index=keep_idx.unsqueeze(2).expand(-1, -1, vision_seqlen, -1),
        )
        if kept_dim > 0
        else keys_vision[..., :0]
    )


    collect_calibration_mode = _is_collect_calibration_mode_enabled()

    if collect_calibration_mode and _calibration_task_enabled("channel_importance") and calibration_channel_importance is None:
        # Accumulate and periodically save per-layer MMStar channel-importance statistics during calibration collection.
        _accumulate_and_maybe_save_channel_importance_calibration(channel_scores, kv_group_size=1) # channel_scores are already grouped -> kv_group_size=1

    if collect_calibration_mode and _calibration_task_enabled("supplementary_matrix"):
        # Run this branch only when supplementary-matrix calibration is explicitly enabled.
        # Collect calibration data to fit a supplementary projection/bias for pruned vision keys.
        _accumulate_and_maybe_save_supplementary_matrix(
            keys_vision,
            channel_scores,
            q_states=q_states,
            num_key_value_groups=num_key_value_groups,
        )

    if collect_calibration_mode and _calibration_task_enabled("modality_score"):
        # Accumulate modality-level attention scores from recent queries and KV states for MMStar analysis.
        _accumulate_and_maybe_save_modality_score_calibration(
            q_states[..., -min(32, q_states.shape[2]):, :],
            kv_states,
            prompt_seqlen=prompt_seqlen,
            query_seqlen=query_seqlen,
        )

    if collect_calibration_mode and _calibration_task_enabled("attention_shift"):
        # Compare recent-32 text-query attention on vision keys before pruning,
        # after channel pruning, and after supplementary reconstruction.
        # The pruning mask must come from precomputed channel-importance calibration.
        _accumulate_and_maybe_save_attention_shift_calibration(
            q_states,
            kv_states,
            prompt_seqlen=prompt_seqlen,
            query_seqlen=query_seqlen,
            calibration_channel_importance=calibration_channel_importance,
            num_key_value_groups=num_key_value_groups,
        )

    if collect_calibration_mode and _calibration_task_enabled("attention_kl"):
        # Cross-modal attention pattern divergence: KL between full and pruned
        # text-to-vision attention distributions (renormalized to the vision
        # span). Captures *shape* / ranking changes, not just mass shift.
        _accumulate_and_maybe_save_attention_kl_calibration(
            q_states,
            kv_states,
            prompt_seqlen=prompt_seqlen,
            query_seqlen=query_seqlen,
            calibration_channel_importance=calibration_channel_importance,
            num_key_value_groups=num_key_value_groups,
        )

    if DEBUG_PRINT_KEY_EFFECTIVE_RANK:
        _print_key_effective_rank_and_maybe_block(
            keys_vision,
            q_states=q_states,
            num_key_value_groups=num_key_value_groups,
            query_window_size=query_window_size,
        )

    if DEBUG_SAVE_SV_CURVES:
        num_layers = getattr(globals().get("current_kv_cluster"), "num_layers", None)
        import sys as _sys
        _sys.stderr.write(
            f"[sv_curve DBG] key_pruner reached: layer_idx={layer_idx}, num_layers={num_layers}\n"
        )
        _sys.stderr.flush()
        if num_layers and num_layers > 0:
            _save_layer_singular_value_curves(
                keys_vision,
                q_states,
                layer_slot=layer_idx % num_layers,
                num_key_value_groups=num_key_value_groups,
                query_window_size=query_window_size,
                num_layers=num_layers,
            )

    layer_idx += 1
    del q_states

    # When called from update_think (return_mask=True), return the per-head
    # bool keep mask instead of the long index tensor — paper-faithful to
    # the original ThinK reference (boolean indexing for decode recovery).
    last_meta = keep_mask if return_mask else keep_idx
    return (
        kept_kv_states,
        kv_states[:, :, :prompt_seqlen, :],
        kv_states[:, :, -query_seqlen:, :],
        keep_mask,
        last_meta,
    )




class VisionZipCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', recent_size = 32, ratio =  0.4, prompt_seqlen = 0, query_seqlen = 0, query_window_size = 32, scoring = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.ratio = ratio
        self.recent_size = recent_size
        self.current_supplementary_matrix = None
        self.prompt_seqlen = prompt_seqlen
        self.query_seqlen = query_seqlen
        self.query_window_size = query_window_size
        self.scoring = scoring
        self.supplementary_matrix_cache = {}
        self.current_supplementary_matrix = None

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', recent_size = 32, ratio =  0.4):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.ratio = ratio
        self.recent_size = recent_size

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
        
    def update_think(self, key_states, query_states, value_states, attention_mask, num_key_value_groups, calibration_channel_importance=None):
        """ThinK: per-head channel pruning with zero fill at decode.

        Stores `think_mask [B, H_kv, D]` bool (True = kept channel) — paper-faithful
        per the original ThinK reference. Decode recovery uses boolean
        indexing: `recovered[mask.expand(...)] = kept_keys`, mirroring the
        author's implementation.
        """
        global layer_idx

        assert key_states.shape[-2] == query_states.shape[-2]

        kv_pruned, kv_prompt, kv_text, mask, keep_mask = key_pruner_query_driven(
            key_states,
            query_states,
            self.prompt_seqlen,
            self.query_seqlen,
            self.query_window_size,
            self.ratio,
            num_key_value_groups=num_key_value_groups,
            calibration_channel_importance=calibration_channel_importance,
            return_mask=True,
        )

        self.current_think_mask = keep_mask

        return kv_pruned, kv_prompt, kv_text, mask, value_states

    # NOTE: `update_visionk` was moved to `legacy/methods/visionk_method.py`
    # on 2026-04-27. VisionK is no longer an active method — its custom
    # Triton sparse-attention kernel still lives in
    # `kernel/sparse_channel_flash_decoding_triton.py` (RotateK reuses it),
    # but the cluster orchestration is no longer wired up.

    def update_spark(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        """SparK (compact-storage, ThinK-parity): per-token channel pruning.

        Stores compact K [B, H, S, D_keep] + per-token keep_indices
        [B, H, S, D_keep] + per-token pruned_mean [B, H, S, 1]. Decode-time
        recovery: pre-fill a [B, H, S, D] buffer with `pruned_mean`, then
        scatter compact K into the kept channel slots — preserves SparK's
        paper-defined mean-fill recovery while keeping cache memory ≈ ThinK
        (vs the 1.5× full-K cost of the bool-mask variant).

        Storage per layer per token (cache-resident):
            K_compact     : D_keep × bf16
            keep_indices  : D_keep × int64
            pruned_mean   : 1      × bf16
        """
        global layer_idx
        from rotatek.baselines.spark import per_token_channel_prune

        bsz, num_kv_heads, total_seq_len, head_dim = key_states.shape
        prompt_seqlen = max(0, min(getattr(self, "prompt_seqlen", 0), total_seq_len))
        query_seqlen = max(0, min(getattr(self, "query_seqlen", 0), max(total_seq_len - prompt_seqlen, 0)))
        vision_end = total_seq_len - query_seqlen if query_seqlen > 0 else total_seq_len

        kv_prompt = key_states[:, :, :prompt_seqlen, :]
        kv_text = key_states[:, :, -query_seqlen:, :] if query_seqlen > 0 else key_states[:, :, :0, :]
        keys_vision = key_states[:, :, prompt_seqlen:vision_end, :]

        # Group queries from H_q to H_kv for scoring
        query_window = min(32, query_states.shape[2])
        queries_recent = query_states[..., -query_window:, :]
        queries_grouped = queries_recent.view(
            bsz, num_kv_heads, num_key_value_groups, query_window, head_dim,
        ).mean(2)  # [B, H_kv, query_window, D]

        K_compact, keep_mask, pruned_mean = per_token_channel_prune(
            queries_grouped, keys_vision, ratio=self.ratio,
        )
        # K_compact   : [B, H_kv, S_vision, D_keep]
        # keep_mask   : [B, H_kv, S_vision, D] bool — True = kept channel
        # pruned_mean : [B, H_kv, S_vision, 1]

        # Stash side-channel artifacts on cluster — adapter appends to
        # past_key_value alongside K_compact (which is returned as kv_pruned).
        self.current_spark_mask = keep_mask
        self.current_spark_pruned_mean = pruned_mean

        # Dummy per-head mask kept for store_pruned signature compatibility
        # (not consulted by the SparK decode path).
        dummy_mask = torch.ones(bsz, num_kv_heads, head_dim, dtype=torch.bool, device=key_states.device)

        layer_idx += 1

        return K_compact, kv_prompt, kv_text, dummy_mask, value_states

    # NOTE: `update_fisherk` was moved to `legacy/methods/fisherk_method.py`
    # on 2026-04-27. FisherK is no longer an active method.

    def update_rotatek(
        self,
        key_states,
        query_states,
        value_states,
        attention_mask,
        num_key_value_groups,
        calibration_rotation_R_partial=None,
    ):
        """RotateK: PCA-aligned structured channel pruning.

        Picks an orthogonal R per (batch, kv-head) such that the first
        D_keep rotated channels carry the largest K-energy, then stores
        the rank-D_keep SVD approximation of K in the original basis:

            K_approx = (K @ R_partial) @ R_partial^T = K @ P

        where P = R_partial R_partial^T is the projection onto the top
        D_keep eigenvectors of K^T K. This is the Frobenius-optimal
        low-rank approximation (strictly ≥ any greedy channel selection
        in the original basis, e.g. ThinK/VisionK).

        Storage is full-D in the original basis so the decode path can
        concatenate prompt/vision/text keys and run standard FA2 — no
        per-step un-rotation matmul, no basis mismatch with the
        untouched prompt/text segments. For actual compute+memory
        savings (store D_keep, split-attention at decode), a follow-up
        kernel is needed; this implementation is accuracy-only.
        """
        global layer_idx

        assert key_states.shape[-2] == query_states.shape[-2]

        bsz, num_kv_heads, total_seq_len, head_dim = key_states.shape
        prompt_seqlen = max(0, min(getattr(self, "prompt_seqlen", 0), total_seq_len))
        query_seqlen = max(0, min(getattr(self, "query_seqlen", 0), max(total_seq_len - prompt_seqlen, 0)))
        vision_end = total_seq_len - query_seqlen if query_seqlen > 0 else total_seq_len
        vision_seqlen = vision_end - prompt_seqlen

        keys_vision = key_states[:, :, prompt_seqlen:vision_end, :]
        kv_prompt = key_states[:, :, :prompt_seqlen, :]
        kv_text = key_states[:, :, -query_seqlen:, :] if query_seqlen > 0 else key_states[:, :, :0, :]

        prune_count = min(head_dim, max(0, int(head_dim * self.ratio)))
        keep_count = head_dim - prune_count

        # ---- timing profile (auto-disables after first few prefills) ----
        _profile = _rotatek_profile_count[0] < _ROTATEK_PROFILE_LIMIT
        _ratio_on = _rotatek_ratio_count[0] < _ROTATEK_RATIO_LIMIT
        _events = []

        def _mark():
            if _profile:
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                _events.append(e)

        # Total-rotatek events for ratio comparison (always recorded when
        # ratio profiling is on; cheap — the sync happens in the adapter).
        _rk_start = _rk_end = None
        if _ratio_on:
            _rk_start = torch.cuda.Event(enable_timing=True)
            _rk_start.record()

        _mark()  # 0: start

        if keep_count == 0 or vision_seqlen == 0:
            # degenerate: prune everything or no vision tokens
            kept_kv_states = torch.zeros_like(keys_vision) if keep_count == 0 else keys_vision
            delta_mu = None
        elif calibration_rotation_R_partial is not None:
            # Offline calibration path: R_partial pre-computed on centered K,
            # so we also center K at inference, project, and add the mean
            # back (vision is one segment of a longer sequence — dropping the
            # mean would skew the softmax vs untouched prompt/text segments).
            R_partial = calibration_rotation_R_partial.to(
                device=keys_vision.device, dtype=keys_vision.dtype,
            )
            if R_partial.dim() == 3:
                R_partial = R_partial.unsqueeze(0).expand(bsz, -1, -1, -1)
            _mark()  # 1 (placeholder)
            _mark()  # 2
            _mark()  # 3
            _mark()  # 4
            _mark()  # 5
            mean_per_head = keys_vision.mean(dim=-2, keepdim=True)
            if _ROTATEK_STORAGE == "truncated":
                # Truncated storage: store K @ R_partial WITHOUT centering. At
                # decode the kernel computes Q @ P @ K^T for vision tokens.
                # Full mode would compute Q @ P @ K^T + Q @ (I - P) @ μ^T;
                # the missing term is a per-head scalar shift on every vision
                # logit that we recover via `sparse_bias_shift`.
                # Stash δμ = μ @ (I - P) for the adapter to dot with Q at decode.
                rotated = torch.matmul(keys_vision, R_partial)
                kept_kv_states = rotated
                mu_proj = torch.matmul(
                    torch.matmul(mean_per_head, R_partial),
                    R_partial.transpose(-2, -1),
                )  # [B, H_kv, 1, D]
                delta_mu = (mean_per_head - mu_proj).squeeze(-2)  # [B, H_kv, D]
            else:
                keys_centered = keys_vision - mean_per_head
                rotated = torch.matmul(keys_centered, R_partial)
                kept_kv_states = torch.matmul(rotated, R_partial.transpose(-2, -1)) + mean_per_head
                delta_mu = None
            _mark()  # 6: matmul1
            _mark()  # 7: matmul2
        else:
            # Online PCA path: compute R per-image from (K-μ)^T (K-μ).
            # Solver switchable via ROTATEK_SOLVER env var:
            #   "eigh"       : full CPU eigh round-trip (exact, slow at large D)
            #   "randomized" : top-k sketch (HMT), ~2-3x faster, ~1-2% excess err
            #
            # Optional subsampling: if ROTATEK_PCA_K_SAMPLE is set and S >
            # K_sample, estimate cov from K_sample uniformly random tokens.
            # This makes the cov einsum O(K_sample · D²) instead of O(S · D²).
            # Storage and δμ still use the full sequence (only the *basis*
            # is approximated), so the only effect is slightly suboptimal
            # eigenvector quality — for visual features with rapid spectrum
            # decay, K_sample = 512 is typically sufficient.
            S_full = keys_vision.shape[-2]
            if _ROTATEK_PCA_K_SAMPLE is not None and S_full > _ROTATEK_PCA_K_SAMPLE:
                idx = torch.randperm(
                    S_full, device=keys_vision.device,
                )[:_ROTATEK_PCA_K_SAMPLE]
                keys_for_cov = keys_vision.index_select(-2, idx)
            else:
                keys_for_cov = keys_vision

            # Skip materializing a full-precision K. The original code did
            # `keys_fp32 = keys_for_cov.float()` (96MB → 192MB write at
            # B=4, S=12K) followed by `keys_centered_fp32 = keys_fp32 -
            # mean` (another 192MB write) before the cov einsum. That cast
            # + center round-trip dominates `score`. We avoid it by:
            #   1. computing the mean in fp32 from a streaming reduction
            #      (output is tiny [B, H_kv, 1, D])
            #   2. centering K *in its native bf16* (one 96MB tensor)
            #   3. running the cov matmul on the bf16 tensor with fp32
            #      output cast — bf16 tensor cores accumulate in fp32
            #      internally, so cov precision is comparable to the old
            #      fp32 path. Final cast just rounds the [D, D] result.
            # Net memory: ~480MB → ~192MB; net compute: fp32-matmul →
            # bf16-tensor-core (8× peak FLOPS on A100).
            full_mean_fp32 = keys_vision.float().mean(dim=-2, keepdim=True) \
                if keys_for_cov is not keys_vision else None
            mean_per_head_fp32 = keys_for_cov.float().mean(dim=-2, keepdim=True)
            _mark()  # 1: mean (was: fp32 cast)
            keys_centered = keys_for_cov - mean_per_head_fp32.to(keys_for_cov.dtype)
            cov = torch.einsum(
                "bhsd,bhse->bhde", keys_centered, keys_centered,
            ).float()
            _mark()  # 2: cov einsum (bf16 input, fp32 output)
            # symmetrise to kill fp noise that would violate eigh's assumption
            cov = 0.5 * (cov + cov.transpose(-1, -2))
            _mark()  # 3: symmetrise

            # ----------- Optional: Query-aware RotateK (env-var gated) -----------
            # Vanilla RotateK does PCA on K^T K only — query-agnostic. If Q's
            # high-importance channels don't align with K's principal directions
            # (model-dependent), RotateK loses to query-aware methods (ThinK,
            # SparK). Solution: weight K by per-channel ||Q||_2 before PCA.
            #
            # Naively this needs an extra K read/write of [B, H, S, D] (heavy
            # at long S). Algebraic trick: Q has no S-dependency, so the
            # weighting commutes through the bilinear sum:
            #
            #   sum_s (K[s,d]·q[d]) (K[s,e]·q[e])  =  q[d]·q[e] · sum_s K[s,d] K[s,e]
            #                                       =  q_outer[d,e] · cov[d,e]
            #
            # → apply outer(q_norm, q_norm) elementwise on the [D, D] cov AFTER
            # the einsum. Cost: O(B·H·D²) extra (16K elements at D=128) — negligible.
            # Q-aware is the default (set ROTATEK_QUERY_AWARE=0 to disable
            # and run the original K-only PCA path).
            if int(os.environ.get("ROTATEK_QUERY_AWARE", "1")):
                # Mirror SparK/ThinK: take last query_window queries.
                q_window = min(32, query_states.shape[2])
                q_recent = query_states[..., -q_window:, :]  # [B, H_q, q_window, D]
                # GQA collapse H_q → H_kv via group-mean.
                bsz_q = q_recent.shape[0]
                h_q = q_recent.shape[1]
                if h_q != num_kv_heads:
                    q_recent = q_recent.view(
                        bsz_q, num_kv_heads, h_q // num_kv_heads, q_window, head_dim,
                    ).mean(dim=2)  # [B, H_kv, q_window, D]
                # Per-channel ||Q||_2 over query window. Add a tiny floor so
                # channels with near-zero query magnitude don't drop to exactly
                # zero — otherwise Q-weighted cov has zero rows/cols, becomes
                # rank-deficient, and downstream power-iter Cholesky fails on
                # the resulting near-singular Gram matrix.
                q_norm = q_recent.float().norm(dim=-2, p=2)  # [B, H_kv, D]
                # Floor at a small fraction of the per-(B, H) median norm,
                # so the floor scales with the data and doesn't dominate when
                # ||Q|| values are themselves small.
                q_floor = 1e-3 * q_norm.median(dim=-1, keepdim=True).values
                q_norm = q_norm.clamp(min=q_floor)
                q_outer = q_norm.unsqueeze(-1) * q_norm.unsqueeze(-2)  # [B, H_kv, D, D]
                cov = cov * q_outer
                # Multiplication by symmetric q_outer preserves cov's symmetry,
                # but re-symmetrize to absorb fp noise (cheap).
                cov = 0.5 * (cov + cov.transpose(-1, -2))
                # Belt-and-suspenders ridge: add eps · I scaled to cov's
                # magnitude. Negligible numerically vs the leading
                # eigenvalues, but makes the matrix strictly PSD so the
                # subsequent Cholesky in power_iteration cannot fail.
                # Higher than vanilla (1e-4 vs 1e-6) because Q-weighting
                # induces highly non-uniform eigenvalue magnitudes which
                # makes the boundary eigenvectors near-degenerate in
                # power-iter and trips Cholesky's PSD check.
                _ridge_eps = 1e-4 * cov.diagonal(dim1=-2, dim2=-1).abs().mean(
                    dim=-1, keepdim=True
                ).clamp(min=1e-6).unsqueeze(-1)
                cov = cov + _ridge_eps * torch.eye(
                    head_dim, device=cov.device, dtype=cov.dtype,
                )

            # DEBUG: cumulative eigenvalue ratio for top-D_keep at layer 0 only
            # (printed once per prefill so output is bounded). Ratio = how
            # much of K's variance the kept rotation subspace captures.
            # >0.9 = K is low-rank → RotateK should work well.
            # <0.5 = K is high-rank → RotateK fundamentally limited.
            if int(os.environ.get("ROTATEK_DEBUG_EIG", "0")) and layer_idx % 8 == 0:
                eigvals = torch.linalg.eigvalsh(cov[0, 0].cpu()).flip(0)
                cum = eigvals.cumsum(0) / eigvals.sum()
                _kept = keep_count
                sys.stderr.write(
                    f"[rotatek-eig L{layer_idx:02d}] keep={_kept}/{head_dim} "
                    f"top-{_kept} captures {cum[_kept-1].item():.3f} of variance "
                    f"(top-1 {cum[0].item():.3f}, top-D/2 {cum[head_dim//2-1].item():.3f})\n"
                )
                sys.stderr.flush()

            if _ROTATEK_SOLVER == "randomized":
                # Halko-Martinsson-Tropp: small sketch + small eigh on
                # projected matrix. Kept entirely on GPU.
                from rotatek.rotation import randomized_topk_eigh
                oversample = 10
                num_power_iters = 1
                _, R_partial_fp32 = randomized_topk_eigh(
                    cov, k=keep_count,
                    oversample=oversample,
                    num_power_iters=num_power_iters,
                )
                R_partial = R_partial_fp32.to(keys_vision.dtype)
                _mark()  # 4: randomized eigh
                _mark()  # 5: (no flip needed — randomized already returns descending)
            elif _ROTATEK_SOLVER == "power_iter":
                # Pure PyTorch-GPU power iteration with QR orthonormalization.
                # Very few kernel launches (matmul + QR per iter), amortizes
                # extremely well with batch size. Approximate: quality depends
                # on eigenvalue-gap at the K-th boundary. 5 iters usually fine
                # for Wishart-like K^T K; may need more for flat spectra.
                from rotatek.rotation import power_iteration_gpu
                num_iters = 5
                R_partial_fp32 = power_iteration_gpu(
                    cov, k=keep_count, num_iters=num_iters, seed=0,
                )
                R_partial = R_partial_fp32.to(keys_vision.dtype)
                _mark()  # 4: power iter
                _mark()  # 5: (no flip)
            else:
                # Run eigh on CPU — for 128×128 symmetric matrices LAPACK
                # beats cuSOLVER by ~10× because GPU launch/sync overhead
                # dominates the actual O(D³) compute at this size.
                cov_cpu = cov.cpu()
                _, eigvecs_cpu = torch.linalg.eigh(cov_cpu)
                eigvecs = eigvecs_cpu.to(keys_vision.device, non_blocking=True)
                _mark()  # 4: eigh (CPU roundtrip)
                # eigvecs[..., :, -1] is top-eigenvalue direction — flip so that
                # column 0 is the largest.
                R_full = eigvecs.flip(dims=[-1])
                R_partial = R_full[..., :keep_count].to(keys_vision.dtype)
                _mark()  # 5: flip+slice+cast

            # Project K to top-D_keep subspace.
            # When subsampling, the cov mean is from K_sample tokens but the
            # δμ-bias must use the *true* full-sequence mean to be exact at
            # decode (else vision logits have a residual constant shift).
            true_mean_fp32 = full_mean_fp32 if full_mean_fp32 is not None else mean_per_head_fp32
            mean_per_head = true_mean_fp32.to(keys_vision.dtype)
            if _ROTATEK_STORAGE == "truncated":
                # Store K @ R_partial WITHOUT centering. Kernel computes
                # Q @ P @ K^T for vision logits; the missing full-mode term
                # is Q @ (I - P) @ μ^T (constant shift per vision token,
                # required to match prompt/text relative scale). Stash
                # δμ = μ @ (I - P) for decode-time bias.
                rotated = torch.matmul(keys_vision, R_partial)
                kept_kv_states = rotated
                R_partial_fp32_for_mu = R_partial.float()
                mu_proj_fp32 = torch.matmul(
                    torch.matmul(true_mean_fp32, R_partial_fp32_for_mu),
                    R_partial_fp32_for_mu.transpose(-2, -1),
                )  # [B, H_kv, 1, D] fp32
                delta_mu = (
                    (true_mean_fp32 - mu_proj_fp32).squeeze(-2).to(keys_vision.dtype)
                )  # [B, H_kv, D]
            else:
                # Un-rotate back to original basis (full D, rank-D_keep) so the
                # standard FA2 decode path can attend without Q rotation.
                keys_centered = keys_vision - mean_per_head
                rotated = torch.matmul(keys_centered, R_partial)
                kept_kv_states = torch.matmul(rotated, R_partial.transpose(-2, -1)) + mean_per_head
                delta_mu = None
            _mark()  # 6: matmul1
            _mark()  # 7: matmul2

        # Note: rotation_matrix calibration accumulation is NOT done here; it
        # lives in the adapter's prefill branch so it runs for any
        # channel_method (the rotation is a property of K, not the pruning
        # scheme). See `_maybe_accumulate_rotation_matrix_calibration`.

        # Stash R_partial on the cluster — the adapter's decode branch needs
        # it (in truncated storage mode) to rotate Q at each decode step.
        # In full mode the adapter ignores it; cheap to always set.
        try:
            self.current_rotatek_R_partial = R_partial
        except UnboundLocalError:
            # degenerate keep_count == 0 path may not produce R_partial
            self.current_rotatek_R_partial = None

        # δμ = μ @ (I - P) — per-head additive bias on vision logits at decode
        # time, required to match full-mode logit Q @ K_approx^T when storing
        # K in rotated truncated form without centering. None in full storage
        # mode and on degenerate paths.
        self.current_rotatek_delta_mu = delta_mu

        # Interface parity: in full storage mode all D channels are present,
        # so keep_mask is all True. In truncated mode we only stored the first
        # D_keep channels; mark them True, rest False (kernels can use this).
        keep_mask = torch.zeros(
            bsz, num_kv_heads, head_dim,
            dtype=torch.bool, device=keys_vision.device,
        )
        if _ROTATEK_STORAGE == "truncated":
            keep_mask[..., :keep_count] = True
        else:
            keep_mask[...] = True

        if _ratio_on:
            _rk_end = torch.cuda.Event(enable_timing=True)
            _rk_end.record()
            # Expose to adapter; it will sync + compute ratio against
            # full attention-forward time.
            self.current_rotatek_evt_start = _rk_start
            self.current_rotatek_evt_end = _rk_end
        else:
            self.current_rotatek_evt_start = None
            self.current_rotatek_evt_end = None

        if _profile and len(_events) >= 8:
            torch.cuda.synchronize()
            stages = ["fp32", "cov", "sym", "eigh", "Rsel", "mm1", "mm2"]
            dts = [_events[i].elapsed_time(_events[i + 1]) for i in range(len(_events) - 1)]
            total = sum(dts)
            # Write to stderr so output survives the driver's stdout->log-file
            # redirection (in visionzip_internvl.py when log=True).
            sys.stderr.write(
                f"[rotatek L{layer_idx:02d} call#{_rotatek_profile_count[0]:03d}] "
                + "  ".join(f"{n}={t:.2f}" for n, t in zip(stages, dts))
                + f"  | total={total:.2f}ms\n"
            )
            sys.stderr.flush()
        _rotatek_profile_count[0] += 1

        # Pause once the profile window closes so the user can inspect the
        # timing output without the rest of the eval overwriting it.
        if _rotatek_profile_count[0] == _ROTATEK_PROFILE_LIMIT:
            sys.stderr.write("here: press ENTER to continue...\n")
            sys.stderr.flush()
            input()

        layer_idx += 1

        return kept_kv_states, kv_prompt, kv_text, keep_mask, value_states


def init_visionzip(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 4096
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'recent_size'):
            self.config.recent_size = 32
        if not hasattr(self.config, 'ratio'):
            self.config.ratio = 0.4
        if not hasattr(self.config, 'prompt_seqlen'):
            self.config.prompt_seqlen = 0
        if not hasattr(self.config, 'query_seqlen'):
            self.config.query_seqlen = 0
    
    self.kv_cluster = VisionZipCluster( 
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        recent_size = self.config.recent_size,
        ratio = self.config.channel_ratio,
        prompt_seqlen = self.config.prompt_seqlen,
        query_seqlen = self.config.query_seqlen,
        # query_window_size = self.config.query_window_size,
        # scoring = self.config.scoring,
        )
    self.kv_cluster.num_layers = getattr(self.config, "num_hidden_layers", None)
    if self.kv_cluster.num_layers is None and hasattr(self, "model") and hasattr(self.model, "layers"):
        self.kv_cluster.num_layers = len(self.model.layers)
    self.kv_cluster.num_layer = self.kv_cluster.num_layers
    globals()["current_kv_cluster"] = self.kv_cluster
    config_result_root = getattr(self.config, "result_root", None)
    if config_result_root is None:
        raise ValueError(
            "config.result_root is not set. The model adapter must inject the "
            "model-specific DEFAULT_CALIBRATION_RESULT_ROOT before init_visionzip runs."
        )
    self.kv_cluster.result_root = str(config_result_root)
    self.kv_cluster.calibration_mode = getattr(self.config, "calibration_mode", None)
    self.kv_cluster.channel_method = getattr(self.config, "channel_method", "visionk")
    self.kv_cluster.channel_reconstruction = str(getattr(self.config, "channel_reconstruction", "off")).strip().lower()
    self.kv_cluster.offline_calibration_tasks = getattr(self.config, "offline_calibration_tasks", "channel_importance")
    if not hasattr(self.config, "dominant_ratio"):
        raise ValueError("config.dominant_ratio must be provided")
    self.kv_cluster.dominant_ratio = self.config.dominant_ratio

    if self.kv_cluster.channel_reconstruction == "matrix" and _parse_calibration_mode(self.kv_cluster.calibration_mode) == "use":
        # Preload once before inference to avoid first-token latency spikes from lazy disk I/O.
        if bool(getattr(self.config, "layer_adaptive_channel_budget", False)):
            preload_sparsities = SUPPLEMENTARY_MATRIX_SPARSITIES
        else:
            preload_sparsities = [getattr(self.config, "channel_ratio", None)]
        _preload_supplementary_matrix_cache(
            num_layers=self.kv_cluster.num_layers,
            sparsities=preload_sparsities,
        )


def recover_cache(key_states_pruned, mask, layer_idx, ratio=0.4, key_states_supplementary=None):
    """
    _, heads, head_dim  = mask.shape
    k = head_dim - int(head_dim * ratio)
    sqlen = int(key_states_pruned.shape[-1]/(k*heads))
    mask = mask.unsqueeze(2).expand(-1, -1, sqlen, -1)
    recovered_key_states = torch.zeros(1, heads, sqlen, head_dim, dtype=key_states_pruned.dtype, device = key_states_pruned.device)
    recovered_key_states[mask] = key_states_pruned
    del mask
    """

    _, heads, head_dim  = mask.shape
    sqlen = key_states_pruned.shape[-2]
    mask = mask.unsqueeze(2).expand(-1, -1, sqlen, -1)
    recovered_key_states = torch.zeros(1, heads, sqlen, head_dim, dtype=key_states_pruned.dtype, device=key_states_pruned.device)
    recovered_key_states[mask] = key_states_pruned.reshape(-1)

    if key_states_supplementary is not None:
        supplementary_mask = ~mask
        recovered_key_states[supplementary_mask] = key_states_supplementary.reshape(-1)
        del supplementary_mask

    del mask

    return recovered_key_states



# --- research instrumentation -------------------------------------------
# The calibration / profiling helpers live in _research_probes so the inference
# path above stays readable. Each is a thin forwarder that imports the module on
# first call, so nothing is loaded unless a flag actually asks for it.

def _probes():
    from lmms_eval.models.model_utils import _research_probes
    return _research_probes

def _accumulate_and_maybe_save_attention_kl_calibration(*args, **kwargs):
    return getattr(_probes(), "_accumulate_and_maybe_save_attention_kl_calibration")(*args, **kwargs)

def _accumulate_and_maybe_save_attention_shift_calibration(*args, **kwargs):
    return getattr(_probes(), "_accumulate_and_maybe_save_attention_shift_calibration")(*args, **kwargs)

def _accumulate_and_maybe_save_channel_importance_calibration(*args, **kwargs):
    return getattr(_probes(), "_accumulate_and_maybe_save_channel_importance_calibration")(*args, **kwargs)

def _accumulate_and_maybe_save_modality_score_calibration(*args, **kwargs):
    return getattr(_probes(), "_accumulate_and_maybe_save_modality_score_calibration")(*args, **kwargs)

def _accumulate_and_maybe_save_supplementary_matrix(*args, **kwargs):
    return getattr(_probes(), "_accumulate_and_maybe_save_supplementary_matrix")(*args, **kwargs)

def _calibration_task_enabled(*args, **kwargs):
    return getattr(_probes(), "_calibration_task_enabled")(*args, **kwargs)

def _is_collect_calibration_mode_enabled(*args, **kwargs):
    return getattr(_probes(), "_is_collect_calibration_mode_enabled")(*args, **kwargs)

def _parse_calibration_mode(*args, **kwargs):
    return getattr(_probes(), "_parse_calibration_mode")(*args, **kwargs)

def _preload_supplementary_matrix_cache(*args, **kwargs):
    return getattr(_probes(), "_preload_supplementary_matrix_cache")(*args, **kwargs)

def _print_key_effective_rank_and_maybe_block(*args, **kwargs):
    return getattr(_probes(), "_print_key_effective_rank_and_maybe_block")(*args, **kwargs)

def _save_layer_singular_value_curves(*args, **kwargs):
    return getattr(_probes(), "_save_layer_singular_value_curves")(*args, **kwargs)
