

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


def _get_calibration_channel_importance_dir():
    return os.path.join(_get_calibration_root_dir(), "channel_importance")


def _get_calibration_modality_score_dir():
    return os.path.join(_get_calibration_root_dir(), "attention_score")


def _get_supplementary_matrix_dir():
    return os.path.join(_get_calibration_root_dir(), "supplementary_matrix")


def _get_attention_shift_dir():
    return os.path.join(_get_calibration_root_dir(), "attention_shift")


def _get_calibration_rotation_matrix_dir():
    return os.path.join(_get_calibration_root_dir(), "rotation_matrix")


def _parse_calibration_mode(mode_spec):
    if mode_spec is None:
        return "off"
    return str(mode_spec).strip().lower()


def _parse_offline_calibration_tasks(task_spec):
    if task_spec is None:
        return {"channel_importance"}

    if isinstance(task_spec, str):
        normalized = task_spec.strip().lower()
        if normalized in ("", "none", "off"):
            return set()
        raw_items = [item.strip().lower() for item in normalized.split(",") if item.strip()]
    elif isinstance(task_spec, (list, tuple, set)):
        raw_items = [str(item).strip().lower() for item in task_spec if str(item).strip()]
    else:
        raw_items = [str(task_spec).strip().lower()]

    valid_tasks = {"channel_importance", "modality_score", "supplementary_matrix", "attention_shift", "attention_kl", "rotation_matrix", "all"}
    tasks = {item for item in raw_items if item in valid_tasks}

    if "all" in tasks:
        return {"channel_importance", "modality_score", "supplementary_matrix", "attention_shift", "attention_kl", "rotation_matrix"}
    return tasks


def _calibration_task_enabled(task_name):
    current_cluster = globals().get("current_kv_cluster")
    task_spec = getattr(current_cluster, "offline_calibration_tasks", None)
    active_tasks = _parse_offline_calibration_tasks(task_spec)
    return task_name in active_tasks


def _is_collect_calibration_mode_enabled():
    current_cluster = globals().get("current_kv_cluster")
    mode_spec = getattr(current_cluster, "calibration_mode", None)
    mode = _parse_calibration_mode(mode_spec)
    return mode == "collect"


def _get_layer_supplementary_components(layer_slot, sparsity):
    global supplementary_matrix_file_cache

    rounded_sparsity = round(float(sparsity), 3)
    supported_sparsity = None
    for candidate in SUPPLEMENTARY_MATRIX_SPARSITIES:
        if abs(candidate - rounded_sparsity) < 1e-6:
            supported_sparsity = candidate
            break
    if supported_sparsity is None:
        upper_or_equal = [candidate for candidate in SUPPLEMENTARY_MATRIX_SPARSITIES if candidate >= rounded_sparsity]
        supported_sparsity = min(upper_or_equal) if upper_or_equal else max(SUPPLEMENTARY_MATRIX_SPARSITIES)

    supplementary_matrix_dir = _get_supplementary_matrix_dir()
    cache_key = (supplementary_matrix_dir, layer_slot, supported_sparsity, SUPPLEMENTARY_MATRIX_RIDGE_LAMBDA)
    if cache_key not in supplementary_matrix_file_cache:
        file_path = os.path.join(
            supplementary_matrix_dir,
            f"sparsity_{supported_sparsity:.3f}_lambda_{SUPPLEMENTARY_MATRIX_RIDGE_LAMBDA:g}",
            f"layer_{layer_slot:02d}.pt",
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Supplementary matrix file not found for layer={layer_slot}, sparsity={supported_sparsity:.3f}: {file_path}"
            )
        supplementary_matrix_file_cache[cache_key] = torch.load(file_path, map_location="cpu")

    supplementary_matrix = supplementary_matrix_file_cache[cache_key]
    return {
        "weight": supplementary_matrix["weight"],
        "bias": supplementary_matrix["bias"],
        "keep_idx": supplementary_matrix["keep_idx"],
        "pruned_idx": supplementary_matrix["pruned_idx"],
        "sparsity": supplementary_matrix.get("sparsity", supported_sparsity),
        "samples": supplementary_matrix.get("samples"),
    }


def _preload_supplementary_matrix_cache(num_layers, sparsities):
    if num_layers is None or num_layers <= 0:
        return

    unique_sparsities = []
    for sparsity in sparsities:
        if sparsity is None:
            continue
        unique_sparsities.append(round(float(sparsity), 3))

    for supported_sparsity in sorted(set(unique_sparsities)):
        for layer_slot in range(int(num_layers)):
            _get_layer_supplementary_components(layer_slot, supported_sparsity)


def _accumulate_and_maybe_save_channel_importance_calibration(channel_scores, kv_group_size=1):
    global layer_idx
    global calibration_channel_importance_accumulator
    global calibration_channel_importance_seen_sample_counts
    global calibration_channel_importance_last_saved_steps

    if channel_scores is None or channel_scores.dim() != 3:
        return

    grouped_scores = channel_scores.detach().to(torch.float32)
    if kv_group_size > 1 and grouped_scores.shape[1] % kv_group_size == 0:
        grouped_scores = grouped_scores.view(
            grouped_scores.shape[0],
            grouped_scores.shape[1] // kv_group_size,
            kv_group_size,
            grouped_scores.shape[2],
        ).mean(dim=2)

    num_layers = getattr(globals().get("current_kv_cluster"), "num_layers", None)
    if num_layers is None:
        raise RuntimeError("current_kv_cluster.num_layers must be set before using kv pruning utilities")
    batch_size, num_heads, head_dim = grouped_scores.shape
    expected_shape = (num_layers, num_heads, head_dim)
    if calibration_channel_importance_accumulator is None or tuple(calibration_channel_importance_accumulator.shape) != expected_shape:
        calibration_channel_importance_accumulator = torch.zeros(expected_shape, dtype=torch.float32, device="cpu")
        calibration_channel_importance_seen_sample_counts = torch.zeros(num_layers, dtype=torch.long, device="cpu")
        calibration_channel_importance_last_saved_steps = torch.full((num_layers,), -1, dtype=torch.long, device="cpu")

    layer_slot = layer_idx % num_layers
    calibration_channel_importance_accumulator[layer_slot] += grouped_scores.sum(dim=0).cpu()
    calibration_channel_importance_seen_sample_counts[layer_slot] += batch_size

    seen_samples = int(calibration_channel_importance_seen_sample_counts[layer_slot].item())
    current_save_step = min(seen_samples, CALIBRATION_CHANNEL_IMPORTANCE_TARGET_SAMPLES) // CALIBRATION_CHANNEL_IMPORTANCE_SAVE_INTERVAL
    last_saved_step = int(calibration_channel_importance_last_saved_steps[layer_slot].item())
    if seen_samples < CALIBRATION_CHANNEL_IMPORTANCE_SAVE_INTERVAL or current_save_step <= last_saved_step:
        return

    channel_importance_dir = _get_calibration_channel_importance_dir()
    os.makedirs(channel_importance_dir, exist_ok=True)
    avg_scores = (calibration_channel_importance_accumulator[layer_slot] / seen_samples).cpu().numpy()
    out_csv_path = os.path.join(channel_importance_dir, f"layer_{layer_slot:02d}_mmstar_calibration.csv")
    # print(f"[channel_importance save] {out_csv_path}", flush=True)
    # input("here")
    with open(out_csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["layer_idx", "head_idx", "channel_idx", "avg_score", "samples"])
        for head_idx in range(avg_scores.shape[0]):
            for channel_idx in range(avg_scores.shape[1]):
                writer.writerow([
                    layer_slot,
                    head_idx,
                    channel_idx,
                    f"{float(avg_scores[head_idx, channel_idx]):.10f}",
                    seen_samples,
                ])
    calibration_channel_importance_last_saved_steps[layer_slot] = current_save_step


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


def _accumulate_and_maybe_save_modality_score_calibration(queries_recent, kv_states, prompt_seqlen=0, query_seqlen=0):
    global layer_idx
    global calibration_modality_score_accumulator
    global calibration_modality_score_seen_sample_counts
    global calibration_modality_score_last_saved_steps

    if queries_recent is None or kv_states is None:
        return
    if queries_recent.dim() != 4 or kv_states.dim() != 4:
        return

    batch_size, num_heads, query_window, head_dim = queries_recent.shape
    _, key_heads, seqlen, key_dim = kv_states.shape
    if head_dim != key_dim or seqlen == 0 or query_window == 0:
        return
    if num_heads != key_heads:
        # GQA: expand H_kv to H_q so attention probabilities are computed per query head.
        # Scores are stored at H_q granularity (consistent with attention_shift).
        if num_heads % key_heads != 0:
            return
        kv_states = repeat_kv(kv_states, num_heads // key_heads)

    vision_end = seqlen - query_seqlen if query_seqlen > 0 else seqlen
    prompt_end = max(0, min(prompt_seqlen, vision_end))
    vision_start = prompt_end
    vision_stop = max(vision_start, min(vision_end, seqlen))
    text_start = max(vision_stop, 0)

    queries_recent = queries_recent.detach().to(torch.float32)
    keys_all = kv_states.detach().to(torch.float32)
    attn_scores = torch.matmul(queries_recent, keys_all.transpose(-2, -1)) / math.sqrt(head_dim)
    attn_probs = torch.softmax(attn_scores, dim=-1)

    modality_scores = torch.zeros(batch_size, num_heads, 3, dtype=torch.float32, device=attn_probs.device)
    if prompt_end > 0:
        modality_scores[..., 0] = attn_probs[..., :prompt_end].sum(dim=-1).mean(dim=-1)
    if vision_stop > vision_start:
        modality_scores[..., 1] = attn_probs[..., vision_start:vision_stop].sum(dim=-1).mean(dim=-1)
    if text_start < seqlen:
        modality_scores[..., 2] = attn_probs[..., text_start:].sum(dim=-1).mean(dim=-1)

    num_layers = getattr(globals().get("current_kv_cluster"), "num_layers", None)
    if num_layers is None:
        raise RuntimeError("current_kv_cluster.num_layers must be set before using kv pruning utilities")
    expected_shape = (num_layers, num_heads, 3)
    if calibration_modality_score_accumulator is None or tuple(calibration_modality_score_accumulator.shape) != expected_shape:
        calibration_modality_score_accumulator = torch.zeros(expected_shape, dtype=torch.float32, device="cpu")
        calibration_modality_score_seen_sample_counts = torch.zeros(num_layers, dtype=torch.long, device="cpu")
        calibration_modality_score_last_saved_steps = torch.full((num_layers,), -1, dtype=torch.long, device="cpu")

    layer_slot = layer_idx % num_layers
    calibration_modality_score_accumulator[layer_slot] += modality_scores.sum(dim=0).cpu()
    calibration_modality_score_seen_sample_counts[layer_slot] += batch_size

    seen_samples = int(calibration_modality_score_seen_sample_counts[layer_slot].item())
    current_save_step = min(seen_samples, CALIBRATION_MODALITY_SCORE_TARGET_SAMPLES) // CALIBRATION_MODALITY_SCORE_SAVE_INTERVAL
    last_saved_step = int(calibration_modality_score_last_saved_steps[layer_slot].item())
    if seen_samples < CALIBRATION_MODALITY_SCORE_SAVE_INTERVAL or current_save_step <= last_saved_step:
        return

    modality_score_dir = _get_calibration_modality_score_dir()
    os.makedirs(modality_score_dir, exist_ok=True)
    avg_scores = (calibration_modality_score_accumulator[layer_slot] / seen_samples).cpu().numpy()
    out_csv_path = os.path.join(modality_score_dir, f"layer_{layer_slot:02d}.csv")
    with open(out_csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["layer_idx", "head_idx", "prompt_score", "vision_score", "text_score", "samples"])
        for head_idx in range(avg_scores.shape[0]):
            writer.writerow([
                layer_slot,
                head_idx,
                f"{float(avg_scores[head_idx, 0]):.10f}",
                f"{float(avg_scores[head_idx, 1]):.10f}",
                f"{float(avg_scores[head_idx, 2]):.10f}",
                seen_samples,
            ])
    calibration_modality_score_last_saved_steps[layer_slot] = current_save_step
    _save_modality_layer_summary_mean()


def _save_modality_layer_summary_mean():
    layer_means = []

    modality_score_dir = _get_calibration_modality_score_dir()
    if not os.path.isdir(modality_score_dir):
        return

    for file_name in os.listdir(modality_score_dir):
        if not file_name.startswith("layer_") or not file_name.endswith(".csv"):
            continue
        if file_name == "layer_summary_mean.csv":
            continue

        layer_token = file_name[len("layer_"):-len(".csv")]
        if not layer_token.isdigit():
            continue
        layer_idx_local = int(layer_token)

        file_path = os.path.join(modality_score_dir, file_name)
        prompt_scores = []
        vision_scores = []
        text_scores = []
        with open(file_path, "r", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if "prompt_score" not in row or "vision_score" not in row or "text_score" not in row:
                    continue
                prompt_scores.append(float(row["prompt_score"]))
                vision_scores.append(float(row["vision_score"]))
                text_scores.append(float(row["text_score"]))

        if not prompt_scores:
            continue

        layer_means.append(
            (
                layer_idx_local,
                sum(prompt_scores) / len(prompt_scores),
                sum(vision_scores) / len(vision_scores),
                sum(text_scores) / len(text_scores),
                len(prompt_scores),
            )
        )

    if not layer_means:
        return

    layer_means.sort(key=lambda x: x[0])
    summary_path = os.path.join(modality_score_dir, "layer_summary_mean.csv")
    with open(summary_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["layer_idx", "prompt_headwise_mean", "vision_headwise_mean", "text_headwise_mean", "num_heads"])
        for layer_idx_local, prompt_mean, vision_mean, text_mean, num_heads in layer_means:
            writer.writerow([
                layer_idx_local,
                f"{prompt_mean:.10f}",
                f"{vision_mean:.10f}",
                f"{text_mean:.10f}",
                num_heads,
            ])


def _initialize_attention_shift_stats(num_layers):
    global attention_shift_before_accumulator
    global attention_shift_after_pruning_accumulator
    global attention_shift_after_reconstruction_accumulator
    global attention_shift_seen_sample_counts
    global attention_shift_last_saved_steps

    expected_shape = (num_layers,)
    if attention_shift_seen_sample_counts is not None and tuple(attention_shift_seen_sample_counts.shape) == expected_shape:
        return

    attention_shift_before_accumulator = {}
    attention_shift_after_pruning_accumulator = {}
    attention_shift_after_reconstruction_accumulator = {}
    attention_shift_seen_sample_counts = torch.zeros(expected_shape, dtype=torch.long, device="cpu")
    attention_shift_last_saved_steps = torch.full(expected_shape, -1, dtype=torch.long, device="cpu")


def _build_recent_query_causal_mask(recent_query_len, total_key_len, device, dtype):
    query_positions = torch.arange(total_key_len - recent_query_len, total_key_len, device=device)
    key_positions = torch.arange(total_key_len, device=device)
    invalid = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
    mask = torch.zeros(recent_query_len, total_key_len, dtype=dtype, device=device)
    mask = mask.masked_fill(invalid, torch.finfo(dtype).min)
    return mask.unsqueeze(0).unsqueeze(0)


def _expand_saved_channel_importance(calibration_channel_importance, num_heads, head_dim, num_key_value_groups, device, dtype):
    if calibration_channel_importance is None:
        raise RuntimeError(
            "attention_shift calibration requires precomputed channel_importance to be loaded and passed in"
        )

    channel_scores = calibration_channel_importance.to(device=device, dtype=dtype)
    if channel_scores.dim() != 2:
        raise ValueError(
            f"Expected calibration_channel_importance to have shape (num_heads, head_dim), got {tuple(channel_scores.shape)}"
        )
    if channel_scores.shape[-1] != head_dim:
        raise ValueError(
            f"channel importance head_dim mismatch: expected {head_dim}, got {channel_scores.shape[-1]}"
        )

    # In GQA models this helper can be called with KV heads while the saved
    # calibration was collected at query-head granularity. Accept both layouts.
    group_size = int(num_key_value_groups) if num_key_value_groups and num_key_value_groups > 0 else 1
    saved_heads = channel_scores.shape[0]
    expected_heads = [num_heads]

    if saved_heads == num_heads:
        pass
    elif group_size > 1 and saved_heads == num_heads * group_size:
        # H_q -> H_kv, matching key_pruner_query_driven's calibration path.
        channel_scores = channel_scores.reshape(num_heads, group_size, head_dim).mean(dim=1)
        expected_heads.append(num_heads * group_size)
    elif group_size > 1 and num_heads % group_size == 0 and saved_heads == num_heads // group_size:
        # H_kv -> H_q for callers that operate after repeat_kv.
        channel_scores = channel_scores.repeat_interleave(group_size, dim=0)
        expected_heads.append(num_heads // group_size)
    else:
        if group_size > 1:
            expected_heads.extend([num_heads * group_size])
            if num_heads % group_size == 0:
                expected_heads.extend([num_heads // group_size])
        expected_heads = sorted(set(expected_heads))
        raise ValueError(
            f"channel importance num_heads mismatch: expected one of {expected_heads}, got {saved_heads}"
        )
    return channel_scores


def _apply_attention_shift_reconstruction(vision_keys, supplementary_matrix, num_heads, num_key_value_groups):
    supplementary_weight = supplementary_matrix["weight"].to(device=vision_keys.device, dtype=vision_keys.dtype)
    supplementary_bias = supplementary_matrix["bias"].to(device=vision_keys.device, dtype=vision_keys.dtype)
    keep_idx = supplementary_matrix["keep_idx"].to(device=vision_keys.device, dtype=torch.long)
    pruned_idx = supplementary_matrix["pruned_idx"].to(device=vision_keys.device, dtype=torch.long)

    if supplementary_weight.shape[0] != num_heads:
        if supplementary_weight.shape[0] * num_key_value_groups != num_heads:
            raise ValueError(
                f"Supplementary matrix head mismatch: matrix_heads={supplementary_weight.shape[0]}, key_heads={num_heads}, num_key_value_groups={num_key_value_groups}"
            )
        supplementary_weight = supplementary_weight.repeat_interleave(num_key_value_groups, dim=0)
        supplementary_bias = supplementary_bias.repeat_interleave(num_key_value_groups, dim=0)
        keep_idx = keep_idx.repeat_interleave(num_key_value_groups, dim=0)
        pruned_idx = pruned_idx.repeat_interleave(num_key_value_groups, dim=0)

    keep_idx_expanded = keep_idx.unsqueeze(0).unsqueeze(2).expand(vision_keys.shape[0], -1, vision_keys.shape[2], -1)
    pruned_idx_expanded = pruned_idx.unsqueeze(0).unsqueeze(2).expand(vision_keys.shape[0], -1, vision_keys.shape[2], -1)

    pruned_vision_keys = vision_keys.detach().clone()
    if pruned_idx.shape[-1] > 0:
        pruned_vision_keys = pruned_vision_keys.scatter(-1, pruned_idx_expanded, torch.zeros_like(torch.gather(pruned_vision_keys, dim=-1, index=pruned_idx_expanded)))

    kept_key_states = torch.gather(vision_keys, dim=-1, index=keep_idx_expanded)
    reconstructed_vision_keys = pruned_vision_keys.detach().clone()
    if pruned_idx.shape[-1] > 0:
        reconstructed_pruned = torch.matmul(kept_key_states, supplementary_weight) + supplementary_bias.unsqueeze(0).unsqueeze(2)
        reconstructed_vision_keys = reconstructed_vision_keys.scatter(-1, pruned_idx_expanded, reconstructed_pruned)

    return pruned_vision_keys, reconstructed_vision_keys


def _compute_visual_attention_mass(queries_recent, full_keys, prompt_seqlen, vision_end):
    q_heads = queries_recent.shape[1]
    kv_heads = full_keys.shape[1]
    if q_heads != kv_heads:
        if q_heads % kv_heads != 0:
            raise ValueError(
                f"GQA head mismatch in attention-shift: q_heads={q_heads}, kv_heads={kv_heads}"
            )
        full_keys = repeat_kv(full_keys, q_heads // kv_heads)
    attn_scores = torch.matmul(queries_recent, full_keys.transpose(-2, -1)) / math.sqrt(queries_recent.shape[-1])
    attn_scores = attn_scores + _build_recent_query_causal_mask(
        recent_query_len=queries_recent.shape[-2],
        total_key_len=full_keys.shape[-2],
        device=full_keys.device,
        dtype=attn_scores.dtype,
    )
    attn_probs = torch.softmax(attn_scores, dim=-1)
    return attn_probs[..., prompt_seqlen:vision_end].sum(dim=-1).mean(dim=-1)


def _save_attention_shift_layer_summary():
    out_dir = _get_attention_shift_dir()
    if not os.path.isdir(out_dir):
        return

    num_layers = getattr(_get_current_cluster(), "num_layers", None)
    if num_layers is None:
        raise RuntimeError(
            "current_kv_cluster.num_layers must be set before saving attention-shift summary"
        )

    summary_rows = []
    for sparsity in SUPPLEMENTARY_MATRIX_SPARSITIES:
        for layer_slot in range(int(num_layers)):
            layer_path = os.path.join(out_dir, f"layer_{layer_slot:02d}.csv")
            if not os.path.exists(layer_path):
                continue

            with open(layer_path, "r", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    if row.get("sparsity") != f"{sparsity:.3f}":
                        continue
                    if row.get("head_idx") != "head_average":
                        continue
                    summary_rows.append([
                        layer_slot,
                        row["sparsity"],
                        row["head_idx"],
                        row["before_pruning"],
                        row["after_pruning"],
                        row["after_reconstruction"],
                        row["delta_pruning"],
                        row["delta_reconstruction"],
                        row["samples"],
                    ])
                    break

    if not summary_rows:
        return

    summary_path = os.path.join(out_dir, "layer_summary.csv")
    with open(summary_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "layer_idx",
            "sparsity",
            "head_average",
            "before_pruning",
            "after_pruning",
            "after_reconstruction",
            "delta_pruning",
            "delta_reconstruction",
            "samples",
        ])
        writer.writerows(summary_rows)


def _save_layer_attention_shift(layer_slot):
    global attention_shift_before_accumulator
    global attention_shift_after_pruning_accumulator
    global attention_shift_after_reconstruction_accumulator
    global attention_shift_seen_sample_counts

    if attention_shift_seen_sample_counts is None:
        return

    seen_samples = int(attention_shift_seen_sample_counts[layer_slot].item())
    if seen_samples <= 0:
        return

    out_dir = _get_attention_shift_dir()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"layer_{layer_slot:02d}.csv")
    with open(out_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "sparsity",
            "head_idx",
            "before_pruning",
            "after_pruning",
            "after_reconstruction",
            "delta_pruning",
            "delta_reconstruction",
            "samples",
        ])

        for sparsity_idx, sparsity in enumerate(SUPPLEMENTARY_MATRIX_SPARSITIES):
            stats_key = (sparsity_idx, layer_slot)
            if stats_key not in attention_shift_before_accumulator:
                continue

            before_avg = attention_shift_before_accumulator[stats_key] / seen_samples
            after_pruning_avg = attention_shift_after_pruning_accumulator[stats_key] / seen_samples
            after_reconstruction_avg = attention_shift_after_reconstruction_accumulator[stats_key] / seen_samples

            for head_idx in range(before_avg.shape[0]):
                writer.writerow([
                    f"{sparsity:.3f}",
                    head_idx,
                    f"{float(before_avg[head_idx]):.10f}",
                    f"{float(after_pruning_avg[head_idx]):.10f}",
                    f"{float(after_reconstruction_avg[head_idx]):.10f}",
                    f"{float(after_pruning_avg[head_idx] - before_avg[head_idx]):.10f}",
                    f"{float(after_reconstruction_avg[head_idx] - before_avg[head_idx]):.10f}",
                    seen_samples,
                ])

            writer.writerow([
                f"{sparsity:.3f}",
                "head_average",
                f"{float(before_avg.mean()):.10f}",
                f"{float(after_pruning_avg.mean()):.10f}",
                f"{float(after_reconstruction_avg.mean()):.10f}",
                f"{float((after_pruning_avg - before_avg).mean()):.10f}",
                f"{float((after_reconstruction_avg - before_avg).mean()):.10f}",
                seen_samples,
            ])

    _save_attention_shift_layer_summary()


def _accumulate_and_maybe_save_attention_shift_calibration(
    q_states,
    kv_states,
    prompt_seqlen=0,
    query_seqlen=0,
    calibration_channel_importance=None,
    num_key_value_groups=1,
):
    global layer_idx
    global attention_shift_before_accumulator
    global attention_shift_after_pruning_accumulator
    global attention_shift_after_reconstruction_accumulator
    global attention_shift_seen_sample_counts
    global attention_shift_last_saved_steps

    if q_states is None or kv_states is None:
        return
    if q_states.dim() != 4 or kv_states.dim() != 4:
        return

    batch_size, num_heads, total_key_len, head_dim = kv_states.shape
    recent_query_len = min(32, query_seqlen)
    if recent_query_len <= 0 or total_key_len <= 0:
        return

    num_layers = getattr(globals().get("current_kv_cluster"), "num_layers", None)
    if num_layers is None:
        raise RuntimeError("current_kv_cluster.num_layers must be set before using attention-shift calibration")
    _initialize_attention_shift_stats(num_layers=num_layers)

    layer_slot = layer_idx % num_layers
    queries_recent = q_states[..., -recent_query_len:, :].detach().to(torch.float32)
    keys_all = kv_states.detach().to(torch.float32)
    channel_importance = _expand_saved_channel_importance(
        calibration_channel_importance,
        num_heads=num_heads,
        head_dim=head_dim,
        num_key_value_groups=num_key_value_groups,
        device=keys_all.device,
        dtype=keys_all.dtype,
    )

    vision_end = total_key_len - query_seqlen if query_seqlen > 0 else total_key_len
    prompt_seqlen = max(0, min(prompt_seqlen, vision_end))
    if vision_end <= prompt_seqlen:
        return
    vision_keys = keys_all[:, :, prompt_seqlen:vision_end, :]

    before_mass = _compute_visual_attention_mass(queries_recent, keys_all, prompt_seqlen, vision_end).sum(dim=0).cpu()

    for sparsity_idx, sparsity in enumerate(SUPPLEMENTARY_MATRIX_SPARSITIES):
        prune_count = min(head_dim, max(0, int(head_dim * sparsity)))
        keep_count = head_dim - prune_count
        if prune_count <= 0 or keep_count <= 0:
            continue

        supplementary_matrix = _get_layer_supplementary_components(layer_slot, sparsity)
        pruned_vision_keys, reconstructed_vision_keys = _apply_attention_shift_reconstruction(
            vision_keys,
            supplementary_matrix,
            num_heads=num_heads,
            num_key_value_groups=num_key_value_groups,
        )

        pruned_full_keys = torch.cat([keys_all[:, :, :prompt_seqlen, :], pruned_vision_keys, keys_all[:, :, vision_end:, :]], dim=-2)
        reconstructed_full_keys = torch.cat([keys_all[:, :, :prompt_seqlen, :], reconstructed_vision_keys, keys_all[:, :, vision_end:, :]], dim=-2)

        after_pruning_mass = _compute_visual_attention_mass(queries_recent, pruned_full_keys, prompt_seqlen, vision_end).sum(dim=0).cpu()
        after_reconstruction_mass = _compute_visual_attention_mass(queries_recent, reconstructed_full_keys, prompt_seqlen, vision_end).sum(dim=0).cpu()

        stats_key = (sparsity_idx, layer_slot)
        if stats_key not in attention_shift_before_accumulator:
            attention_shift_before_accumulator[stats_key] = torch.zeros_like(before_mass)
            attention_shift_after_pruning_accumulator[stats_key] = torch.zeros_like(after_pruning_mass)
            attention_shift_after_reconstruction_accumulator[stats_key] = torch.zeros_like(after_reconstruction_mass)

        attention_shift_before_accumulator[stats_key] += before_mass
        attention_shift_after_pruning_accumulator[stats_key] += after_pruning_mass
        attention_shift_after_reconstruction_accumulator[stats_key] += after_reconstruction_mass

    attention_shift_seen_sample_counts[layer_slot] += batch_size

    seen_samples = int(attention_shift_seen_sample_counts[layer_slot].item())
    current_save_step = seen_samples // ATTENTION_SHIFT_SAVE_INTERVAL
    last_saved_step = int(attention_shift_last_saved_steps[layer_slot].item())
    if seen_samples >= ATTENTION_SHIFT_SAVE_INTERVAL and current_save_step > last_saved_step:
        _save_layer_attention_shift(layer_slot)
        attention_shift_last_saved_steps[layer_slot] = current_save_step


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

def _initialize_attention_kl_stats(num_layers):
    global attention_kl_v_pruning_accumulator
    global attention_kl_v_reconstruction_accumulator
    global attention_kl_full_pruning_accumulator
    global attention_kl_full_reconstruction_accumulator
    global attention_kl_seen_sample_counts
    global attention_kl_last_saved_steps

    expected_shape = (num_layers,)
    if attention_kl_seen_sample_counts is not None and tuple(attention_kl_seen_sample_counts.shape) == expected_shape:
        return
    attention_kl_v_pruning_accumulator = {}
    attention_kl_v_reconstruction_accumulator = {}
    attention_kl_full_pruning_accumulator = {}
    attention_kl_full_reconstruction_accumulator = {}
    attention_kl_seen_sample_counts = torch.zeros(expected_shape, dtype=torch.long, device="cpu")
    attention_kl_last_saved_steps = torch.full(expected_shape, -1, dtype=torch.long, device="cpu")


def _compute_visual_attention_kl(queries_recent, full_keys_ref, full_keys_variant, prompt_seqlen, vision_end):
    """Per-head KL for two normalization conventions, averaged over the
    recent query window.

    Two KL values per (batch, head):
    - kl_v:     vision-only renormalized distributions (shape-shift only).
                Each vector is attn[..., vision] / attn[..., vision].sum().
    - kl_full:  vision-portion contribution to full-sequence KL (attn over
                all keys sums to 1). Retains mass information in addition
                to shape. Computed as Σ_{k ∈ vision} p_ref[k] · log(p_ref[k] / p_var[k]).

    Returns: (kl_v [B, H_q], kl_full [B, H_q]).
    """
    q_heads = queries_recent.shape[1]
    kv_heads = full_keys_ref.shape[1]
    if q_heads != kv_heads:
        if q_heads % kv_heads != 0:
            raise ValueError(f"GQA head mismatch: q={q_heads}, kv={kv_heads}")
        full_keys_ref = repeat_kv(full_keys_ref, q_heads // kv_heads)
        full_keys_variant = repeat_kv(full_keys_variant, q_heads // kv_heads)

    scale = 1.0 / math.sqrt(queries_recent.shape[-1])
    causal = _build_recent_query_causal_mask(
        recent_query_len=queries_recent.shape[-2],
        total_key_len=full_keys_ref.shape[-2],
        device=queries_recent.device,
        dtype=queries_recent.dtype,
    )

    scores_ref = torch.matmul(queries_recent, full_keys_ref.transpose(-2, -1)) * scale + causal
    scores_var = torch.matmul(queries_recent, full_keys_variant.transpose(-2, -1)) * scale + causal
    probs_ref = torch.softmax(scores_ref, dim=-1)
    probs_var = torch.softmax(scores_var, dim=-1)

    # --- Full-sequence normalization: slice vision portion of full p ---
    p_full_ref = probs_ref[..., prompt_seqlen:vision_end]                   # [B, H_q, Q, L_vis]
    p_full_var = probs_var[..., prompt_seqlen:vision_end]
    log_full_ref = p_full_ref.clamp_min(1e-12).log()
    log_full_var = p_full_var.clamp_min(1e-12).log()
    kl_full = (p_full_ref * (log_full_ref - log_full_var)).sum(dim=-1).mean(dim=-1)  # [B, H_q]

    # --- Vision-only renormalization: p_vision / sum ---
    denom_ref = p_full_ref.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    denom_var = p_full_var.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    p_v_ref = p_full_ref / denom_ref
    p_v_var = p_full_var / denom_var
    log_v_ref = p_v_ref.clamp_min(1e-12).log()
    log_v_var = p_v_var.clamp_min(1e-12).log()
    kl_v = (p_v_ref * (log_v_ref - log_v_var)).sum(dim=-1).mean(dim=-1)               # [B, H_q]

    return kl_v, kl_full


def _save_layer_attention_kl(layer_slot):
    """Save per-layer KL CSV + update summary. Records both
    (1) vision-only renormalized KL and (2) full-sequence normalized KL,
    for both pruning-only and after-reconstruction variants."""
    global attention_kl_v_pruning_accumulator
    global attention_kl_v_reconstruction_accumulator
    global attention_kl_full_pruning_accumulator
    global attention_kl_full_reconstruction_accumulator
    global attention_kl_seen_sample_counts

    if attention_kl_seen_sample_counts is None:
        return
    seen = int(attention_kl_seen_sample_counts[layer_slot].item())
    if seen <= 0:
        return

    out_dir = os.path.join(_get_calibration_root_dir(), "attention_kl")
    os.makedirs(out_dir, exist_ok=True)
    layer_path = os.path.join(out_dir, f"layer_{layer_slot:02d}.csv")
    summary_path = os.path.join(out_dir, "layer_summary.csv")

    field_names = [
        "sparsity", "head_idx",
        "kl_v_pruning", "kl_v_reconstruction",
        "kl_full_pruning", "kl_full_reconstruction",
        "samples",
    ]

    import csv as _csv
    rows = []
    for sparsity_idx, sparsity in enumerate(SUPPLEMENTARY_MATRIX_SPARSITIES):
        stats_key = (sparsity_idx, layer_slot)
        if stats_key not in attention_kl_v_pruning_accumulator:
            continue
        kl_v_p = attention_kl_v_pruning_accumulator[stats_key] / seen
        kl_v_r = attention_kl_v_reconstruction_accumulator[stats_key] / seen
        kl_f_p = attention_kl_full_pruning_accumulator[stats_key] / seen
        kl_f_r = attention_kl_full_reconstruction_accumulator[stats_key] / seen
        for h in range(kl_v_p.shape[-1]):
            rows.append({
                "sparsity": f"{sparsity:.3f}",
                "head_idx": h,
                "kl_v_pruning":        f"{kl_v_p[h].item():.10f}",
                "kl_v_reconstruction": f"{kl_v_r[h].item():.10f}",
                "kl_full_pruning":     f"{kl_f_p[h].item():.10f}",
                "kl_full_reconstruction": f"{kl_f_r[h].item():.10f}",
                "samples": seen,
            })
        rows.append({
            "sparsity": f"{sparsity:.3f}",
            "head_idx": "head_average",
            "kl_v_pruning":        f"{kl_v_p.mean().item():.10f}",
            "kl_v_reconstruction": f"{kl_v_r.mean().item():.10f}",
            "kl_full_pruning":     f"{kl_f_p.mean().item():.10f}",
            "kl_full_reconstruction": f"{kl_f_r.mean().item():.10f}",
            "samples": seen,
        })

    with open(layer_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=field_names)
        w.writeheader(); w.writerows(rows)

    summary_fields = [
        "layer_idx", "sparsity", "head_average_tag",
        "kl_v_pruning", "kl_v_reconstruction",
        "kl_full_pruning", "kl_full_reconstruction",
        "samples",
    ]
    existing = {}
    if os.path.exists(summary_path):
        with open(summary_path, newline="") as f:
            for row in _csv.DictReader(f):
                try:
                    key = (int(row["layer_idx"]), float(row["sparsity"]))
                    existing[key] = row
                except (KeyError, ValueError):
                    pass
    for r in rows:
        if r["head_idx"] != "head_average":
            continue
        existing[(layer_slot, float(r["sparsity"]))] = {
            "layer_idx": layer_slot,
            "sparsity": r["sparsity"],
            "head_average_tag": "head_average",
            "kl_v_pruning":        r["kl_v_pruning"],
            "kl_v_reconstruction": r["kl_v_reconstruction"],
            "kl_full_pruning":     r["kl_full_pruning"],
            "kl_full_reconstruction": r["kl_full_reconstruction"],
            "samples": seen,
        }
    with open(summary_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for key in sorted(existing.keys(), key=lambda k: (k[1], k[0])):
            w.writerow(existing[key])


def _accumulate_and_maybe_save_attention_kl_calibration(
    q_states, kv_states, prompt_seqlen=0, query_seqlen=0,
    calibration_channel_importance=None, num_key_value_groups=1,
):
    """Collect KL divergence between full and pruned text-to-vision
    attention distributions. Two variants per (layer, sparsity, head):
    vision-only renormalized (shape) and full-sequence normalized
    (shape + mass)."""
    global layer_idx
    global attention_kl_v_pruning_accumulator
    global attention_kl_v_reconstruction_accumulator
    global attention_kl_full_pruning_accumulator
    global attention_kl_full_reconstruction_accumulator
    global attention_kl_seen_sample_counts
    global attention_kl_last_saved_steps

    if q_states is None or kv_states is None:
        return
    if q_states.dim() != 4 or kv_states.dim() != 4:
        return

    batch_size, num_heads, total_key_len, head_dim = kv_states.shape
    recent_query_len = min(32, query_seqlen)
    if recent_query_len <= 0 or total_key_len <= 0:
        return

    num_layers = getattr(globals().get("current_kv_cluster"), "num_layers", None)
    if num_layers is None:
        raise RuntimeError("current_kv_cluster.num_layers must be set before using attention_kl calibration")
    _initialize_attention_kl_stats(num_layers=num_layers)

    layer_slot = layer_idx % num_layers
    queries_recent = q_states[..., -recent_query_len:, :].detach().to(torch.float32)
    keys_all = kv_states.detach().to(torch.float32)
    _expand_saved_channel_importance(  # raises if missing; ensures channel_importance loaded
        calibration_channel_importance,
        num_heads=num_heads, head_dim=head_dim,
        num_key_value_groups=num_key_value_groups,
        device=keys_all.device, dtype=keys_all.dtype,
    )

    vision_end = total_key_len - query_seqlen if query_seqlen > 0 else total_key_len
    prompt_seqlen = max(0, min(prompt_seqlen, vision_end))
    if vision_end <= prompt_seqlen:
        return
    vision_keys = keys_all[:, :, prompt_seqlen:vision_end, :]

    for sparsity_idx, sparsity in enumerate(SUPPLEMENTARY_MATRIX_SPARSITIES):
        prune_count = min(head_dim, max(0, int(head_dim * sparsity)))
        keep_count = head_dim - prune_count
        if prune_count <= 0 or keep_count <= 0:
            continue

        supplementary_matrix = _get_layer_supplementary_components(layer_slot, sparsity)
        pruned_vision_keys, reconstructed_vision_keys = _apply_attention_shift_reconstruction(
            vision_keys, supplementary_matrix,
            num_heads=num_heads, num_key_value_groups=num_key_value_groups,
        )
        pruned_full = torch.cat([keys_all[:, :, :prompt_seqlen, :], pruned_vision_keys, keys_all[:, :, vision_end:, :]], dim=-2)
        recon_full  = torch.cat([keys_all[:, :, :prompt_seqlen, :], reconstructed_vision_keys, keys_all[:, :, vision_end:, :]], dim=-2)

        kl_v_p, kl_f_p = _compute_visual_attention_kl(queries_recent, keys_all, pruned_full, prompt_seqlen, vision_end)
        kl_v_r, kl_f_r = _compute_visual_attention_kl(queries_recent, keys_all, recon_full,  prompt_seqlen, vision_end)
        kl_v_p = kl_v_p.sum(dim=0).cpu()
        kl_v_r = kl_v_r.sum(dim=0).cpu()
        kl_f_p = kl_f_p.sum(dim=0).cpu()
        kl_f_r = kl_f_r.sum(dim=0).cpu()

        stats_key = (sparsity_idx, layer_slot)
        if stats_key not in attention_kl_v_pruning_accumulator:
            attention_kl_v_pruning_accumulator[stats_key] = torch.zeros_like(kl_v_p)
            attention_kl_v_reconstruction_accumulator[stats_key] = torch.zeros_like(kl_v_r)
            attention_kl_full_pruning_accumulator[stats_key] = torch.zeros_like(kl_f_p)
            attention_kl_full_reconstruction_accumulator[stats_key] = torch.zeros_like(kl_f_r)
        attention_kl_v_pruning_accumulator[stats_key] += kl_v_p
        attention_kl_v_reconstruction_accumulator[stats_key] += kl_v_r
        attention_kl_full_pruning_accumulator[stats_key] += kl_f_p
        attention_kl_full_reconstruction_accumulator[stats_key] += kl_f_r

    attention_kl_seen_sample_counts[layer_slot] += batch_size
    seen = int(attention_kl_seen_sample_counts[layer_slot].item())
    step = seen // ATTENTION_KL_SAVE_INTERVAL
    last = int(attention_kl_last_saved_steps[layer_slot].item())
    if seen >= ATTENTION_KL_SAVE_INTERVAL and step > last:
        _save_layer_attention_kl(layer_slot)
        attention_kl_last_saved_steps[layer_slot] = step


def _initialize_supplementary_matrix_stats(num_layers):
    global supplementary_matrix_gram_accumulator
    global supplementary_matrix_cross_accumulator
    global supplementary_matrix_keep_indices
    global supplementary_matrix_pruned_indices
    global supplementary_matrix_target_sq_accumulator
    global supplementary_matrix_row_count_accumulator
    global supplementary_matrix_q_gram_accumulator
    global supplementary_matrix_seen_sample_counts
    global supplementary_matrix_last_saved_steps

    expected_shape = (len(SUPPLEMENTARY_MATRIX_SPARSITIES), num_layers)
    if supplementary_matrix_seen_sample_counts is not None and tuple(supplementary_matrix_seen_sample_counts.shape) == expected_shape:
        return

    supplementary_matrix_gram_accumulator = {}
    supplementary_matrix_cross_accumulator = {}
    supplementary_matrix_keep_indices = {}
    supplementary_matrix_pruned_indices = {}
    supplementary_matrix_target_sq_accumulator = {}
    supplementary_matrix_row_count_accumulator = {}
    supplementary_matrix_q_gram_accumulator = {}
    supplementary_matrix_q_position_count_accumulator = {}
    supplementary_matrix_seen_sample_counts = torch.zeros(expected_shape, dtype=torch.long, device="cpu")
    supplementary_matrix_last_saved_steps = torch.full(expected_shape, -1, dtype=torch.long, device="cpu")


def _save_layer_supplementary_matrix(layer_slot, sparsity_idx):
    global supplementary_matrix_gram_accumulator
    global supplementary_matrix_cross_accumulator
    global supplementary_matrix_keep_indices
    global supplementary_matrix_pruned_indices
    global supplementary_matrix_target_sq_accumulator
    global supplementary_matrix_row_count_accumulator
    global supplementary_matrix_q_gram_accumulator
    global supplementary_matrix_q_position_count_accumulator
    global supplementary_matrix_seen_sample_counts

    stats_key = (sparsity_idx, layer_slot)
    if stats_key not in supplementary_matrix_gram_accumulator or supplementary_matrix_seen_sample_counts is None:
        return

    seen_samples = int(supplementary_matrix_seen_sample_counts[sparsity_idx, layer_slot].item())
    if seen_samples <= 0:
        return

    gram = supplementary_matrix_gram_accumulator[stats_key]
    cross = supplementary_matrix_cross_accumulator[stats_key]
    keep_idx = supplementary_matrix_keep_indices[stats_key]
    pruned_idx = supplementary_matrix_pruned_indices[stats_key]
    target_sq = supplementary_matrix_target_sq_accumulator[stats_key]
    row_count = supplementary_matrix_row_count_accumulator[stats_key]

    keep_dim_plus_bias = gram.shape[-1]
    keep_dim = keep_dim_plus_bias - 1
    prune_dim = cross.shape[-1]
    if keep_dim <= 0 or prune_dim == 0:
        return

    q_gram_full = supplementary_matrix_q_gram_accumulator.get(layer_slot)

    objective = str(SUPPLEMENTARY_MATRIX_OBJECTIVE).strip().lower()
    if objective not in ("k_mse", "qk_weighted"):
        raise ValueError(
            f"SUPPLEMENTARY_MATRIX_OBJECTIVE must be 'k_mse' or 'qk_weighted', got {SUPPLEMENTARY_MATRIX_OBJECTIVE!r}"
        )
    if objective == "qk_weighted" and q_gram_full is None:
        raise RuntimeError(
            f"Q^T Q accumulator missing for layer {layer_slot} under qk_weighted objective. "
            "Re-collect supplementary_matrix calibration with the updated helper "
            "that passes q_states, or switch SUPPLEMENTARY_MATRIX_OBJECTIVE to 'k_mse'."
        )

    if objective == "k_mse":
        # Original: column-wise ridge minimizing ||K_pruned - X β||²
        regularizer = torch.eye(keep_dim_plus_bias, dtype=gram.dtype, device=gram.device)
        regularizer[-1, -1] = 0
        params = torch.linalg.solve(
            gram + SUPPLEMENTARY_MATRIX_RIDGE_LAMBDA * regularizer.unsqueeze(0),
            cross,
        )
    else:  # qk_weighted
        # Restrict G_Q to pruned channel indices per head: (H_kv, |P|, |P|).
        D_full = q_gram_full.shape[-1]
        pruned_idx_long = pruned_idx.to(q_gram_full.device).long()
        idx_rows = pruned_idx_long.unsqueeze(-1).expand(-1, -1, D_full)
        idx_cols = pruned_idx_long.unsqueeze(-2).expand(-1, prune_dim, -1)
        q_gram_pruned = torch.gather(
            torch.gather(q_gram_full, dim=-2, index=idx_rows),
            dim=-1,
            index=idx_cols,
        )
        q_gram_pruned = 0.5 * (q_gram_pruned + q_gram_pruned.transpose(-2, -1))

        # Eigendecompose G_Q (ascending). (H_kv, |P|), (H_kv, |P|, |P|).
        lam_g, U = torch.linalg.eigh(q_gram_pruned)
        lam_g = lam_g.clamp_min(0.0)  # G_Q is PSD; clip fp noise

        # Valid-direction mask: skip eigenvectors where Q has ~zero mass.
        max_lam = lam_g.amax(dim=-1, keepdim=True).clamp_min(1e-30)
        valid_mask = lam_g > max_lam * 1e-10

        # Transform cross into G_Q's eigenbasis: cross̃ = cross @ U.
        cross_tilde = torch.matmul(cross, U)

        # Per-eigen-column closed form via gram's eigendecomposition:
        #   (Λ_k · gram + λ·I) β̃[:, k] = Λ_k · cross̃[:, k]
        # ⇒ β̃[:, k] = V · ((d + λ/Λ_k)^-1 ⊙ (V^T cross̃[:, k]))
        d_g, V = torch.linalg.eigh(0.5 * (gram + gram.transpose(-2, -1)))
        d_g = d_g.clamp_min(0.0)  # gram is PSD; clip fp noise
        Vt_cross_tilde = torch.matmul(V.transpose(-2, -1), cross_tilde)
        # Replace invalid Λ_k with max_lam so α stays finite and well-conditioned;
        # we'll zero those β̃ columns out via valid_mask afterward.
        lam_for_alpha = torch.where(valid_mask, lam_g, max_lam.expand_as(lam_g))
        alphas = SUPPLEMENTARY_MATRIX_RIDGE_LAMBDA / lam_for_alpha.clamp_min(1e-30)
        # Cap gram's condition number by floor-clamping its eigenvalues to a
        # small fraction of the largest. Without this, rank-deficient gram
        # (common at low calibration sample counts) produces tiny `denom` in
        # directions where Λ is large, blowing β̃ up to 1e12+.
        d_g_max = d_g.amax(dim=-1, keepdim=True).clamp_min(1e-30)
        d_g_floor = d_g_max * 1e-6                                # condition number cap ≈ 1e6
        d_g_safe = torch.maximum(d_g, d_g_floor.expand_as(d_g))
        denom = (d_g_safe.unsqueeze(-1) + alphas.unsqueeze(-2)).clamp_min(1e-30)
        beta_tilde = torch.matmul(V, Vt_cross_tilde / denom)
        beta_tilde = torch.where(valid_mask.unsqueeze(-2), beta_tilde, torch.zeros_like(beta_tilde))
        # Final NaN/Inf guard — any numerical hiccup gets replaced with zero
        # (equivalent to "don't reconstruct that direction").
        beta_tilde = torch.nan_to_num(beta_tilde, nan=0.0, posinf=0.0, neginf=0.0)

        # Back to original basis: β = β̃ U^T.
        params = torch.matmul(beta_tilde, U.transpose(-2, -1))

    weights = params[:, :-1, :]
    bias = params[:, -1, :]

    # K-MSE residual (always computed; interpretable as average per-element
    # squared error of raw K reconstruction over calibration rows).
    linear_term = torch.einsum("hkp,hkp->hp", params, cross)
    gram_params = torch.matmul(gram, params)
    quadratic_term = torch.einsum("hkp,hkp->hp", params, gram_params)
    sse = (target_sq - 2.0 * linear_term + quadratic_term).clamp_min(0.0)
    reconstruction_target_mse = (target_sq.sum() / max(int(row_count), 1) / max(target_sq.shape[-1], 1)).item()
    reconstruction_residual_mse = (sse.sum() / max(int(row_count), 1) / max(sse.shape[-1], 1)).item()

    # QK-weighted residual (only if q_gram available). Reports:
    #   target_mass = tr(G_Q diag(Y^T Y))     — approximation using only the
    #                                           target_sq accumulator diagonal.
    #   residual_mass = weighted residual     — consistent approximation.
    # Ratio residual/target is the correct attention-oriented quality metric.
    reconstruction_qk_target_mass = None
    reconstruction_qk_residual_mass = None
    if q_gram_full is not None:
        if objective == "k_mse":
            D_full = q_gram_full.shape[-1]
            pruned_idx_long = pruned_idx.to(q_gram_full.device).long()
            idx_rows = pruned_idx_long.unsqueeze(-1).expand(-1, -1, D_full)
            idx_cols = pruned_idx_long.unsqueeze(-2).expand(-1, prune_dim, -1)
            q_gram_pruned = torch.gather(
                torch.gather(q_gram_full, dim=-2, index=idx_rows),
                dim=-1,
                index=idx_cols,
            )
            q_gram_pruned = 0.5 * (q_gram_pruned + q_gram_pruned.transpose(-2, -1))
            lam_g, U = torch.linalg.eigh(q_gram_pruned)
            cross_tilde = torch.matmul(cross, U)
            beta_tilde = torch.matmul(params, U)  # β̃ = β U
        target_sq_tilde_diag = torch.einsum("hp,hpk->hk", target_sq, U.pow(2))  # (H_kv, |P|)
        qk_target_energy = (lam_g * target_sq_tilde_diag).sum(dim=-1)
        linear_qk = torch.einsum("hkp,hkp->hp", beta_tilde, cross_tilde)
        gram_beta_tilde = torch.matmul(gram, beta_tilde)
        quadratic_qk = torch.einsum("hkp,hkp->hp", beta_tilde, gram_beta_tilde)
        sse_qk_weighted = (lam_g * (target_sq_tilde_diag - 2.0 * linear_qk + quadratic_qk).clamp_min(0.0)).sum(dim=-1)
        # Normalize by (row_count × total Q positions) so absolute values are
        # per-(K-pos, Q-pos) squared-logit units (comparable to an average
        # squared attention score contribution from pruned channels). The
        # ratio residual/target is unchanged by this extra factor.
        q_pos_count = supplementary_matrix_q_position_count_accumulator.get(layer_slot, 0)
        qk_norm = max(int(row_count), 1) * max(int(q_pos_count), 1)
        reconstruction_qk_target_mass = qk_target_energy.sum().item() / qk_norm
        reconstruction_qk_residual_mass = sse_qk_weighted.sum().item() / qk_norm

    sparsity = SUPPLEMENTARY_MATRIX_SPARSITIES[sparsity_idx]
    supplementary_matrix_dir = _get_supplementary_matrix_dir()
    out_dir = os.path.join(
        supplementary_matrix_dir,
        f"sparsity_{sparsity:.3f}_lambda_{SUPPLEMENTARY_MATRIX_RIDGE_LAMBDA:g}",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"layer_{layer_slot:02d}.pt")
    torch.save(
        {
            "layer_idx": layer_slot,
            "sparsity": sparsity,
            "samples": seen_samples,
            "ridge_lambda": SUPPLEMENTARY_MATRIX_RIDGE_LAMBDA,
            "weight": weights,
            "bias": bias,
            "keep_idx": keep_idx,
            "pruned_idx": pruned_idx,
            "reconstruction_target_mse": reconstruction_target_mse,
            "reconstruction_residual_mse": reconstruction_residual_mse,
            "reconstruction_qk_target_mass": reconstruction_qk_target_mass,
            "reconstruction_qk_residual_mass": reconstruction_qk_residual_mass,
            "objective": objective,
        },
        out_path,
    )
    return {
        "weight": weights,
        "bias": bias,
        "keep_idx": keep_idx,
        "pruned_idx": pruned_idx,
    }


def _accumulate_and_maybe_save_supplementary_matrix(keys_vision, channel_scores, q_states=None, num_key_value_groups=1):
    global layer_idx
    global supplementary_matrix_gram_accumulator
    global supplementary_matrix_cross_accumulator
    global supplementary_matrix_keep_indices
    global supplementary_matrix_pruned_indices
    global supplementary_matrix_target_sq_accumulator
    global supplementary_matrix_row_count_accumulator
    global supplementary_matrix_q_gram_accumulator
    global supplementary_matrix_q_position_count_accumulator
    global supplementary_matrix_seen_sample_counts
    global supplementary_matrix_last_saved_steps

    if keys_vision is None or channel_scores is None:
        return
    if keys_vision.dim() != 4 or channel_scores.dim() != 3:
        return

    num_layers = getattr(globals().get("current_kv_cluster"), "num_layers", None)
    if num_layers is None:
        raise RuntimeError("current_kv_cluster.num_layers must be set before using supplementary-matrix calibration")

    batch_size, num_heads, vision_seqlen, head_dim = keys_vision.shape
    _initialize_supplementary_matrix_stats(num_layers=num_layers)

    layer_slot = layer_idx % num_layers
    keys_vision = keys_vision.detach().to(torch.float32)
    channel_scores = channel_scores.detach().to(torch.float32)

    # Accumulate Σ Q^T Q at H_kv granularity, summed over the recent query
    # window (matches how channel_importance/Q^2·K^2 uses recent queries).
    # Needed for the QK-weighted ridge at save time.
    if q_states is not None and q_states.dim() == 4:
        query_window = min(32, q_states.shape[2])
        if query_window > 0:
            q_recent = q_states[..., -query_window:, :].detach().to(torch.float32)
            bsz_q, num_q_heads, L_q, D_q = q_recent.shape
            if num_q_heads == num_heads:
                q_flat = q_recent
            elif num_q_heads % num_heads == 0:
                group_size = num_q_heads // num_heads
                q_flat = q_recent.reshape(bsz_q, num_heads, group_size, L_q, D_q).reshape(
                    bsz_q, num_heads, group_size * L_q, D_q,
                )
            else:
                raise ValueError(
                    f"Cannot group {num_q_heads} query heads to {num_heads} KV heads for Q^T Q accumulation"
                )
            q_gram_contrib = torch.matmul(q_flat.transpose(-2, -1), q_flat).sum(dim=0).cpu()  # (H_kv, D, D)
            if layer_slot not in supplementary_matrix_q_gram_accumulator:
                supplementary_matrix_q_gram_accumulator[layer_slot] = torch.zeros_like(q_gram_contrib)
            supplementary_matrix_q_gram_accumulator[layer_slot] += q_gram_contrib
            # Track total Q positions that contributed to this layer's G_Q, so
            # qk_target/residual_mass can be normalized to per-(Q-pos, K-pos) units.
            q_positions_here = int(q_flat.shape[0] * q_flat.shape[2])  # bsz * (group_size * L_q)
            supplementary_matrix_q_position_count_accumulator[layer_slot] = (
                supplementary_matrix_q_position_count_accumulator.get(layer_slot, 0) + q_positions_here
            )

    # keys_vision and channel_scores are already at H_kv granularity (the
    # upstream kv_states has num_kv_heads heads; the GQA collapse already
    # happened in `key_pruner_query_driven`). No further grouping needed.

    for sparsity_idx, sparsity in enumerate(SUPPLEMENTARY_MATRIX_SPARSITIES):
        prune_count = min(head_dim, max(0, int(head_dim * sparsity)))
        keep_count = head_dim - prune_count
        if prune_count <= 0 or keep_count <= 0:
            continue

        pruned_idx = torch.topk(channel_scores, prune_count, dim=-1, largest=False, sorted=False).indices.sort(dim=-1).values
        pruned_mask = torch.zeros_like(channel_scores, dtype=torch.bool)
        pruned_mask.scatter_(-1, pruned_idx, True)
        keep_mask = ~pruned_mask
        keep_idx = _get_sorted_mask_indices(keep_mask, selected_dim=keep_count)

        source = torch.gather(
            keys_vision,
            dim=-1,
            index=keep_idx.unsqueeze(2).expand(-1, -1, vision_seqlen, -1),
        )
        target = torch.gather(
            keys_vision,
            dim=-1,
            index=pruned_idx.unsqueeze(2).expand(-1, -1, vision_seqlen, -1),
        )
        ones = torch.ones(batch_size, num_heads, vision_seqlen, 1, dtype=source.dtype, device=source.device)
        source_augmented = torch.cat([source, ones], dim=-1)

        gram = torch.matmul(source_augmented.transpose(-2, -1), source_augmented).sum(dim=0).cpu()
        cross = torch.matmul(source_augmented.transpose(-2, -1), target).sum(dim=0).cpu()
        target_sq = target.pow(2).sum(dim=2).sum(dim=0).cpu()
        row_count = batch_size * vision_seqlen

        stats_key = (sparsity_idx, layer_slot)
        if stats_key not in supplementary_matrix_gram_accumulator:
            supplementary_matrix_gram_accumulator[stats_key] = torch.zeros_like(gram)
            supplementary_matrix_cross_accumulator[stats_key] = torch.zeros_like(cross)
            supplementary_matrix_keep_indices[stats_key] = keep_idx[0].detach().cpu()
            supplementary_matrix_pruned_indices[stats_key] = pruned_idx[0].detach().cpu()
            supplementary_matrix_target_sq_accumulator[stats_key] = torch.zeros_like(target_sq)
            supplementary_matrix_row_count_accumulator[stats_key] = 0

        supplementary_matrix_gram_accumulator[stats_key] += gram
        supplementary_matrix_cross_accumulator[stats_key] += cross
        supplementary_matrix_target_sq_accumulator[stats_key] += target_sq
        supplementary_matrix_row_count_accumulator[stats_key] += row_count
        supplementary_matrix_seen_sample_counts[sparsity_idx, layer_slot] += batch_size

        seen_samples = int(supplementary_matrix_seen_sample_counts[sparsity_idx, layer_slot].item())
        current_save_step = min(seen_samples, SUPPLEMENTARY_MATRIX_TARGET_SAMPLES) // SUPPLEMENTARY_MATRIX_SAVE_INTERVAL
        last_saved_step = int(supplementary_matrix_last_saved_steps[sparsity_idx, layer_slot].item())
        if seen_samples >= SUPPLEMENTARY_MATRIX_SAVE_INTERVAL and current_save_step > last_saved_step:
            _save_layer_supplementary_matrix(layer_slot, sparsity_idx)
            supplementary_matrix_last_saved_steps[sparsity_idx, layer_slot] = current_save_step

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


def _rank_stats_from_singular_values(sv):
    """sv shape [H, D], non-negative values (sort order doesn't matter).
    Returns (stable_rank, entropy_rank, participation) per head, shape [H].
    Treats `sv` as "singular values" of whatever signal we measured."""
    fro_sq = sv.pow(2).sum(dim=-1)
    spec_sq = sv.max(dim=-1).values.pow(2).clamp_min(1e-20)
    stable_rank = fro_sq / spec_sq
    p = sv / sv.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    entropy = -(p * p.clamp_min(1e-20).log()).sum(dim=-1)
    entropy_rank = torch.exp(entropy)
    participation = fro_sq.pow(2) / sv.pow(4).sum(dim=-1).clamp_min(1e-20)
    return stable_rank, entropy_rank, participation


def _compute_effective_rank_stats(K):
    """Effective rank of matrix K[H, L, D] via singular values of K^T K."""
    gram = torch.matmul(K.transpose(-2, -1), K)
    eigvals = torch.linalg.eigvalsh(gram).flip(-1).clamp_min(0.0)
    sv = eigvals.sqrt()
    return _rank_stats_from_singular_values(sv)


def _compute_key_singular_values(K):
    """Return per-head singular values of K[H, L, D], sorted descending.
    Shape [H, D]."""
    gram = torch.matmul(K.transpose(-2, -1), K)
    eigvals = torch.linalg.eigvalsh(gram).flip(-1).clamp_min(0.0)
    return eigvals.sqrt()


def _save_layer_singular_value_curves(
    keys_vision,
    q_states,
    layer_slot,
    num_key_value_groups,
    query_window_size,
    num_layers=None,
):
    """Dump per-layer, per-head singular-value decay curves for K and for the
    Q·K^T channel-contribution spectrum (w_c = ||Q[:,c]|| · ||K[:,c]||).

    File: {result_root}/singular_values/layer_NN.csv
    Columns: head_idx, rank, sv_k_only, sv_qk_contrib

    After saving the final layer (layer_slot == num_layers - 1), blocks with
    `input("here")` so the process pauses for inspection. Pass `num_layers`
    to enable this behavior; if None, no blocking.
    """
    global _sv_curves_saved_layers
    if layer_slot in _sv_curves_saved_layers:
        return
    if keys_vision is None or keys_vision.dim() != 4:
        return

    K = keys_vision[0].detach().to(torch.float32)
    num_heads, vision_seqlen, head_dim = K.shape
    if vision_seqlen == 0 or head_dim == 0:
        return

    sv_k = _compute_key_singular_values(K)  # [H_kv, D], sorted desc

    sv_qk = None
    if q_states is not None and q_states.dim() == 4:
        query_window = min(int(query_window_size), q_states.shape[2])
        if query_window > 0:
            q_recent = q_states[..., -query_window:, :].detach().to(torch.float32)
            q_norm_sq = q_recent.pow(2).mean(dim=2)
            bsz_q, num_q_heads, _ = q_norm_sq.shape
            if num_q_heads != num_heads and num_q_heads % num_heads == 0:
                q_norm_sq = q_norm_sq.view(
                    bsz_q, num_heads, num_q_heads // num_heads, head_dim,
                ).mean(dim=2)
            k_norm_sq = keys_vision.detach().to(torch.float32).pow(2).mean(dim=2)
            w = (q_norm_sq * k_norm_sq).clamp_min(0.0).sqrt()[0]  # [H_kv, D]
            sv_qk = w.sort(dim=-1, descending=True).values

    import csv as _csv
    if isinstance(DEBUG_SV_CURVES_DIR, str) and DEBUG_SV_CURVES_DIR:
        out_dir = DEBUG_SV_CURVES_DIR
    else:
        # Default: {result_root}/mmstar_calibration_dominant_ratio_X.XX/singular_values/
        # — co-located with the other calibration outputs (channel_importance,
        # attention_shift, supplementary_matrix) for the same dominant_ratio.
        out_dir = os.path.join(_get_calibration_root_dir(), "singular_values")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"layer_{layer_slot:02d}.csv")

    with open(out_path, "w", newline="") as f:
        w = _csv.writer(f)
        header = ["head_idx", "rank", "sv_k_only"]
        if sv_qk is not None:
            header.append("sv_qk_contrib")
        w.writerow(header)
        sv_k_cpu = sv_k.cpu()
        sv_qk_cpu = sv_qk.cpu() if sv_qk is not None else None
        for h in range(num_heads):
            for r in range(head_dim):
                row = [h, r, f"{sv_k_cpu[h, r].item():.6g}"]
                if sv_qk_cpu is not None:
                    row.append(f"{sv_qk_cpu[h, r].item():.6g}")
                w.writerow(row)

    _sv_curves_saved_layers.add(layer_slot)
    import sys as _sys
    msg = f"[sv_curve] saved layer {layer_slot:02d} → {out_path}"
    print(msg, flush=True)
    _sys.stderr.write(msg + "\n")
    _sys.stderr.flush()

    # Block after the last layer so the user can inspect the CSVs.
    if num_layers is not None and int(num_layers) > 0 and layer_slot == int(num_layers) - 1:
        input("[sv_curve] all layers saved. Press Enter to continue... ")


def _format_rank_line(tag, layer_slot, stable_rank, entropy_rank, participation, num_heads, vision_seqlen, head_dim):
    return (
        f"[eff_rank/{tag:<10}] layer={layer_slot:02d}  "
        f"stable(mean/min/max)={stable_rank.mean():.2f}/{stable_rank.min():.2f}/{stable_rank.max():.2f}  "
        f"entropy(mean/min/max)={entropy_rank.mean():.2f}/{entropy_rank.min():.2f}/{entropy_rank.max():.2f}  "
        f"participation(mean/min/max)={participation.mean():.2f}/{participation.min():.2f}/{participation.max():.2f}  "
        f"(H={num_heads}, L_vis={vision_seqlen}, D={head_dim})"
    )


def _print_key_effective_rank_and_maybe_block(keys_vision, q_states=None, num_key_value_groups=1, query_window_size=32):
    """Prefill debug hook: print per-head effective-rank metrics for the
    current layer, then block via input() after the final layer. Uses the
    first batch sample. Prints up to two lines per layer:

      k_only     — effective rank of K (post-RoPE vision keys).
      qk_contrib — effective number of channels that actually contribute to
                   Q·K^T, measured via pseudo-singular-values
                   w_c = ||Q[:, c]|| · ||K[:, c]|| per head and channel.
    """
    global layer_idx
    if keys_vision is None or keys_vision.dim() != 4:
        return
    num_layers = getattr(globals().get("current_kv_cluster"), "num_layers", None)
    if num_layers is None or num_layers <= 0:
        return

    layer_slot = layer_idx % num_layers

    K = keys_vision[0].detach().to(torch.float32)             # [H_kv, L_vis, D]
    num_heads, vision_seqlen, head_dim = K.shape
    if vision_seqlen == 0 or head_dim == 0:
        return

    stable_k, entropy_k, participation_k = _compute_effective_rank_stats(K)
    print(
        _format_rank_line("k_only", layer_slot, stable_k, entropy_k, participation_k, num_heads, vision_seqlen, head_dim),
        flush=True,
    )

    if q_states is not None and q_states.dim() == 4:
        query_window = min(int(query_window_size), q_states.shape[2])
        if query_window > 0:
            q_recent = q_states[..., -query_window:, :].detach().to(torch.float32)
            q_norm_sq = q_recent.pow(2).mean(dim=2)                    # [B, H_q, D]
            bsz_q, num_q_heads, _ = q_norm_sq.shape
            if num_q_heads != num_heads and num_q_heads % num_heads == 0:
                q_norm_sq = q_norm_sq.view(
                    bsz_q, num_heads, num_q_heads // num_heads, head_dim,
                ).mean(dim=2)                                          # [B, H_kv, D]
            k_norm_sq = keys_vision.detach().to(torch.float32).pow(2).mean(dim=2)  # [B, H_kv, D]
            w = (q_norm_sq * k_norm_sq).clamp_min(0.0).sqrt()[0]                   # [H_kv, D]
            stable_qk, entropy_qk, participation_qk = _rank_stats_from_singular_values(w)
            print(
                _format_rank_line("qk_contrib", layer_slot, stable_qk, entropy_qk, participation_qk, num_heads, vision_seqlen, head_dim),
                flush=True,
            )

    if layer_slot == num_layers - 1:
        input("here")


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


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


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
