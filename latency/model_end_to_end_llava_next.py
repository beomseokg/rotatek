"""End-to-end latency benchmark for channel-pruning methods on LLaVA-NeXT.

Mirrors `model_end_to_end_internvl.py` (same SUMMARY format, same hooks
infrastructure) but adapted for `llava-hf/llama3-llava-next-8b-hf`.
Inputs are synthetic — random `inputs_embeds` of layout
`[prompt(30) | vision(--prefill_length) | text(30)]` fed directly to the
LlamaForCausalLM backbone (vision tower bypassed). Only the vision span
is channel-pruned.

Usage:
  python -m latency.model_end_to_end_llava_next \\
      --methods full,think,spark,rotatek \\
      --prefill_length 8k,16k,32k,64k \\
      --batch_sizes 1,2,4 \\
      --decoding_length 128 \\
      --channel_ratio 0.75
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from typing import Dict, List, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:512",
)

import torch
torch.set_num_threads(8)

# Ensure lmms-eval imports work regardless of cwd
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
LMMS_PKG = os.path.join(_REPO, "lmms-eval")
if LMMS_PKG not in sys.path:
    sys.path.insert(0, LMMS_PKG)


PRETRAINED = "llava-hf/llama3-llava-next-8b-hf"
DEFAULT_LOG_DIR = os.path.join(_REPO, "results", "latency_breakdown")

# Synthetic-input framing: [prompt | vision | text]. Vision spans
# `--prefill_length` tokens (the channel-pruned range); prompt / text are
# fixed framing (not pruned). Total prefill = vision + 60.
PROMPT_TOKENS = 30
TEXT_TOKENS = 30

# LLaMA3-8B numerical config (LLaVA-NeXT's text backbone) — used for
# theoretical KV-cache size in the SUMMARY.
LLAVA_NEXT_CFG = {
    "num_layers": 32,
    "num_kv_heads": 8,    # LLaMA3-8B uses GQA with 8 KV heads
    "head_dim": 128,
    "dtype_bytes": 2,     # bf16
}


class _Tee:
    """Mirror writes to multiple file-like targets (stdout + log file)."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, s):
        for st in self._streams:
            st.write(s)
            st.flush()
    def flush(self):
        for st in self._streams:
            st.flush()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_full_llava_next():
    """No-compression baseline. Uses the same `llava_next_visionzip`
    adapter as the compressed methods, with token pruning OFF and
    channel pruning OFF (channel_ratio=0). Implemented as
    `channel_method=think, channel_ratio=0`.
    """
    return _load_visionzip("think", "0.0")


def _load_visionzip(method: str, channel_ratio: str):
    """LLaMA3-LLaVA-NeXT-8B with **only** channel pruning enabled.

    Reuses the `llava_next_visionzip` adapter for its patch chain (matches
    production), with VisionZip token pruning explicitly disabled
    (dominant=1.0, contextual=0.0) so the sole measured intervention is
    the channel-pruning method.
    """
    from lmms_eval import models
    print(f"[load] LLaMA3-LLaVA-NeXT-8B + channel pruning [{method}, ratio={channel_ratio}] ...", flush=True)
    model_args = (
        f"pretrained={PRETRAINED},"
        f"visionzip_dominant_ratio=1.0,"
        f"visionzip_contextual_ratio=0.0,"
        f"attn_implementation=flash_attention_2,"
        f"channel_ratio={channel_ratio},"
        f"channel_method={method},"
        f"layer_adaptive_channel_budget=False,"
        f"channel_reconstruction=off,"
        f"reconstruction_constant=0.1,"
        f"custom_kernel=False,"
        f"decode_attention_backend=triton,"
        f"calibration_mode=off,"
        f"offline_calibration_tasks=channel_importance"
    )
    LM = models.get_model("llava_next_visionzip", force_simple=False)
    lm = LM.create_from_arg_string(
        model_args, {"batch_size": 1, "max_batch_size": None, "device": "cuda:0"},
    )
    _disable_grad_ckpt(lm.model)
    return lm


def _attention_config(model):
    """Return the config that LlamaVisionZipAttention sees as `self.config`.
    For LLaVA-Next that's `model.config.text_config` (the inner LlamaConfig).
    """
    return getattr(model.config, "text_config", model.config)


_DECODE_BREAKDOWN_KEYS = (
    "qkv_proj_ms",
    "rope_ms",
    "cache_update_ms",
    "recovery_ms",
    "cat_ms",
    "custom_decode_kernel_ms",
    "transpose_ms",
    "fa2_ms",
    "attn_out_proj_ms",
)


def _aggregate_decode_breakdown(model) -> Dict[int, Dict[str, float]]:
    """Average per-step decode timings into per-layer dicts.

    Reads from `_attention_config(model)._decode_timings_by_layer` (populated
    by LlamaVisionZipAttention.forward when `_decode_profile=True`).
    """
    cfg = _attention_config(model)
    raw = getattr(cfg, "_decode_timings_by_layer", None) or {}
    out: Dict[int, Dict[str, float]] = {}
    for layer_idx, t in raw.items():
        if not t.get("qkv_proj_ms"):
            continue
        n = len(t["qkv_proj_ms"])
        avgs = {
            k: (sum(t.get(k, [])) / len(t[k])) if t.get(k) else 0.0
            for k in _DECODE_BREAKDOWN_KEYS
        }
        avgs["total_per_step_ms"] = sum(avgs.values())
        avgs["num_steps"] = n
        out[layer_idx] = {
            k: round(v, 4) if isinstance(v, float) else v for k, v in avgs.items()
        }
    return out


def _print_breakdown_table(breakdown_by_layer: Dict[int, Dict[str, float]]):
    """Print decode-stage breakdown summarized as mean±std across layers.

    Each row in `breakdown_by_layer` is a per-layer average over decode steps;
    we then aggregate across layers (Tukey-trim already happens at the layer
    timing level — here we just take population mean and std).
    """
    if not breakdown_by_layer:
        print("[breakdown] no decode timings recorded", flush=True)
        return
    layer_ids = sorted(breakdown_by_layer.keys())
    n_layers = len(layer_ids)
    n_steps = breakdown_by_layer[layer_ids[0]].get("num_steps", "?")
    keys = _DECODE_BREAKDOWN_KEYS + ("total_per_step_ms",)
    print(
        f"[breakdown] decode per-token ms (mean ± std across "
        f"{n_layers} layers, {n_steps} steps each):",
        flush=True,
    )
    for k in keys:
        vals = [breakdown_by_layer[lid].get(k, 0.0) for lid in layer_ids]
        m, s = _mean_std(vals)
        label = k.replace("_ms", "").replace("_per_step", "")
        print(f"  {label:<24}  {m:>8.3f} ± {s:>6.3f} ms", flush=True)


def _disable_grad_ckpt(model):
    """Belt-and-suspenders: turn off gradient checkpointing on the loaded
    model. We're in inference; checkpointing only inflates memory."""
    candidates = [model]
    if hasattr(model, "language_model"):
        candidates.append(model.language_model)
        if hasattr(model.language_model, "model"):
            candidates.append(model.language_model.model)
    if hasattr(model, "vision_tower"):
        candidates.append(model.vision_tower)
    for m in candidates:
        if hasattr(m, "gradient_checkpointing_disable"):
            try:
                m.gradient_checkpointing_disable()
            except Exception:
                pass
        if hasattr(m, "gradient_checkpointing"):
            try:
                m.gradient_checkpointing = False
            except Exception:
                pass


def _teardown():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _gpu_mem_str() -> str:
    if not torch.cuda.is_available():
        return "(no CUDA)"
    a = torch.cuda.memory_allocated() / 1024**3
    r = torch.cuda.memory_reserved() / 1024**3
    p = torch.cuda.max_memory_allocated() / 1024**3
    return f"alloc={a:.2f}GB peak_alloc={p:.2f}GB reserved={r:.2f}GB"


def _reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _model_param_bytes(model) -> int:
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total


def _kv_cache_bytes(method: str, channel_ratio: float,
                    input_tokens: int, output_tokens: int,
                    vision_tokens: int) -> int:
    """Theoretical KV cache size — full path: K, V both [L, H_kv, S, D].
    Channel pruning at ratio r shrinks K-vision from D to D_keep =
    D - floor(D*r); non-vision K and all V stay full.
    """
    L = LLAVA_NEXT_CFG["num_layers"]
    H = LLAVA_NEXT_CFG["num_kv_heads"]
    D = LLAVA_NEXT_CFG["head_dim"]
    dt = LLAVA_NEXT_CFG["dtype_bytes"]
    S = input_tokens + output_tokens
    v_bytes = L * H * S * D * dt
    if method == "full":
        k_bytes = L * H * S * D * dt
    else:
        D_keep = D - int(D * channel_ratio)
        non_vision_S = max(0, S - vision_tokens)
        k_bytes = L * H * (vision_tokens * D_keep + non_vision_S * D) * dt
    return k_bytes + v_bytes


def _gb(b: int) -> float:
    return b / 1024**3


# ---------------------------------------------------------------------------
# Layer-level timing hooks (uniform across all methods)
# ---------------------------------------------------------------------------

class LayerLatencyHooks:
    """Per-layer pre/post hooks: capture (start, end) cuda events per call.

    Tracks prefill (first call, q_len > 1) vs decode (subsequent calls,
    q_len == 1) by inspecting the input hidden_states' seq dim.
    """

    def __init__(self, layers, mem_trace: bool = False):
        self.mem_trace = mem_trace
        self._layers = layers
        self._events: Dict[int, Dict[str, List[Tuple]]] = {
            i: {"prefill": [], "decode": []} for i in range(len(layers))
        }
        self._handles = []
        self._pending_start: Dict[int, torch.cuda.Event] = {}
        self._pending_phase: Dict[int, str] = {}
        self.prefill_seqlen: int = 0
        for i, layer in enumerate(layers):
            self._handles.append(
                layer.register_forward_pre_hook(self._make_pre(i), with_kwargs=True)
            )
            self._handles.append(
                layer.register_forward_hook(self._make_post(i), with_kwargs=True)
            )

    def _make_pre(self, i):
        def _pre(module, args, kwargs):
            hs = args[0] if args else kwargs.get("hidden_states")
            if hs is None:
                return
            seq_len = hs.shape[1]
            phase = "prefill" if seq_len > 1 else "decode"
            evt = torch.cuda.Event(enable_timing=True)
            evt.record()
            self._pending_start[i] = evt
            self._pending_phase[i] = phase
            if phase == "prefill" and i == 0 and self.prefill_seqlen == 0:
                self.prefill_seqlen = int(seq_len)
            if self.mem_trace and phase == "prefill":
                a = torch.cuda.memory_allocated() / 1024**3
                p = torch.cuda.max_memory_allocated() / 1024**3
                print(f"  [mem_trace] pre  L{i:02d}: alloc={a:.2f}GB peak={p:.2f}GB", flush=True)
        return _pre

    def _make_post(self, i):
        def _post(module, args, kwargs, output):
            start = self._pending_start.pop(i, None)
            phase = self._pending_phase.pop(i, None)
            if start is None or phase is None:
                return
            evt = torch.cuda.Event(enable_timing=True)
            evt.record()
            self._events[i][phase].append((start, evt))
            if self.mem_trace and phase == "prefill":
                a = torch.cuda.memory_allocated() / 1024**3
                p = torch.cuda.max_memory_allocated() / 1024**3
                print(f"  [mem_trace] post L{i:02d}: alloc={a:.2f}GB peak={p:.2f}GB", flush=True)
        return _post

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def summarize(self) -> Dict[int, Dict[str, List[float]]]:
        torch.cuda.synchronize()
        out: Dict[int, Dict[str, List[float]]] = {}
        for i, phases in self._events.items():
            out[i] = {p: [s.elapsed_time(e) for s, e in pairs] for p, pairs in phases.items()}
        return out


def _set_visionzip_seqlens(model, prompt_seqlen: int, query_seqlen: int):
    """Stamp prompt_seqlen / query_seqlen on both top-level and text sub-config.

    The synthetic-input path uses (0, 1) so the entire prefill range is
    treated as "vision" by VisionZip — channel pruning applies to S-1
    positions. Matches the long-S regime where >99% of input is vision.
    """
    cfg = model.config
    cfg.prompt_seqlen = prompt_seqlen
    cfg.query_seqlen = query_seqlen
    sub = getattr(cfg, "text_config", None)
    if sub is not None:
        sub.prompt_seqlen = prompt_seqlen
        sub.query_seqlen = query_seqlen


def _run_generation(model, vision_tokens: int, batch_size: int,
                              max_new_tokens: int):
    """Bypass vision tower entirely. Build random `inputs_embeds` of layout
    [prompt(30) | vision(`vision_tokens`) | text(30)] and feed directly to
    language_model.generate. Total prefill seq_len = vision_tokens + 60.

    LLaVA-Next's `language_model` is `LlamaForCausalLM`; calling its
    generate with `inputs_embeds=` short-circuits the parent's
    image_features → scatter logic.
    """
    hidden = model.config.text_config.hidden_size
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    total_S = PROMPT_TOKENS + vision_tokens + TEXT_TOKENS
    _set_visionzip_seqlens(model, prompt_seqlen=PROMPT_TOKENS, query_seqlen=TEXT_TOKENS)
    inputs_embeds = torch.randn(
        batch_size, total_S, hidden, dtype=dtype, device=device,
    ) * 0.5
    attention_mask = torch.ones(
        batch_size, total_S, dtype=torch.long, device=device,
    )
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "min_new_tokens": max_new_tokens,
        "output_hidden_states": False,
        "output_attentions": False,
        "return_dict_in_generate": False,
    }
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for f in ("output_hidden_states", "output_attentions"):
            if getattr(cfg, f, False):
                setattr(cfg, f, False)
    with torch.inference_mode():
        out_ids = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            **generation_config,
        )
    return out_ids


def _mean_std(values):
    """Population mean and std. Returns (0.0, 0.0) when empty, (m, 0.0) when single."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / n
    return m, var ** 0.5


def _trimmed_stats(values):
    """IQR-based trim of upper outliers, then mean+std on the remaining values.

    Tukey's upper fence: drop layers whose mean > Q3 + 1.5 * IQR. Lower
    outliers are kept (they're physically implausible — no GPU event makes
    a layer faster than baseline). Returns (mean, std, n_dropped, n_kept).
    """
    if not values:
        return 0.0, 0.0, 0, 0
    sv = sorted(values)
    n = len(sv)
    q1 = sv[n // 4]
    q3 = sv[(3 * n) // 4]
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr
    kept = [v for v in values if v <= threshold]
    m, s = _mean_std(kept)
    return m, s, n - len(kept), len(kept)


def _per_layer_stats(timings):
    """Per-phase stats across layers, after IQR-based outlier removal.

    Returns (pf_mean, pf_std, pf_n_dropped, dec_mean, dec_std, dec_n_dropped)
    in ms. Layers with prefill/decode time > Q3 + 1.5*IQR are dropped —
    these are sporadic GPU events (allocator fragmentation, GC, scheduling
    jitter) not structural to the layer.
    """
    pf, dec = [], []
    for i, phases in timings.items():
        if phases["prefill"]:
            pf.append(sum(phases["prefill"]) / len(phases["prefill"]))
        if phases["decode"]:
            dec.append(sum(phases["decode"]) / len(phases["decode"]))
    pf_mean, pf_std, pf_drop, _ = _trimmed_stats(pf)
    dec_mean, dec_std, dec_drop, _ = _trimmed_stats(dec)
    return pf_mean, pf_std, pf_drop, dec_mean, dec_std, dec_drop


def _resolve_layers(model):
    """Find the transformer-layer module list. LLaVA-NeXT's path is
    `model.language_model.model.layers` (LlamaModel decoder layers).
    Robust to both pure-HF and lmms-eval-wrapped models."""
    candidate_paths = [
        "language_model.model.layers",
        "language_model.layers",
        "model.layers",
        "transformer.layers",
    ]
    for path in candidate_paths:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__") and len(obj) > 0:
                return obj
        except AttributeError:
            continue
    raise RuntimeError("Could not locate transformer layer list on model")


# ---------------------------------------------------------------------------
# One measurement
# ---------------------------------------------------------------------------

def measure_one(method: str, prefill_length: int,
                          channel_ratio: str, max_new_tokens: int,
                          batch_size: int = 1, n_repeats: int = 1,
                          mem_trace: bool = False,
                          profile_breakdown: bool = False,
                          profile_mode: str = "cuda") -> dict:
    """Synthetic-input measurement at exact `prefill_length` prefill length.

    Skips the vision tower; feeds random inputs_embeds of shape
    [B, prefill_length, hidden] to the LLaMA backbone. Whole prefill range
    is marked as vision so VisionZip channel pruning still applies.
    """
    if method == "full":
        lm = _load_full_llava_next()
    else:
        lm = _load_visionzip(method, channel_ratio)
    model = lm.model

    print(
        f"[input] synthetic prefill_length={prefill_length}  B={batch_size}  "
        f"(bypassing vision tower)",
        flush=True,
    )

    layers = _resolve_layers(model)
    hooks = LayerLatencyHooks(layers, mem_trace=mem_trace)
    _reset_peak()
    print(f"[mem]    after model load: {_gpu_mem_str()}", flush=True)

    # Warmup at full max_new_tokens — primes JIT AND lets the caching
    # allocator reserve all the KV memory it'll need at peak. Running with
    # a tiny max_new_tokens (e.g., 4) leaves the allocator under-sized,
    # triggering cudaMalloc syncs mid-measurement that spike a few layers'
    # prefill times by 100s of ms.
    #
    # 3 warmup passes (was 1): single warmup leaves Triton autotune jitter
    # and cuBLAS handle init in run 1 of the timed measurement. Three passes
    # ensures every kernel shape variant is autotuned and cached before the
    # measurement window opens — material for sequence-length sweeps where
    # each S triggers fresh autotune of the dense decode kernel.
    _reset_peak()
    _N_WARMUP = 3
    for _wi in range(_N_WARMUP):
        try:
            _ = _run_generation(
                model, prefill_length, batch_size, max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            print(f"[warmup {_wi+1}/{_N_WARMUP}] failed: {e}", flush=True)
            raise
        print(
            f"[mem]    after warmup {_wi+1}/{_N_WARMUP}: {_gpu_mem_str()}",
            flush=True,
        )
    torch.cuda.empty_cache()

    hooks.remove()
    hooks = LayerLatencyHooks(layers, mem_trace=mem_trace)

    _reset_peak()
    t0 = time.perf_counter()
    for rep in range(n_repeats):
        _ = _run_generation(
            model, prefill_length, batch_size, max_new_tokens=max_new_tokens,
        )
    torch.cuda.synchronize()
    wall_s = (time.perf_counter() - t0) / n_repeats
    print(f"[mem]    after generation (×{n_repeats}): {_gpu_mem_str()}", flush=True)

    timings = hooks.summarize()
    pf_ms, pf_std, pf_drop, dec_ms, dec_std, dec_drop = _per_layer_stats(timings)
    input_tokens = hooks.prefill_seqlen
    decode_calls_per_layer = len(timings.get(0, {}).get("decode", []))
    output_tokens = decode_calls_per_layer // max(1, n_repeats)
    hooks.remove()

    breakdown_by_layer: Dict[int, Dict[str, float]] = {}
    if profile_breakdown:
        # Lazy import: only loaded when --profile_breakdown is set, since the
        # qwen module imports triton kernels.
        from lmms_eval.models.model_utils.qwen.qwen2_5vl_visionzip import (
            run_custom_decode_kernel,
        )

        print(
            f"[breakdown] running profile pass (mode={profile_mode}, "
            "excluded from timing)...",
            flush=True,
        )
        attn_cfg = _attention_config(model)
        n_layers = LLAVA_NEXT_CFG["num_layers"]
        attn_cfg._decode_profile = True
        attn_cfg._decode_profile_mode = profile_mode
        attn_cfg._decode_profile_layers = set(range(n_layers))
        attn_cfg._decode_timings = None
        attn_cfg._decode_timings_by_layer = None
        run_custom_decode_kernel._profile_kernel = True
        run_custom_decode_kernel._profile_layer_idx = None
        run_custom_decode_kernel._kernel_timings = None
        run_custom_decode_kernel._kernel_timings_by_layer = None

        try:
            _ = _run_generation(
                model, prefill_length, batch_size, max_new_tokens=max_new_tokens,
            )
            torch.cuda.synchronize()
        finally:
            attn_cfg._decode_profile = False
            run_custom_decode_kernel._profile_kernel = False

        breakdown_by_layer = _aggregate_decode_breakdown(model)
        _print_breakdown_table(breakdown_by_layer)

    # Synthetic layout: [prompt | vision | text]. Vision is the channel-pruned
    # range; prompt/text are fixed framing tokens (not pruned).
    prompt_tokens = PROMPT_TOKENS
    text_tokens = TEXT_TOKENS
    vision_tokens_est = max(0, input_tokens - prompt_tokens - text_tokens)
    print(
        f"[tokens]  prompt={prompt_tokens}  vision={vision_tokens_est}  "
        f"text={text_tokens}  output={output_tokens}  "
        f"=> input_tokens(prefill)={input_tokens}",
        flush=True,
    )
    param_bytes = _model_param_bytes(model)
    kv_bytes = _kv_cache_bytes(
        method, float(channel_ratio), input_tokens, output_tokens, vision_tokens_est,
    )
    total_bytes = param_bytes + kv_bytes
    peak_alloc_bytes = (
        torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    )
    activation_bytes = max(0, peak_alloc_bytes - total_bytes)
    print(
        f"[memory]  param={_gb(param_bytes):.2f}GB  "
        f"kv_cache={_gb(kv_bytes):.2f}GB  "
        f"total(theory)={_gb(total_bytes):.2f}GB  | "
        f"peak_alloc={_gb(peak_alloc_bytes):.2f}GB  "
        f"activation(est)={_gb(activation_bytes):.2f}GB",
        flush=True,
    )
    n_layers = LLAVA_NEXT_CFG["num_layers"]

    # Build per-layer arrays of (mean across n_repeats) prefill/decode ms.
    pf_per_layer = []
    dec_per_layer = []
    for i in range(n_layers):
        ph = timings.get(i, {})
        pf_vals = ph.get("prefill", [])
        dec_vals = ph.get("decode", [])
        pf_per_layer.append(sum(pf_vals) / len(pf_vals) if pf_vals else float("nan"))
        dec_per_layer.append(
            sum(dec_vals) / len(dec_vals) if dec_vals else float("nan")
        )

    def _median(xs):
        ys = sorted(v for v in xs if v == v)
        if not ys:
            return float("nan")
        n = len(ys)
        return ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])

    pf_median = _median(pf_per_layer)
    dec_median = _median(dec_per_layer)

    print(
        f"[result]  prefill_per_layer mean±std={pf_ms:.3f}±{pf_std:.3f}ms  "
        f"median={pf_median:.3f}ms (dropped {pf_drop}/{n_layers})  "
        f"decode_per_layer mean±std={dec_ms:.3f}±{dec_std:.3f}ms  "
        f"median={dec_median:.3f}ms (dropped {dec_drop}/{n_layers})  "
        f"wall={wall_s:.2f}s",
        flush=True,
    )
    print(
        f"[prefill_ms_per_layer] "
        + " ".join(f"{v:.2f}" for v in pf_per_layer),
        flush=True,
    )
    print(
        f"[decode_ms_per_layer]  "
        + " ".join(f"{v:.2f}" for v in dec_per_layer),
        flush=True,
    )

    if prefill_length >= 1024 and prefill_length % 1024 == 0:
        s_label = f"{prefill_length // 1024}k"
    else:
        s_label = str(prefill_length)
    result = {
        "method": method,
        "image_tag": s_label,
        "num_images": 0,
        "batch_size": batch_size,
        "vision_tokens_est": vision_tokens_est,
        "text_question_tokens": 0,
        "chat_template_tokens": 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "prefill_per_layer_ms": pf_ms,
        "prefill_per_layer_std_ms": pf_std,
        "prefill_per_layer_median_ms": pf_median,
        "prefill_per_layer_n_dropped": pf_drop,
        "decode_per_layer_ms": dec_ms,
        "decode_per_layer_std_ms": dec_std,
        "decode_per_layer_median_ms": dec_median,
        "decode_per_layer_n_dropped": dec_drop,
        "prefill_ms_per_layer": pf_per_layer,
        "decode_ms_per_layer": dec_per_layer,
        "raw_timings_per_layer": {
            i: {
                "prefill": list(timings.get(i, {}).get("prefill", [])),
                "decode": list(timings.get(i, {}).get("decode", [])),
            } for i in range(n_layers)
        },
        "wall_clock_s": wall_s,
        "max_new_tokens": max_new_tokens,
        "param_bytes": param_bytes,
        "kv_cache_bytes": kv_bytes,
        "total_memory_bytes": total_bytes,
        "peak_alloc_bytes": peak_alloc_bytes,
        "activation_bytes_est": activation_bytes,
        "decode_breakdown_by_layer": breakdown_by_layer,
    }
    try:
        model.cpu()
    except Exception:
        pass
    del lm, model, layers, hooks
    _teardown()
    print(f"[mem]    after teardown: {_gpu_mem_str()}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods", default="full,think,spark,rotatek",
        help="comma-separated subset of {full, think, spark, rotatek}",
    )
    parser.add_argument(
        "--prefill_length", required=True,
        help=(
            "Comma-separated vision-token counts (e.g. `8192,16384,32768`) or "
            "shorthand (`8k,16k,32k,64k,128k`). Specifies the *vision* span; "
            "the actual prefill = prompt(30) + vision + text(30). Bypasses the "
            "vision tower and feeds random inputs_embeds; only the vision "
            "span is channel-pruned (prompt/text are not)."
        ),
    )
    parser.add_argument(
        "--batch_sizes", default="1",
        help="comma-separated batch sizes",
    )
    parser.add_argument(
        "--decoding_length", type=int, default=128,
        help="number of decode tokens per run",
    )
    parser.add_argument(
        "--mem_trace", action="store_true",
        help="Print cuda.memory_allocated() before/after each layer.",
    )
    parser.add_argument(
        "--n_repeats", type=int, default=3,
        help="Repeat each measurement N times to reduce variance.",
    )
    parser.add_argument(
        "--channel_ratio", default="0.75",
        help="channel pruning ratio (kept fraction) — ignored for full",
    )
    parser.add_argument(
        "--log_dir", default=DEFAULT_LOG_DIR,
        help=f"directory for log + JSON output (default {DEFAULT_LOG_DIR})",
    )
    parser.add_argument(
        "--tag", default="",
        help="optional run tag appended to log filename",
    )
    parser.add_argument(
        "--profile_breakdown", action="store_true",
        help=(
            "Run an extra (non-timed) generation pass with `_decode_profile=True` "
            "to collect per-layer module breakdown (qkv_proj/rope/cache_update/"
            "recovery/cat/custom_kernel/transpose/fa2/attn_out_proj). Stored in "
            "`decode_breakdown_by_layer` of each result row."
        ),
    )
    parser.add_argument(
        "--profile_mode", choices=["cuda", "wall"], default="cuda",
        help=(
            "Timing primitive for `--profile_breakdown`. 'cuda' uses "
            "torch.cuda.Event (low overhead, GPU-side). 'wall' calls "
            "torch.cuda.synchronize()+time.perf_counter() at every stage "
            "boundary (serializes the pipeline; reports pure wall-clock per "
            "stage)."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    methods_short = "_".join(m[:2] for m in args.methods.split(","))
    batches_short = args.batch_sizes.replace(",", "-")
    input_short = f"syn_{args.prefill_length.replace(',', '-')}"
    base = (
        f"llava_next_methods_{methods_short}_{input_short}"
        f"_B_{batches_short}_dec_{args.decoding_length}_ratio_{args.channel_ratio}"
    )
    if args.tag:
        base += f"_{args.tag}"
    log_path = os.path.join(args.log_dir, base + ".txt")
    json_path = os.path.join(args.log_dir, base + ".json")
    log_fh = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]

    def _parse_length(s: str) -> int:
        s = s.strip().lower()
        if s.endswith("k"):
            return int(float(s[:-1]) * 1024)
        return int(s)

    prefill_lengths = [
        _parse_length(t) for t in args.prefill_length.split(",") if t.strip()
    ]

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(no CUDA)"
    print(f"model         : {PRETRAINED}")
    print(f"GPU           : {gpu_name}")
    print(f"[log] writing to {log_path}", flush=True)
    print(f"[log] structured results: {json_path}", flush=True)
    print(f"methods       : {methods}")
    print(f"prefill_length: {prefill_lengths}  (synthetic, vision tower bypassed)")
    print(f"batch sizes   : {batch_sizes}")
    print(f"decoding_len  : {args.decoding_length}")
    print(f"channel_ratio : {args.channel_ratio}")
    print(f"n_repeats     : {args.n_repeats}")
    import os as _os_for_flags
    print(
        "compile flags : "
        f"THINK={_os_for_flags.environ.get('THINK_COMPILE', '')!r} "
        f"SPARK={_os_for_flags.environ.get('SPARK_COMPILE', '')!r} "
        f"ROTATEK={_os_for_flags.environ.get('ROTATEK_COMPILE', '')!r}"
    )
    print()

    rows = []
    for S in prefill_lengths:
        s_label = (
            f"{S // 1024}k" if S >= 1024 and S % 1024 == 0 else str(S)
        )
        for B in batch_sizes:
            for method in methods:
                _teardown()
                print(
                    f"=== {method} | S={s_label} (={S} tokens) | B={B} === "
                    f"({_gpu_mem_str()})",
                    flush=True,
                )
                try:
                    r = measure_one(
                        method, S, args.channel_ratio,
                        args.decoding_length, batch_size=B,
                        n_repeats=args.n_repeats,
                        profile_breakdown=args.profile_breakdown,
                        profile_mode=args.profile_mode,
                    )
                    rows.append(r)
                except torch.cuda.OutOfMemoryError as e:
                    print(f"[OOM]   {method} | S={s_label} | B={B}: {e}", flush=True)
                    rows.append({
                        "method": method, "image_tag": s_label, "num_images": 0,
                        "batch_size": B, "status": "OOM",
                    })
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except Exception as e:
                    print(f"[error] {method} | S={s_label} | B={B}: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    rows.append({
                        "method": method, "image_tag": s_label, "num_images": 0,
                        "batch_size": B, "status": f"error: {type(e).__name__}",
                    })
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                print()

    # Per-image totals on successful rows
    NUM_LAYERS = LLAVA_NEXT_CFG["num_layers"]
    for r in rows:
        if "status" in r:
            continue
        r["per_image_prefill_ms"] = NUM_LAYERS * r["prefill_per_layer_ms"]
        r["per_image_decode_ms"] = (
            NUM_LAYERS * r["output_tokens"] * r["decode_per_layer_ms"]
        )
        r["per_image_total_ms"] = (
            r["per_image_prefill_ms"] + r["per_image_decode_ms"]
        )

    print("=" * 200)
    print(
        "SUMMARY: per-layer latency (ms), per-image total "
        f"(× {NUM_LAYERS} layers, decode × output_tokens), and memory (GB)"
    )
    print("=" * 200)
    print(
        f"{'method':>8}  {'image':>6}  {'N':>2}  {'B':>2}  "
        f"{'in_tok':>7}  {'out_tok':>7}  "
        f"{'pf/layer (trim)':>17}  {'pf_med':>7}  {'pf_drop':>7}  "
        f"{'dec/layer (trim)':>18}  {'dec_med':>8}  {'dec_drop':>8}  | "
        f"{'pf/img':>9}  {'dec/img':>10}  {'tot/img':>10}  "
        f"{'wall(s)':>8}  | "
        f"{'param':>6}  {'kv':>6}  {'theory':>7}  {'peak':>6}  {'activ':>6}"
    )
    print("-" * 245)
    for r in rows:
        if "status" in r:
            tag = "OOM" if r["status"] == "OOM" else "err"
            print(
                f"{r['method']:>8}  {r['image_tag']:>6}  {r['num_images']:>2}  "
                f"{r['batch_size']:>2}  "
                f"{tag:>7}  {tag:>7}  "
                f"{tag:>17}  {tag:>7}  {tag:>7}  "
                f"{tag:>18}  {tag:>8}  {tag:>8}  | "
                f"{tag:>8}  {tag:>9}  {tag:>9}  "
                f"{tag:>7}  | "
                f"{tag:>5}  {tag:>5}  {tag:>6}  {tag:>5}  {tag:>5}"
            )
            continue
        pf_cell = f"{r['prefill_per_layer_ms']:.2f}±{r.get('prefill_per_layer_std_ms', 0):.2f}"
        pf_med_cell = f"{r.get('prefill_per_layer_median_ms', 0):.2f}"
        pf_drop_cell = f"{r.get('prefill_per_layer_n_dropped', 0)}"
        dec_cell = f"{r['decode_per_layer_ms']:.3f}±{r.get('decode_per_layer_std_ms', 0):.3f}"
        dec_med_cell = f"{r.get('decode_per_layer_median_ms', 0):.3f}"
        dec_drop_cell = f"{r.get('decode_per_layer_n_dropped', 0)}"
        print(
            f"{r['method']:>8}  {r['image_tag']:>6}  {r['num_images']:>2}  "
            f"{r['batch_size']:>2}  "
            f"{r['input_tokens']:>7}  {r['output_tokens']:>7}  "
            f"{pf_cell:>17}  {pf_med_cell:>7}  {pf_drop_cell:>7}  "
            f"{dec_cell:>18}  {dec_med_cell:>8}  {dec_drop_cell:>8}  | "
            f"{r['per_image_prefill_ms']:>8.1f}  {r['per_image_decode_ms']:>9.1f}  "
            f"{r['per_image_total_ms']:>9.1f}  "
            f"{r['wall_clock_s']:>7.2f}  | "
            f"{_gb(r['param_bytes']):>5.2f}  "
            f"{_gb(r['kv_cache_bytes']):>5.2f}  "
            f"{_gb(r['total_memory_bytes']):>6.2f}  "
            f"{_gb(r.get('peak_alloc_bytes', 0)):>5.2f}  "
            f"{_gb(r.get('activation_bytes_est', 0)):>5.2f}"
        )

    import json
    with open(json_path, "w") as fh:
        json.dump(
            {
                "config": {
                    "model": PRETRAINED,
                    "gpu": gpu_name,
                    "methods": methods,
                    "prefill_length": prefill_lengths,
                    "batch_sizes": batch_sizes,
                    "decoding_length": args.decoding_length,
                    "channel_ratio": args.channel_ratio,
                    "n_repeats": args.n_repeats,
                    "num_layers": NUM_LAYERS,
                },
                "rows": rows,
            }, fh, indent=2,
        )
    print(f"\n[log] wrote {json_path}")


if __name__ == "__main__":
    main()
