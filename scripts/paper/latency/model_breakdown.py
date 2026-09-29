"""Per-stage latency breakdown for prefill AND decode (single pass).

Loads InternVL with one channel-pruning method, enables both
`_prefill_profile` and `_decode_profile` on the LLM config so that the
patched attention / decoder-layer forwards record cuda events at every
internal stage. Runs `model.chat` once per `n_repeats` and prints two
tables: prefill (per-layer avg across all 32 layers) and decode
(per-token avg on a representative layer — layer 14 by default).

Prefill stages:
    qkv_proj, rope, score, kv_write, fa2, out_proj, attn_total, ffn,
    layer_total

Decode stages (recorded only on `_decode_profile_layers`, default {14}):
    qkv+rope, cache_upd, recovery, cat, triton_attn, fa2, out_proj,
    attn_total, ffn, layer_total

Usage:
    python -m latency.model_breakdown \\
        --methods full,think,spark,rotatek \\
        --image 1k --num_images 1 --max_num 48 --n_repeats 3 \\
        --max_new_tokens 16
"""
from __future__ import annotations

import os
import sys

# Make the repo root importable so `rotatek` resolves no matter where this
# script is launched from.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List, Tuple

# Match end_to_end.py defaults — set before `import torch`.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:512",
)

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latency.model_end_to_end_internvl import (  # noqa: E402
    _load_visionzip, _load_test_image, _build_pixel_values, _build_prompt,
)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

PREFILL_STAGES: List[Tuple[str, str]] = [
    ("qkv_proj_ms",            "qkv_proj"),
    ("rope_ms",                "rope"),
    ("score_ms",               "score"),
    ("kv_write_ms",            "kv_write"),
    ("fa2_ms",                 "fa2"),
    ("attn_out_proj_ms",       "out_proj"),
    ("prefill_attn_ms",        "attn_total"),
    ("prefill_ffn_ms",         "ffn"),
    ("prefill_layer_total_ms", "layer_total"),
]

DECODE_STAGES: List[Tuple[str, str]] = [
    ("qkv_proj_ms",             "qkv_proj"),
    ("rope_ms",                 "rope"),
    ("cache_update_ms",         "cache_upd"),
    ("recovery_ms",             "recovery"),
    ("cat_ms",                  "cat"),
    ("custom_decode_kernel_ms", "triton_attn"),
    ("fa2_ms",                  "fa2"),
    ("attn_out_proj_ms",        "out_proj"),
    ("decode_attn_ms",          "attn_total"),
    ("decode_ffn_ms",           "ffn"),
    ("decode_layer_total_ms",   "layer_total"),
]

ATTN_LAYER_LEVEL_KEYS = {
    "prefill_attn_ms", "prefill_ffn_ms", "prefill_layer_total_ms",
    "decode_attn_ms", "decode_ffn_ms", "decode_layer_total_ms",
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _enable_profile(model, decode_layer: int = 14):
    cfg = model.config.llm_config
    cfg._prefill_profile = True
    cfg._decode_profile = True
    cfg._decode_profile_layers = {decode_layer}
    cfg._prefill_attn_events_by_layer = None
    cfg._decode_event_pairs_by_layer = None
    cfg._layer_event_pairs = None


def _reset_buckets(model):
    cfg = model.config.llm_config
    cfg._prefill_attn_events_by_layer = None
    cfg._decode_event_pairs_by_layer = None
    cfg._layer_event_pairs = None


def _collect_pairs(model, key: str, phase: str,
                   attn_layer_filter=None) -> List[Tuple]:
    """Return list of (start, end) cuda event pairs for the given stage key.

    `prefill_*` / `decode_*` keys (the layer-block-level ones — see
    ATTN_LAYER_LEVEL_KEYS) live in `_layer_event_pairs`. The
    attention-internal keys live in phase-specific dicts:
      - prefill: `_prefill_attn_events_by_layer`
      - decode : `_decode_event_pairs_by_layer`
    so prefill and decode events for the SAME stage key (e.g. `fa2_ms`)
    don't pollute each other.
    """
    cfg = model.config.llm_config
    if key in ATTN_LAYER_LEVEL_KEYS:
        events = getattr(cfg, "_layer_event_pairs", None) or {}
        return list(events.get(key, []))
    bucket_attr = (
        "_prefill_attn_events_by_layer" if phase == "prefill"
        else "_decode_event_pairs_by_layer"
    )
    by_layer = getattr(cfg, bucket_attr, None) or {}
    pairs: List[Tuple] = []
    for li, evts in by_layer.items():
        if attn_layer_filter is not None and li not in attn_layer_filter:
            continue
        pairs.extend(evts.get(key, []))
    return pairs


def _sum_ms(pairs: List[Tuple]) -> float:
    total = 0.0
    for s, e in pairs:
        if s is None or e is None:
            continue
        total += s.elapsed_time(e)
    return total


def _gpu_burner(seconds: float = 5.0) -> None:
    """Run a dense fp32 matmul loop to bring the GPU from base clock to
    its sustained boost clock and equalize thermal state. Without this,
    the first method measured each run pays a cold-clock penalty
    (especially on launch-overhead-bound stages like RoPE), making
    cross-method comparisons noisy in a systematic way."""
    if not torch.cuda.is_available():
        return
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a = a @ b
    torch.cuda.synchronize()
    del a, b


def _cooldown_and_reset(seconds: float = 15.0) -> None:
    """Drain stream + drop allocator pools + sit idle so the next method's
    measurement starts from a comparable GPU state. Combined with
    `_gpu_burner` immediately after this, the next method enters its
    measurement loop at the same boost-clock thermal steady-state as the
    previous method did."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    time.sleep(seconds)


def _microbench_dispatch_artifacts(model, vision_len: int = 12000,
                                    prompt_len: int = 50,
                                    reps: int = 1000) -> Dict[str, float]:
    """Tight-loop microbench for RoPE and KV-cache cat.

    Both stages light up large in `model_breakdown` because the breakdown
    table records cuda events around tiny per-op sequences, so the elapsed
    time captures CPU-dispatch idle gaps on the stream rather than just
    real GPU work. Running the same code in a tight loop with a single
    event pair amortizes those gaps; the resulting per-call number is the
    actual GPU-side cost.

    Returns a dict of `{stage}_us` per-call steady-state times.
    """
    from lmms_eval.models.model_utils.internvl.internvl2_5_visionzip import (
        _apply_rotary_pos_emb,
    )

    layer = model.language_model.model.layers[14]
    attn = layer.attention
    head_dim = attn.head_dim
    num_kv_groups = attn.num_key_value_groups
    # Q heads = num_kv_heads * num_kv_groups; we get q-head count from wqkv
    # output split via rearrange in the patched forward (h gs d, gs = 2+groups).
    out_features = attn.wqkv.weight.shape[0]
    h_q_plus_2_kv = out_features // head_dim
    h_kv = h_q_plus_2_kv // (num_kv_groups + 2)
    h_q = h_kv * num_kv_groups

    device = next(model.parameters()).device
    dtype = torch.bfloat16
    kv_seq_len = prompt_len + vision_len
    bsz = 1

    q = torch.randn(bsz, h_q, 1, head_dim, device=device, dtype=dtype)
    k = torch.randn(bsz, h_kv, 1, head_dim, device=device, dtype=dtype)
    v = torch.randn(bsz, h_kv, 1, head_dim, device=device, dtype=dtype)
    pos = torch.arange(kv_seq_len - 1, kv_seq_len, device=device).unsqueeze(0)

    # ---- RoPE ---------------------------------------------------------
    for _ in range(20):
        cos, sin = attn.rotary_emb(v, seq_len=kv_seq_len)
        _ = _apply_rotary_pos_emb(q, k, cos, sin, pos)
    torch.cuda.synchronize()

    s, e = (torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True))
    s.record()
    for _ in range(reps):
        cos, sin = attn.rotary_emb(v, seq_len=kv_seq_len)
        _ = _apply_rotary_pos_emb(q, k, cos, sin, pos)
    e.record()
    torch.cuda.synchronize()
    rope_us = s.elapsed_time(e) * 1000.0 / reps  # ms→μs

    # ---- KV cache cat (the two cats inside DynamicCache.update) -------
    k_cache = torch.randn(bsz, h_kv, kv_seq_len, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(bsz, h_kv, kv_seq_len, head_dim, device=device, dtype=dtype)
    for _ in range(20):
        _ = torch.cat([k_cache, k], dim=-2)
        _ = torch.cat([v_cache, v], dim=-2)
    torch.cuda.synchronize()

    s, e = (torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True))
    s.record()
    for _ in range(reps):
        _ = torch.cat([k_cache, k], dim=-2)
        _ = torch.cat([v_cache, v], dim=-2)
    e.record()
    torch.cuda.synchronize()
    cache_us = s.elapsed_time(e) * 1000.0 / reps

    # ---- FFN (isolated, no preceding PCA / attention) -----------------
    # Goal: tell apart "FFN inflation in rotatek's prefill is intrinsic
    # to the model" vs "it's spillover from PCA running just before FFN
    # in the same forward pass". If both methods report the same FFN time
    # here, the inflation is intra-forward state spillover (thermal,
    # cache, scheduler), not anything about FFN itself.
    ffn = layer.feed_forward
    hidden_size = ffn.w1.weight.shape[1] if hasattr(ffn, "w1") else (
        next(p for p in ffn.parameters()).shape[-1]
    )
    # Use a realistic prefill-shape input (S=kv_seq_len) so FFN sees the
    # same matmul size as in the actual breakdown.
    x = torch.randn(bsz, kv_seq_len, hidden_size, device=device, dtype=dtype)
    for _ in range(20):
        _ = ffn(x)
    torch.cuda.synchronize()

    s, e = (torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True))
    s.record()
    # FFN is heavy (multi-GFLOP matmuls); fewer reps to keep total runtime
    # reasonable.
    ffn_reps = max(20, reps // 50)
    for _ in range(ffn_reps):
        _ = ffn(x)
    e.record()
    torch.cuda.synchronize()
    ffn_ms = s.elapsed_time(e) / ffn_reps    # ms per call

    return {
        "rope_us": rope_us,
        "cache_cat_us": cache_us,
        "ffn_ms": ffn_ms,
    }


def _measure_method(method: str, ratio: str, image: str, num_images: int,
                    max_num: int, max_new_tokens: int, n_repeats: int,
                    decode_layer: int,
                    batch_size: int = 1,
                    label: str | None = None) -> Dict[str, Dict[str, float]]:
    # `method` is the internal channel-pruning name passed to the model
    # (e.g. "think" with ratio=0 implements user-facing "full"). `label`
    # is the user-facing name to display in headers.
    display = label if label is not None else method
    print(f"\n=== {display} | image={image}×{num_images} (ratio={ratio}, "
          f"B={batch_size}) ===", flush=True)
    # Force GPU into sustained boost-clock / steady thermal state before we
    # touch any model code. This eliminates the cold-clock penalty the
    # first measured method otherwise eats (and which propagates as
    # apparent method-vs-method differences in the otherwise method-
    # agnostic stages like qkv_proj / rope / ffn).
    _gpu_burner(seconds=10.0)
    model, tok = _load_visionzip(method, ratio)
    model.eval()

    img = _load_test_image(image)
    pixel_values, num_sub = _build_pixel_values([img] * num_images, max_num=max_num)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    prompt = _build_prompt(num_images)

    # Token-count diagnostics. Replicate InternVL's chat-time prompt
    # construction (chat template + image-token expansion) so we can
    # report exact prefill input length, broken down by source.
    num_image_token = getattr(
        model, "num_image_token",
        getattr(getattr(model, "config", None), "num_image_token", 256),
    )
    vision_tokens = sum(num_sub) * num_image_token

    # User question text (without image tags); shows the pure text contribution.
    prompt_no_image_tags = prompt.replace("<image>\n", "").replace("<image>", "")
    text_question_tokens = len(
        tok(prompt_no_image_tags, add_special_tokens=False).input_ids
    )

    # Chat-template overhead (system message + role prefixes/suffixes +
    # `<img>`/`</img>` markers around each image block). Build the same
    # template the model.chat() call uses, then tokenize a version where
    # each `<image>` tag is replaced with an empty string AND a version
    # with markers but no IMG_CONTEXT padding. Diff gives the wrapper cost.
    chat_template_tokens = None
    try:
        from copy import deepcopy
        template = deepcopy(model.conv_template)
        template.system_message = model.system_message
        template.append_message(template.roles[0], prompt)
        template.append_message(template.roles[1], None)
        query_with_image_tags = template.get_prompt()
        # Replace `<image>` with just `<img></img>` markers (no IMG_CONTEXT
        # padding). Diff between this and the user-text-only count gives
        # the chat-template overhead (system + role wrappers + marker tags).
        query_for_overhead = query_with_image_tags.replace(
            "<image>", "<img></img>",
        )
        n_overhead = len(
            tok(query_for_overhead, add_special_tokens=False).input_ids
        )
        chat_template_tokens = n_overhead - text_question_tokens
    except Exception:
        pass

    prefill_total = (
        vision_tokens
        + text_question_tokens
        + (chat_template_tokens if chat_template_tokens is not None else 0)
    )

    # Print in actual input-sequence order:
    #   [chat prefix]  [vision]  [text question]  [chat suffix]
    # (chat_template count covers prefix + suffix combined.)
    print(f"  tokens (per batch-slot, in input order):", flush=True)
    if chat_template_tokens is not None:
        print(f"    chat_template = {chat_template_tokens:>6d}   "
              f"(system + role markers + <img></img> wrappers)", flush=True)
    else:
        print(f"    chat_template ≈ 50-80   (couldn't extract template)",
              flush=True)
    print(f"    vision        = {vision_tokens:>6d}   "
          f"({sum(num_sub)} sub-images × {num_image_token} tokens)", flush=True)
    print(f"    text_question = {text_question_tokens:>6d}   "
          f"(user prompt only, image tags stripped)", flush=True)
    print(f"    --------------")
    print(f"    prefill_total ≈ {prefill_total:>6d}", flush=True)
    print(f"    output        = {max_new_tokens:>6d}   "
          f"(forced via min_new_tokens)", flush=True)
    if batch_size > 1:
        print(f"  × B={batch_size}: "
              f"prefill_total≈{prefill_total * batch_size}, "
              f"output={max_new_tokens * batch_size}", flush=True)

    gen_cfg = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "min_new_tokens": max_new_tokens,
        "pad_token_id": tok.pad_token_id,
    }

    # Pre-build batched inputs once (replicated across batch slots) so the
    # warmup / measurement loop only does the chat call. Mirrors
    # model_end_to_end_internvl.py's batch_chat path.
    if batch_size > 1:
        pixel_values_run = pixel_values.repeat(batch_size, 1, 1, 1)
        num_sub_run = num_sub * batch_size
        questions = [prompt] * batch_size
    else:
        pixel_values_run = pixel_values
        num_sub_run = num_sub

    def _do_chat():
        if batch_size == 1:
            return model.chat(
                tok, pixel_values_run, prompt,
                generation_config=gen_cfg,
                num_patches_list=num_sub_run,
                history=None, return_history=True,
            )
        return model.batch_chat(
            tok, pixel_values_run, questions,
            generation_config=gen_cfg,
            num_patches_list=num_sub_run,
        )

    # Warmup (no profile yet) — JIT-compiles Triton kernels and brings the
    # model weights into HBM.
    with torch.inference_mode():
        _ = _do_chat()

    # Microbench RoPE + KV-cache cat in a tight loop. The per-call event
    # pairs in the breakdown table capture CPU-dispatch idle gaps on the
    # stream; these tight-loop numbers are the real GPU cost. Comparing
    # them to the table makes the artifact magnitude visible.
    try:
        with torch.inference_mode():
            mb = _microbench_dispatch_artifacts(model)
        print(f"  [microbench]    rope: {mb['rope_us']:7.1f} us/call  "
              f"(steady-state, dispatch-idle amortized)")
        print(f"  [microbench]  kv_cat: {mb['cache_cat_us']:7.1f} us/call  "
              f"(steady-state)")
        print(f"  [microbench]     ffn: {mb['ffn_ms']:7.3f} ms/call  "
              f"(isolated, no PCA / attention preceding)")
    except Exception as ex:
        print(f"  [microbench] skipped: {ex}")

    # Enable both profiles for the real run(s).
    _enable_profile(model, decode_layer=decode_layer)
    num_layers = len(model.language_model.model.layers)

    # Profile-enabled warmup. Pays for cuda-event creation, allocator
    # first-alloc on the event-recording path, and GPU clock ramp-up.
    # Bumped to 5 because each method runs in its own process slot with a
    # fresh model load between them — that gap (~30-60s checkpoint load +
    # GC) lets GPU thermal/clock state drift, so the first few profiled
    # runs of a new method are still settling. Without enough warmup, the
    # method-agnostic rows (qkv_proj / rope / ffn) show 1.5-2x differences
    # across methods that purely reflect that drift, not algorithmic diff.
    n_warmup_profiled = 5
    for _ in range(n_warmup_profiled):
        _reset_buckets(model)
        with torch.inference_mode():
            _ = _do_chat()
        torch.cuda.synchronize()
    _reset_buckets(model)

    # Accumulate stage totals across all reps.
    pf_total = {label: 0.0 for _, label in PREFILL_STAGES}
    dec_total = {label: 0.0 for _, label in DECODE_STAGES}
    pf_count_per_rep = num_layers      # one prefill event per layer per rep
    dec_calls_per_rep = max_new_tokens  # decoded tokens per rep on the chosen layer

    # Belt-and-suspenders: drop rep 0 from accumulation regardless of warmup.
    n_kept = 0
    for rep in range(n_repeats):
        _reset_buckets(model)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = _do_chat()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        if rep == 0:
            print(f"  rep0 wall={wall*1000:.1f}ms (dropped from accumulation)")
            continue
        n_kept += 1

        # Prefill: per-stage events fired on ALL layers (gate: q_len > 1).
        # Read from `_prefill_attn_events_by_layer` (or `_layer_event_pairs`
        # for the prefill_* layer-block keys).
        for key, label in PREFILL_STAGES:
            pairs = _collect_pairs(model, key, phase="prefill")
            pf_total[label] += _sum_ms(pairs)

        # Decode: events fired only on `decode_layer`. Read from
        # `_decode_event_pairs_by_layer` (or `_layer_event_pairs` for the
        # decode_* layer-block keys).
        for key, label in DECODE_STAGES:
            if key in ATTN_LAYER_LEVEL_KEYS:
                pairs = _collect_pairs(model, key, phase="decode")
            else:
                pairs = _collect_pairs(
                    model, key, phase="decode",
                    attn_layer_filter={decode_layer},
                )
            dec_total[label] += _sum_ms(pairs)

    # Per-layer-per-call avg.
    pf_per_layer = {
        label: ms / max(1, num_layers * n_kept)
        for label, ms in pf_total.items()
    }
    # Per-token avg (on the representative decode_layer).
    dec_per_token = {
        label: ms / max(1, n_kept * dec_calls_per_rep)
        for label, ms in dec_total.items()
    }

    del model, tok
    gc.collect()
    # Cooldown so the next method's GPU state isn't influenced by this
    # method's allocator / thermal residue. Burner happens at the start of
    # the next method to bring clock back up.
    # Long decode runs (max_new_tokens >= 64) leave the GPU hot enough
    # that 15s cooldown isn't sufficient — method-agnostic stages
    # (qkv_proj, rope) creep up across consecutive methods. 45s lets
    # thermal/clock state actually return to baseline.
    _cooldown_and_reset(seconds=45.0)
    return {"prefill": pf_per_layer, "decode": dec_per_token}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_table(title: str, stages: List[Tuple[str, str]],
                 results: Dict[str, Dict[str, float]], section: str):
    print()
    print("=" * 130)
    print(title)
    print("=" * 130)
    labels = [label for _, label in stages]
    header = f"{'method':>9}  " + "  ".join(f"{l:>10s}" for l in labels)
    print(header)
    print("-" * len(header))
    for method, secs in results.items():
        per_layer = secs[section]
        cells = "  ".join(f"{per_layer.get(l, 0.0):>10.3f}" for l in labels)
        print(f"{method:>9}  {cells}")


def _gpu_info() -> str:
    if not torch.cuda.is_available():
        return "(no CUDA)"
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    mem_gb = props.total_memory / 1024**3
    return f"{name} ({mem_gb:.0f}GB)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="full,think,spark,rotatek")
    ap.add_argument("--image", default="1k")
    ap.add_argument("--num_images", default="1",
                    help="Single int (e.g. 4) or comma-separated list "
                         "(e.g. 1,3,5,11) to sweep over multiple counts.")
    ap.add_argument("--max_num", type=int, default=48)
    ap.add_argument("--batch_size", type=int, default=1,
                    help="Number of batch slots for the chat call. B>1 "
                         "uses model.batch_chat with replicated inputs.")
    ap.add_argument("--n_repeats", type=int, default=3)
    ap.add_argument("--channel_ratio", default="0.75")
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--decode_layer", type=int, default=14,
                    help="Which layer to record decode-stage events on "
                         "(decode profile only fires on a single layer).")
    ap.add_argument("--tag", default="breakdown")
    args = ap.parse_args()

    # Parse num_images into list (single value or comma-separated sweep).
    num_images_list = [int(n.strip()) for n in str(args.num_images).split(",")
                        if n.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    print(f"GPU           : {_gpu_info()}")
    print(f"methods       : {args.methods}")
    print(f"image         : {args.image}×{num_images_list}  "
          f"max_num={args.max_num}")
    print(f"batch_size    : {args.batch_size}")
    print(f"channel_ratio : {args.channel_ratio}")
    print(f"max_new_tokens: {args.max_new_tokens}  n_repeats: {args.n_repeats}")
    print(f"decode_layer  : {args.decode_layer}")

    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "latency_breakdown")
    os.makedirs(log_dir, exist_ok=True)

    # Outer loop: num_images. Inner loop: methods. Print + save per
    # num_images so each scenario gets its own table & JSON.
    for n_img in num_images_list:
        if len(num_images_list) > 1:
            print(f"\n\n########## num_images = {n_img} ##########", flush=True)
        results: Dict[str, Dict[str, Dict[str, float]]] = {}
        for method in methods:
            # "full" is implemented as think + ratio=0.
            m, r = ("think", "0.0") if method == "full" else (
                method, args.channel_ratio,
            )
            results[method] = _measure_method(
                m, r, args.image, n_img,
                args.max_num, args.max_new_tokens, args.n_repeats,
                args.decode_layer,
                batch_size=args.batch_size,
                label=method,
            )

        _print_table(
            f"PREFILL per-layer breakdown (ms)  —  image={args.image}×{n_img}, "
            f"ratio={args.channel_ratio}, max_num={args.max_num}, "
            f"n_reps={args.n_repeats}",
            PREFILL_STAGES, results, "prefill",
        )
        _print_table(
            f"DECODE per-token breakdown (ms)  —  image={args.image}×{n_img}, "
            f"layer={args.decode_layer}, max_new_tokens={args.max_new_tokens}, "
            f"n_reps={args.n_repeats}",
            DECODE_STAGES, results, "decode",
        )

        json_path = os.path.join(
            log_dir,
            f"breakdown_{args.image}x{n_img}_{args.tag}.json",
        )
        with open(json_path, "w") as f:
            json.dump({
                "args": vars(args),
                "num_images": n_img,
                "gpu": _gpu_info(),
                "prefill_stages": [label for _, label in PREFILL_STAGES],
                "decode_stages": [label for _, label in DECODE_STAGES],
                "results": results,
            }, f, indent=2)
        print(f"\n[log] saved {json_path}")


if __name__ == "__main__":
    main()
