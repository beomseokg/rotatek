"""Find the maximum sustainable batch size for each pruning method at a
fixed prefill length.

Two modes (one file, one process per attempt for OOM isolation):

  measure: load model, run warmup + recorded gen at the given batch.
           prints `PEAK_GB=<float>` on success; exits 2 on CUDA OOM,
           3 on any other failure.

  sweep:   driver. For each method, exponential probe (1, 2, 4, ...)
           until OOM, then linear bisect between last_ok and first_oom.
           Spawns one subprocess per attempt — process-level isolation
           guarantees the allocator/cuda context is reset cleanly.

Usage:
  # Single attempt (called by the driver, but also useful manually):
  python max_batch_sweep.py measure --method think --batch 4 --prefill_length 64k

  # Full sweep across all four methods, 64K prefill:
  python max_batch_sweep.py sweep --prefill_length 64k \\
      --methods full,think,spark,rotatek --output sweep_64k.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Optional


def _measure(method: str, batches, prefill_length: str, channel_ratio: str,
             decode_tokens: int = 4) -> None:
    """Load model once, sweep through `batches` in order.

    For each batch, runs two generations and records:
      - peak_gb / reserved_gb        : memory at the long generation
      - prefill_s                     : time to first token (~prefill cost)
      - total_s                       : time for the full `decode_tokens` gen
      - decode_s, per_token_ms        : (total - prefill) / (N-1) — decode only
      - dec_tps                       : decode-only throughput (B*(N-1)/decode_s)
      - e2e_tps                       : end-to-end (B*N/total_s)

    With decode_tokens=4 (default), the timing fields are noisy but the memory
    measurement is the same — keep small to make pure-memory sweeps fast.
    Use decode_tokens>=64 for meaningful throughput numbers.

    Prints one structured line per batch:
      RESULT batch=N status=ok peak_gb=... [more fields]
      RESULT batch=N status=oom
    Exits 0 on clean completion of the list, 2 if any batch OOMed (allocator
    state may be unstable past an OOM, so the loop exits early).
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:512",
    )
    _REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)

    import torch

    PRETRAINED = "llava-hf/llama3-llava-next-8b-hf"
    PROMPT_TOKENS = 30
    TEXT_TOKENS = 30

    def parse_len(s: str) -> int:
        s = s.strip().lower()
        return int(float(s[:-1]) * 1024) if s.endswith("k") else int(s)

    prefill = parse_len(prefill_length)

    # Load model — same recipe as profile_memory.py.
    from lmms_eval import models
    method_arg = "think" if method == "full" else method
    ratio_arg = "0.0" if method == "full" else channel_ratio
    model_args = (
        f"pretrained={PRETRAINED},"
        f"visionzip_dominant_ratio=1.0,"
        f"visionzip_contextual_ratio=0.0,"
        f"attn_implementation=flash_attention_2,"
        f"channel_ratio={ratio_arg},"
        f"channel_method={method_arg},"
        f"decode_attention_backend=triton"
    )
    LM = models.get_model("llava_next_visionzip")
    lm = LM.create_from_arg_string(
        model_args, {"batch_size": 1, "max_batch_size": None, "device": "cuda:0"},
    )
    model = lm.model

    cfg = model.config
    sub = getattr(cfg, "text_config", cfg)
    sub.prompt_seqlen = PROMPT_TOKENS
    sub.query_seqlen = TEXT_TOKENS
    cfg.prompt_seqlen = PROMPT_TOKENS
    cfg.query_seqlen = TEXT_TOKENS
    hidden = sub.hidden_size
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    total_S = PROMPT_TOKENS + prefill + TEXT_TOKENS

    def build(B: int):
        e = torch.randn(B, total_S, hidden, dtype=dtype, device=device) * 0.5
        m = torch.ones(B, total_S, dtype=torch.long, device=device)
        return e, m

    import time as _time

    def gen(e, m, num_new):
        with torch.inference_mode():
            _ = model.language_model.generate(
                inputs_embeds=e, attention_mask=m,
                max_new_tokens=num_new, min_new_tokens=num_new,
                do_sample=False, use_cache=True,
            )

    def time_gen(e, m, num_new):
        torch.cuda.synchronize()
        t0 = _time.perf_counter()
        gen(e, m, num_new)
        torch.cuda.synchronize()
        return _time.perf_counter() - t0

    print("MODEL_READY", flush=True)

    # One-time warmup at the smallest batch — primes lazy initialisation
    # (cuDNN benchmark, Triton autotune workspaces, etc.) so the first
    # *measured* call's peak is not inflated by one-off allocations.
    # Warm with the actual decode length so triton autotune sees the same
    # decode-step shapes that the measured runs will use.
    if batches:
        e, m = build(min(batches))
        gen(e, m, decode_tokens)
        del e, m
        torch.cuda.empty_cache()

    saw_oom = False
    for B in batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            # First run: 1 decode token → time ≈ prefill + 1 decode step.
            e, m = build(B)
            t_prefill = time_gen(e, m, 1)
            del e, m
            torch.cuda.empty_cache()

            # Second run: full decode_tokens. Peak memory captured here
            # includes any decode-time KV growth.
            e, m = build(B)
            t_total = time_gen(e, m, decode_tokens)
        except torch.cuda.OutOfMemoryError:
            print(f"RESULT batch={B} status=oom", flush=True)
            saw_oom = True
            break

        peak = torch.cuda.max_memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3

        decode_s = max(t_total - t_prefill, 1e-9)
        n_decode_steps = max(decode_tokens - 1, 1)  # tokens generated in decode_s
        per_token_ms = decode_s / n_decode_steps * 1000.0
        dec_tps = (B * n_decode_steps) / decode_s
        e2e_tps = (B * decode_tokens) / max(t_total, 1e-9)

        print(
            f"RESULT batch={B} status=ok peak_gb={peak:.3f} reserved_gb={reserved:.3f} "
            f"prefill_s={t_prefill:.3f} total_s={t_total:.3f} decode_s={decode_s:.3f} "
            f"per_token_ms={per_token_ms:.2f} dec_tps={dec_tps:.2f} e2e_tps={e2e_tps:.2f}",
            flush=True,
        )
        del e, m

    if saw_oom:
        sys.exit(2)


def _spawn(method: str, batches, prefill_length: str, channel_ratio: str,
           python_bin: str, timeout_s: float) -> list:
    """Run one subprocess measuring `batches` in order.

    Returns a list of dicts (one per attempted batch), each with at minimum
    `batch` and `status` ('ok' | 'oom' | 'error' | 'timeout'). Successful
    entries also have `peak_gb` and `reserved_gb`.
    """
    cmd = [
        python_bin, __file__, "measure",
        "--method", method,
        "--batches", ",".join(str(b) for b in batches),
        "--prefill_length", prefill_length,
        "--channel_ratio", channel_ratio,
    ]
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
    except subprocess.TimeoutExpired:
        return [{"batch": batches[0] if batches else None,
                 "status": "timeout", "elapsed_s": time.time() - t0}]

    elapsed = time.time() - t0
    out = (p.stdout or "") + "\n" + (p.stderr or "")

    results = []
    for line in (p.stdout or "").splitlines():
        if not line.startswith("RESULT "):
            continue
        # RESULT batch=N status=ok peak_gb=X reserved_gb=Y
        # RESULT batch=N status=oom
        kv = {}
        for tok in line[len("RESULT "):].split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                kv[k] = v
        try:
            entry = {"batch": int(kv["batch"]), "status": kv.get("status", "?")}
        except (KeyError, ValueError):
            continue
        if "peak_gb" in kv:
            entry["peak_gb"] = float(kv["peak_gb"])
        if "reserved_gb" in kv:
            entry["reserved_gb"] = float(kv["reserved_gb"])
        results.append(entry)

    # If subprocess died with non-zero rc and we didn't see an oom RESULT
    # line, classify the trailing batch attempt as error.
    if p.returncode not in (0, 2):
        # Identify which batch never completed.
        attempted = {r["batch"] for r in results}
        for b in batches:
            if b not in attempted:
                if (
                    "OutOfMemoryError" in out
                    or "CUDA out of memory" in out
                    or "out of memory" in out.lower()
                ):
                    results.append({"batch": b, "status": "oom"})
                else:
                    results.append({
                        "batch": b, "status": "error",
                        "returncode": p.returncode,
                        "stderr_tail": out[-2000:],
                    })
                break  # only mark the first one that didn't run
    if results:
        results[-1]["elapsed_s_total"] = elapsed
    else:
        results.append({"batch": batches[0] if batches else None,
                        "status": "error", "elapsed_s_total": elapsed,
                        "stderr_tail": out[-2000:]})
    return results


def _sweep(methods, prefill_length, channel_ratio, output, python_bin, timeout_s,
           max_probe):
    """Hybrid sweep: exponential probe to find OOM bracket, then
    linear-dense within bracket for exact OOM boundary + cliff data.

    One subprocess per phase per method (instead of per-attempt) — model is
    loaded once and reused across all batches in the phase, which is the
    bulk of the per-attempt overhead.
    """
    results = {}
    for method in methods:
        print(f"\n========== METHOD: {method} ==========", flush=True)
        method_log = []
        last_ok: Optional[int] = None
        first_oom: Optional[int] = None

        # Phase 1: exponential probe — 1, 2, 4, 8, ..., capped at max_probe.
        probe = []
        b = 1
        while b <= max_probe:
            probe.append(b)
            b *= 2
        print(f"[probe] {method} batches={probe}", flush=True)
        phase1 = _spawn(method, probe, prefill_length, channel_ratio,
                        python_bin, timeout_s)
        for r in phase1:
            print(f"  → {r}", flush=True)
            method_log.append(r)
            if r["status"] == "ok":
                last_ok = max(last_ok or 0, r["batch"])
            elif r["status"] == "oom":
                first_oom = r["batch"]
                break
            else:
                print(f"[abort] {method}: unexpected status {r['status']}", flush=True)
                break

        # Phase 2: linear dense within (last_ok, first_oom).
        if last_ok is not None and first_oom is not None and first_oom - last_ok > 1:
            dense = list(range(last_ok + 1, first_oom))
            print(f"[dense] {method} batches={dense}", flush=True)
            phase2 = _spawn(method, dense, prefill_length, channel_ratio,
                            python_bin, timeout_s)
            for r in phase2:
                print(f"  → {r}", flush=True)
                method_log.append(r)
                if r["status"] == "ok":
                    last_ok = max(last_ok, r["batch"])
                elif r["status"] == "oom":
                    first_oom = min(first_oom, r["batch"])
                    break

        results[method] = {
            "max_batch": last_ok,
            "first_oom": first_oom,
            "log": method_log,
        }
        # Persist incrementally so we don't lose progress.
        with open(output, "w") as f:
            json.dump({
                "prefill_length": prefill_length,
                "channel_ratio": channel_ratio,
                "results": results,
            }, f, indent=2)
        print(f"[save] {output}", flush=True)

    print("\n========== SUMMARY ==========")
    for m, r in results.items():
        print(f"  {m:8s}  max_batch = {r['max_batch']}  (first_oom = {r['first_oom']})")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    pm = sub.add_parser("measure")
    pm.add_argument("--method", required=True, choices=["full", "think", "spark", "rotatek"])
    pm.add_argument("--batches", required=True,
                    help="Comma-separated batch list, e.g. '1,2,4,8,16'")
    pm.add_argument("--prefill_length", default="64k")
    pm.add_argument("--channel_ratio", default="0.75")
    pm.add_argument("--decode_tokens", type=int, default=4,
                    help="Decode tokens per generation. 4 = fast pure-memory "
                         "sweep. >=64 to get meaningful throughput numbers.")

    ps = sub.add_parser("sweep")
    ps.add_argument("--methods", default="full,think,spark,rotatek")
    ps.add_argument("--prefill_length", default="64k")
    ps.add_argument("--channel_ratio", default="0.75")
    ps.add_argument("--output", default="sweep_64k.json")
    ps.add_argument("--python_bin", default=sys.executable)
    ps.add_argument("--timeout_s", type=float, default=1800.0,
                    help="Per-subprocess timeout (one subprocess can sweep "
                         "many batches, so allow generously).")
    ps.add_argument("--max_probe", type=int, default=64,
                    help="Cap exponential probe at this batch size.")

    args = parser.parse_args()

    if args.mode == "measure":
        batches = [int(x) for x in args.batches.split(",") if x.strip()]
        _measure(args.method, batches, args.prefill_length, args.channel_ratio,
                 args.decode_tokens)
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        _sweep(methods, args.prefill_length, args.channel_ratio,
               args.output, args.python_bin, args.timeout_s, args.max_probe)


if __name__ == "__main__":
    main()
