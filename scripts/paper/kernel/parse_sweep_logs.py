"""Parse one or more sweep log files (RESULT lines from `_measure`) and
write a unified sweep JSON.

A log file is plain text containing:
  === METHOD_NAME ===          (section delimiter; underscores split off,
                                first token lowercased becomes the method)
  RESULT batch=N status=ok|oom [field=value ...]

When the same (method, batch) appears in multiple files, fields are merged
(later files override on key collision).

Usage:
  python parse_sweep_logs.py \\
      --logs path1.output path2.output path3.output \\
      --output sweep_16k.json \\
      --prefill_length 16k --channel_ratio 0.75
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


SECTION_RE = re.compile(r"^=== ([A-Z_]+) ===\s*$")


def _parse_result_line(line: str) -> dict | None:
    """RESULT batch=N status=ok foo=bar baz=qux  →  {batch:N, status:ok, ...}"""
    if not line.startswith("RESULT "):
        return None
    out = {}
    for tok in line[len("RESULT "):].strip().split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        # Try int → float → str.
        try:
            out[k] = int(v)
            continue
        except ValueError:
            pass
        try:
            out[k] = float(v)
            continue
        except ValueError:
            pass
        out[k] = v
    return out if "batch" in out else None


def parse_log(path: Path) -> dict:
    """{method: {batch: entry}}"""
    out = defaultdict(dict)
    if not path.exists():
        return out
    cur = None
    for line in path.read_text().splitlines():
        m = SECTION_RE.match(line)
        if m:
            cur = m.group(1).split("_")[0].lower()
            continue
        entry = _parse_result_line(line)
        if entry and cur:
            b = entry["batch"]
            # Merge with any existing entry for this (method, batch).
            existing = out[cur].get(b, {})
            existing.update(entry)
            out[cur][b] = existing
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prefill_length", default="16k")
    ap.add_argument("--channel_ratio", default="0.75")
    ap.add_argument("--seed_full",
                    default="1:19.406,2:23.240,4:30.909,8:46.248",
                    help="Optional batch:peak_gb seed for full (captured "
                         "before the sweep was rebuilt). Empty to disable.")
    ap.add_argument("--seed_think",
                    default="1:18.656,2:21.740,4:27.909",
                    help="Same, for think (smoke-test data).")
    args = ap.parse_args()

    by_method = defaultdict(dict)
    for p in args.logs:
        for m, b2e in parse_log(Path(p)).items():
            for b, entry in b2e.items():
                base = by_method[m].get(b, {})
                base.update(entry)
                by_method[m][b] = base

    # Apply seeds for any (method, batch) not already present.
    def apply_seed(method: str, spec: str):
        for chunk in spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            b_str, peak_str = chunk.split(":")
            b = int(b_str)
            if b not in by_method[method]:
                by_method[method][b] = {
                    "batch": b, "status": "ok", "peak_gb": float(peak_str),
                }
    if args.seed_full:
        apply_seed("full", args.seed_full)
    if args.seed_think:
        apply_seed("think", args.seed_think)

    methods = ["full", "think", "spark", "rotatek"]
    results = {}
    for m in methods:
        log = sorted(by_method[m].values(), key=lambda r: r["batch"])
        oks = [r["batch"] for r in log if r.get("status") == "ok"]
        ooms = [r["batch"] for r in log if r.get("status") == "oom"]
        results[m] = {
            "max_batch": max(oks) if oks else None,
            "first_oom": min(ooms) if ooms else None,
            "log": log,
        }

    with open(args.output, "w") as f:
        json.dump({
            "prefill_length": args.prefill_length,
            "channel_ratio": args.channel_ratio,
            "results": results,
        }, f, indent=2)
    print(f"Wrote {args.output}")
    print()
    print(f"{'method':<10}{'max_batch':>12}{'first_oom':>12}{'#points':>10}")
    print("-" * 44)
    for m in methods:
        r = results[m]
        n = sum(1 for x in r["log"] if x.get("status") == "ok")
        print(f"{m:<10}{str(r['max_batch']):>12}{str(r['first_oom']):>12}{n:>10}")


if __name__ == "__main__":
    main()
