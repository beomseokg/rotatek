"""Plot peak memory vs batch size for each method from a sweep JSON.

Reads `sweep_<len>.json` produced by max_batch_sweep_*.py and writes:
  - <output>.png : peak_GB vs batch (one line per method, dots at measured
    points; vertical marker for max sustainable batch).
  - prints a summary table to stdout.

Usage:
  python plot_max_batch_sweep.py sweep_16k.json --output sweep_16k.png
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="sweep_<len>.json")
    parser.add_argument("--output", default=None,
                        help="PNG output path (default: <input>.png)")
    args = parser.parse_args()

    out_png = args.output or args.input.replace(".json", ".png")

    with open(args.input) as f:
        data = json.load(f)
    prefill = data.get("prefill_length", "?")
    ratio = data.get("channel_ratio", "?")
    results = data["results"]

    # Collect (batch, peak_gb) for each method, dedup keeping the OK measurements.
    points = defaultdict(dict)  # method → {batch: peak_gb}
    oom = {}                    # method → first OOM batch
    max_ok = {}                 # method → max sustainable batch
    for method, r in results.items():
        max_ok[method] = r.get("max_batch")
        oom[method] = r.get("first_oom")
        for entry in r["log"]:
            if entry["status"] == "ok" and entry.get("peak_gb") is not None:
                points[method][entry["batch"]] = entry["peak_gb"]

    # Print summary table.
    print(f"\nPrefill={prefill}, ratio={ratio}\n")
    print(f"{'method':<10}{'max_batch':>12}{'first_oom':>12}{'peak@B=1':>14}"
          f"{'slope GB/B':>14}")
    print("-" * 62)
    summary = {}
    for method in results:
        pts = sorted(points[method].items())
        if not pts:
            continue
        b1 = pts[0][0]
        peak1 = pts[0][1]
        # Linear fit slope from first to last OK point.
        if len(pts) >= 2:
            b_last, peak_last = pts[-1]
            slope = (peak_last - peak1) / max(b_last - b1, 1)
        else:
            slope = float("nan")
        summary[method] = (max_ok[method], oom[method], peak1, slope, pts)
        print(f"{method:<10}{str(max_ok[method]):>12}{str(oom[method]):>12}"
              f"{peak1:>13.2f}G{slope:>13.2f}G")

    # Plot.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    color_cycle = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    method_order = ["full", "think", "rotatek", "spark"]
    methods_plot = [m for m in method_order if m in summary] + \
                   [m for m in summary if m not in method_order]

    for i, method in enumerate(methods_plot):
        max_b, oom_b, peak1, slope, pts = summary[method]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        c = color_cycle[i % len(color_cycle)]
        ax.plot(xs, ys, "o-", color=c, label=f"{method} (max B={max_b})", linewidth=1.6)
        if max_b is not None:
            ax.axvline(max_b, color=c, alpha=0.25, linestyle="--", linewidth=1.0)

    ax.axhline(80.0, color="black", alpha=0.4, linestyle=":", linewidth=1.0,
               label="A100 80GB")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Peak GPU memory (GB)")
    ax.set_title(f"LLaVA-Next peak memory vs batch (prefill={prefill}, ratio={ratio})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"\nWrote {out_png}")


if __name__ == "__main__":
    main()
