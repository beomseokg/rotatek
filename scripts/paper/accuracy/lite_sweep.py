"""TextVQA / InfoVQA / ChartQA (lite): token-only vs ThinK / SparK / RotateK
at matched KV budgets, Qwen2.5-VL-7B. Runs in the existing `rotatek` env
against the original repo, one cell per process, sharded over GPUs."""
import itertools, os, subprocess, sys, time
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PY = sys.executable
OUT = Path(REPO) / "results" / "lite_sweep"
DATASETS = ["textvqa_val_lite", "infovqa_val_lite", "chartqa_lite"]
ARMS = [  # (framework, channel method, dominant|keep, channel_ratio)
    ("vz", "think",   "0.23", "0.00"),   ("vz", "think",   "0.40", "0.750"),
    ("vz", "spark",   "0.40", "0.750"),  ("vz", "rotatek", "0.40", "0.750"),
    ("fv", "think",   "0.25", "0.000"),  ("fv", "think",   "0.40", "0.750"),
    ("fv", "spark",   "0.40", "0.750"),  ("fv", "rotatek", "0.40", "0.750"),
]
GPUS = [0, 1, 2, 3, 4, 5, 6]
CODE = '''
import os, sys
sys.path.insert(0, "{repo}")
sys.path.insert(0, "{repo}/scripts/paper/accuracy")
os.chdir("{repo}")
import {mod} as M
M.run_evaluate("{out}", dataset="{ds}", method="{method}", log=False,
               {kw}=["{ratio}"], channel_ratios_override=["{chan}"])
'''

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(a, d) for a, d in itertools.product(ARMS, DATASETS)]
    print(f"{len(jobs)} cells over {len(GPUS)} GPUs", flush=True)
    running, idx = [], 0
    while idx < len(jobs) or running:
        busy = {g for _, _, _, g in running}
        free = [g for g in GPUS if g not in busy]
        while idx < len(jobs) and free:
            (fw, method, ratio, chan), ds = jobs[idx]; gpu = free.pop(0)
            tag = f"{fw}_{method}_r{ratio}_c{chan}_{ds}"
            code = CODE.format(repo=REPO, out=OUT, ds=ds, method=method,
                               mod="visionzip_qwen" if fw == "vz" else "fastv_qwen",
                               kw="dominant_ratios_override" if fw == "vz" else "fast_v_keep_ratios_override",
                               ratio=ratio, chan=chan)
            log = open(OUT / f"{tag}.log", "w")
            p = subprocess.Popen([PY, "-c", code], stdout=log, stderr=subprocess.STDOUT, cwd=REPO,
                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
            running.append((p, tag, log, gpu)); print(f"  [gpu{gpu} start] {tag}", flush=True); idx += 1
        time.sleep(15)
        for r in running[:]:
            p, tag, log, gpu = r
            if p.poll() is not None:
                log.close(); running.remove(r)
                print(f"  [gpu{gpu} done rc={p.returncode}] {tag}", flush=True)
    print("ALL CELLS DONE", flush=True)

main()
