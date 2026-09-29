"""TextVQA / InfoVQA / ChartQA (lite): token-only vs ThinK / SparK / RotateK
at matched KV budgets, Qwen2.5-VL-7B. One `eval_accuracy.py` process per cell,
sharded over the GPUs in CUDA_VISIBLE_DEVICES (all GPUs if unset). Cells whose
log already holds a result are skipped, so re-running only redoes the ones
that failed."""
import itertools, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "lite_sweep"
DATASETS = ["textvqa_val_lite", "infovqa_val_lite", "chartqa_lite"]
ARMS = [  # (pruner, channel method, token ratio, channel ratio)
    ("visionzip", "think",   "0.23", "0.00"),  ("visionzip", "think",   "0.40", "0.75"),
    ("visionzip", "spark",   "0.40", "0.75"),  ("visionzip", "rotatek", "0.40", "0.75"),
    ("fastv",     "think",   "0.25", "0.00"),  ("fastv",     "think",   "0.40", "0.75"),
    ("fastv",     "spark",   "0.40", "0.75"),  ("fastv",     "rotatek", "0.40", "0.75"),
]


def visible_gpus():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        return [g.strip() for g in vis.split(",") if g.strip()]
    import torch
    return [str(i) for i in range(torch.cuda.device_count())]


def finished(log_path):
    return log_path.exists() and "results: " in log_path.read_text(errors="ignore")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gpus = visible_gpus()
    jobs = []
    for (pruner, method, ratio, chan), ds in itertools.product(ARMS, DATASETS):
        tag = f"{pruner}_{method}_t{ratio}_c{chan}_{ds}"
        if not finished(OUT / f"{tag}.log"):
            jobs.append((tag, [sys.executable, str(REPO / "scripts/paper/accuracy/eval_accuracy.py"),
                               "--model", "qwen", "--pruner", pruner, "--method", method,
                               "--token_ratio", ratio, "--channel_ratio", chan,
                               "--tasks", ds, "--output_dir", str(OUT)]))
    print(f"{len(jobs)} cells to run over {len(gpus)} GPUs", flush=True)
    running = []
    while jobs or running:
        busy = {g for *_, g in running}
        for gpu in [g for g in gpus if g not in busy]:
            if not jobs:
                break
            tag, argv = jobs.pop(0)
            log = open(OUT / f"{tag}.log", "w")
            p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=REPO,
                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
            running.append((p, tag, log, gpu))
            print(f"  [gpu{gpu} start] {tag}", flush=True)
        time.sleep(15)
        for r in running[:]:
            p, tag, log, gpu = r
            if p.poll() is not None:
                log.close()
                running.remove(r)
                print(f"  [gpu{gpu} done rc={p.returncode}] {tag}", flush=True)
    print("ALL CELLS DONE", flush=True)


if __name__ == "__main__":
    main()
