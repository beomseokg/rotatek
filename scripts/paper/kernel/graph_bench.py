"""Eager-launch vs CUDA-graph-captured timing for the RotateK and dense kernels.

The repo's profiler already brackets the whole rep loop with one CUDA-event pair,
so it is not per-call wall clock -- but it still includes per-launch CPU cost,
which dominates once the kernels are small. Capturing the call in a CUDA graph
removes launch cost entirely and isolates GPU work.
"""
import argparse, os, sys
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts", "paper", "kernel"))
import torch
import kernel_profile as KP


def bench_eager(fn, reps, warmup):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / reps * 1000.0  # us


def bench_graph(fn, reps, warmup):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup): fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    for _ in range(5): g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(reps): g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / reps * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="1000x1,4000x1,16000x1,1000x32,16000x32")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    a = ap.parse_args()
    print(f"{'S':>6} {'B':>3} | {'eager rotatek':>13} {'eager dense':>11} {'eager x':>7}"
          f" | {'graph rotatek':>13} {'graph dense':>11} {'graph x':>7}")
    print("-" * 92)
    for cfg in a.configs.split(","):
        S, B = (int(x) for x in cfg.split("x"))
        inp = KP._make_inputs(S, bsz=B)
        # rebuild the closures the profiler uses
        from rotatek.kernels.fused_decode import rotatek_decode_fused_triton
        from rotatek.kernels.full_channel_flash_decoding import full_channel_decode_triton
        def rot():
            return rotatek_decode_fused_triton(
                q_full=inp["q_full"], R_partial=inp["R_partial"], delta_mu=inp["delta_mu"],
                k_full=inp["k_full"], v_full=inp["v_full"], mask_full=inp["mask_full"],
                k_sparse=inp["k_sparse"], v_sparse=inp["v_sparse"],
                num_kv_groups=KP.H_Q // KP.H_KV)
        def den():
            return full_channel_decode_triton(
                inp["q_full"], inp["k_dense"], inp["v_dense"],
                num_kv_groups=KP.H_Q // KP.H_KV)
        er, ed = bench_eager(rot, a.reps, a.warmup), bench_eager(den, a.reps, a.warmup)
        try:
            gr, gd = bench_graph(rot, a.reps, a.warmup), bench_graph(den, a.reps, a.warmup)
            gs = f"{gr:13.1f} {gd:11.1f} {gd/gr:6.2f}x"
        except Exception as ex:
            gs = f"  graph capture failed: {type(ex).__name__}: {str(ex)[:40]}"
        print(f"{S:>6} {B:>3} | {er:13.1f} {ed:11.1f} {ed/er:6.2f}x | {gs}")


main()
