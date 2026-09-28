"""Isolated microbench for rotatek vs dense decode kernels — for ncu profiling.

Synthesizes tensors matching the real decode shape at S=12K (image=1k×1
max_num=48 with InternVL2.5-8B: H_q=32, H_kv=8, D=128, D_keep=32 at
channel_ratio=0.75) and runs each kernel in a tight steady-state loop.

The script is structured for two use modes:

1. **Wall-time bench** (default):
       python -m latency.kernel_profile --kernel both --reps 200

   Prints per-call ms for each kernel after warmup.

2. **ncu kernel-internal profiling**:
       ncu --target-processes all \\
           --kernel-name regex:'_rotatek_sparse_kernel|_dense_phase1_kernel' \\
           --launch-skip 50 --launch-count 2 \\
           --set detailed -o /tmp/rotatek_profile \\
           python -m latency.kernel_profile --kernel both --reps 100

   Then open /tmp/rotatek_profile.ncu-rep in Nsight Compute UI, OR
   summarize the SOL roof / stall reasons via:
       ncu --import /tmp/rotatek_profile.ncu-rep --print-summary per-kernel

Key metrics to compare in the ncu output:
  - SM Throughput / Memory Throughput (% of peak)
  - Achieved Occupancy
  - Warp Cycles Per Issued Instruction + top stall reasons
  - L1/L2 Hit Rate
  - DRAM Throughput
  - Registers Per Thread

If rotatek_sparse hits the roof on a different bottleneck than dense_phase1
at the same input bytes, we know exactly what to fix.
"""
import os
import sys

# Make the repo root importable so `rotatek` resolves no matter where this
# script is launched from.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:512",
)

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rotatek.kernels.full_channel_flash_decoding import (  # noqa: E402
    full_channel_decode_triton,
)
from rotatek.kernels.fused_decode import rotatek_decode_fused_triton  # noqa: E402
from rotatek.decode import rotatek_decode_triton  # noqa: E402


# InternVL2.5-8B decode shape at channel_ratio=0.75
H_Q = 32
H_KV = 8
NUM_KV_GROUPS = H_Q // H_KV
HEAD_DIM = 128
HEAD_DIM_KEEP = 32  # 75% sparsity → D_keep = 128 - floor(128*0.75) = 32
S_PROMPT = 50       # typical InternVL prompt span
S_TEXT = 30         # mid-decode text span
DTYPE = torch.bfloat16


def _make_inputs(s_vision: int, bsz: int = 1, device: str = "cuda"):
    """Tensors with shapes matching the InternVL adapter's actual decode call."""
    s_full = S_PROMPT + S_TEXT  # k_full / v_full = prompt + text only
    s_total_dense = S_PROMPT + s_vision + S_TEXT

    q_full = torch.randn(bsz, H_Q, HEAD_DIM, device=device, dtype=DTYPE)

    # ---- rotatek inputs (truncated K, full V) -------------------------
    R_partial = torch.randn(bsz, H_KV, HEAD_DIM, HEAD_DIM_KEEP,
                            device=device, dtype=DTYPE)
    delta_mu = torch.randn(bsz, H_KV, HEAD_DIM, device=device, dtype=DTYPE)
    k_full = torch.randn(bsz, H_KV, s_full, HEAD_DIM, device=device, dtype=DTYPE)
    v_full = torch.randn(bsz, H_KV, s_full, HEAD_DIM, device=device, dtype=DTYPE)
    mask_full = torch.ones(bsz, s_full, device=device, dtype=torch.uint8)
    k_sparse = torch.randn(bsz, H_KV, s_vision, HEAD_DIM_KEEP,
                           device=device, dtype=DTYPE)
    v_sparse = torch.randn(bsz, H_KV, s_vision, HEAD_DIM,
                           device=device, dtype=DTYPE)

    # ---- dense inputs (all tokens at full D, recovered) --------------
    k_dense = torch.randn(bsz, H_KV, s_total_dense, HEAD_DIM,
                          device=device, dtype=DTYPE)
    v_dense = torch.randn(bsz, H_KV, s_total_dense, HEAD_DIM,
                          device=device, dtype=DTYPE)

    return {
        "q_full": q_full,
        "R_partial": R_partial,
        "delta_mu": delta_mu,
        "k_full": k_full,
        "v_full": v_full,
        "mask_full": mask_full,
        "k_sparse": k_sparse,
        "v_sparse": v_sparse,
        "k_dense": k_dense,
        "v_dense": v_dense,
    }


def _bench(name: str, fn, reps: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    ms_per_call = s.elapsed_time(e) / reps
    print(f"  {name:>10s}: {ms_per_call*1000:7.1f} us/call  "
          f"(reps={reps}, warmup={warmup})")
    return ms_per_call


def _bench_rotatek_breakdown(inp: dict, reps: int, warmup: int) -> None:
    """Time each rotatek sub-kernel (prelude → sparse → full → merge)
    separately. Mirrors the launcher in `methods/rotatek/kernel.py` but
    wraps each phase in its own cuda-event pair so we can see where the
    time actually goes.

    Loops the FOUR sub-kernels back-to-back inside one timed window per
    stage so launch latency is amortized."""
    import math
    from rotatek.kernels.sparse_channel_flash_decoding import _next_power_of_2
    from rotatek.kernels.fused_decode import (
        _rotatek_sparse_kernel,
        _rotatek_combined_kernel,
    )

    q_full = inp["q_full"].contiguous()
    R_partial = inp["R_partial"].contiguous()
    delta_mu = inp["delta_mu"].contiguous()
    k_full = inp["k_full"]
    v_full = inp["v_full"]
    mask_full = inp["mask_full"]
    k_sparse = inp["k_sparse"]
    v_sparse = inp["v_sparse"]

    bsz, heads, head_dim = q_full.shape
    head_dim_keep = R_partial.shape[-1]
    seq_full = k_full.shape[-2]
    seq_sparse = k_sparse.shape[-2]
    num_kv_groups = NUM_KV_GROUPS

    scale = 1.0 / math.sqrt(head_dim)
    block_d = _next_power_of_2(head_dim)
    block_dk = _next_power_of_2(head_dim_keep)
    BLOCK_N = 64

    num_splits_sparse = max(1, min((seq_sparse + BLOCK_N - 1) // BLOCK_N, 64))
    sparse_per_split = (seq_sparse + num_splits_sparse - 1) // num_splits_sparse

    total_partials = bsz * heads * num_splits_sparse
    partial_m = torch.empty(total_partials, device=q_full.device, dtype=torch.float32)
    partial_l = torch.empty(total_partials, device=q_full.device, dtype=torch.float32)
    partial_acc = torch.empty(total_partials, head_dim, device=q_full.device, dtype=torch.float32)
    out = torch.empty((bsz, heads, head_dim), device=q_full.device, dtype=q_full.dtype)

    def _launch_sparse():
        _rotatek_sparse_kernel[(bsz, heads, num_splits_sparse)](
            q_full, R_partial, delta_mu,
            k_sparse, v_sparse,
            partial_m, partial_l, partial_acc,
            q_full.stride(0), q_full.stride(1), q_full.stride(2),
            R_partial.stride(0), R_partial.stride(1),
            R_partial.stride(2), R_partial.stride(3),
            delta_mu.stride(0), delta_mu.stride(1), delta_mu.stride(2),
            k_sparse.stride(0), k_sparse.stride(1), k_sparse.stride(2), k_sparse.stride(3),
            v_sparse.stride(0), v_sparse.stride(1), v_sparse.stride(2), v_sparse.stride(3),
            partial_m.stride(0), partial_l.stride(0),
            partial_acc.stride(0), partial_acc.stride(1),
            seq_sparse, head_dim, head_dim_keep,
            sparse_per_split, scale,
            HAS_BIAS=True,
            NUM_KV_GROUPS=num_kv_groups,
            NUM_SPLITS_SPARSE=num_splits_sparse,
            BLOCK_N=BLOCK_N, BLOCK_D=block_d, BLOCK_DK=block_dk,
            num_warps=4, num_stages=3,
        )

    def _launch_combined():
        _rotatek_combined_kernel[(bsz, heads)](
            q_full, k_full, v_full, mask_full,
            partial_m, partial_l, partial_acc, out,
            q_full.stride(0), q_full.stride(1), q_full.stride(2),
            k_full.stride(0), k_full.stride(1), k_full.stride(2), k_full.stride(3),
            v_full.stride(0), v_full.stride(1), v_full.stride(2), v_full.stride(3),
            mask_full.stride(0), mask_full.stride(1),
            partial_m.stride(0), partial_l.stride(0),
            partial_acc.stride(0), partial_acc.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            seq_full, head_dim, scale,
            NUM_KV_GROUPS=num_kv_groups,
            NUM_SPLITS_SPARSE=num_splits_sparse,
            BLOCK_N=BLOCK_N, BLOCK_D=block_d,
            num_warps=4, num_stages=3,
        )

    stages = [
        ("sparse",   _launch_sparse),
        ("combined", _launch_combined),
    ]

    # Warmup all four kernels to avoid compile cost in the timed runs.
    for _ in range(warmup):
        for _, fn in stages:
            fn()
    torch.cuda.synchronize()

    times_us = {}
    for name, fn in stages:
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        torch.cuda.synchronize()
        times_us[name] = s.elapsed_time(e) * 1000.0 / reps

    total = sum(times_us.values())
    print(f"  {'stage':>10s}  {'us/call':>10s}   {'%':>6s}")
    for name, _ in stages:
        t = times_us[name]
        print(f"  {name:>10s}  {t:10.1f}   {t/total*100:5.1f}%")
    print(f"  {'sum':>10s}  {total:10.1f}")


def _run_scan_config(inp: dict, reps: int, warmup: int) -> None:
    """Top-level: scan (num_warps, num_stages) for sparse, combined, and
    dense_phase1 kernels."""
    import math
    from rotatek.kernels.sparse_channel_flash_decoding import _next_power_of_2
    from rotatek.kernels.full_channel_flash_decoding import _dense_phase1_kernel
    from rotatek.kernels.fused_decode import (
        _rotatek_sparse_kernel,
        _rotatek_combined_kernel,
    )

    q_full = inp["q_full"].contiguous()
    R_partial = inp["R_partial"].contiguous()
    delta_mu = inp["delta_mu"].contiguous()
    k_full = inp["k_full"]
    v_full = inp["v_full"]
    mask_full = inp["mask_full"]
    k_sparse = inp["k_sparse"]
    v_sparse = inp["v_sparse"]
    k_dense = inp["k_dense"]
    v_dense = inp["v_dense"]

    bsz, heads, head_dim = q_full.shape
    head_dim_keep = R_partial.shape[-1]
    seq_full = k_full.shape[-2]
    seq_sparse = k_sparse.shape[-2]
    seq_dense = k_dense.shape[-2]
    num_kv_groups = NUM_KV_GROUPS

    scale = 1.0 / math.sqrt(head_dim)
    block_d = _next_power_of_2(head_dim)
    block_dk = _next_power_of_2(head_dim_keep)
    BLOCK_N = 64

    num_splits_sparse = max(1, min((seq_sparse + BLOCK_N - 1) // BLOCK_N, 64))
    sparse_per_split = (seq_sparse + num_splits_sparse - 1) // num_splits_sparse
    num_splits_dense = max(1, min((seq_dense + BLOCK_N - 1) // BLOCK_N, 64))
    tokens_per_split_dense = (seq_dense + num_splits_dense - 1) // num_splits_dense

    total_partials = bsz * heads * num_splits_sparse
    partial_m = torch.empty(total_partials, device=q_full.device, dtype=torch.float32)
    partial_l = torch.empty(total_partials, device=q_full.device, dtype=torch.float32)
    partial_acc = torch.empty(total_partials, head_dim, device=q_full.device, dtype=torch.float32)
    out = torch.empty((bsz, heads, head_dim), device=q_full.device, dtype=q_full.dtype)

    total_partials_d = bsz * heads * num_splits_dense
    partial_m_d = torch.empty(total_partials_d, device=q_full.device, dtype=torch.float32)
    partial_l_d = torch.empty(total_partials_d, device=q_full.device, dtype=torch.float32)
    partial_acc_d = torch.empty(total_partials_d, head_dim, device=q_full.device, dtype=torch.float32)

    def _sparse_factory(num_warps, num_stages):
        def _go():
            _rotatek_sparse_kernel[(bsz, heads, num_splits_sparse)](
                q_full, R_partial, delta_mu,
                k_sparse, v_sparse,
                partial_m, partial_l, partial_acc,
                q_full.stride(0), q_full.stride(1), q_full.stride(2),
                R_partial.stride(0), R_partial.stride(1),
                R_partial.stride(2), R_partial.stride(3),
                delta_mu.stride(0), delta_mu.stride(1), delta_mu.stride(2),
                k_sparse.stride(0), k_sparse.stride(1), k_sparse.stride(2), k_sparse.stride(3),
                v_sparse.stride(0), v_sparse.stride(1), v_sparse.stride(2), v_sparse.stride(3),
                partial_m.stride(0), partial_l.stride(0),
                partial_acc.stride(0), partial_acc.stride(1),
                seq_sparse, head_dim, head_dim_keep,
                sparse_per_split, scale,
                HAS_BIAS=True,
                NUM_KV_GROUPS=num_kv_groups,
                NUM_SPLITS_SPARSE=num_splits_sparse,
                BLOCK_N=BLOCK_N, BLOCK_D=block_d, BLOCK_DK=block_dk,
                num_warps=num_warps, num_stages=num_stages,
            )
        return _go

    def _combined_factory(num_warps, num_stages):
        def _go():
            _rotatek_combined_kernel[(bsz, heads)](
                q_full, k_full, v_full, mask_full,
                partial_m, partial_l, partial_acc, out,
                q_full.stride(0), q_full.stride(1), q_full.stride(2),
                k_full.stride(0), k_full.stride(1), k_full.stride(2), k_full.stride(3),
                v_full.stride(0), v_full.stride(1), v_full.stride(2), v_full.stride(3),
                mask_full.stride(0), mask_full.stride(1),
                partial_m.stride(0), partial_l.stride(0),
                partial_acc.stride(0), partial_acc.stride(1),
                out.stride(0), out.stride(1), out.stride(2),
                seq_full, head_dim, scale,
                NUM_KV_GROUPS=num_kv_groups,
                NUM_SPLITS_SPARSE=num_splits_sparse,
                BLOCK_N=BLOCK_N, BLOCK_D=block_d,
                num_warps=num_warps, num_stages=num_stages,
            )
        return _go

    def _dense_factory(num_warps, num_stages):
        def _go():
            _dense_phase1_kernel[(bsz, heads, num_splits_dense)](
                q_full, k_dense, v_dense,
                partial_m_d, partial_l_d, partial_acc_d,
                q_full.stride(0), q_full.stride(1), q_full.stride(2),
                k_dense.stride(0), k_dense.stride(1), k_dense.stride(2), k_dense.stride(3),
                v_dense.stride(0), v_dense.stride(1), v_dense.stride(2), v_dense.stride(3),
                partial_m_d.stride(0), partial_l_d.stride(0),
                partial_acc_d.stride(0), partial_acc_d.stride(1),
                seq_dense, head_dim,
                tokens_per_split_dense, scale,
                NUM_KV_GROUPS=num_kv_groups,
                NUM_SPLITS=num_splits_dense,
                BLOCK_N=BLOCK_N, BLOCK_D=block_d,
                num_warps=num_warps, num_stages=num_stages,
            )
        return _go

    configs = [(w, s) for w in (1, 2, 4, 8) for s in (1, 2, 3, 4)]

    _scan_kernel_config("rotatek_sparse", _sparse_factory, configs, reps, warmup)
    _scan_kernel_config("rotatek_combined", _combined_factory, configs, reps, warmup)
    _scan_kernel_config("dense_phase1", _dense_factory, configs, reps, warmup)


def _scan_kernel_config(name: str, launch_factory, configs: list,
                          reps: int, warmup: int) -> None:
    """Scan (num_warps, num_stages) combos for one kernel and print
    a sorted table of per-call us. `launch_factory(num_warps, num_stages)`
    returns a zero-arg callable that launches the kernel with that config.
    """
    print(f"\n--- {name} (num_warps × num_stages) ---")
    print(f"  {'warps':>5s} {'stages':>6s}   {'us/call':>10s}")
    results = []
    for num_warps, num_stages in configs:
        try:
            fn = launch_factory(num_warps, num_stages)
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(reps):
                fn()
            e.record()
            torch.cuda.synchronize()
            us = s.elapsed_time(e) * 1000.0 / reps
            results.append(((num_warps, num_stages), us))
            print(f"  {num_warps:>5d} {num_stages:>6d}   {us:>10.2f}")
        except Exception as ex:
            print(f"  {num_warps:>5d} {num_stages:>6d}   FAIL: {ex.__class__.__name__}: {ex}")
    if results:
        best_cfg, best_us = min(results, key=lambda r: r[1])
        print(f"  best: num_warps={best_cfg[0]} num_stages={best_cfg[1]}  →  {best_us:.2f} us")


def _bench_with_nvtx(name: str, fn, reps: int, warmup: int) -> float:
    """Same as `_bench` but wraps each measured call in an nvtx range, so
    ncu's `--kernel-name` filter pairs cleanly with the named region."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        torch.cuda.nvtx.range_push(name)
        fn()
        torch.cuda.nvtx.range_pop()
    e.record()
    torch.cuda.synchronize()
    ms_per_call = s.elapsed_time(e) / reps
    print(f"  {name:>10s}: {ms_per_call*1000:7.1f} us/call  "
          f"(reps={reps}, warmup={warmup})")
    return ms_per_call


def _gather_caches(jit_fn) -> list:
    """Triton 3.x stores compiled kernels in `JITFunction.cache` as
    `{device_id: {signature: CompiledKernel}}` (two-level dict). Older
    versions used a flat `{signature: CompiledKernel}`. Flatten either to
    a single list of (key, compiled_kernel) tuples."""
    out = []

    def _consume_dict(d):
        if not isinstance(d, dict):
            return
        for k, v in d.items():
            if isinstance(v, dict):
                # one level deeper: device → {sig → kernel}
                out.extend(v.items())
            else:
                out.append((k, v))

    _consume_dict(getattr(jit_fn, "cache", None))
    nested = getattr(jit_fn, "device_caches", None)
    if isinstance(nested, dict):
        for _dev, sub in nested.items():
            if isinstance(sub, dict):
                _consume_dict(sub) if any(isinstance(v, dict) for v in sub.values()) else out.extend(sub.items())
            else:
                inner = getattr(sub, "cache", None)
                _consume_dict(inner)
    return out


def _extract_metrics(compiled) -> dict:
    """Pull register/spill/shared from a CompiledKernel across triton
    versions. The metadata may live as direct attrs, in `.metadata` (dict
    or namedtuple), or in `.asm`/`.metadata.target` etc."""
    fields = {"n_regs": None, "n_spills": None, "shared": None,
              "name": None, "num_warps": None, "num_stages": None}

    def _set(name, val):
        if val is not None and fields[name] is None:
            fields[name] = val

    # 1. Direct attrs (older triton).
    for k_attr, f in [("n_regs", "n_regs"), ("num_regs", "n_regs"),
                      ("n_spills", "n_spills"), ("num_spills", "n_spills"),
                      ("shared", "shared"), ("smem", "shared"),
                      ("name", "name"),
                      ("num_warps", "num_warps"),
                      ("num_stages", "num_stages")]:
        _set(f, getattr(compiled, k_attr, None))

    # 2. .metadata as dict or dataclass.
    md = getattr(compiled, "metadata", None)
    if md is not None:
        if isinstance(md, dict):
            getter = md.get
        else:
            getter = lambda x: getattr(md, x, None)
        for src, dst in [("num_regs", "n_regs"), ("n_regs", "n_regs"),
                         ("num_spills", "n_spills"), ("n_spills", "n_spills"),
                         ("shared", "shared"), ("smem", "shared"),
                         ("name", "name"),
                         ("num_warps", "num_warps"),
                         ("num_stages", "num_stages")]:
            _set(dst, getter(src))

    # 3. .kernel.metadata (some builds wrap CompiledKernel).
    inner = getattr(compiled, "kernel", None)
    if inner is not None:
        md2 = getattr(inner, "metadata", None)
        if md2 is not None:
            getter = (md2.get if isinstance(md2, dict)
                      else (lambda x: getattr(md2, x, None)))
            for src, dst in [("num_regs", "n_regs"), ("n_regs", "n_regs"),
                             ("num_spills", "n_spills"),
                             ("shared", "shared"),
                             ("name", "name")]:
                _set(dst, getter(src))

    return fields


def _kernel_info(jit_fn, label: str, debug: bool = False) -> None:
    """Print per-variant metrics for all compiled instances of a triton.jit
    function. Works without GPU performance-counter permissions — pulls
    everything from the in-process compile cache."""
    items = _gather_caches(jit_fn)
    if not items:
        print(f"  [{label}] no compile cache; was the kernel launched?")
        return

    print(f"  [{label}] {len(items)} compiled variant(s):")
    for i, (_key, k) in enumerate(items):
        f = _extract_metrics(k)
        if debug and i == 0:
            attrs = sorted(a for a in dir(k) if not a.startswith("_"))
            print(f"    [debug] type={type(k).__name__} attrs={attrs}")
            md = getattr(k, "metadata", None)
            if md is not None:
                md_attrs = (sorted(md.keys()) if isinstance(md, dict)
                            else sorted(a for a in dir(md) if not a.startswith("_")))
                print(f"    [debug] metadata type={type(md).__name__} "
                      f"keys/attrs={md_attrs}")
        nw = f["num_warps"] or 4
        n_regs = f["n_regs"]
        occ_str = ""
        if isinstance(n_regs, int) and n_regs > 0:
            threads = nw * 32
            regs_per_block = n_regs * threads
            # A100: 65536 regs/SM, max 64 warps/SM, max 32 blocks/SM.
            max_blocks_reg = 65536 // max(1, regs_per_block)
            max_blocks_warp = 64 // nw
            max_blocks = min(max_blocks_reg, max_blocks_warp, 32)
            warps = nw * max_blocks
            occ_str = f"  → ~{warps}/64 warps = {warps/64:.0%} occ"
        print(f"    variant[{i}]: regs={n_regs} spills={f['n_spills']} "
              f"shared={f['shared']}B  num_warps={f['num_warps']} "
              f"num_stages={f['num_stages']}{occ_str}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", choices=["rotatek", "dense", "both"],
                    default="both",
                    help="Which kernel to bench")
    ap.add_argument("--seqlen", type=int, default=12000,
                    help="Vision tokens (= rotatek sparse seqlen)")
    ap.add_argument("--batch_size", type=int, default=1,
                    help="Batch dim for synthetic inputs.")
    ap.add_argument("--reps", type=int, default=200,
                    help="Tight-loop reps per measurement")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--nvtx", action="store_true",
                    help="Wrap each call in an nvtx range")
    ap.add_argument("--inspect", action="store_true",
                    help="After warmup, dump compiled-kernel metadata "
                         "(registers/thread, shared memory, theoretical "
                         "occupancy) for each kernel variant — works "
                         "without GPU performance-counter permissions.")
    ap.add_argument("--breakdown", action="store_true",
                    help="Time each rotatek sub-kernel (prelude, sparse, "
                         "full, merge) separately so we can see where the "
                         "rotatek-vs-dense gap actually lives.")
    ap.add_argument("--scan-config", action="store_true",
                    help="Scan (num_warps × num_stages) combos for each "
                         "kernel and report best. Slow (~5min).")
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"shape: B=1 H_q={H_Q} H_kv={H_KV} D={HEAD_DIM} D_keep={HEAD_DIM_KEEP}")
    print(f"S_vision={args.seqlen} S_prompt={S_PROMPT} S_text={S_TEXT} B={args.batch_size}")
    print()

    bench = _bench_with_nvtx if args.nvtx else _bench
    inp = _make_inputs(args.seqlen, bsz=args.batch_size)

    def _rotatek_call():
        rotatek_decode_fused_triton(
            q_full=inp["q_full"],
            R_partial=inp["R_partial"],
            delta_mu=inp["delta_mu"],
            k_full=inp["k_full"],
            v_full=inp["v_full"],
            mask_full=inp["mask_full"],
            k_sparse=inp["k_sparse"],
            v_sparse=inp["v_sparse"],
            num_kv_groups=NUM_KV_GROUPS,
        )

    def _dense_call():
        full_channel_decode_triton(
            q=inp["q_full"],
            k=inp["k_dense"],
            v=inp["v_dense"],
            num_kv_groups=NUM_KV_GROUPS,
        )

    if args.kernel in ("rotatek", "both"):
        bench("rotatek", _rotatek_call, args.reps, args.warmup)
    if args.kernel in ("dense", "both"):
        bench("dense", _dense_call, args.reps, args.warmup)

    if args.breakdown:
        print("\n--- rotatek sub-kernel breakdown ---")
        _bench_rotatek_breakdown(inp, args.reps, args.warmup)

    if args.scan_config:
        _run_scan_config(inp, args.reps, args.warmup)

    if args.inspect:
        from rotatek.kernels.full_channel_flash_decoding import (
            _dense_phase1_kernel,
        )
        from rotatek.kernels.fused_decode import (
            _rotatek_sparse_kernel,
            _rotatek_combined_kernel,
        )
        print("\n--- compiled kernel introspection ---")
        first = True
        if args.kernel in ("rotatek", "both"):
            _kernel_info(_rotatek_sparse_kernel,    "rotatek_sparse",
                         debug=first); first = False
            _kernel_info(_rotatek_combined_kernel,  "rotatek_combined")
        if args.kernel in ("dense", "both"):
            _kernel_info(_dense_phase1_kernel,      "dense_phase1",
                         debug=first)


if __name__ == "__main__":
    main()
