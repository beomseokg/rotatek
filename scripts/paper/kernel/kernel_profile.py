"""Decode attention kernel microbenchmark: fused RotateK vs full-channel.

Synthesizes one decode step at an 8B-class GQA shape (H_q=32, H_kv=8, D=128,
D_keep=32 at channel_ratio=0.75) and times each kernel in a tight loop.
`--impl paper` (default) is the kernel pair the paper measured; `--impl gqa`
shares each K/V tile across the query heads of a GQA group.

    python scripts/paper/kernel/kernel_profile.py --kernel both --seqlen 16000 --batch_size 32
"""

from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import argparse

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:512",
)

import torch  # noqa: E402

from rotatek.kernels.full_channel_flash_decoding import full_channel_decode_triton  # noqa: E402
from rotatek.kernels.fused_decode import rotatek_decode_fused_triton  # noqa: E402
from rotatek.kernels.gqa_decode import full_channel_decode_gqa, rotatek_decode_gqa  # noqa: E402

IMPLS = {
    "paper": (rotatek_decode_fused_triton, full_channel_decode_triton),
    "gqa": (rotatek_decode_gqa, full_channel_decode_gqa),
}


H_Q = 32
H_KV = 8
NUM_KV_GROUPS = H_Q // H_KV
HEAD_DIM = 128
HEAD_DIM_KEEP = 32  # 75% sparsity → D_keep = 128 - floor(128*0.75) = 32
S_PROMPT = 50       # system prompt span before the image
S_TEXT = 30         # mid-decode text span
DTYPE = torch.bfloat16


def _make_inputs(s_vision: int, bsz: int = 1, device: str = "cuda"):
    """RotateK keeps vision Keys rotated and truncated (k_sparse) next to the
    full-width prompt/text Keys (k_full); the dense baseline holds every token
    at full width (k_dense)."""
    s_full = S_PROMPT + S_TEXT
    s_total_dense = S_PROMPT + s_vision + S_TEXT

    def rand(*shape):
        return torch.randn(*shape, device=device, dtype=DTYPE)

    return {
        "q_full": rand(bsz, H_Q, HEAD_DIM),
        "R_partial": rand(bsz, H_KV, HEAD_DIM, HEAD_DIM_KEEP),
        "delta_mu": rand(bsz, H_KV, HEAD_DIM),
        "k_full": rand(bsz, H_KV, s_full, HEAD_DIM),
        "v_full": rand(bsz, H_KV, s_full, HEAD_DIM),
        "mask_full": torch.ones(bsz, s_full, device=device, dtype=torch.uint8),
        "k_sparse": rand(bsz, H_KV, s_vision, HEAD_DIM_KEEP),
        "v_sparse": rand(bsz, H_KV, s_vision, HEAD_DIM),
        "k_dense": rand(bsz, H_KV, s_total_dense, HEAD_DIM),
        "v_dense": rand(bsz, H_KV, s_total_dense, HEAD_DIM),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", choices=["rotatek", "dense", "both"], default="both")
    ap.add_argument("--impl", choices=sorted(IMPLS), default="paper")
    ap.add_argument("--seqlen", type=int, default=12000,
                    help="Vision tokens (= rotatek sparse seqlen)")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--reps", type=int, default=200, help="Tight-loop reps per measurement")
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    rotatek_fn, dense_fn = IMPLS[args.impl]
    print(f"GPU: {torch.cuda.get_device_name(0)}   kernels: {args.impl}")
    print(f"shape: H_q={H_Q} H_kv={H_KV} D={HEAD_DIM} D_keep={HEAD_DIM_KEEP}")
    print(f"S_vision={args.seqlen} S_prompt={S_PROMPT} S_text={S_TEXT} B={args.batch_size}")
    print()

    inp = _make_inputs(args.seqlen, bsz=args.batch_size)

    def _rotatek_call():
        rotatek_fn(
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
        dense_fn(
            q=inp["q_full"],
            k=inp["k_dense"],
            v=inp["v_dense"],
            num_kv_groups=NUM_KV_GROUPS,
        )

    if args.kernel in ("rotatek", "both"):
        _bench("rotatek", _rotatek_call, args.reps, args.warmup)
    if args.kernel in ("dense", "both"):
        _bench("dense", _dense_call, args.reps, args.warmup)


if __name__ == "__main__":
    main()
