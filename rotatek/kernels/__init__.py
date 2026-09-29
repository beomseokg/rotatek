"""Decode kernels, selected by ``ROTATEK_KERNEL``:

    paper (default)  fused_decode.py + full_channel_flash_decoding.py -- the
                     kernels the paper's latency figures were measured with;
                     one program per query head.
    gqa              gqa_decode.py -- the query heads of a GQA group share
                     each K/V tile; faster at long context / large batch.

The model adapters call ``rotatek_decode`` (RotateK) and ``full_channel_decode``
(Full / ThinK / SparK with decode_attention_backend="triton"), so one switch
moves every method to the same kernel family.
"""
import os

ROTATEK_KERNEL = os.environ.get("ROTATEK_KERNEL", "paper").strip().lower()

if ROTATEK_KERNEL == "paper":
    from rotatek.kernels.fused_decode import rotatek_decode_fused_triton as rotatek_decode
    from rotatek.kernels.full_channel_flash_decoding import full_channel_decode_triton as full_channel_decode
elif ROTATEK_KERNEL == "gqa":
    from rotatek.kernels.gqa_decode import rotatek_decode_gqa as rotatek_decode
    from rotatek.kernels.gqa_decode import full_channel_decode_gqa as full_channel_decode
else:
    raise ValueError(f"ROTATEK_KERNEL must be 'paper' or 'gqa', got {ROTATEK_KERNEL!r}")

__all__ = ["ROTATEK_KERNEL", "rotatek_decode", "full_channel_decode"]
