"""RotateK channel-pruning method.

Online PCA-aligned channel pruning for visual KV caches.

Submodules:
    decode  -- public attention-decode entrypoints (`rotatek_decode_fused`,
               `rotatek_decode_triton`).
    kernel  -- the fused Triton phase-1 kernel + launcher
               (`rotatek_decode_fused_triton`).
    jacobi  -- GPU power iteration / randomized SVD helpers
               (`power_iteration_gpu`, `randomized_topk_eigh`).
"""
from methods.rotatek.decode import rotatek_decode_fused, rotatek_decode_triton
from methods.rotatek.kernel import rotatek_decode_fused_triton
from methods.rotatek.jacobi import power_iteration_gpu, randomized_topk_eigh

__all__ = [
    "rotatek_decode_fused",
    "rotatek_decode_triton",
    "rotatek_decode_fused_triton",
    "power_iteration_gpu",
    "randomized_topk_eigh",
]
