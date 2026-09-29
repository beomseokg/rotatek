"""RotateK: rotation-aligned Key channel pruning for VLM KV caches.

    rotation.power_iteration_gpu   top-k eigenbasis of the Key covariance
    kernels.rotatek_decode         decode over rotated-truncated visual Keys
    kernels.full_channel_decode    full-width split-K decode (baselines)

The two decode entry points follow ``ROTATEK_KERNEL`` (``paper`` | ``gqa``);
see ``rotatek/kernels/__init__.py``.
"""
from rotatek.kernels import full_channel_decode, rotatek_decode
from rotatek.rotation import power_iteration_gpu

__all__ = ["full_channel_decode", "rotatek_decode", "power_iteration_gpu"]
