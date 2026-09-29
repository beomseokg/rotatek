"""Top-k eigenbasis of the (query-weighted) Key covariance for RotateK.

``power_iteration_gpu`` runs truncated subspace iteration with Cholesky-QR
orthonormalisation: it computes only the top-k eigenvectors, with a fixed
sequence of small ops that CUDA-graph capture can replay cheaply. The exact
alternative (``torch.linalg.eigh``, selected by ``ROTATEK_SOLVER=eigh``) is
dispatched in ``kv_pruning_utils.py``.
"""
from __future__ import annotations

import os

import torch

# Q-aware RotateK weights the covariance by outer(q, q), which needs a larger
# Cholesky ridge (see `_power_iter_loop`). Default ON; ROTATEK_QUERY_AWARE=0
# ablates to K-only PCA.
_ROTATEK_QUERY_AWARE_ACTIVE = bool(int(os.environ.get("ROTATEK_QUERY_AWARE", "1")))


def _power_iter_loop(cov: torch.Tensor, V: torch.Tensor, num_iters: int) -> torch.Tensor:
    """Inner subspace-iteration loop: matmul → Cholesky-orthonormalise.

    Factored out so torch.compile can wrap it as a single CUDA-graph-replayable
    region. Each iter is just (matmul + 2 small linalg ops).

    Numerical guard: as iterations push V columns toward dominant eigenvectors,
    G = V^T V can become ill-conditioned (or even positive semi-definite,
    breaking Cholesky). We add a small diagonal regularisation `eps * I` to
    G before Cholesky. Mathematically: V_orth^T V_orth ≈ I (off by O(eps)),
    so the orthonormalisation quality is preserved.
    """
    K = V.shape[-1]
    eye_K = torch.eye(K, device=V.device, dtype=V.dtype)
    # Cholesky ridge. Vanilla RotateK's K^T K is well-conditioned and
    # 1e-6 is plenty. Q-aware RotateK weights cov by outer(q, q) which
    # induces highly anisotropic spectra at the keep-count boundary,
    # making G ill-conditioned and tripping Cholesky's PSD check. Bump
    # to 1e-4 when Q-aware is active. Numerically still negligible vs
    # the leading eigenvalues; orthonormalisation quality is preserved.
    eps = 1e-4 if _ROTATEK_QUERY_AWARE_ACTIVE else 1e-6
    for _ in range(num_iters):
        V = torch.matmul(cov, V)
        G = torch.matmul(V.transpose(-2, -1), V)
        # Scale eps by the magnitude of G's diagonal so we don't perturb
        # tiny matrices too much. trace(G)/K is the average diag entry.
        trace_avg = torch.diagonal(G, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
        G = G + (eps * trace_avg) * eye_K
        L = torch.linalg.cholesky(G)
        V = torch.linalg.solve_triangular(
            L.transpose(-2, -1), V, upper=True, left=False,
        )
    return V


# ROTATEK_COMPILE=1 captures the loop as a CUDA graph (`reduce-overhead`), which
# removes per-op launch cost at small batch; the latency sweeps set it. Off by
# default because a shape change (e.g. a new channel_ratio) forces a recompile.
_ROTATEK_COMPILE_MODE = os.environ.get("ROTATEK_COMPILE", "").strip().lower()
if _ROTATEK_COMPILE_MODE in ("1", "reduce", "reduce-overhead"):
    _power_iter_loop_compiled = torch.compile(_power_iter_loop, mode="reduce-overhead")
elif _ROTATEK_COMPILE_MODE in ("default", "true"):
    _power_iter_loop_compiled = torch.compile(_power_iter_loop)
else:
    _power_iter_loop_compiled = _power_iter_loop


def power_iteration_gpu(
    cov: torch.Tensor,
    k: int,
    num_iters: int = 5,
    seed: int = 0,
) -> torch.Tensor:
    """Top-k eigenvectors of `cov` ([..., D, D]) as columns, [..., D, k].

    Each iteration:
        V = cov @ V                 # [B, H, D, k]
        G = V^T @ V                 # [B, H, k, k]
        L = cholesky(G)             # G = L L^T, L lower triangular
        V = V @ L^{-T}              # so V^T V = L^{-1} G L^{-T} = I

    Cholesky on the k×k Gram matrix replaces QR on [D, k], which is several
    times cheaper per iteration. The result spans the same subspace as
    QR-based subspace iteration (up to column signs / rotations within
    invariant subspaces).
    """
    assert cov.is_cuda
    D = cov.shape[-1]
    g = torch.Generator(device=cov.device).manual_seed(seed)
    V = torch.randn(*cov.shape[:-2], D, k, device=cov.device, dtype=cov.dtype, generator=g)
    return _power_iter_loop_compiled(cov, V, num_iters)
