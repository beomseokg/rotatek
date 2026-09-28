"""RotateK eigensolvers for online channel-rotation construction.

Implements three GPU-resident solvers used in the channel-pruning pipeline:

* ``power_iteration_gpu`` — truncated subspace iteration with shifted
  Cholesky-QR orthonormalisation (the default solver). Computes only the
  top-k eigenvectors of the (possibly query-weighted) covariance, amenable
  to CUDA-graph capture for low-dispatch decode.
* ``randomized_topk_eigh`` — Halko-Martinsson-Tropp randomized SVD: small
  Gaussian sketch + projected eigendecomposition. Selectable via
  ``ROTATEK_SOLVER=randomized``.
* ``jacobi_eigh_gpu`` / ``jacobi_eigh_reference`` — parallel Brent-Luk
  Jacobi eigendecomposition; kept as a correctness reference for the
  other paths.

Switchable via the ``ROTATEK_SOLVER`` environment variable; see
``kv_pruning_utils.py`` for the dispatch site.
"""
from __future__ import annotations

import os

import torch

# Read once at import time; controls Cholesky ridge magnitude in
# `_power_iter_loop`. Q-aware RotateK weights cov by outer(q, q) which
# requires a larger ridge to keep the Gram matrix strictly PSD.
# Default is ON (1) — set ROTATEK_QUERY_AWARE=0 to disable.
_ROTATEK_QUERY_AWARE_ACTIVE = bool(int(os.environ.get("ROTATEK_QUERY_AWARE", "1")))


def jacobi_eigh_reference(
    cov: torch.Tensor,
    max_sweeps: int = 20,
    tol: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference Jacobi eigendecomposition for symmetric PSD matrices.

    Used as a correctness baseline for the Triton kernel we'll write in Phase 1.
    This is the classic cyclic Jacobi rotation method.

    Args:
        cov: [..., D, D] symmetric matrices.
        max_sweeps: hard cap on sweeps.
        tol: stop when max off-diagonal |entry| < tol.

    Returns:
        (eigvals, eigvecs) with eigvals ascending, eigvecs as columns.
    """
    if cov.dim() < 2 or cov.shape[-1] != cov.shape[-2]:
        raise ValueError(f"cov must be square, got {tuple(cov.shape)}")

    D = cov.shape[-1]
    A = cov.clone().float()  # work in fp32
    V = torch.eye(D, device=A.device, dtype=A.dtype).expand_as(A).clone()

    for _sweep in range(max_sweeps):
        off_diag_max = _max_off_diagonal_abs(A)
        if bool((off_diag_max < tol).all()):
            break
        for p in range(D - 1):
            for q in range(p + 1, D):
                _apply_jacobi_rotation(A, V, p, q)

    # Diagonal of A is eigenvalues (descending? no — cyclic Jacobi gives unsorted).
    # Sort ascending to match torch.linalg.eigh convention.
    eigvals = torch.diagonal(A, dim1=-2, dim2=-1)
    sort_idx = eigvals.argsort(dim=-1)
    eigvals_sorted = torch.gather(eigvals, -1, sort_idx)
    eigvecs_sorted = torch.gather(
        V, -1, sort_idx.unsqueeze(-2).expand_as(V),
    )
    return eigvals_sorted, eigvecs_sorted


def _max_off_diagonal_abs(A: torch.Tensor) -> torch.Tensor:
    """Max |A[i,j]| over i != j, per leading-batch element."""
    D = A.shape[-1]
    mask = ~torch.eye(D, device=A.device, dtype=torch.bool)
    # broadcast over leading dims
    masked = A.masked_fill(~mask, 0.0)
    return masked.abs().reshape(*A.shape[:-2], -1).max(dim=-1).values


def _apply_jacobi_rotation(A: torch.Tensor, V: torch.Tensor, p: int, q: int) -> None:
    """In-place: apply Givens rotation to zero out A[p, q] (and A[q, p])."""
    App = A[..., p, p]
    Aqq = A[..., q, q]
    Apq = A[..., p, q]

    # Skip if already zero to avoid divide by zero.
    tol = 1e-14
    if bool((Apq.abs() < tol).all()):
        return

    theta = (Aqq - App) / (2.0 * Apq)
    t = torch.where(
        theta >= 0,
        1.0 / (theta + torch.sqrt(1.0 + theta * theta)),
        1.0 / (theta - torch.sqrt(1.0 + theta * theta)),
    )
    c = 1.0 / torch.sqrt(1.0 + t * t)
    s = t * c

    # Update A: rotate rows p, q and cols p, q.
    Ap_row = A[..., p, :].clone()
    Aq_row = A[..., q, :].clone()
    A[..., p, :] = c.unsqueeze(-1) * Ap_row - s.unsqueeze(-1) * Aq_row
    A[..., q, :] = s.unsqueeze(-1) * Ap_row + c.unsqueeze(-1) * Aq_row

    Ap_col = A[..., :, p].clone()
    Aq_col = A[..., :, q].clone()
    A[..., :, p] = c.unsqueeze(-1) * Ap_col - s.unsqueeze(-1) * Aq_col
    A[..., :, q] = s.unsqueeze(-1) * Ap_col + c.unsqueeze(-1) * Aq_col

    # Update V: only columns p and q change.
    Vp = V[..., :, p].clone()
    Vq = V[..., :, q].clone()
    V[..., :, p] = c.unsqueeze(-1) * Vp - s.unsqueeze(-1) * Vq
    V[..., :, q] = s.unsqueeze(-1) * Vp + c.unsqueeze(-1) * Vq


def _brent_luk_pairs(D: int, round_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Brent-Luk round-robin schedule.

    Returns two length-(D//2) index tensors (p_idx, q_idx) such that the pairs
    (p_idx[k], q_idx[k]) form a disjoint matching of {0, ..., D-1} at this
    round. Over D-1 rounds, every unordered (i, j) pair is visited exactly
    once (assuming D even).

    Standard "circle" construction: fix vertex 0, rotate the rest on a cycle.
    At round r, pair vertex i with the vertex that sits opposite on the
    partially-rotated cycle.
    """
    assert D % 2 == 0, "Brent-Luk schedule assumes D is even"
    # Cycle layout: [0, 1, 2, ..., D-1]. At round r, rotate positions 1..D-1
    # by r steps. Then pair position k with position (D-1-k) for k in 0..D/2-1.
    positions = list(range(D))
    # Rotate indices 1..D-1 by round_idx
    ring = positions[1:]
    r = round_idx % (D - 1)
    ring = ring[r:] + ring[:r]
    layout = [0] + ring  # length D
    p_idx = torch.tensor([layout[k] for k in range(D // 2)], dtype=torch.long)
    q_idx = torch.tensor([layout[D - 1 - k] for k in range(D // 2)], dtype=torch.long)
    return p_idx, q_idx


def jacobi_eigh_gpu(
    cov: torch.Tensor,
    num_sweeps: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parallel Brent-Luk Jacobi eigendecomposition on GPU (PyTorch ops only).

    Works for even D. For our target D=128, fires D/2 = 64 rotations in
    parallel per round, D-1 = 127 rounds per sweep.

    Args:
        cov: [..., D, D] symmetric matrices (typically PSD for our use case).
        num_sweeps: fixed sweep count. 10 is comfortable for convergence on
            reasonably conditioned matrices.

    Returns:
        (eigvals, eigvecs) with eigvals ascending (to match torch.linalg.eigh).
    """
    if cov.dim() < 2 or cov.shape[-1] != cov.shape[-2]:
        raise ValueError(f"cov must be square, got {tuple(cov.shape)}")
    D = cov.shape[-1]
    if D % 2 != 0:
        raise NotImplementedError("Brent-Luk assumes even D; pad if needed")

    device = cov.device
    dtype = torch.float32  # stability
    A = cov.to(dtype).contiguous()
    V = (
        torch.eye(D, device=device, dtype=dtype)
        .expand(*cov.shape[:-2], D, D)
        .contiguous()
    )

    # Precompute schedule once (on the right device).
    rounds = []
    for r in range(D - 1):
        p_idx, q_idx = _brent_luk_pairs(D, r)
        rounds.append((p_idx.to(device), q_idx.to(device)))

    for _sweep in range(num_sweeps):
        for p_idx, q_idx in rounds:
            # Gather A[..., p, p], A[..., q, q], A[..., p, q]  shape [..., D/2]
            App = A[..., p_idx, p_idx]
            Aqq = A[..., q_idx, q_idx]
            Apq = A[..., p_idx, q_idx]

            # Givens angle: tan(2θ) = 2·Apq / (Aqq - App)
            # Numerically stable branch.
            denom = Aqq - App
            # Avoid div-by-zero when Apq == 0 (already diagonal at that entry).
            safe_mask = Apq.abs() > 1e-20
            theta = torch.zeros_like(Apq)
            theta = torch.where(
                safe_mask,
                denom / (2.0 * Apq.where(safe_mask, torch.ones_like(Apq))),
                torch.zeros_like(Apq),
            )
            t = torch.where(
                theta >= 0,
                1.0 / (theta + torch.sqrt(1.0 + theta * theta)),
                1.0 / (theta - torch.sqrt(1.0 + theta * theta)),
            )
            t = torch.where(safe_mask, t, torch.zeros_like(t))
            c = 1.0 / torch.sqrt(1.0 + t * t)
            s = t * c

            # Apply rotation to A: rotate rows (p_idx, q_idx) and cols (p_idx, q_idx).
            # Gather rows [..., D/2, D] and columns [..., D, D/2]. Broadcast c, s
            # to [..., D/2, 1].
            c_b = c.unsqueeze(-1)  # [..., D/2, 1]
            s_b = s.unsqueeze(-1)

            # Row update: new_Ap = c * Ap - s * Aq, new_Aq = s * Ap + c * Aq
            Ap_row = A.index_select(-2, p_idx)  # [..., D/2, D]
            Aq_row = A.index_select(-2, q_idx)
            new_Ap_row = c_b * Ap_row - s_b * Aq_row
            new_Aq_row = s_b * Ap_row + c_b * Aq_row
            A = A.index_copy(-2, p_idx, new_Ap_row)
            A = A.index_copy(-2, q_idx, new_Aq_row)

            # Column update: after row rotation, still need to rotate columns of
            # the (now-updated) A. Since pairs are disjoint, the column update
            # is applied to the updated rows as well (A is symmetric only up to
            # fp noise during rotation, so we apply col update independently).
            Ap_col = A.index_select(-1, p_idx)  # [..., D, D/2]
            Aq_col = A.index_select(-1, q_idx)
            new_Ap_col = c.unsqueeze(-2) * Ap_col - s.unsqueeze(-2) * Aq_col
            new_Aq_col = s.unsqueeze(-2) * Ap_col + c.unsqueeze(-2) * Aq_col
            A = A.index_copy(-1, p_idx, new_Ap_col)
            A = A.index_copy(-1, q_idx, new_Aq_col)

            # Update V columns p_idx, q_idx the same way.
            Vp = V.index_select(-1, p_idx)
            Vq = V.index_select(-1, q_idx)
            new_Vp = c.unsqueeze(-2) * Vp - s.unsqueeze(-2) * Vq
            new_Vq = s.unsqueeze(-2) * Vp + c.unsqueeze(-2) * Vq
            V = V.index_copy(-1, p_idx, new_Vp)
            V = V.index_copy(-1, q_idx, new_Vq)

    # Eigenvalues = diagonal; sort ascending to match torch.linalg.eigh.
    eigvals = torch.diagonal(A, dim1=-2, dim2=-1)
    sort_idx = eigvals.argsort(dim=-1)
    eigvals_sorted = torch.gather(eigvals, -1, sort_idx)
    eigvecs_sorted = torch.gather(V, -1, sort_idx.unsqueeze(-2).expand_as(V))
    return eigvals_sorted.to(cov.dtype), eigvecs_sorted.to(cov.dtype)


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


# torch.compile path: gated by env var. `reduce-overhead` mode uses CUDA
# graphs which dramatically cuts kernel launch overhead at B=1, but can
# misbehave under shape changes (e.g., switching channel_ratio mid-run
# triggers recompilation; in some envs it recompiles per-call → hours).
# Default OFF for safety; set ROTATEK_COMPILE=1 (or =reduce / =default) to enable.
import os as _os_for_rotatek
_ROTATEK_COMPILE_MODE = _os_for_rotatek.environ.get("ROTATEK_COMPILE", "").strip().lower()
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
    """GPU power iteration with **Cholesky-based** orthonormalization,
    optionally wrapped by torch.compile with CUDA-graph capture.

    Key optimizations vs naive subspace iteration:
      1. Replace QR on [D, k] (cuSOLVER QR ≈ 1ms launch overhead) with
         Cholesky on the much smaller k×k Gram matrix V^T V (≈ 0.12ms).
         8× cheaper per iter.
      2. torch.compile with `mode="reduce-overhead"` captures the entire
         iteration sequence as a CUDA graph, replayed in microseconds.
         Eliminates per-op launch overhead (the dominant cost at B=1).

    Algorithm (each iter):
        V = cov @ V                 # [B, H, D, k]
        G = V^T @ V                 # [B, H, k, k]
        L = cholesky(G)             # G = L L^T, L lower triangular
        V = V @ L^{-T}              # so V^T V = L^{-1} G L^{-T} = I

    Result is mathematically equivalent to QR-based subspace iteration
    (same orthonormal basis up to column signs / rotations within
    invariant subspaces). At B=1: ~3.5× faster than eager Cholesky.
    """
    assert cov.is_cuda
    D = cov.shape[-1]
    g = torch.Generator(device=cov.device).manual_seed(seed)
    V = torch.randn(*cov.shape[:-2], D, k, device=cov.device, dtype=cov.dtype, generator=g)
    return _power_iter_loop_compiled(cov, V, num_iters)


def randomized_topk_eigh(
    cov: torch.Tensor,
    k: int,
    oversample: int = 10,
    num_power_iters: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomized top-k eigendecomposition of symmetric PSD matrices.

    Uses the Halko-Martinsson-Tropp sketch: project onto a random range,
    orthogonalise, then do eigh on a small (k+p)×(k+p) projected matrix.
    For RotateK we only need the top-k eigenvectors (not all D), so this
    side-steps the full 128×128 eigh entirely.

    Args:
        cov: [..., D, D] symmetric PSD matrices.
        k: number of top eigenvectors to return (the eigenvectors of the
           largest k eigenvalues).
        oversample: extra sketch dimensions to improve accuracy. 5-10 is
           standard. Larger = more accurate, slightly slower.
        num_power_iters: subspace iterations (0 = pure sketch, 1-2 = much
           better for slowly-decaying spectra). Each iter is a matmul.

    Returns:
        (top_eigvals, top_eigvecs) with eigvals descending (largest first)
        and eigvecs as columns, shape [..., k] and [..., D, k].
    """
    if cov.dim() < 2 or cov.shape[-1] != cov.shape[-2]:
        raise ValueError(f"cov must be square, got {tuple(cov.shape)}")
    D = cov.shape[-1]
    l = min(k + oversample, D)  # sketch dimension

    # Step 1: random Gaussian sketch onto l columns.
    sketch = torch.randn(
        *cov.shape[:-2], D, l,
        device=cov.device, dtype=cov.dtype,
    )
    Y = torch.matmul(cov, sketch)  # [..., D, l]

    # Step 2: optional subspace iterations. Y := cov^q · Omega for q iters,
    # with interleaved orthogonalisation to keep conditioning healthy.
    for _ in range(num_power_iters):
        Y, _ = torch.linalg.qr(Y)
        Y = torch.matmul(cov, Y)

    # Step 3: QR to get orthonormal basis Q of range(Y).
    Q, _ = torch.linalg.qr(Y)  # Q: [..., D, l]

    # Step 4: project cov into the l-dimensional subspace.
    # B_small = Q^T · cov · Q  — an (l × l) symmetric matrix.
    B_small = torch.matmul(Q.transpose(-2, -1), torch.matmul(cov, Q))
    B_small = 0.5 * (B_small + B_small.transpose(-2, -1))  # symmetrise

    # Step 5: small eigh in l-dim space.
    eigvals_small, U_small = torch.linalg.eigh(B_small)  # ascending

    # Step 6: extract top-k and map back to original basis.
    top_eigvals = eigvals_small[..., -k:].flip(-1)  # descending
    top_U = U_small[..., -k:].flip(-1)              # [..., l, k]
    top_eigvecs = torch.matmul(Q, top_U)             # [..., D, k]

    return top_eigvals, top_eigvecs


# Placeholder for the actual Triton kernel we'll write in Phase 1b.
def jacobi_eigh_triton(cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """TODO(Phase 1b): Triton Jacobi eigendecomposition kernel.

    For now this falls back to the GPU PyTorch implementation so the test
    harness can exercise the full pipeline and give us speed/accuracy numbers
    before we commit to Triton-level effort.
    """
    return jacobi_eigh_gpu(cov)
