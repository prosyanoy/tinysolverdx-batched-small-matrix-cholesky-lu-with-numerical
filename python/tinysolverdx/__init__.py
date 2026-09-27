"""TinySolverDx: batched small-matrix Cholesky / LU on NumPy arrays.

Inputs are (batch, n, n) arrays (a single (n, n) matrix is also accepted) with
n in 1..64. dtype selects the precision (float32 -> fp32, float64 -> fp64,
float16 -> emulated fp16). For bfloat16 emulation pass a float32 array with
``precision="bf16"``; results are float32 arrays holding bf16-representable values.
"""
import numpy as np

from . import _core
from ._core import (  # noqa: F401  (re-exported)
    has_cuda, has_openmp, num_threads, set_num_threads, unit_roundoff,
    chol_backward_error, lu_backward_error, solve_backward_error,
    solve_forward_error, reference_solve, measure_peak_gflops, measure_bandwidth_gbs,
)

__all__ = ["cholesky", "lu", "cholesky_solve", "lu_solve", "posv", "gesv", "round_to",
           "has_cuda", "has_openmp", "num_threads", "set_num_threads", "unit_roundoff",
           "chol_backward_error", "lu_backward_error", "solve_backward_error",
           "solve_forward_error", "reference_solve", "measure_peak_gflops", "measure_bandwidth_gbs"]

_DEFAULT_PREC = {np.dtype(np.float32): "fp32", np.dtype(np.float64): "fp64", np.dtype(np.float16): "fp16"}


def _fn(name, arr):
    # the core exposes one entry point per storage dtype (no implicit casting)
    return getattr(_core, f"{name}_{'f64' if arr.dtype == np.float64 else 'f32'}")


def _prep(a, precision, ndim_single):
    a = np.asarray(a)
    single = a.ndim == ndim_single
    if single:
        a = a[None]
    out_dtype = a.dtype
    if precision is None:
        if a.dtype not in _DEFAULT_PREC:
            raise TypeError(f"unsupported dtype {a.dtype}; use float16/32/64")
        precision = _DEFAULT_PREC[a.dtype]
    if precision not in ("fp16", "bf16", "fp32", "fp64"):
        raise ValueError("precision must be one of fp16, bf16, fp32, fp64")
    work = np.float64 if (precision == "fp64" and a.dtype == np.float64) else np.float32
    if a.dtype == np.float64 and precision != "fp64":
        work = np.float64  # keep wide storage; the core rounds into the target format
    return np.ascontiguousarray(a, dtype=work), precision, single, out_dtype


def _finish(x, out_dtype, precision):
    if precision == "fp16" and out_dtype == np.float16:
        return x.astype(np.float16)
    return x


def cholesky(A, precision=None, backend="auto"):
    """Lower Cholesky factors and LAPACK-style info (0 = ok, k>0: pivot k not positive)."""
    a, p, single, dt = _prep(A, precision, 2)
    L, info = _fn('cholesky', a)(a, p, backend)
    L = _finish(L, dt, p)
    return (L[0], info[0]) if single else (L, info)


def lu(A, precision=None, backend="auto"):
    """Packed LU with partial pivoting: (LU, piv, info). piv is scipy/LAPACK style (0-based)."""
    a, p, single, dt = _prep(A, precision, 2)
    LU, piv, info = _fn('lu', a)(a, p, backend)
    LU = _finish(LU, dt, p)
    return (LU[0], piv[0], info[0]) if single else (LU, piv, info)


def _rhs(B, ref, single):
    B = np.asarray(B)
    if single:
        B = B[:, None] if B.ndim == 1 else B
        B = B[None]
    elif B.ndim == 2:
        B = B[:, :, None]
    return np.ascontiguousarray(B, dtype=ref.dtype)


def cholesky_solve(L, B, precision=None):
    """Solve L L^T x = B given factors from :func:`cholesky`."""
    Ln = np.asarray(L)
    l, p, single, dt = _prep(Ln, precision, 2)
    vec = np.asarray(B).ndim == (1 if single else 2)
    X = _fn('cholesky_solve', l)(l, _rhs(B, l, single), p)
    X = _finish(X, dt, p)
    X = X[..., 0] if vec else X
    return X[0] if single else X


def lu_solve(LU, piv, B, precision=None):
    """Solve A x = B given (LU, piv) from :func:`lu`."""
    l, p, single, dt = _prep(LU, precision, 2)
    pv = np.ascontiguousarray(np.asarray(piv)[None] if single else piv, dtype=np.int32)
    vec = np.asarray(B).ndim == (1 if single else 2)
    X = _fn('lu_solve', l)(l, pv, _rhs(B, l, single), p)
    X = _finish(X, dt, p)
    X = X[..., 0] if vec else X
    return X[0] if single else X


def posv(A, B, precision=None, backend="auto"):
    """Solve SPD systems A x = B. Returns (x, info)."""
    L, info = cholesky(A, precision, backend)
    return cholesky_solve(L, B, precision), info


def gesv(A, B, precision=None, backend="auto"):
    """Solve general systems A x = B with partial pivoting. Returns (x, info)."""
    LU, piv, info = lu(A, precision, backend)
    return lu_solve(LU, piv, B, precision), info


def round_to(x, precision):
    """Round values to the nearest number representable in ``precision`` (float64 in/out)."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    return _core.round_to_f64(x, precision)
