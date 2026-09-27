import numpy as np
import pytest
import scipy.linalg as sla

import tinysolverdx as tsdx
from tinysolverdx import matrices

SIZES = [2, 3, 5, 8, 16, 17, 32, 64]
BACKENDS = ["naive", "scalar", "simd", "auto"]


def rng(seed=0):
    return np.random.default_rng(seed)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype,tol", [(np.float64, 1e-12), (np.float32, 1e-4)])
def test_cholesky_matches_numpy(n, backend, dtype, tol):
    A = matrices.spd(rng(n), 21, n, 1e2).astype(dtype)
    L, info = tsdx.cholesky(A, backend=backend)
    assert L.dtype == dtype and (info == 0).all()
    np.testing.assert_allclose(L, np.linalg.cholesky(A.astype(np.float64)), atol=tol, rtol=tol)
    assert np.allclose(np.triu(L, 1), 0)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_lu_matches_scipy(n, backend):
    A = matrices.general(rng(n), 13, n, 1e3)
    LU, piv, info = tsdx.lu(A, backend=backend)
    assert (info == 0).all()
    for b in range(A.shape[0]):
        ref_lu, ref_piv = sla.lu_factor(A[b])
        np.testing.assert_array_equal(piv[b], ref_piv)
        np.testing.assert_allclose(LU[b], ref_lu, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("n", [2, 7, 32, 64])
def test_solvers_match_numpy_solve(n):
    r = rng(1)
    A = matrices.general(r, 9, n, 1e3)
    S = matrices.spd(r, 9, n, 1e3)
    b = r.standard_normal((9, n))
    x, info = tsdx.gesv(A, b)
    np.testing.assert_allclose(x, np.linalg.solve(A, b[..., None])[..., 0], rtol=1e-8, atol=1e-8)
    x, info = tsdx.posv(S, b)
    assert (info == 0).all()
    np.testing.assert_allclose(x, np.linalg.solve(S, b[..., None])[..., 0], rtol=1e-8, atol=1e-8)
    B = r.standard_normal((9, n, 3))
    x, _ = tsdx.gesv(A, B)
    np.testing.assert_allclose(A @ x, B, atol=1e-9)


def test_single_matrix_and_vector_rhs():
    A = matrices.spd(rng(), 1, 6, 10.0)[0]
    b = np.arange(6.0)
    x, info = tsdx.posv(A, b)
    assert x.shape == (6,) and info == 0
    np.testing.assert_allclose(x, np.linalg.solve(A, b), atol=1e-12)
    x, info = tsdx.gesv(A, b)
    np.testing.assert_allclose(x, np.linalg.solve(A, b), atol=1e-12)


def test_input_not_modified_and_noncontiguous():
    A = matrices.spd(rng(), 4, 8, 10.0)
    keep = A.copy()
    tsdx.cholesky(A)
    np.testing.assert_array_equal(A, keep)
    big = np.zeros((4, 8, 16)); big[:, :, ::2] = A
    L, _ = tsdx.cholesky(big[:, :, ::2])
    np.testing.assert_allclose(L, np.linalg.cholesky(A), atol=1e-12)


def test_not_positive_definite_reports_info():
    A = matrices.spd(rng(), 3, 5, 10.0)
    A[1] = -np.eye(5)
    _, info = tsdx.cholesky(A)
    assert list(info) == [0, 1, 0]


def test_singular_lu_reports_info():
    A = matrices.general(rng(), 2, 4, 10.0)
    _, _, info = tsdx.lu(A)
    assert list(info) == [0, 0]
    A[1] = 0
    _, _, info = tsdx.lu(A)
    assert info[1] == 1


@pytest.mark.parametrize("prec", ["fp16", "bf16"])
def test_emulated_precisions_have_expected_accuracy(prec):
    u = tsdx.unit_roundoff(prec)
    A = tsdx.round_to(matrices.spd(rng(3), 20, 16, 10.0), prec)
    L, info = tsdx.cholesky(A.astype(np.float32), precision=prec)
    assert (info == 0).all()
    assert (tsdx.round_to(L, prec) == L).all()  # outputs representable in the format
    err = tsdx.chol_backward_error(A, L.astype(np.float64))
    assert err.max() < 20 * u * 16 and err.max() > tsdx.unit_roundoff("fp32")


def test_float16_dtype_roundtrip():
    A = matrices.spd(rng(), 5, 8, 10.0).astype(np.float16)
    L, info = tsdx.cholesky(A)
    assert L.dtype == np.float16 and (info == 0).all()


def test_errors():
    with pytest.raises(ValueError):
        tsdx.cholesky(np.zeros((1, 65, 65), np.float64))
    with pytest.raises(ValueError):
        tsdx.cholesky(np.zeros((1, 3, 4), np.float64))
    with pytest.raises(ValueError):
        tsdx.cholesky(np.eye(3)[None], precision="fp8")
    with pytest.raises(TypeError):
        tsdx.cholesky(np.eye(3, dtype=np.int32))
    if not tsdx.has_cuda():
        with pytest.raises(RuntimeError):
            tsdx.cholesky(np.eye(3)[None], backend="cuda")


def test_error_metrics_against_numpy():
    r = rng(5)
    A = matrices.general(r, 6, 10, 1e4)
    b = r.standard_normal((6, 10))
    LU, piv, _ = tsdx.lu(A)
    be = tsdx.lu_backward_error(A, LU, piv)
    assert be.max() < 1e-14
    x = np.linalg.solve(A, b[..., None])[..., 0]
    assert tsdx.solve_backward_error(A, x, b).max() < 1e-14
    assert tsdx.solve_forward_error(A, x, b).max() < 1e-10
    np.testing.assert_allclose(tsdx.reference_solve(A, b), x, rtol=1e-9)
    # a perturbed solution must show a large forward error
    assert tsdx.solve_forward_error(A, x * 1.01, b).min() > 5e-3


def test_cuda_matches_cpu_if_available():
    if not tsdx.has_cuda():
        pytest.skip("no CUDA backend")
    A = matrices.spd(rng(), 50, 24, 10.0)
    np.testing.assert_allclose(tsdx.cholesky(A, backend="cuda")[0], tsdx.cholesky(A, backend="scalar")[0], atol=1e-10)
