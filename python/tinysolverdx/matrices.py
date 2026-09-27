"""Seeded test-matrix generators with prescribed 2-norm condition number."""
import numpy as np


def _orthogonal(rng, batch, n):
    q, r = np.linalg.qr(rng.standard_normal((batch, n, n)))
    return q * np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]


def spd(rng, batch, n, kappa):
    """SPD matrices Q diag(lambda) Q^T with eigenvalues geometric in [1/kappa, 1]."""
    lam = np.geomspace(1.0, 1.0 / kappa, n)
    Q = _orthogonal(rng, batch, n)
    A = np.einsum("bij,j,bkj->bik", Q, lam, Q)
    return (A + A.transpose(0, 2, 1)) / 2


def general(rng, batch, n, kappa):
    """General matrices U diag(sigma) V^T with singular values geometric in [1/kappa, 1]."""
    s = np.geomspace(1.0, 1.0 / kappa, n)
    return np.einsum("bij,j,bkj->bik", _orthogonal(rng, batch, n), s, _orthogonal(rng, batch, n))
