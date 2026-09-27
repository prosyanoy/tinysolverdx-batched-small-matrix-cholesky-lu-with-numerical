// Single-matrix and lane-interleaved ("tile") factorization kernels.
//
// All kernels are header-only, templated on the scalar type and written so that
// the matrix dimension can be a compile-time constant after inlining
// (see dispatch in batched.hpp). Row-major storage, lower triangle for Cholesky.
// Failure semantics follow LAPACK: info == 0 on success, otherwise the 1-based
// index of the first non-positive Cholesky pivot / exactly-zero LU pivot. The
// contents of the factors are unspecified when info > 0 (except LU, which
// completes like dgetrf).
#pragma once

#include "precision.hpp"

namespace tsdx {
namespace kernel {

// ---------------------------------------------------------------- Cholesky ---
template <class T>
TSDX_HD TSDX_INLINE int cholesky_rm(T* TSDX_RESTRICT a, int n) {
  for (int i = 0; i < n; ++i) {
    T* ai = a + i * n;
    for (int j = 0; j <= i; ++j) {
      const T* aj = a + j * n;
      T s = ai[j];
      for (int k = 0; k < j; ++k) s -= ai[k] * aj[k];
      if (j == i) {
        if (!(s > T(0))) return i + 1;
        ai[i] = t_sqrt(s);
      } else {
        ai[j] = s / aj[j];
      }
    }
  }
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) a[i * n + j] = T(0);
  return 0;
}

// Textbook column-oriented right-looking variant with strided access. Used as
// the "naive" baseline in benchmarks.
template <class T>
TSDX_HD int cholesky_naive(T* a, int n) {
  for (int j = 0; j < n; ++j) {
    for (int k = 0; k < j; ++k)
      for (int i = j; i < n; ++i) a[i * n + j] -= a[i * n + k] * a[j * n + k];
    if (!(a[j * n + j] > T(0))) return j + 1;
    const T d = t_sqrt(a[j * n + j]);
    a[j * n + j] = d;
    for (int i = j + 1; i < n; ++i) a[i * n + j] /= d;
  }
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) a[i * n + j] = T(0);
  return 0;
}

// W matrices at once; element (i,j) of lane l lives at t[(i*n+j)*W + l].
template <class T, int W>
TSDX_INLINE void cholesky_tile(T* TSDX_RESTRICT t, int n, int* TSDX_RESTRICT info) {
  for (int l = 0; l < W; ++l) info[l] = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      T s[W];
      T* dst = t + (i * n + j) * W;
      for (int l = 0; l < W; ++l) s[l] = dst[l];
      for (int k = 0; k < j; ++k) {
        const T* x = t + (i * n + k) * W;
        const T* y = t + (j * n + k) * W;
        for (int l = 0; l < W; ++l) s[l] -= x[l] * y[l];
      }
      if (j == i) {
        for (int l = 0; l < W; ++l) {
          const bool bad = !(s[l] > T(0));
          if (bad && info[l] == 0) info[l] = i + 1;
          s[l] = bad ? T(1) : s[l];
        }
        for (int l = 0; l < W; ++l) s[l] = t_sqrt(s[l]);
      } else {
        const T* d = t + (j * n + j) * W;
        for (int l = 0; l < W; ++l) s[l] /= d[l];
      }
      for (int l = 0; l < W; ++l) dst[l] = s[l];
    }
  }
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j)
      for (int l = 0; l < W; ++l) t[(i * n + j) * W + l] = T(0);
}

// ---------------------------------------------------------------------- LU ---
// Partial pivoting; piv[k] = row swapped with row k (LAPACK 0-based ipiv).
template <class T>
TSDX_HD TSDX_INLINE int lu_rm(T* TSDX_RESTRICT a, int n, int* TSDX_RESTRICT piv) {
  int info = 0;
  for (int k = 0; k < n; ++k) {
    int p = k;
    T best = t_abs(a[k * n + k]);
    for (int i = k + 1; i < n; ++i) {
      const T v = t_abs(a[i * n + k]);
      if (v > best) { best = v; p = i; }
    }
    piv[k] = p;
    if (p != k)
      for (int j = 0; j < n; ++j) {
        const T tmp = a[k * n + j];
        a[k * n + j] = a[p * n + j];
        a[p * n + j] = tmp;
      }
    const T pk = a[k * n + k];
    if (pk == T(0)) { if (info == 0) info = k + 1; continue; }
    const T* rk = a + k * n;
    for (int i = k + 1; i < n; ++i) {
      T* ri = a + i * n;
      const T f = ri[k] / pk;
      ri[k] = f;
      for (int j = k + 1; j < n; ++j) ri[j] -= f * rk[j];
    }
  }
  return info;
}

template <class T>
TSDX_HD int lu_naive(T* a, int n, int* piv) {
  int info = 0;
  for (int k = 0; k < n; ++k) {
    int p = k;
    for (int i = k + 1; i < n; ++i)
      if (t_abs(a[i * n + k]) > t_abs(a[p * n + k])) p = i;
    piv[k] = p;
    if (p != k)
      for (int j = 0; j < n; ++j) { T tmp = a[k * n + j]; a[k * n + j] = a[p * n + j]; a[p * n + j] = tmp; }
    if (a[k * n + k] == T(0)) { if (info == 0) info = k + 1; continue; }
    for (int j = k + 1; j < n; ++j)          // column-oriented updates (strided)
      for (int i = k + 1; i < n; ++i)
        a[i * n + j] -= (a[i * n + k] / a[k * n + k]) * a[k * n + j];
    for (int i = k + 1; i < n; ++i) a[i * n + k] /= a[k * n + k];
  }
  return info;
}

// Lane-interleaved LU with per-lane pivoting. piv is tile-local: piv[k*W + l].
template <class T, int W>
TSDX_INLINE void lu_tile(T* TSDX_RESTRICT t, int n, int* TSDX_RESTRICT piv, int* TSDX_RESTRICT info) {
  for (int l = 0; l < W; ++l) info[l] = 0;
  for (int k = 0; k < n; ++k) {
    int p[W];
    T best[W];
    for (int l = 0; l < W; ++l) { best[l] = t_abs(t[(k * n + k) * W + l]); p[l] = k; }
    for (int i = k + 1; i < n; ++i)
      for (int l = 0; l < W; ++l) {
        const T v = t_abs(t[(i * n + k) * W + l]);
        const bool gt = v > best[l];
        best[l] = gt ? v : best[l];
        p[l] = gt ? i : p[l];
      }
    for (int l = 0; l < W; ++l) {
      piv[k * W + l] = p[l];
      if (p[l] != k)  // divergent row swap: scalar per lane, O(n) vs O(n^2) update
        for (int j = 0; j < n; ++j) {
          const T tmp = t[(k * n + j) * W + l];
          t[(k * n + j) * W + l] = t[(p[l] * n + j) * W + l];
          t[(p[l] * n + j) * W + l] = tmp;
        }
    }
    T d[W];
    for (int l = 0; l < W; ++l) {
      const T pk = t[(k * n + k) * W + l];
      const bool zero = (pk == T(0));
      if (zero && info[l] == 0) info[l] = k + 1;
      d[l] = zero ? T(1) : pk;   // a zero pivot implies an all-zero column: skip scaling
    }
    const T* rk = t + (k * n) * W;
    for (int i = k + 1; i < n; ++i) {
      T* ri = t + (i * n) * W;
      T f[W];
      for (int l = 0; l < W; ++l) { f[l] = ri[k * W + l] / d[l]; ri[k * W + l] = f[l]; }
      for (int j = k + 1; j < n; ++j)
        for (int l = 0; l < W; ++l) ri[j * W + l] -= f[l] * rk[j * W + l];
    }
  }
}

// ------------------------------------------------------------------ solves ---
// B is n x nrhs row-major; L from cholesky_rm (lower).
template <class T>
TSDX_HD void cholesky_solve_rm(const T* TSDX_RESTRICT L, int n, T* TSDX_RESTRICT B, int nrhs) {
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < i; ++k) {
      const T lik = L[i * n + k];
      for (int r = 0; r < nrhs; ++r) B[i * nrhs + r] -= lik * B[k * nrhs + r];
    }
    const T d = L[i * n + i];
    for (int r = 0; r < nrhs; ++r) B[i * nrhs + r] /= d;
  }
  for (int i = n - 1; i >= 0; --i) {
    for (int k = i + 1; k < n; ++k) {
      const T lki = L[k * n + i];
      for (int r = 0; r < nrhs; ++r) B[i * nrhs + r] -= lki * B[k * nrhs + r];
    }
    const T d = L[i * n + i];
    for (int r = 0; r < nrhs; ++r) B[i * nrhs + r] /= d;
  }
}

template <class T>
TSDX_HD void lu_solve_rm(const T* TSDX_RESTRICT LU, const int* TSDX_RESTRICT piv, int n,
                         T* TSDX_RESTRICT B, int nrhs) {
  for (int k = 0; k < n; ++k)
    if (piv[k] != k)
      for (int r = 0; r < nrhs; ++r) {
        const T tmp = B[k * nrhs + r];
        B[k * nrhs + r] = B[piv[k] * nrhs + r];
        B[piv[k] * nrhs + r] = tmp;
      }
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < i; ++k) {
      const T lik = LU[i * n + k];
      for (int r = 0; r < nrhs; ++r) B[i * nrhs + r] -= lik * B[k * nrhs + r];
    }
  for (int i = n - 1; i >= 0; --i) {
    for (int k = i + 1; k < n; ++k) {
      const T uik = LU[i * n + k];
      for (int r = 0; r < nrhs; ++r) B[i * nrhs + r] -= uik * B[k * nrhs + r];
    }
    const T d = LU[i * n + i];
    for (int r = 0; r < nrhs; ++r) B[i * nrhs + r] /= d;
  }
}

}  // namespace kernel
}  // namespace tsdx
