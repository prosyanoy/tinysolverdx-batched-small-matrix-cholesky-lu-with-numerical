// Batched small-matrix solvers.
//
// Layout: A is (batch, n, n) row-major, matrices contiguous. Cholesky uses and
// returns the lower triangle (upper is zeroed on success). LU returns the packed
// L\U factors plus LAPACK-style sequential pivots piv (batch, n), 0-based.
// info is (batch) ints or nullptr.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kernels.hpp"
#include "parallel.hpp"
#include "precision.hpp"

#ifdef TSDX_WITH_CUDA
#include "cuda_backend.hpp"
#endif

namespace tsdx {

constexpr int kMinN = 1;
constexpr int kMaxN = 64;
// Sizes up to this are instantiated with a compile-time N (fully unrolled /
// register-resident); larger sizes share one runtime-N kernel to bound code size.
constexpr int kMaxUnrolled = 16;
// Largest n for which Backend::Auto picks the SIMD tile kernel for LU.
constexpr int kLuSimdMaxN = 24;

template <class T> constexpr int lanes() { return sizeof(T) <= 4 ? 8 : 4; }

namespace detail {

inline void check_dims(int n, std::int64_t batch) {
  if (n < kMinN || n > kMaxN)
    throw std::invalid_argument("matrix size n=" + std::to_string(n) + " outside supported range [1, 64]");
  if (batch < 0) throw std::invalid_argument("negative batch");
}

template <class T>
Backend resolve(Backend b) {
  if (b == Backend::Auto || (b == Backend::Simd && !is_native<T>::value))
    return is_native<T>::value ? Backend::Simd : Backend::Scalar;
  return b;
}

// Matrices per parallel chunk: enough work per task to amortize scheduling.
inline std::int64_t chunk_size(int n, int multiple) {
  std::int64_t c = 32768 / (static_cast<std::int64_t>(n) * n);
  if (c < multiple) c = multiple;
  return (c + multiple - 1) / multiple * multiple;
}

// ---- compile-time-N dispatch tables ------------------------------------------
template <class T> using ScalarChol = int (*)(T*, int);
template <class T> using ScalarLU = int (*)(T*, int, int*);

template <class T> TSDX_NOINLINE int chol_rt(T* a, int n) { return kernel::cholesky_rm<T>(a, n); }
template <class T> TSDX_NOINLINE int lu_rt(T* a, int n, int* p) { return kernel::lu_rm<T>(a, n, p); }

template <class T, int N> ScalarChol<T> pick_chol() {
  if constexpr (N <= kMaxUnrolled) return [](T* a, int) { return kernel::cholesky_rm<T>(a, N); };
  else return &chol_rt<T>;
}
template <class T, int N> ScalarLU<T> pick_lu() {
  if constexpr (N <= kMaxUnrolled) return [](T* a, int, int* p) { return kernel::lu_rm<T>(a, N, p); };
  else return &lu_rt<T>;
}
template <class T, std::size_t... I>
std::array<ScalarChol<T>, kMaxN + 1> chol_table(std::index_sequence<I...>) {
  return {{(I == 0 ? &chol_rt<T> : pick_chol<T, static_cast<int>(I)>())...}};
}
template <class T, std::size_t... I>
std::array<ScalarLU<T>, kMaxN + 1> lu_table(std::index_sequence<I...>) {
  return {{(I == 0 ? &lu_rt<T> : pick_lu<T, static_cast<int>(I)>())...}};
}

template <class T> using TileChol = void (*)(T*, int, int*);
template <class T> using TileLU = void (*)(T*, int, int*, int*);
template <class T> TSDX_NOINLINE void chol_tile_rt(T* t, int n, int* i) { kernel::cholesky_tile<T, lanes<T>()>(t, n, i); }
template <class T> TSDX_NOINLINE void lu_tile_rt(T* t, int n, int* p, int* i) { kernel::lu_tile<T, lanes<T>()>(t, n, p, i); }

template <class T, int N> TileChol<T> pick_chol_tile() {
  if constexpr (N <= kMaxUnrolled) return [](T* t, int, int* i) { kernel::cholesky_tile<T, lanes<T>()>(t, N, i); };
  else return &chol_tile_rt<T>;
}
template <class T, int N> TileLU<T> pick_lu_tile() {
  if constexpr (N <= kMaxUnrolled) return [](T* t, int, int* p, int* i) { kernel::lu_tile<T, lanes<T>()>(t, N, p, i); };
  else return &lu_tile_rt<T>;
}
template <class T, std::size_t... I>
std::array<TileChol<T>, kMaxN + 1> chol_tile_table(std::index_sequence<I...>) {
  return {{(I == 0 ? &chol_tile_rt<T> : pick_chol_tile<T, static_cast<int>(I)>())...}};
}
template <class T, std::size_t... I>
std::array<TileLU<T>, kMaxN + 1> lu_tile_table(std::index_sequence<I...>) {
  return {{(I == 0 ? &lu_tile_rt<T> : pick_lu_tile<T, static_cast<int>(I)>())...}};
}

template <class T> const std::array<ScalarChol<T>, kMaxN + 1>& chol_fns() {
  static const auto t = chol_table<T>(std::make_index_sequence<kMaxN + 1>{}); return t;
}
template <class T> const std::array<ScalarLU<T>, kMaxN + 1>& lu_fns() {
  static const auto t = lu_table<T>(std::make_index_sequence<kMaxN + 1>{}); return t;
}
template <class T> const std::array<TileChol<T>, kMaxN + 1>& chol_tile_fns() {
  static const auto t = chol_tile_table<T>(std::make_index_sequence<kMaxN + 1>{}); return t;
}
template <class T> const std::array<TileLU<T>, kMaxN + 1>& lu_tile_fns() {
  static const auto t = lu_tile_table<T>(std::make_index_sequence<kMaxN + 1>{}); return t;
}

}  // namespace detail

// ------------------------------------------------------------------ factor ---
template <class T>
void cholesky_batched(T* A, int n, std::int64_t batch, int* info, Backend backend = Backend::Auto) {
  detail::check_dims(n, batch);
  const Backend be = detail::resolve<T>(backend);
  const std::int64_t nn = static_cast<std::int64_t>(n) * n;
  if (batch == 0) return;

  if (be == Backend::Cuda) {
#ifdef TSDX_WITH_CUDA
    if constexpr (is_native<T>::value) { cuda::cholesky_batched(A, n, batch, info); return; }
#endif
    throw std::runtime_error("CUDA backend unavailable for this build/precision");
  }
  if (be == Backend::Naive) {
    for (std::int64_t b = 0; b < batch; ++b) {
      const int r = kernel::cholesky_naive<T>(A + b * nn, n);
      if (info) info[b] = r;
    }
    return;
  }
  if (be == Backend::Scalar) {
    const auto fn = detail::chol_fns<T>()[n];
    const std::int64_t cs = detail::chunk_size(n, 1);
    parallel_for((batch + cs - 1) / cs, [&](std::int64_t c) {
      const std::int64_t e = std::min(batch, (c + 1) * cs);
      for (std::int64_t b = c * cs; b < e; ++b) {
        const int r = fn(A + b * nn, n);
        if (info) info[b] = r;
      }
    });
    return;
  }
  // Simd: W matrices interleaved per tile, vectorized across the batch.
  constexpr int W = lanes<T>();
  const auto fn = detail::chol_tile_fns<T>()[n];
  const std::int64_t cs = detail::chunk_size(n, W);
  parallel_for((batch + cs - 1) / cs, [&](std::int64_t c) {
    std::vector<T> tile(static_cast<std::size_t>(nn) * W);
    int linfo[W];
    const std::int64_t e = std::min(batch, (c + 1) * cs);
    for (std::int64_t m0 = c * cs; m0 < e; m0 += W) {
      const int cnt = static_cast<int>(std::min<std::int64_t>(W, batch - m0));
      for (int l = 0; l < W; ++l) {
        if (l < cnt) {
          const T* src = A + (m0 + l) * nn;
          for (std::int64_t k = 0; k < nn; ++k) tile[k * W + l] = src[k];
        } else {
          for (std::int64_t k = 0; k < nn; ++k) tile[k * W + l] = (k % (n + 1) == 0) ? T(1) : T(0);
        }
      }
      fn(tile.data(), n, linfo);
      for (int l = 0; l < cnt; ++l) {
        T* dst = A + (m0 + l) * nn;
        for (std::int64_t k = 0; k < nn; ++k) dst[k] = tile[k * W + l];
        if (info) info[m0 + l] = linfo[l];
      }
    }
  });
}

template <class T>
void lu_batched(T* A, int n, std::int64_t batch, int* piv, int* info, Backend backend = Backend::Auto) {
  detail::check_dims(n, batch);
  if (!piv) throw std::invalid_argument("piv must not be null");
  // Measured: per-lane divergent row swaps make the interleaved tile slower than
  // the plain per-matrix kernel from n ~ 32 upwards (see docs/report/bench_*.csv).
  if (backend == Backend::Auto && n >= kLuSimdMaxN + 1) backend = Backend::Scalar;
  const Backend be = detail::resolve<T>(backend);
  const std::int64_t nn = static_cast<std::int64_t>(n) * n;
  if (batch == 0) return;

  if (be == Backend::Cuda) {
#ifdef TSDX_WITH_CUDA
    if constexpr (is_native<T>::value) { cuda::lu_batched(A, n, batch, piv, info); return; }
#endif
    throw std::runtime_error("CUDA backend unavailable for this build/precision");
  }
  if (be == Backend::Naive) {
    for (std::int64_t b = 0; b < batch; ++b) {
      const int r = kernel::lu_naive<T>(A + b * nn, n, piv + b * n);
      if (info) info[b] = r;
    }
    return;
  }
  if (be == Backend::Scalar) {
    const auto fn = detail::lu_fns<T>()[n];
    const std::int64_t cs = detail::chunk_size(n, 1);
    parallel_for((batch + cs - 1) / cs, [&](std::int64_t c) {
      const std::int64_t e = std::min(batch, (c + 1) * cs);
      for (std::int64_t b = c * cs; b < e; ++b) {
        const int r = fn(A + b * nn, n, piv + b * n);
        if (info) info[b] = r;
      }
    });
    return;
  }
  constexpr int W = lanes<T>();
  const auto fn = detail::lu_tile_fns<T>()[n];
  const std::int64_t cs = detail::chunk_size(n, W);
  parallel_for((batch + cs - 1) / cs, [&](std::int64_t c) {
    std::vector<T> tile(static_cast<std::size_t>(nn) * W);
    std::vector<int> tpiv(static_cast<std::size_t>(n) * W);
    int linfo[W];
    const std::int64_t e = std::min(batch, (c + 1) * cs);
    for (std::int64_t m0 = c * cs; m0 < e; m0 += W) {
      const int cnt = static_cast<int>(std::min<std::int64_t>(W, batch - m0));
      for (int l = 0; l < W; ++l) {
        if (l < cnt) {
          const T* src = A + (m0 + l) * nn;
          for (std::int64_t k = 0; k < nn; ++k) tile[k * W + l] = src[k];
        } else {
          for (std::int64_t k = 0; k < nn; ++k) tile[k * W + l] = (k % (n + 1) == 0) ? T(1) : T(0);
        }
      }
      fn(tile.data(), n, tpiv.data(), linfo);
      for (int l = 0; l < cnt; ++l) {
        T* dst = A + (m0 + l) * nn;
        for (std::int64_t k = 0; k < nn; ++k) dst[k] = tile[k * W + l];
        for (int k = 0; k < n; ++k) piv[(m0 + l) * n + k] = tpiv[k * W + l];
        if (info) info[m0 + l] = linfo[l];
      }
    }
  });
}

// ------------------------------------------------------------------ solves ---
// B is (batch, n, nrhs) row-major and is overwritten with the solution.
template <class T>
void cholesky_solve_batched(const T* L, int n, std::int64_t batch, T* B, int nrhs, Backend = Backend::Auto) {
  detail::check_dims(n, batch);
  const std::int64_t nn = static_cast<std::int64_t>(n) * n, bn = static_cast<std::int64_t>(n) * nrhs;
  const std::int64_t cs = detail::chunk_size(n, 1);
  parallel_for((batch + cs - 1) / cs, [&](std::int64_t c) {
    const std::int64_t e = std::min(batch, (c + 1) * cs);
    for (std::int64_t b = c * cs; b < e; ++b) kernel::cholesky_solve_rm<T>(L + b * nn, n, B + b * bn, nrhs);
  });
}

template <class T>
void lu_solve_batched(const T* LU, const int* piv, int n, std::int64_t batch, T* B, int nrhs, Backend = Backend::Auto) {
  detail::check_dims(n, batch);
  const std::int64_t nn = static_cast<std::int64_t>(n) * n, bn = static_cast<std::int64_t>(n) * nrhs;
  const std::int64_t cs = detail::chunk_size(n, 1);
  parallel_for((batch + cs - 1) / cs, [&](std::int64_t c) {
    const std::int64_t e = std::min(batch, (c + 1) * cs);
    for (std::int64_t b = c * cs; b < e; ++b) kernel::lu_solve_rm<T>(LU + b * nn, piv + b * n, n, B + b * bn, nrhs);
  });
}

// One-shot drivers: A is overwritten by its factor, B by the solution.
template <class T>
void posv_batched(T* A, int n, std::int64_t batch, T* B, int nrhs, int* info, Backend be = Backend::Auto) {
  cholesky_batched<T>(A, n, batch, info, be);
  cholesky_solve_batched<T>(A, n, batch, B, nrhs, be);
}
template <class T>
void gesv_batched(T* A, int n, std::int64_t batch, T* B, int nrhs, int* piv, int* info, Backend be = Backend::Auto) {
  lu_batched<T>(A, n, batch, piv, info, be);
  lu_solve_batched<T>(A, piv, n, batch, B, nrhs, be);
}

}  // namespace tsdx
