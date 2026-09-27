#include "tinysolverdx/tinysolverdx.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <thread>

#ifdef TSDX_HAVE_OPENMP
#include <omp.h>
#endif

namespace tsdx {

// ----------------------------------------------------------------- threading ---
namespace {
int default_threads() {
  if (const char* e = std::getenv("TSDX_NUM_THREADS")) {
    const int v = std::atoi(e);
    if (v > 0) return v;
  }
  const unsigned h = std::thread::hardware_concurrency();
  return h ? static_cast<int>(h) : 1;
}
std::atomic<int> g_threads{0};
}  // namespace

int num_threads() {
  int t = g_threads.load();
  if (t <= 0) { t = default_threads(); g_threads = t; }
  return t;
}
void set_num_threads(int n) { g_threads = n > 0 ? n : default_threads(); }

bool has_openmp() {
#ifdef TSDX_HAVE_OPENMP
  return true;
#else
  return false;
#endif
}

void parallel_for(std::int64_t n_chunks, const std::function<void(std::int64_t)>& f) {
  const int nt = static_cast<int>(std::min<std::int64_t>(num_threads(), n_chunks));
  if (nt <= 1) { for (std::int64_t c = 0; c < n_chunks; ++c) f(c); return; }
#ifdef TSDX_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(nt)
  for (std::int64_t c = 0; c < n_chunks; ++c) f(c);
#else
  std::atomic<std::int64_t> next{0};
  std::vector<std::thread> pool;
  for (int t = 0; t < nt; ++t)
    pool.emplace_back([&] {
      for (std::int64_t c; (c = next.fetch_add(1)) < n_chunks;) f(c);
    });
  for (auto& th : pool) th.join();
#endif
}

bool has_cuda() {
#ifdef TSDX_WITH_CUDA
  return cuda::available();
#else
  return false;
#endif
}

// ------------------------------------------------------- runtime precision ---
namespace {

template <class F>
void with_precision(Precision p, F&& f) {
  switch (p) {
    case Precision::FP32: f(float{}); break;
    case Precision::FP64: f(double{}); break;
    case Precision::FP16: f(Half{}); break;
    case Precision::BF16: f(BFloat16{}); break;
  }
}

template <class T, class S> std::vector<T> to_t(const S* x, std::int64_t count) {
  std::vector<T> v(static_cast<std::size_t>(count));
  for (std::int64_t i = 0; i < count; ++i) v[i] = T(x[i]);
  return v;
}
template <class T> double as_d(T x) { return static_cast<double>(x); }
inline double as_d(Half x) { return x.v; }
inline double as_d(BFloat16 x) { return x.v; }
template <class T, class S> void from_t(const std::vector<T>& v, S* x) {
  for (std::size_t i = 0; i < v.size(); ++i) x[i] = static_cast<S>(as_d(v[i]));
}

}  // namespace

template <class S>
void cholesky(Precision p, S* A, int n, std::int64_t batch, int* info, Backend be) {
  with_precision(p, [&](auto tag) {
    using T = decltype(tag);
    if constexpr (std::is_same<T, S>::value) cholesky_batched<T>(A, n, batch, info, be);
    else {
      auto v = to_t<T>(A, batch * n * n);
      cholesky_batched<T>(v.data(), n, batch, info, be);
      from_t(v, A);
    }
  });
}

template <class S>
void lu(Precision p, S* A, int n, std::int64_t batch, int* piv, int* info, Backend be) {
  with_precision(p, [&](auto tag) {
    using T = decltype(tag);
    if constexpr (std::is_same<T, S>::value) lu_batched<T>(A, n, batch, piv, info, be);
    else {
      auto v = to_t<T>(A, batch * n * n);
      lu_batched<T>(v.data(), n, batch, piv, info, be);
      from_t(v, A);
    }
  });
}

template <class S>
void cholesky_solve(Precision p, const S* L, int n, std::int64_t batch, S* B, int nrhs) {
  with_precision(p, [&](auto tag) {
    using T = decltype(tag);
    auto l = to_t<T>(L, batch * n * n);
    auto b = to_t<T>(B, batch * n * nrhs);
    cholesky_solve_batched<T>(l.data(), n, batch, b.data(), nrhs);
    from_t(b, B);
  });
}

template <class S>
void lu_solve(Precision p, const S* LU, const int* piv, int n, std::int64_t batch, S* B, int nrhs) {
  with_precision(p, [&](auto tag) {
    using T = decltype(tag);
    auto l = to_t<T>(LU, batch * n * n);
    auto b = to_t<T>(B, batch * n * nrhs);
    lu_solve_batched<T>(l.data(), piv, n, batch, b.data(), nrhs);
    from_t(b, B);
  });
}

template <class S>
void round_to(Precision p, S* x, std::int64_t count) {
  if (p == Precision::FP64) return;
  for (std::int64_t i = 0; i < count; ++i) {
    const float f = static_cast<float>(x[i]);
    x[i] = static_cast<S>(p == Precision::FP32 ? f : p == Precision::FP16 ? detail::round_fp16(f) : detail::round_bf16(f));
  }
}

#define TSDX_INST(S)                                                                                       \
  template void cholesky<S>(Precision, S*, int, std::int64_t, int*, Backend);                              \
  template void lu<S>(Precision, S*, int, std::int64_t, int*, int*, Backend);                              \
  template void cholesky_solve<S>(Precision, const S*, int, std::int64_t, S*, int);                        \
  template void lu_solve<S>(Precision, const S*, const int*, int, std::int64_t, S*, int);                  \
  template void round_to<S>(Precision, S*, std::int64_t);
TSDX_INST(float)
TSDX_INST(double)
#undef TSDX_INST

// ------------------------------------------------------------- ground truth ---
namespace {
using LD = long double;

// Long-double LU solve with partial pivoting on a copy.
void ld_solve(const double* A, const double* b, int n, LD* x) {
  std::vector<LD> a(static_cast<std::size_t>(n) * n);
  for (int i = 0; i < n * n; ++i) a[i] = A[i];
  for (int i = 0; i < n; ++i) x[i] = b[i];
  for (int k = 0; k < n; ++k) {
    int p = k;
    for (int i = k + 1; i < n; ++i) if (std::fabs(a[i * n + k]) > std::fabs(a[p * n + k])) p = i;
    if (p != k) {
      for (int j = 0; j < n; ++j) std::swap(a[k * n + j], a[p * n + j]);
      std::swap(x[k], x[p]);
    }
    for (int i = k + 1; i < n; ++i) {
      const LD f = a[i * n + k] / a[k * n + k];
      for (int j = k + 1; j < n; ++j) a[i * n + j] -= f * a[k * n + j];
      x[i] -= f * x[k];
    }
  }
  for (int i = n - 1; i >= 0; --i) {
    LD s = x[i];
    for (int j = i + 1; j < n; ++j) s -= a[i * n + j] * x[j];
    x[i] = s / a[i * n + i];
  }
}

LD fro(const double* a, int cnt) { LD s = 0; for (int i = 0; i < cnt; ++i) s += (LD)a[i] * a[i]; return std::sqrt(s); }
}  // namespace

void chol_backward_error(const double* A, const double* L, int n, std::int64_t batch, double* out) {
  parallel_for(batch, [&](std::int64_t b) {
    const double* a = A + b * n * n; const double* l = L + b * n * n;
    LD num = 0;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        LD s = 0;
        for (int k = 0; k <= std::min(i, j); ++k) s += (LD)l[i * n + k] * l[j * n + k];
        const LD d = (LD)a[i * n + j] - s;
        num += d * d;
      }
    out[b] = static_cast<double>(std::sqrt(num) / fro(a, n * n));
  });
}

void lu_backward_error(const double* A, const double* LU, const int* piv, int n, std::int64_t batch, double* out) {
  parallel_for(batch, [&](std::int64_t b) {
    const double* a = A + b * n * n; const double* f = LU + b * n * n; const int* pv = piv + b * n;
    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i) perm[i] = i;
    for (int k = 0; k < n; ++k) std::swap(perm[k], perm[pv[k]]);  // row i of PA is row perm[i] of A
    LD num = 0;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        LD s = 0;
        for (int k = 0; k <= std::min(i, j); ++k) {
          const LD lik = (k == i) ? (LD)1 : (LD)f[i * n + k];
          s += lik * f[k * n + j];
        }
        const LD d = (LD)a[perm[i] * n + j] - s;
        num += d * d;
      }
    out[b] = static_cast<double>(std::sqrt(num) / fro(a, n * n));
  });
}

void solve_backward_error(const double* A, const double* x, const double* bb, int n, std::int64_t batch, double* out) {
  parallel_for(batch, [&](std::int64_t b) {
    const double* a = A + b * n * n; const double* xv = x + b * n; const double* rhs = bb + b * n;
    LD num = 0;
    for (int i = 0; i < n; ++i) {
      LD s = rhs[i];
      for (int j = 0; j < n; ++j) s -= (LD)a[i * n + j] * xv[j];
      num += s * s;
    }
    const LD den = fro(a, n * n) * fro(xv, n) + fro(rhs, n);
    out[b] = static_cast<double>(std::sqrt(num) / den);
  });
}

void reference_solve(const double* A, const double* b, int n, std::int64_t batch, double* x) {
  parallel_for(batch, [&](std::int64_t m) {
    std::vector<LD> xr(n);
    ld_solve(A + m * n * n, b + m * n, n, xr.data());
    for (int i = 0; i < n; ++i) x[m * n + i] = static_cast<double>(xr[i]);
  });
}

void solve_forward_error(const double* A, const double* x, const double* b, int n, std::int64_t batch, double* out) {
  parallel_for(batch, [&](std::int64_t m) {
    std::vector<LD> xr(n);
    ld_solve(A + m * n * n, b + m * n, n, xr.data());
    LD num = 0, den = 0;
    for (int i = 0; i < n; ++i) { const LD d = (LD)x[m * n + i] - xr[i]; num += d * d; den += xr[i] * xr[i]; }
    out[m] = static_cast<double>(std::sqrt(num / den));
  });
}

// ------------------------------------------------------------ machine probes ---
namespace {
using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

template <class T>
double peak_t() {
  constexpr int L = 64 / sizeof(T), R = 8;
  const long iters = 2000000;
  const int nt = num_threads();
  std::vector<double> sink(nt);
  const auto t0 = Clock::now();
  parallel_for(nt, [&](std::int64_t id) {
    T acc[R][L];
    for (int r = 0; r < R; ++r) for (int l = 0; l < L; ++l) acc[r][l] = T(1) + T(r * L + l) * T(1e-3);
    const volatile T mv = T(0.999999), cv = T(1e-7);
    const T m = mv, c = cv;
    for (long i = 0; i < iters; ++i)
      for (int r = 0; r < R; ++r) for (int l = 0; l < L; ++l) acc[r][l] = acc[r][l] * m + c;
    double s = 0;
    for (int r = 0; r < R; ++r) for (int l = 0; l < L; ++l) s += acc[r][l];
    sink[id] = s;
  });
  const double dt = secs(t0, Clock::now());
  return 2.0 * R * L * iters * nt / dt * 1e-9;
}
}  // namespace

double measure_peak_gflops(Precision p) {
  double best = 0;
  for (int rep = 0; rep < 3; ++rep) best = std::max(best, p == Precision::FP64 ? peak_t<double>() : peak_t<float>());
  return best;
}

double measure_bandwidth_gbs() {
  const std::size_t cnt = 8u << 20;  // 32 MiB per float array, well beyond cache
  std::vector<float> a(cnt, 1.f), b(cnt, 2.f), c(cnt, 0.f);
  const std::int64_t chunks = 64, per = cnt / chunks;
  double best = 0;
  for (int rep = 0; rep < 5; ++rep) {
    const auto t0 = Clock::now();
    parallel_for(chunks, [&](std::int64_t k) {
      for (std::int64_t i = k * per; i < (k + 1) * per; ++i) c[i] = a[i] + 3.0f * b[i];
    });
    best = std::max(best, 3.0 * cnt * sizeof(float) / secs(t0, Clock::now()) * 1e-9);
  }
  volatile float s = c[cnt / 2]; (void)s;
  return best;
}

}  // namespace tsdx
