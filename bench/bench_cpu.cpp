// Google Benchmark micro-benchmarks: Cholesky/LU per backend and size (fp32).
#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "tinysolverdx/tinysolverdx.hpp"

using namespace tsdx;

static std::vector<float> spd(int n, int batch) {
  std::mt19937 g(1);
  std::normal_distribution<float> d;
  std::vector<float> A(static_cast<size_t>(batch) * n * n), M(n * n);
  for (int b = 0; b < batch; ++b) {
    for (auto& v : M) v = d(g);
    float* a = &A[static_cast<size_t>(b) * n * n];
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        float s = (i == j) ? n : 0;
        for (int k = 0; k < n; ++k) s += M[i * n + k] * M[j * n + k];
        a[i * n + j] = s;
      }
  }
  return A;
}

template <bool Chol>
static void BM_factor(benchmark::State& st, Backend be) {
  const int n = static_cast<int>(st.range(0));
  const int batch = 4096;
  const auto A0 = spd(n, batch);
  auto A = A0;
  std::vector<int> piv(static_cast<size_t>(batch) * n), info(batch);
  for (auto _ : st) {
    st.PauseTiming();
    A = A0;
    st.ResumeTiming();
    if (Chol) cholesky_batched<float>(A.data(), n, batch, info.data(), be);
    else lu_batched<float>(A.data(), n, batch, piv.data(), info.data(), be);
    benchmark::DoNotOptimize(A.data());
  }
  const double flops = (Chol ? 1.0 / 3 : 2.0 / 3) * n * n * n;
  st.counters["GFLOP/s"] = benchmark::Counter(flops * batch, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

#define REG(NAME, CHOL, BE) \
  BENCHMARK_CAPTURE(BM_factor<CHOL>, NAME, BE)->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(64)->Unit(benchmark::kMicrosecond)
REG(chol_naive, true, Backend::Naive);
REG(chol_scalar, true, Backend::Scalar);
REG(chol_simd, true, Backend::Simd);
REG(lu_naive, false, Backend::Naive);
REG(lu_scalar, false, Backend::Scalar);
REG(lu_simd, false, Backend::Simd);

BENCHMARK_MAIN();
