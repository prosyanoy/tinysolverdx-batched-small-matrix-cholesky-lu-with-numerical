#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "tinysolverdx/tinysolverdx.hpp"

using namespace tsdx;

namespace {

// Well-conditioned SPD matrices: M M^T + n I.
std::vector<double> make_spd(int n, int batch, unsigned seed) {
  std::mt19937_64 g(seed);
  std::normal_distribution<double> d;
  std::vector<double> A(static_cast<size_t>(batch) * n * n);
  std::vector<double> M(n * n);
  for (int b = 0; b < batch; ++b) {
    for (auto& v : M) v = d(g);
    double* a = &A[static_cast<size_t>(b) * n * n];
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        double s = (i == j) ? n : 0;
        for (int k = 0; k < n; ++k) s += M[i * n + k] * M[j * n + k];
        a[i * n + j] = s;
      }
  }
  return A;
}
std::vector<double> make_general(int n, int batch, unsigned seed) {
  std::mt19937_64 g(seed);
  std::normal_distribution<double> d;
  std::vector<double> A(static_cast<size_t>(batch) * n * n);
  for (auto& v : A) v = d(g);
  return A;
}
template <class S> std::vector<S> cast(const std::vector<double>& x) { return std::vector<S>(x.begin(), x.end()); }
std::vector<double> to_d(const std::vector<float>& x) { return std::vector<double>(x.begin(), x.end()); }
std::vector<double> to_d(const std::vector<double>& x) { return x; }

double max_of(const std::vector<double>& v) { double m = 0; for (double x : v) m = std::max(m, x); return m; }

const Backend kBackends[] = {Backend::Naive, Backend::Scalar, Backend::Simd, Backend::Auto};

template <class S>
void check_cholesky(Precision p, double tol_factor, int n, int batch, Backend be) {
  auto A = make_spd(n, batch, 100 + n);
  if (p == Precision::FP32) { auto f = cast<float>(A); A = to_d(f); }
  std::vector<S> F = cast<S>(A);
  round_to<S>(p, F.data(), F.size());
  auto Ar = to_d(std::vector<S>(F));
  std::vector<int> info(batch, -1);
  cholesky<S>(p, F.data(), n, batch, info.data(), be);
  for (int b = 0; b < batch; ++b) ASSERT_EQ(info[b], 0);
  std::vector<double> err(batch);
  chol_backward_error(Ar.data(), to_d(F).data(), n, batch, err.data());
  EXPECT_LT(max_of(err), tol_factor * unit_roundoff(p) * n) << "n=" << n << " backend=" << int(be);
}

}  // namespace

TEST(Cholesky, AllSizesFp32AllBackends) {
  for (Backend be : kBackends)
    for (int n = 1; n <= 64; ++n) check_cholesky<float>(Precision::FP32, 4.0, n, 19, be);
}
TEST(Cholesky, AllSizesFp64AllBackends) {
  for (Backend be : kBackends)
    for (int n = 1; n <= 64; ++n) check_cholesky<double>(Precision::FP64, 4.0, n, 19, be);
}
TEST(Cholesky, EmulatedPrecisions) {
  for (int n : {2, 5, 16, 33, 64}) {
    check_cholesky<float>(Precision::FP16, 4.0, n, 11, Backend::Auto);
    check_cholesky<float>(Precision::BF16, 4.0, n, 11, Backend::Auto);
  }
}

TEST(Cholesky, DetectsNonSpd) {
  const int n = 5, batch = 11;  // odd batch exercises the SIMD tail
  for (Backend be : kBackends) {
    auto A = make_spd(n, batch, 7);
    A[3 * n * n + 2 * n + 2] = -1.0;  // matrix 3: pivot 3 breaks (info = 3)
    for (int k = 0; k < n * n; ++k) A[9 * n * n + k] = 0;  // matrix 9: zero matrix (info = 1)
    std::vector<int> info(batch);
    cholesky<double>(Precision::FP64, A.data(), n, batch, info.data(), be);
    for (int b = 0; b < batch; ++b) {
      const int want = b == 3 ? 3 : b == 9 ? 1 : 0;
      if (b == 3) EXPECT_GT(info[b], 0);
      else EXPECT_EQ(info[b], want) << "b=" << b << " backend=" << int(be);
    }
  }
}

TEST(Cholesky, BackendsAgree) {
  for (int n : {3, 8, 16, 17, 40, 64}) {
    const int batch = 37;
    auto A = make_spd(n, batch, 5 + n);
    auto s = A, v = A, nv = A;
    std::vector<int> info(batch);
    cholesky<double>(Precision::FP64, s.data(), n, batch, info.data(), Backend::Scalar);
    cholesky<double>(Precision::FP64, v.data(), n, batch, info.data(), Backend::Simd);
    cholesky<double>(Precision::FP64, nv.data(), n, batch, info.data(), Backend::Naive);
    for (size_t i = 0; i < A.size(); ++i) {
      ASSERT_NEAR(s[i], v[i], 1e-11 * (1 + std::fabs(s[i])));
      ASSERT_NEAR(s[i], nv[i], 1e-11 * (1 + std::fabs(s[i])));
    }
  }
}

namespace {
template <class S>
void check_lu(Precision p, double tol_factor, int n, int batch, Backend be, bool general_rhs = false) {
  auto A = make_general(n, batch, 300 + n);
  std::vector<S> F = cast<S>(A);
  round_to<S>(p, F.data(), F.size());
  auto Ar = to_d(std::vector<S>(F));
  std::vector<int> piv(static_cast<size_t>(batch) * n), info(batch, -1);
  lu<S>(p, F.data(), n, batch, piv.data(), info.data(), be);
  for (int b = 0; b < batch; ++b) { ASSERT_EQ(info[b], 0); for (int k = 0; k < n; ++k) ASSERT_TRUE(piv[b * n + k] >= k && piv[b * n + k] < n); }
  std::vector<double> err(batch);
  lu_backward_error(Ar.data(), to_d(F).data(), piv.data(), n, batch, err.data());
  // partial pivoting growth factor is tiny for Gaussian matrices
  EXPECT_LT(max_of(err), tol_factor * unit_roundoff(p) * n) << "n=" << n << " backend=" << int(be);
  (void)general_rhs;
}
}  // namespace

TEST(LU, AllSizesFp32AllBackends) {
  for (Backend be : kBackends)
    for (int n = 1; n <= 64; ++n) check_lu<float>(Precision::FP32, 8.0, n, 19, be);
}
TEST(LU, AllSizesFp64AllBackends) {
  for (Backend be : kBackends)
    for (int n = 1; n <= 64; ++n) check_lu<double>(Precision::FP64, 8.0, n, 19, be);
}
TEST(LU, EmulatedPrecisions) {
  for (int n : {2, 7, 16, 32, 64}) {
    check_lu<float>(Precision::FP16, 8.0, n, 9, Backend::Auto);
    check_lu<float>(Precision::BF16, 8.0, n, 9, Backend::Auto);
  }
}

TEST(LU, PivotingMatchesKnownExample) {
  // [[0,1],[1,0]] must swap rows.
  std::vector<double> A = {0, 1, 1, 0};
  for (Backend be : kBackends) {
    auto a = A;
    int piv[2], info;
    lu<double>(Precision::FP64, a.data(), 2, 1, piv, &info, be);
    EXPECT_EQ(info, 0);
    EXPECT_EQ(piv[0], 1);
    EXPECT_EQ(piv[1], 1);
    EXPECT_EQ(a[0], 1); EXPECT_EQ(a[3], 1);
  }
}

TEST(LU, SingularReportsInfo) {
  const int n = 4, batch = 9;
  for (Backend be : kBackends) {
    auto A = make_general(n, batch, 3);
    for (int j = 0; j < n; ++j) A[4 * n * n + 2 * n + j] = A[4 * n * n + 1 * n + j];  // matrix 4: duplicate rows
    for (int k = 0; k < n * n; ++k) A[6 * n * n + k] = 0;                         // matrix 6: zero
    std::vector<int> piv(batch * n), info(batch);
    lu<double>(Precision::FP64, A.data(), n, batch, piv.data(), info.data(), be);
    for (int b = 0; b < batch; ++b) {
      if (b == 6) EXPECT_EQ(info[b], 1);
      else if (b == 4) EXPECT_GT(info[b], 0);
      else EXPECT_EQ(info[b], 0) << "b=" << b;
    }
  }
}

TEST(Solve, GesvAndPosvMatchReference) {
  for (int n : {2, 9, 32, 64}) {
    const int batch = 13;
    auto Ag = make_general(n, batch, 21), As = make_spd(n, batch, 22);
    std::mt19937_64 g(1);
    std::normal_distribution<double> d;
    std::vector<double> b(batch * n);
    for (auto& v : b) v = d(g);
    std::vector<double> xref(batch * n), err(batch);

    auto a = Ag; auto x = b;
    std::vector<int> piv(batch * n), info(batch);
    gesv_batched<double>(a.data(), n, batch, x.data(), 1, piv.data(), info.data());
    reference_solve(Ag.data(), b.data(), n, batch, xref.data());
    for (size_t i = 0; i < x.size(); ++i) ASSERT_NEAR(x[i], xref[i], 1e-8 * (1 + std::fabs(xref[i])));
    solve_backward_error(Ag.data(), x.data(), b.data(), n, batch, err.data());
    EXPECT_LT(max_of(err), 1e-14);

    a = As; x = b;
    posv_batched<double>(a.data(), n, batch, x.data(), 1, info.data());
    reference_solve(As.data(), b.data(), n, batch, xref.data());
    for (size_t i = 0; i < x.size(); ++i) ASSERT_NEAR(x[i], xref[i], 1e-9 * (1 + std::fabs(xref[i])));
  }
}

TEST(Solve, MultipleRightHandSides) {
  const int n = 6, batch = 4, nrhs = 3;
  auto A = make_spd(n, batch, 8);
  std::vector<double> B(batch * n * nrhs);
  for (size_t i = 0; i < B.size(); ++i) B[i] = std::sin(0.3 * i);
  auto a = A; auto x = B;
  std::vector<int> info(batch);
  posv_batched<double>(a.data(), n, batch, x.data(), nrhs, info.data());
  for (int m = 0; m < batch; ++m)
    for (int r = 0; r < nrhs; ++r)
      for (int i = 0; i < n; ++i) {
        double s = 0;
        for (int j = 0; j < n; ++j) s += A[m * n * n + i * n + j] * x[m * n * nrhs + j * nrhs + r];
        ASSERT_NEAR(s, B[m * n * nrhs + i * nrhs + r], 1e-10);
      }
}

TEST(Api, RejectsBadSizes) {
  std::vector<double> a(100 * 100);
  int info;
  EXPECT_THROW(cholesky<double>(Precision::FP64, a.data(), 0, 1, &info), std::invalid_argument);
  EXPECT_THROW(cholesky<double>(Precision::FP64, a.data(), 65, 1, &info), std::invalid_argument);
#ifndef TSDX_WITH_CUDA
  EXPECT_THROW(cholesky<double>(Precision::FP64, a.data(), 4, 1, &info, Backend::Cuda), std::runtime_error);
#endif
}

TEST(Threads, ResultIndependentOfThreadCount) {
  const int n = 12, batch = 5000;
  auto A = make_spd(n, batch, 4);
  auto a1 = A, a4 = A;
  std::vector<int> info(batch);
  set_num_threads(1);
  cholesky<double>(Precision::FP64, a1.data(), n, batch, info.data());
  set_num_threads(4);
  cholesky<double>(Precision::FP64, a4.data(), n, batch, info.data());
  set_num_threads(0);
  EXPECT_EQ(a1, a4);
}

TEST(Cuda, MatchesCpuWhenAvailable) {
#ifdef TSDX_WITH_CUDA
  if (!has_cuda()) GTEST_SKIP() << "no CUDA device";
  for (int n : {4, 16, 17, 48}) {
    const int batch = 100;
    auto A = make_spd(n, batch, 2);
    auto c = A, g = A;
    std::vector<int> i1(batch), i2(batch);
    cholesky<double>(Precision::FP64, c.data(), n, batch, i1.data(), Backend::Scalar);
    cholesky<double>(Precision::FP64, g.data(), n, batch, i2.data(), Backend::Cuda);
    for (size_t k = 0; k < A.size(); ++k) ASSERT_NEAR(c[k], g[k], 1e-10);
  }
#else
  GTEST_SKIP() << "built without CUDA";
#endif
}
