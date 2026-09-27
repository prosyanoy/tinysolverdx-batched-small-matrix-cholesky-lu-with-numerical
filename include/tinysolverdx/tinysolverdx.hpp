// Umbrella header: typed templates (batched.hpp) plus runtime-precision entry
// points over float/double buffers, reference metrics and machine probes.
#pragma once

#include "batched.hpp"
#include "parallel.hpp"
#include "precision.hpp"

namespace tsdx {

bool has_cuda();  // compiled with the CUDA backend and a device is visible

// Runtime-precision API. S is the storage type of the caller's buffers (float or
// double). For FP16/BF16 the data are rounded into the emulated format, factored
// with per-operation rounding, and written back (exactly representable in S).
template <class S> void cholesky(Precision p, S* A, int n, std::int64_t batch, int* info, Backend be = Backend::Auto);
template <class S> void lu(Precision p, S* A, int n, std::int64_t batch, int* piv, int* info, Backend be = Backend::Auto);
template <class S> void cholesky_solve(Precision p, const S* L, int n, std::int64_t batch, S* B, int nrhs);
template <class S> void lu_solve(Precision p, const S* LU, const int* piv, int n, std::int64_t batch, S* B, int nrhs);
template <class S> void round_to(Precision p, S* x, std::int64_t count);

// ---- long-double ground truth & error metrics (inputs are doubles) -----------
// All return one value per matrix.
void chol_backward_error(const double* A, const double* L, int n, std::int64_t batch, double* out);  // ||A-LL^T||_F/||A||_F
void lu_backward_error(const double* A, const double* LU, const int* piv, int n, std::int64_t batch, double* out);  // ||PA-LU||_F/||A||_F
void solve_backward_error(const double* A, const double* x, const double* b, int n, std::int64_t batch, double* out);  // ||b-Ax||/(||A||_F||x||+||b||)
void solve_forward_error(const double* A, const double* x, const double* b, int n, std::int64_t batch, double* out);  // ||x-x_ref||/||x_ref||
void reference_solve(const double* A, const double* b, int n, std::int64_t batch, double* x);

// ---- machine probes for the roofline chart -----------------------------------
double measure_peak_gflops(Precision p);   // multiply-add loop, all threads
double measure_bandwidth_gbs();            // triad, all threads

}  // namespace tsdx
