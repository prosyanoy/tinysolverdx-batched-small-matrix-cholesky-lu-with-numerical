// One-matrix-per-thread kernels (n <= 16, matrix held in per-thread registers /
// local memory, compile-time N via the shared header kernels) and
// one-matrix-per-block kernels (n <= 64, matrix staged in shared memory).
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

#include "tinysolverdx/cuda_backend.hpp"
#include "tinysolverdx/kernels.hpp"

namespace tsdx {
namespace cuda {

namespace {
constexpr int kThreadMaxN = 16;

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// ---- thread-per-matrix -------------------------------------------------------
template <class T, int N>
__device__ void chol_thread_n(T* g, int* info) {
  T a[N * N];
  for (int i = 0; i < N * N; ++i) a[i] = g[i];
  *info = kernel::cholesky_rm<T>(a, N);
  for (int i = 0; i < N * N; ++i) g[i] = a[i];
}
template <class T, int N>
__device__ void lu_thread_n(T* g, int* piv, int* info) {
  T a[N * N];
  for (int i = 0; i < N * N; ++i) a[i] = g[i];
  *info = kernel::lu_rm<T>(a, N, piv);
  for (int i = 0; i < N * N; ++i) g[i] = a[i];
}

template <class T, int N>
__device__ void chol_dispatch(int n, T* g, int* info) {
  if (n == N) chol_thread_n<T, N>(g, info);
  else if constexpr (N > 1) chol_dispatch<T, N - 1>(n, g, info);
}
template <class T, int N>
__device__ void lu_dispatch(int n, T* g, int* piv, int* info) {
  if (n == N) lu_thread_n<T, N>(g, piv, info);
  else if constexpr (N > 1) lu_dispatch<T, N - 1>(n, g, piv, info);
}

template <class T>
__global__ void chol_thread_kernel(T* A, int n, long batch, int* info) {
  const long b = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  chol_dispatch<T, kThreadMaxN>(n, A + b * n * n, info + b);
}
template <class T>
__global__ void lu_thread_kernel(T* A, int n, long batch, int* piv, int* info) {
  const long b = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  lu_dispatch<T, kThreadMaxN>(n, A + b * n * n, piv + b * n, info + b);
}

// ---- block-per-matrix (shared memory) ---------------------------------------
template <class T>
__global__ void chol_block_kernel(T* A, int n, int* info) {
  extern __shared__ unsigned char smem_raw[];
  T* s = reinterpret_cast<T*>(smem_raw);
  __shared__ int sinfo;
  T* g = A + static_cast<long>(blockIdx.x) * n * n;
  for (int i = threadIdx.x; i < n * n; i += blockDim.x) s[i] = g[i];
  if (threadIdx.x == 0) sinfo = 0;
  __syncthreads();
  for (int k = 0; k < n; ++k) {
    if (threadIdx.x == 0) {
      const T d = s[k * n + k];
      if (!(d > T(0))) { if (sinfo == 0) sinfo = k + 1; s[k * n + k] = T(1); }
      else s[k * n + k] = sqrt(d);
    }
    __syncthreads();
    const T d = s[k * n + k];
    for (int i = k + 1 + threadIdx.x; i < n; i += blockDim.x) s[i * n + k] /= d;
    __syncthreads();
    const int m = n - k - 1;  // trailing lower triangle: pairs (i, j), k < j <= i
    for (int e = threadIdx.x; e < m * m; e += blockDim.x) {
      const int i = k + 1 + e / m, j = k + 1 + e % m;
      if (j <= i) s[i * n + j] -= s[i * n + k] * s[j * n + k];
    }
    __syncthreads();
  }
  for (int i = threadIdx.x; i < n * n; i += blockDim.x) {
    const int r = i / n, c = i % n;
    g[i] = (c > r) ? T(0) : s[i];
  }
  if (threadIdx.x == 0) info[blockIdx.x] = sinfo;
}

template <class T>
__global__ void lu_block_kernel(T* A, int n, int* piv, int* info) {
  extern __shared__ unsigned char smem_raw[];
  T* s = reinterpret_cast<T*>(smem_raw);
  __shared__ int sinfo, sp;
  T* g = A + static_cast<long>(blockIdx.x) * n * n;
  for (int i = threadIdx.x; i < n * n; i += blockDim.x) s[i] = g[i];
  if (threadIdx.x == 0) sinfo = 0;
  __syncthreads();
  for (int k = 0; k < n; ++k) {
    if (threadIdx.x == 0) {  // pivot search over <= 64 entries
      int p = k;
      T best = t_abs(s[k * n + k]);
      for (int i = k + 1; i < n; ++i) {
        const T v = t_abs(s[i * n + k]);
        if (v > best) { best = v; p = i; }
      }
      sp = p;
      piv[static_cast<long>(blockIdx.x) * n + k] = p;
      if (s[p * n + k] == T(0) && sinfo == 0) sinfo = k + 1;
    }
    __syncthreads();
    const int p = sp;
    if (p != k)
      for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const T tmp = s[k * n + j]; s[k * n + j] = s[p * n + j]; s[p * n + j] = tmp;
      }
    __syncthreads();
    const T pk = s[k * n + k];
    if (pk != T(0)) {
      for (int i = k + 1 + threadIdx.x; i < n; i += blockDim.x) s[i * n + k] /= pk;
      __syncthreads();
      const int m = n - k - 1;
      for (int e = threadIdx.x; e < m * m; e += blockDim.x) {
        const int i = k + 1 + e / m, j = k + 1 + e % m;
        s[i * n + j] -= s[i * n + k] * s[k * n + j];
      }
    }
    __syncthreads();
  }
  for (int i = threadIdx.x; i < n * n; i += blockDim.x) g[i] = s[i];
  if (threadIdx.x == 0) info[blockIdx.x] = sinfo;
}

template <class T> struct DevBuf {
  T* p = nullptr;
  explicit DevBuf(std::size_t count) { check(cudaMalloc(&p, count * sizeof(T)), "cudaMalloc"); }
  ~DevBuf() { cudaFree(p); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
};
}  // namespace

bool available() {
  int c = 0;
  return cudaGetDeviceCount(&c) == cudaSuccess && c > 0;
}

template <class T>
void cholesky_batched(T* A, int n, std::int64_t batch, int* info) {
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  DevBuf<T> dA(nn * batch);
  DevBuf<int> dI(batch);
  check(cudaMemcpy(dA.p, A, nn * batch * sizeof(T), cudaMemcpyHostToDevice), "H2D");
  if (n <= kThreadMaxN) {
    const int tpb = 128;
    chol_thread_kernel<T><<<static_cast<unsigned>((batch + tpb - 1) / tpb), tpb>>>(dA.p, n, batch, dI.p);
  } else {
    chol_block_kernel<T><<<static_cast<unsigned>(batch), 128, nn * sizeof(T)>>>(dA.p, n, dI.p);
  }
  check(cudaGetLastError(), "launch");
  check(cudaMemcpy(A, dA.p, nn * batch * sizeof(T), cudaMemcpyDeviceToHost), "D2H");
  if (info) check(cudaMemcpy(info, dI.p, batch * sizeof(int), cudaMemcpyDeviceToHost), "D2H info");
}

template <class T>
void lu_batched(T* A, int n, std::int64_t batch, int* piv, int* info) {
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  DevBuf<T> dA(nn * batch);
  DevBuf<int> dP(static_cast<std::size_t>(n) * batch), dI(batch);
  check(cudaMemcpy(dA.p, A, nn * batch * sizeof(T), cudaMemcpyHostToDevice), "H2D");
  if (n <= kThreadMaxN) {
    const int tpb = 128;
    lu_thread_kernel<T><<<static_cast<unsigned>((batch + tpb - 1) / tpb), tpb>>>(dA.p, n, batch, dP.p, dI.p);
  } else {
    lu_block_kernel<T><<<static_cast<unsigned>(batch), 128, nn * sizeof(T)>>>(dA.p, n, dP.p, dI.p);
  }
  check(cudaGetLastError(), "launch");
  check(cudaMemcpy(A, dA.p, nn * batch * sizeof(T), cudaMemcpyDeviceToHost), "D2H");
  check(cudaMemcpy(piv, dP.p, static_cast<std::size_t>(n) * batch * sizeof(int), cudaMemcpyDeviceToHost), "D2H piv");
  if (info) check(cudaMemcpy(info, dI.p, batch * sizeof(int), cudaMemcpyDeviceToHost), "D2H info");
}

template void cholesky_batched<float>(float*, int, std::int64_t, int*);
template void cholesky_batched<double>(double*, int, std::int64_t, int*);
template void lu_batched<float>(float*, int, std::int64_t, int*, int*);
template void lu_batched<double>(double*, int, std::int64_t, int*, int*);

}  // namespace cuda
}  // namespace tsdx
