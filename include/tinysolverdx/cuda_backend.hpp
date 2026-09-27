// CUDA backend (compiled only when nvcc is found). Host-pointer API: data are
// staged to the device, factored, and copied back. float and double only.
#pragma once
#include <cstdint>

namespace tsdx {
namespace cuda {

bool available();
template <class T> void cholesky_batched(T* A, int n, std::int64_t batch, int* info);
template <class T> void lu_batched(T* A, int n, std::int64_t batch, int* piv, int* info);

}  // namespace cuda
}  // namespace tsdx
