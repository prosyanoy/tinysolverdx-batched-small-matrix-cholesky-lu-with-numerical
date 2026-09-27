// Minimal parallel_for: OpenMP when available, std::thread otherwise.
#pragma once

#include <cstdint>
#include <functional>

namespace tsdx {

int num_threads();
void set_num_threads(int n);  // n <= 0 restores the default (TSDX_NUM_THREADS or hardware concurrency)
bool has_openmp();

// Runs f(0..n_chunks-1), possibly concurrently. f must be thread-safe.
void parallel_for(std::int64_t n_chunks, const std::function<void(std::int64_t)>& f);

}  // namespace tsdx
