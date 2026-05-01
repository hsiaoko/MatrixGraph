#ifndef MATRIXGRAPH_CORE_UTIL_CUDA_PREFETCH_CUH_
#define MATRIXGRAPH_CORE_UTIL_CUDA_PREFETCH_CUH_

#include "core/util/cuda_check.cuh"
#include <algorithm>
#include <cuda_runtime.h>
#include <utility>
#include <vector>

namespace sics {
namespace matrixgraph {
namespace core {
namespace util {

// Prefetch managed-memory ranges to `device` using round-robin CUDA streams,
// then synchronize and destroy streams. Overlaps driver migration when multiple
// buffers are large.
inline void MatrixGraphPrefetchManagedToDevice(
    int device, int n_streams,
    const std::vector<std::pair<void*, size_t>>& ptr_size) {
  if (ptr_size.empty()) {
    return;
  }
  n_streams = std::max(1, n_streams);
  std::vector<cudaStream_t> streams(static_cast<size_t>(n_streams));
  for (int i = 0; i < n_streams; ++i) {
    CUDA_CHECK(cudaStreamCreate(&streams[static_cast<size_t>(i)]));
  }
  for (size_t i = 0; i < ptr_size.size(); ++i) {
    void* p = ptr_size[i].first;
    size_t bytes = ptr_size[i].second;
    if (p != nullptr && bytes > 0) {
      CUDA_CHECK(cudaMemPrefetchAsync(
          p, bytes, device,
          streams[i % static_cast<size_t>(n_streams)]));
    }
  }
  for (int i = 0; i < n_streams; ++i) {
    CUDA_CHECK(cudaStreamSynchronize(streams[static_cast<size_t>(i)]));
    CUDA_CHECK(cudaStreamDestroy(streams[static_cast<size_t>(i)]));
  }
}

}  // namespace util
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_UTIL_CUDA_PREFETCH_CUH_
