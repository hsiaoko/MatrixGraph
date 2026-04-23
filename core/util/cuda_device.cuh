#ifndef MATRIXGRAPH_CORE_UTIL_CUDA_DEVICE_CUH_
#define MATRIXGRAPH_CORE_UTIL_CUDA_DEVICE_CUH_

#include <cstdlib>
#include <cuda_runtime.h>

namespace sics {
namespace matrixgraph {
namespace core {
namespace util {

// Pick CUDA device: env MATRIXGRAPH_CUDA_DEVICE (0-based), else 0.
// Clamped to [0, device_count). Falls back to 0 if CUDA fails or count is 0.
inline int MatrixGraphCudaDevice() {
  int want = 0;
  const char* e = std::getenv("MATRIXGRAPH_CUDA_DEVICE");
  if (e != nullptr && e[0] != '\0') {
    want = std::atoi(e);
  }
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) {
    return 0;
  }
  if (want < 0) {
    want = 0;
  }
  if (want >= n) {
    want = 0;
  }
  return want;
}

}  // namespace util
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_UTIL_CUDA_DEVICE_CUH_
