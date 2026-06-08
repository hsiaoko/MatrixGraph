#ifndef MATRIXGRAPH_CORE_UTIL_CUDA_DEVICE_CUH_
#define MATRIXGRAPH_CORE_UTIL_CUDA_DEVICE_CUH_

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <numeric>
#include <vector>

namespace sics {
namespace matrixgraph {
namespace core {
namespace util {

// Primary GPU index from MATRIXGRAPH_CUDA_DEVICE (0-based), else 0.
// Clamped to [0, device_count). Used when MATRIXGRAPH_CUDA_DEVICES is unset.
inline int MatrixGraphCudaPrimaryDeviceIndex() {
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

// Parse comma-separated non-negative integers (spaces allowed).
inline void MatrixGraphParseDeviceList(const char* str, std::vector<int>* out) {
  out->clear();
  if (str == nullptr) {
    return;
  }
  while (*str != '\0') {
    while (*str == ' ' || *str == ',' || *str == '\t') {
      ++str;
    }
    if (*str == '\0') {
      break;
    }
    char* end = nullptr;
    long v = std::strtol(str, &end, 10);
    if (end == str) {
      break;
    }
    out->push_back(static_cast<int>(v));
    str = end;
  }
}

// Active CUDA devices for multi-GPU execution.
// Precedence:
// - MATRIXGRAPH_CUDA_DEVICES="0,1,2" (non-empty after parsing valid ids) wins over
//   MATRIXGRAPH_CUDA_ALL_DEVICES so an explicit list is never overridden by “use all”.
// - MATRIXGRAPH_CUDA_ALL_DEVICES=1 — every visible device 0..count-1 (only when the
//   explicit list env is unset, empty, or parses to zero valid GPUs).
// - Otherwise — single device { MatrixGraphCudaPrimaryDeviceIndex() }.
inline std::vector<int> MatrixGraphCudaDeviceList() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) {
    return {0};
  }

  const char* list_env = std::getenv("MATRIXGRAPH_CUDA_DEVICES");
  if (list_env != nullptr && list_env[0] != '\0') {
    std::vector<int> parsed;
    MatrixGraphParseDeviceList(list_env, &parsed);
    std::vector<int> valid;
    valid.reserve(parsed.size());
    for (int id : parsed) {
      if (id >= 0 && id < n) {
        valid.push_back(id);
      }
    }
    if (!valid.empty()) {
      return valid;
    }
  }

  const char* use_all = std::getenv("MATRIXGRAPH_CUDA_ALL_DEVICES");
  if (use_all != nullptr && use_all[0] == '1' && use_all[1] == '\0') {
    std::vector<int> all(static_cast<size_t>(n));
    std::iota(all.begin(), all.end(), 0);
    return all;
  }

  return {MatrixGraphCudaPrimaryDeviceIndex()};
}

// Number of non-default streams per GPU for host-side overlap (prefetch, etc.).
// Env MATRIXGRAPH_CUDA_STREAMS (default 2). Clamped to >= 1.
inline int MatrixGraphCudaStreamsPerGpu() {
  const char* e = std::getenv("MATRIXGRAPH_CUDA_STREAMS");
  int s = (e != nullptr && e[0] != '\0') ? std::atoi(e) : 2;
  return std::max(1, s);
}

// Same as MatrixGraphCudaPrimaryDeviceIndex (first / default GPU).
inline int MatrixGraphCudaDevice() { return MatrixGraphCudaPrimaryDeviceIndex(); }

}  // namespace util
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_UTIL_CUDA_DEVICE_CUH_
