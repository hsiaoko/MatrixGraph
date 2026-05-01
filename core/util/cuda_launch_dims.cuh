#ifndef MATRIXGRAPH_CORE_UTIL_CUDA_LAUNCH_DIMS_CUH_
#define MATRIXGRAPH_CORE_UTIL_CUDA_LAUNCH_DIMS_CUH_

#include <cstdlib>

namespace sics {
namespace matrixgraph {
namespace core {
namespace util {

// Optional launch geometry for WCC / BFS / PageRank kernels (host-side reads at
// kernel launch). When unset or invalid, callers pass their compile-time defaults.
inline unsigned MatrixGraphEnvLaunchGridDim(unsigned default_grid) {
  const char* s = std::getenv("MG_GPU_GRID");
  if (s != nullptr && s[0] != '\0') {
    int v = std::atoi(s);
    if (v > 0) {
      return static_cast<unsigned>(v);
    }
  }
  return default_grid;
}

inline unsigned MatrixGraphEnvLaunchBlockDim(unsigned default_block) {
  const char* s = std::getenv("MG_GPU_BLOCK");
  if (s != nullptr && s[0] != '\0') {
    int v = std::atoi(s);
    if (v > 0) {
      return static_cast<unsigned>(v);
    }
  }
  return default_block;
}

}  // namespace util
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_UTIL_CUDA_LAUNCH_DIMS_CUH_
