#ifndef MATRIXGRAPH_CORE_COMMON_HOST_ALGORITHMS_CUH_
#define MATRIXGRAPH_CORE_COMMON_HOST_ALGORITHMS_CUH_

#include <cstdint>

namespace sics {
namespace matrixgraph {
namespace core {
namespace common {

inline uint32_t HashFunction(uint64_t x) {
  x ^= x >> 16;
  x *= 0x85ebca6b;
  x ^= x >> 13;
  x *= 0xc2b2ae35;
  x ^= x >> 16;
  return x;
}

}  // namespace common
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_COMMON_HOST_ALGORITHMS_CUH_
