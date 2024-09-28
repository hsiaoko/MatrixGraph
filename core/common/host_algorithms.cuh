#ifndef MATRIXGRAPH_CORE_COMMON_HOST_ALGORITHMS_CUH_
#define MATRIXGRAPH_CORE_COMMON_HOST_ALGORITHMS_CUH_

namespace sics {
namespace matrixgraph {
namespace core {
namespace common {

static uint32_t hash_function(uint64_t x) {
  x ^= x >> 16;
  x *= 0x85ebca6b;
  x ^= x >> 13;
  x *= 0xc2b2ae35;
  x ^= x >> 16;
  return x;
}

} // namespace common
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_COMMON_HOST_ALGORITHMS_CUH_
