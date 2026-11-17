#ifndef MATRIXGRAPH_CORE_COMMON_TYPES_H_
#define MATRIXGRAPH_CORE_COMMON_TYPES_H_

#include <cstdint>





namespace sics {
namespace matrixgraph {
namespace core {
namespace common {

using GraphID = uint32_t;  // uint32_t: 0 ~ 4,294,967,295
using VertexID = uint32_t;
using VertexIndex = uint32_t;
using TileIndex = uint16_t;
using VertexLabel = uint32_t;
using VertexDegree = uint32_t;
using VertexCount = uint32_t;
using EdgeIndex = uint32_t;



enum StoreStrategy {
  kUnconstrained,  // default
  kIncomingOnly,
  kOutgoingOnly,
  kUndefinedStrategy
};

}  // namespace common
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_COMMON_TYPES_H_