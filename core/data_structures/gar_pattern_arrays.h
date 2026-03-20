#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_GAR_PATTERN_ARRAYS_H_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_GAR_PATTERN_ARRAYS_H_

#include <cstdint>

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

// Serialized pattern arrays (input p).
struct GARPatternArrays {
  const int32_t* node_label_idx = nullptr;
  int n_nodes = 0;

  const int32_t* edge_src = nullptr;
  const int32_t* edge_dst = nullptr;
  const int32_t* edge_label_idx = nullptr;
  int n_edges = 0;
};

}  // namespace data_structures
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_DATA_STRUCTURES_GAR_PATTERN_ARRAYS_H_
