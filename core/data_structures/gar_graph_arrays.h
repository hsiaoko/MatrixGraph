#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_GAR_GRAPH_ARRAYS_H_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_GAR_GRAPH_ARRAYS_H_

#include <cstdint>

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

// Serialized graph arrays (input g).
struct GARGraphArrays {
  const uint32_t* v_id = nullptr;
  const int32_t* v_label_idx = nullptr;
  int n_vertices = 0;

  const uint32_t* e_src = nullptr;
  const uint32_t* e_dst = nullptr;
  const uint32_t* e_id = nullptr;
  const int32_t* e_label_idx = nullptr;
  int n_edges = 0;
};

}  // namespace data_structures
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_DATA_STRUCTURES_GAR_GRAPH_ARRAYS_H_
