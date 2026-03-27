#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_GAR_PATTERN_ARRAYS_H_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_GAR_PATTERN_ARRAYS_H_

#include <algorithm>
#include <cstdint>
#include <iostream>

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

  void Print(int k = 3) const {
    const int top_n = std::max(0, k);
    std::cout << "[GARPatternArrays] n_nodes=" << n_nodes
              << " n_edges=" << n_edges << std::endl;
    if (node_label_idx && n_nodes > 0) {
      const int limit = std::min(n_nodes, top_n);
      std::cout << "p.node_label_idx (n=" << n_nodes << ", top=" << limit
                << "): [";
      for (int i = 0; i < limit; ++i) {
        if (i) std::cout << ", ";
        std::cout << node_label_idx[i];
      }
      if (limit < n_nodes) std::cout << ", ...";
      std::cout << "]" << std::endl;
    }
    if (edge_src && n_edges > 0) {
      const int limit = std::min(n_edges, top_n);
      std::cout << "p.edge_src (n=" << n_edges << ", top=" << limit << "): [";
      for (int i = 0; i < limit; ++i) {
        if (i) std::cout << ", ";
        std::cout << edge_src[i];
      }
      if (limit < n_edges) std::cout << ", ...";
      std::cout << "]" << std::endl;
    }
    if (edge_dst && n_edges > 0) {
      const int limit = std::min(n_edges, top_n);
      std::cout << "p.edge_dst (n=" << n_edges << ", top=" << limit << "): [";
      for (int i = 0; i < limit; ++i) {
        if (i) std::cout << ", ";
        std::cout << edge_dst[i];
      }
      if (limit < n_edges) std::cout << ", ...";
      std::cout << "]" << std::endl;
    }
    if (edge_label_idx && n_edges > 0) {
      const int limit = std::min(n_edges, top_n);
      std::cout << "p.edge_label_idx (n=" << n_edges << ", top=" << limit
                << "): [";
      for (int i = 0; i < limit; ++i) {
        if (i) std::cout << ", ";
        std::cout << edge_label_idx[i];
      }
      if (limit < n_edges) std::cout << ", ...";
      std::cout << "]" << std::endl;
    }
  }
};

}  // namespace data_structures
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_DATA_STRUCTURES_GAR_PATTERN_ARRAYS_H_
