#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_GAR_GRAPH_ARRAYS_H_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_GAR_GRAPH_ARRAYS_H_

#include <algorithm>
#include <cstdint>
#include <iostream>

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

  void Print(int k = 3) const {
    const int top_n = std::max(0, k);
    std::cout << "[GARGraphArrays] n_vertices=" << n_vertices
              << " n_edges=" << n_edges << std::endl;
    if (v_id && n_vertices > 0) {
      const int limit = std::min(n_vertices, top_n);
      std::cout << "g.v_id (n=" << n_vertices << ", top=" << limit << "): [";
      for (int i = 0; i < limit; ++i) {
        if (i) std::cout << ", ";
        std::cout << v_id[i];
      }
      if (limit < n_vertices) std::cout << ", ...";
      std::cout << "]" << std::endl;
    }
    if (v_label_idx && n_vertices > 0) {
      const int limit = std::min(n_vertices, top_n);
      std::cout << "g.v_label_idx (n=" << n_vertices << ", top=" << limit
                << "): [";
      for (int i = 0; i < limit; ++i) {
        if (i) std::cout << ", ";
        std::cout << v_label_idx[i];
      }
      if (limit < n_vertices) std::cout << ", ...";
      std::cout << "]" << std::endl;
    }
    if (e_src && n_edges > 0) {
      const int limit = std::min(n_edges, top_n);
      std::cout << "g.e_src (n=" << n_edges << ", top=" << limit << "): [";
      for (int i = 0; i < limit; ++i) {
        if (i) std::cout << ", ";
        std::cout << e_src[i];
      }
      if (limit < n_edges) std::cout << ", ...";
      std::cout << "]" << std::endl;
    }
    if (e_dst && n_edges > 0) {
      const int limit = std::min(n_edges, top_n);
      std::cout << "g.e_dst (n=" << n_edges << ", top=" << limit << "): [";
      for (int i = 0; i < limit; ++i) {
        if (i) std::cout << ", ";
        std::cout << e_dst[i];
      }
      if (limit < n_edges) std::cout << ", ...";
      std::cout << "]" << std::endl;
    }
    if (e_id && n_edges > 0) {
      const int limit = std::min(n_edges, top_n);
      std::cout << "g.e_id (n=" << n_edges << ", top=" << limit << "): [";
      for (int i = 0; i < limit; ++i) {
        if (i) std::cout << ", ";
        std::cout << e_id[i];
      }
      if (limit < n_edges) std::cout << ", ...";
      std::cout << "]" << std::endl;
    }
    if (e_label_idx && n_edges > 0) {
      const int limit = std::min(n_edges, top_n);
      std::cout << "g.e_label_idx (n=" << n_edges << ", top=" << limit
                << "): [";
      for (int i = 0; i < limit; ++i) {
        if (i) std::cout << ", ";
        std::cout << e_label_idx[i];
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

#endif  // MATRIXGRAPH_CORE_DATA_STRUCTURES_GAR_GRAPH_ARRAYS_H_
