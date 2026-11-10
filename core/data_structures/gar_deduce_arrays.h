#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_GAR_DEDUCE_ARRAYS_H_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_GAR_DEDUCE_ARRAYS_H_

#include <cstdint>
#include <iostream>

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

// Device-friendly X-predicate record for the GPU deduce (chase) task.
// One predicate over pattern positions; attribute columns are referenced by
// their index into GARAttrColumnSet. `kind`:
//   0 = const   node.col <op> cval
//   1 = var     node.col <op> rhs_node.rhs_col
//   2 = edge    edge_label(node, rhs_node) exists in the graph
//               (edge_label is already folded to a base relation id)
//   3 = exists  node.col has a valid value
// `op` (comparison): 0=eq 1=ne 2=lt 3=le 4=gt 5=ge.
struct GARPredicate {
  int32_t kind = 0;
  int32_t node = -1;
  int32_t col = -1;      // index into GARAttrColumnSet (-1 for edge)
  int32_t op = 0;
  double cval = 0.0;     // const right-hand side (kind 0)
  int32_t rhs_node = -1;
  int32_t rhs_col = -1;  // var right-hand column (kind 1)
  int32_t edge_label = -1;  // base relation id (kind 2)
};

// Host views of the vertex attribute columns referenced by a rule's
// predicates. Values are normalized to double by the caller (int64 columns
// converted value-by-value, mirroring the CPU chase comparison semantics).
// A nullptr entry in `valid` means the column is dense (every row valid).
// `col_ids` carries a caller-chosen stable id per column so the task can
// cache device uploads across rules and chase rounds.
struct GARAttrColumnSet {
  static constexpr int kMaxCols = 16;
  int n_cols = 0;
  int32_t col_ids[kMaxCols] = {};
  const double* values[kMaxCols] = {};
  const uint8_t* valid[kMaxCols] = {};
};

// Output of the GPU deduce match for a single rule in a single chase round.
struct GARDeduceArrays {
  // Host scalars written by the task.
  uint64_t* num_embeddings = nullptr;   // topology matches (label-filtered)
  uint64_t* num_x_satisfied = nullptr;  // matches passing X
  int* match_capped = nullptr;          // 1 when the embedding cap truncated

  // Derived head candidates (head_src[i], head_dst[i]) for x-ok embeddings.
  uint32_t* head_src = nullptr;
  uint32_t* head_dst = nullptr;
  int head_capacity = 0;
  int* head_size = nullptr;  // written; may exceed capacity when truncated

  // Optional embedding dump for the matches output: row t is
  // embeddings[t*n_nodes + j], with one x_ok flag per row. Pass
  // embeddings = nullptr to skip the copy-back.
  int32_t* embeddings = nullptr;
  uint8_t* emb_x_ok = nullptr;
  int emb_capacity = 0;  // rows
  int* emb_rows = nullptr;

  void Print() const {
    std::cout << "[GARDeduceArrays] embeddings="
              << (num_embeddings ? *num_embeddings : 0)
              << " x_satisfied="
              << (num_x_satisfied ? *num_x_satisfied : 0)
              << " heads=" << (head_size ? *head_size : 0)
              << " capped=" << (match_capped ? *match_capped : 0)
              << std::endl;
  }
};

}  // namespace data_structures
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_DATA_STRUCTURES_GAR_DEDUCE_ARRAYS_H_
