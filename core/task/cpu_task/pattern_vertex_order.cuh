#ifndef MATRIXGRAPH_CORE_TASK_CPU_PATTERN_VERTEX_ORDER_CUH_
#define MATRIXGRAPH_CORE_TASK_CPU_PATTERN_VERTEX_ORDER_CUH_

#include <algorithm>
#include <array>
#include <cstring>
#include <numeric>
#include <vector>

#include "core/common/consts.h"
#include "core/common/types.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/matrix.cuh"
#include "core/task/gpu_task/kernel/algorithms/hash.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using sics::matrixgraph::core::common::kMaxVertexID;

class PatternVertexOrder {
 private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;
  using Matrix = sics::matrixgraph::core::data_structures::Matrix;

 public:
  // Compute a permutation of pattern local ids such that the vertex with the
  // strongest one-min filtering power is placed at new local id 0, the next
  // strongest at id 1, and so on.
  // Returns new_to_old[new_local_id] = old_local_id.
  static std::vector<VertexID> ComputeOrder(const ImmutableCSR& pattern,
                                            const ImmutableCSR& data_graph) {
    // 1. Bucket data vertices by their one-min hash.
    std::array<uint32_t, 16> buckets{};
    for (VertexID v = 0; v < data_graph.get_num_vertices(); ++v) {
      VertexID h = ComputeOneMinHash(v, data_graph);
      if (h < 16) buckets[h]++;
    }

    // 2. Prefix sum: prefix[h] = number of data vertices with hash <= h.
    std::array<uint32_t, 16> prefix{};
    uint32_t acc = 0;
    for (int i = 0; i < 16; ++i) {
      acc += buckets[i];
      prefix[i] = acc;
    }

    // 3. Score pattern vertices.
    const VertexID n = pattern.get_num_vertices();
    std::vector<uint32_t> score(n, data_graph.get_num_vertices());
    for (VertexID u = 0; u < n; ++u) {
      VertexID h = ComputeOneMinHash(u, pattern);
      if (h < 16) score[u] = prefix[h];
    }

    // 4. Sort by score ascending (lower score => stronger pruning).
    std::vector<VertexID> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&pattern, &score](VertexID a, VertexID b) {
                       if (score[a] != score[b]) return score[a] < score[b];
                       // Tie-break: higher degree first, then lower label,
                       // then original id for determinism.
                       VertexID deg_a = pattern.GetOutDegreeByLocalID(a) +
                                        pattern.GetInDegreeByLocalID(a);
                       VertexID deg_b = pattern.GetOutDegreeByLocalID(b) +
                                        pattern.GetInDegreeByLocalID(b);
                       if (deg_a != deg_b) return deg_a > deg_b;
                       VertexLabel lbl_a = pattern.GetVLabelBasePointer()[a];
                       VertexLabel lbl_b = pattern.GetVLabelBasePointer()[b];
                       if (lbl_a != lbl_b) return lbl_a < lbl_b;
                       return a < b;
                     });
    return order;
  }

  // Reorder a pattern CSR in-place. new_to_old[new_local_id] = old_local_id.
  static void Reorder(ImmutableCSR* pattern,
                      const std::vector<VertexID>& new_to_old) {
    const VertexID n = pattern->get_num_vertices();
    if (n == 0) return;
    if (new_to_old.size() != n) return;

    std::vector<VertexID> old_to_new(n);
    for (VertexID i = 0; i < n; ++i) {
      old_to_new[new_to_old[i]] = i;
    }

    const EdgeIndex num_in = pattern->get_num_incoming_edges();
    const EdgeIndex num_out = pattern->get_num_outgoing_edges();
    const VertexID max_vid = pattern->get_max_vid();

    // Temporary buffers.
    std::vector<VertexID> new_globalid(n);
    std::vector<VertexID> new_indegree(n);
    std::vector<VertexID> new_outdegree(n);
    std::vector<EdgeIndex> new_in_offset(n + 1);
    std::vector<EdgeIndex> new_out_offset(n + 1);
    std::vector<VertexID> new_in_edges(num_in);
    std::vector<VertexID> new_out_edges(num_out);
    std::vector<VertexID> new_edges_globalid(max_vid + 1, kMaxVertexID);
    std::vector<VertexID> new_localid(max_vid + 1, kMaxVertexID);
    std::vector<VertexLabel> new_label(n);

    const VertexID* old_globalid = pattern->GetGloablIDBasePointer();
    const VertexID* old_indegree = pattern->GetInDegreeBasePointer();
    const VertexID* old_outdegree = pattern->GetOutDegreeBasePointer();
    const EdgeIndex* old_in_offset = pattern->GetInOffsetBasePointer();
    const EdgeIndex* old_out_offset = pattern->GetOutOffsetBasePointer();
    const VertexID* old_in_edges = pattern->GetIncomingEdgesBasePointer();
    const VertexID* old_out_edges = pattern->GetOutgoingEdgesBasePointer();
    const VertexLabel* old_label = pattern->GetVLabelBasePointer();

    // Compute degrees and offsets for the new ordering.
    new_in_offset[0] = 0;
    new_out_offset[0] = 0;
    for (VertexID i = 0; i < n; ++i) {
      VertexID old = new_to_old[i];
      new_globalid[i] = old_globalid[old];
      new_label[i] = old_label[old];
      new_indegree[i] = old_indegree[old];
      new_outdegree[i] = old_outdegree[old];
      new_in_offset[i + 1] = new_in_offset[i] + new_indegree[i];
      new_out_offset[i + 1] = new_out_offset[i] + new_outdegree[i];
      new_edges_globalid[i] = new_globalid[i];
    }

    // Remap edges using the old_to_new local id mapping.
    for (VertexID i = 0; i < n; ++i) {
      VertexID old = new_to_old[i];

      // Outgoing edges.
      EdgeIndex out_start = new_out_offset[i];
      EdgeIndex old_out_start = old_out_offset[old];
      for (EdgeIndex j = 0; j < new_outdegree[i]; ++j) {
        VertexID old_nbr = old_out_edges[old_out_start + j];
        new_out_edges[out_start + j] = old_to_new[old_nbr];
      }

      // Incoming edges.
      EdgeIndex in_start = new_in_offset[i];
      EdgeIndex old_in_start = old_in_offset[old];
      for (EdgeIndex j = 0; j < new_indegree[i]; ++j) {
        VertexID old_nbr = old_in_edges[old_in_start + j];
        new_in_edges[in_start + j] = old_to_new[old_nbr];
      }
    }

    // Rebuild global id <-> local id lookups.
    for (VertexID i = 0; i < n; ++i) {
      new_localid[new_globalid[i]] = i;
    }

    // Write back into the existing pattern buffers.
    std::memcpy(pattern->GetGloablIDBasePointer(), new_globalid.data(),
                sizeof(VertexID) * n);
    std::memcpy(pattern->GetInDegreeBasePointer(), new_indegree.data(),
                sizeof(VertexID) * n);
    std::memcpy(pattern->GetOutDegreeBasePointer(), new_outdegree.data(),
                sizeof(VertexID) * n);
    std::memcpy(pattern->GetInOffsetBasePointer(), new_in_offset.data(),
                sizeof(EdgeIndex) * (n + 1));
    std::memcpy(pattern->GetOutOffsetBasePointer(), new_out_offset.data(),
                sizeof(EdgeIndex) * (n + 1));
    std::memcpy(pattern->GetIncomingEdgesBasePointer(), new_in_edges.data(),
                sizeof(VertexID) * num_in);
    std::memcpy(pattern->GetOutgoingEdgesBasePointer(), new_out_edges.data(),
                sizeof(VertexID) * num_out);
    std::memcpy(pattern->GetEdgesGloablIDBasePointer(), new_edges_globalid.data(),
                sizeof(VertexID) * (max_vid + 1));
    std::memcpy(pattern->GetLocalIDBasePointer(), new_localid.data(),
                sizeof(VertexID) * (max_vid + 1));
    std::memcpy(pattern->GetVLabelBasePointer(), new_label.data(),
                sizeof(VertexLabel) * n);
  }

  // Convenience method: compute order and apply it.
  static void Apply(ImmutableCSR* pattern, const ImmutableCSR& data_graph) {
    auto order = ComputeOrder(*pattern, data_graph);
    Reorder(pattern, order);
  }

  // Permute the rows of a pattern embedding matrix to match a pattern reorder.
  // new_to_old[new_local_id] = old_local_id.
  static void PermuteMatrixRows(Matrix* matrix,
                                const std::vector<VertexID>& new_to_old) {
    if (matrix == nullptr || matrix->GetPtr() == nullptr) return;
    const uint32_t x = matrix->get_x();
    const uint32_t y = matrix->get_y();
    if (x != new_to_old.size()) return;

    const float* old_data = matrix->GetPtr();
    std::vector<float> new_data(static_cast<size_t>(x) * y);
    for (uint32_t i = 0; i < x; ++i) {
      VertexID old_row = new_to_old[i];
      std::memcpy(new_data.data() + static_cast<size_t>(i) * y,
                  old_data + static_cast<size_t>(old_row) * y,
                  sizeof(float) * y);
    }
    std::memcpy(matrix->GetPtr(), new_data.data(),
                sizeof(float) * x * y);
  }

 private:
  // One-min hash: minimum HashTable(label) over all outgoing and incoming
  // neighbors.  This matches the hash used by the one-min filter.
  static VertexID ComputeOneMinHash(VertexID vid, const ImmutableCSR& g) {
    VertexID min_hash = kMaxVertexID;
    auto labels = g.GetVLabelBasePointer();

    auto out_degree = g.GetOutDegreeByLocalID(vid);
    auto out_edges = g.GetOutgoingEdgesByLocalID(vid);
    for (VertexID i = 0; i < out_degree; ++i) {
      VertexID h = sics::matrixgraph::core::task::kernel::HashTable(
          labels[out_edges[i]]);
      if (h < min_hash) min_hash = h;
    }

    auto in_degree = g.GetInDegreeByLocalID(vid);
    auto in_edges = g.GetIncomingEdgesByLocalID(vid);
    for (VertexID i = 0; i < in_degree; ++i) {
      VertexID h = sics::matrixgraph::core::task::kernel::HashTable(
          labels[in_edges[i]]);
      if (h < min_hash) min_hash = h;
    }
    return min_hash;
  }
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_CPU_PATTERN_VERTEX_ORDER_CUH_
