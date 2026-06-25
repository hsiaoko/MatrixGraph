#ifndef MATRIXGRAPH_CORE_TASK_KERNEL_DATA_STRUCTURES_EXEC_PLAN_CUH_
#define MATRIXGRAPH_CORE_TASK_KERNEL_DATA_STRUCTURES_EXEC_PLAN_CUH_

#include <algorithm>
#include <numeric>
#include <queue>
#include <vector>

#include "core/common/consts.h"
#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/host_buffer.cuh"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/metadata.h"
#include "core/data_structures/unified_buffer.cuh"
#include "core/util/bitmap_no_ownership.h"
#include "core/util/bitmap_ownership.h"
#include "core/task/cpu_task/pattern_vertex_order.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

using sics::matrixgraph::core::common::kMaxVertexID;

class ExecutionPlan {
 private:
  using BitmapOwnership = sics::matrixgraph::core::util::BitmapOwnership;
  using BitmapNoOwnerShip = sics::matrixgraph::core::util::BitmapNoOwnerShip;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using UnifiedOwnedBufferVertexID =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexID>;
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;

 public:
  ExecutionPlan() = default;

  void SetUseCostModelOrder(bool use) { use_cost_model_order_ = use; }

  ~ExecutionPlan() {
    delete sequential_exec_path_;
    delete sequential_exec_path_in_edges_;
    delete inverted_index_of_sequential_exec_path_;

    delete[] exec_path_;
    delete[] exec_path_in_edges_;
  };

  void DFSTraverse(VertexID vid, BitmapNoOwnerShip& visited_src,
                   const ImmutableCSR& g, std::vector<VertexID>& output,
                   std::vector<VertexID>& output_in_edges, VertexID depth,
                   VertexID& max_depth,
                   const std::vector<VertexID>& rank_by_local_id) {
    max_depth = std::max(depth, max_depth);
    if (visited_src.GetBit(vid)) {
      return;
    }
    visited_src.SetBit(vid);

    auto u = g.GetVertexByLocalID(vid);
    auto globalid = g.GetGlobalIDByLocalID(vid);
    output.emplace_back(globalid);

    // Visit outgoing neighbors in order of pruning power (best first).
    std::vector<VertexID> neighbors(u.outgoing_edges,
                                    u.outgoing_edges + u.outdegree);
    std::stable_sort(neighbors.begin(), neighbors.end(),
                     [&rank_by_local_id](VertexID a, VertexID b) {
                       return rank_by_local_id[a] < rank_by_local_id[b];
                     });

    for (VertexID neighbor : neighbors) {
      auto neighbor_global = g.GetGlobalIDByLocalID(neighbor);

      output_in_edges.emplace_back(globalid);
      output_in_edges.emplace_back(neighbor_global);

      DFSTraverse(neighbor, visited_src, g, output, output_in_edges, depth + 1,
                  max_depth, rank_by_local_id);
    }
  }

  __host__ void GenerateDFSExecutionPlan(const ImmutableCSR& p,
                                         const ImmutableCSR& g) {
    n_vertices_ = p.get_num_vertices();

    // Compute pruning-power order using the one-min filter cost model.
    auto order = sics::matrixgraph::core::task::PatternVertexOrder::ComputeOrder(
        p, g);
    std::vector<VertexID> rank_by_local_id(n_vertices_);
    for (VertexID i = 0; i < n_vertices_; ++i) {
      rank_by_local_id[order[i]] = i;
    }

    // Pick the strongest-filtering vertex that can reach all vertices via
    // outgoing edges as the single root.  This preserves the matching
    // algorithm's assumption of one root edge.
    auto reaches_all = [&p](VertexID root) {
      std::vector<bool> visited(p.get_num_vertices(), false);
      std::vector<VertexID> stack;
      stack.push_back(root);
      visited[root] = true;
      size_t count = 0;
      while (!stack.empty()) {
        VertexID u = stack.back();
        stack.pop_back();
        ++count;
        auto out_degree = p.GetOutDegreeByLocalID(u);
        auto out_edges = p.GetOutgoingEdgesByLocalID(u);
        for (VertexID i = 0; i < out_degree; ++i) {
          VertexID v = out_edges[i];
          if (!visited[v]) {
            visited[v] = true;
            stack.push_back(v);
          }
        }
      }
      return count == p.get_num_vertices();
    };

    VertexID root = kMaxVertexID;
    for (VertexID candidate : order) {
      if (reaches_all(candidate)) {
        root = candidate;
        break;
      }
    }
    if (root == kMaxVertexID) {
      // Fall back to original local id 0 if no vertex reaches everyone.
      root = 0;
    }

    if (!use_cost_model_order_) {
      // Default local-id order: no cost-model sorting, root at local id 0.
      std::iota(order.begin(), order.end(), 0);
      std::iota(rank_by_local_id.begin(), rank_by_local_id.end(), 0);
      root = 0;
    }

    uint64_t* visited_data = new uint64_t[WORD_OFFSET(p.get_max_vid())]();
    BitmapNoOwnerShip visited(p.get_max_vid(), visited_data);
    visited.Clear();

    std::vector<VertexID> output;
    std::vector<VertexID> output_in_edges;
    output.reserve(p.get_max_vid());
    output_in_edges.reserve(p.get_max_vid());

    // Start from the chosen best root, then continue with any remaining
    // unvisited vertices in pruning order (mirrors original multi-root DFS
    // while still prioritizing high-pruning vertices).
    DFSTraverse(root, visited, p, output, output_in_edges, 0, depth_,
                rank_by_local_id);
    for (VertexID candidate : order) {
      if (!visited.GetBit(candidate)) {
        DFSTraverse(candidate, visited, p, output, output_in_edges, 0, depth_,
                    rank_by_local_id);
      }
    }

    sequential_exec_path_in_edges_ = new UnifiedOwnedBufferVertexID();
    sequential_exec_path_ = new UnifiedOwnedBufferVertexID();
    inverted_index_of_sequential_exec_path_ = new UnifiedOwnedBufferVertexID();

    sequential_exec_path_->Init(sizeof(VertexID) * p.get_num_vertices());
    n_edges_ = output_in_edges.size() / 2 + 1;

    sequential_exec_path_in_edges_->Init(sizeof(VertexID) * 2 * n_edges_);
    inverted_index_of_sequential_exec_path_->Init(sizeof(VertexID) * n_edges_ *
                                                  2);

    sequential_exec_path_in_edges_->GetPtr()[0] = kMaxVertexID;
    sequential_exec_path_in_edges_->GetPtr()[1] =
        p.GetGlobalIDByLocalID(root);
    cudaMemcpy(sequential_exec_path_->GetPtr(), output.data(),
               sizeof(VertexID) * output.size(), cudaMemcpyHostToHost);
    cudaMemcpy(sequential_exec_path_in_edges_->GetPtr() + 2,
               output_in_edges.data(),
               sizeof(VertexID) * output_in_edges.size(), cudaMemcpyHostToHost);

    for (VertexID _ = 0; _ < p.get_num_vertices(); _++) {
      inverted_index_of_sequential_exec_path_
          ->GetPtr()[sequential_exec_path_->GetPtr()[_]] = _;
    }

    exec_path_ = new VertexID[p.get_num_vertices()]();
    exec_path_in_edges_ = new VertexID[n_edges_ * 2]();

    exec_path_in_edges_[0] = kMaxVertexID;
    exec_path_in_edges_[1] = p.GetGlobalIDByLocalID(root);

    memcpy(exec_path_, output.data(), sizeof(VertexID) * p.get_num_vertices());
    memcpy(exec_path_in_edges_ + 2, output_in_edges.data(),
           sizeof(VertexID) * output_in_edges.size());
    delete[] visited_data;
  }

  VertexID* get_exec_path_ptr() const { return exec_path_; }

  VertexID* get_exec_path_in_edges_ptr() { return exec_path_in_edges_; }

  VertexID get_exec_path_in_edges_val(VertexID idx) const {
    return exec_path_in_edges_[idx];
  }

  UnifiedOwnedBufferVertexID* get_sequential_exec_path_ptr() const {
    return sequential_exec_path_;
  }

  UnifiedOwnedBufferVertexID* get_sequential_exec_path_in_edges_ptr() const {
    return sequential_exec_path_in_edges_;
  }

  UnifiedOwnedBufferVertexID* get_inverted_index_of_sequential_exec_path_ptr()
      const {
    return inverted_index_of_sequential_exec_path_;
  }

  inline VertexID get_depth() const { return depth_; }

  inline VertexID get_n_vertices() const { return n_vertices_; }

  inline VertexID get_n_edges() const { return n_edges_; }

  void Print() const {
    std::cout << "Print ExecPlan - n_vertices: " << n_vertices_
              << " n_edges_: " << n_edges_ << " depth_ " << get_depth()
              << std::endl;
    std::cout << "\t sequential_exec_path" << std::endl;
    std::cout << "\t";
    for (int i = 0; i < n_vertices_; i++) {
      auto ptr = get_sequential_exec_path_ptr()->GetPtr();
      std::cout << ptr[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "\t sequential_exec_path_in_edges:" << std::endl;
    // auto ptr = get_sequential_exec_path_in_edges_ptr()->GetPtr();
    auto ptr = exec_path_in_edges_;
    for (int i = 0; i < n_edges_; i++) {
      std::cout << "\t* " << ptr[i * 2] << "->" << ptr[i * 2 + 1] << std::endl;
    }
  }

  bool IsInExecPathInEdges(VertexID src, VertexID dst) const {
    auto ptr = get_sequential_exec_path_in_edges_ptr()->GetPtr();
    for (int i = 0; i < n_vertices_; i++) {
      if (ptr[i * 2] == src && ptr[i * 2 + 1] == dst) {
        return true;
      }
    }
    return false;
  }

 public:
  bool use_cost_model_order_ = true;

  UnifiedOwnedBufferVertexID* sequential_exec_path_ = nullptr;
  UnifiedOwnedBufferVertexID* sequential_exec_path_in_edges_ = nullptr;
  UnifiedOwnedBufferVertexID* inverted_index_of_sequential_exec_path_ = nullptr;

  VertexID* exec_path_ = nullptr;
  VertexID* exec_path_in_edges_ = nullptr;

  VertexID n_vertices_ = 0;
  VertexID depth_ = 0;
  VertexID n_edges_ = 0;
};

}  // namespace data_structures
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_KERNEL_DATA_STRUCTURES_EXEC_PLAN_CUH_