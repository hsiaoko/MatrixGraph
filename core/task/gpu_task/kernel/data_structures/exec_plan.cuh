#ifndef MATRIXGRAPH_CORE_TASK_KERNEL_DATA_STRUCTURES_EXEC_PLAN_CUH_
#define MATRIXGRAPH_CORE_TASK_KERNEL_DATA_STRUCTURES_EXEC_PLAN_CUH_

#include "core/common/consts.h"
#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/host_buffer.cuh"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/metadata.h"
#include "core/data_structures/unified_buffer.cuh"
#include "core/util/bitmap_no_ownership.h"
#include "core/util/bitmap_ownership.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

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

  ~ExecutionPlan() {
    delete sequential_exec_path_;
    delete sequential_exec_path_in_edges_;
    delete inverted_index_of_sequential_exec_path_;

    delete[] exec_path_;
    delete[] exec_path_in_edges_;
  };

  void DFSTraverse(VertexID vid, BitmapNoOwnerShip& visited,
                   const ImmutableCSR& g, std::vector<VertexID>& output,
                   std::vector<VertexID>& output_in_edges, VertexID depth,
                   VertexID& max_depth) {
    auto u = g.GetVertexByLocalID(vid);

    auto globalid = g.GetGlobalIDByLocalID(vid);
    output.emplace_back(globalid);

    visited.SetBit(vid);

    max_depth = std::max(depth, max_depth);

    for (VertexID _ = 0; _ < u.outdegree; _++) {
      if (!visited.GetBit(u.outgoing_edges[_])) {
        output_in_edges.emplace_back(globalid);
        output_in_edges.emplace_back(u.outgoing_edges[_]);
        DFSTraverse(u.outgoing_edges[_], visited, g, output, output_in_edges,
                    depth + 1, max_depth);
      }
    }
  }

  __host__ void GenerateDFSExecutionPlan(const ImmutableCSR& p,
                                         const ImmutableCSR& g) {
    n_vertices_ = p.get_num_vertices();

    // BitmapOwnership visited(p.get_max_vid());
    uint64_t* visited_data = new uint64_t[WORD_OFFSET(p.get_max_vid())]();
    BitmapNoOwnerShip visited(p.get_max_vid(), visited_data);

    auto global_vid = p.GetGlobalIDByLocalID(0);
    std::vector<VertexID> output;
    std::vector<VertexID> output_in_edges;
    output.reserve(p.get_max_vid());
    output_in_edges.reserve(p.get_max_vid());

    DFSTraverse(0, visited, p, output, output_in_edges, 0, depth_);

    sequential_exec_path_in_edges_ = new UnifiedOwnedBufferVertexID();
    sequential_exec_path_ = new UnifiedOwnedBufferVertexID();
    inverted_index_of_sequential_exec_path_ = new UnifiedOwnedBufferVertexID();

    sequential_exec_path_->Init(sizeof(VertexID) * p.get_num_vertices());
    sequential_exec_path_in_edges_->Init(sizeof(VertexID) *
                                         p.get_num_vertices() * 2);
    inverted_index_of_sequential_exec_path_->Init(sizeof(VertexID) *
                                                  p.get_num_vertices());

    cudaMemcpy(sequential_exec_path_->GetPtr(), output.data(),
               sizeof(VertexID) * p.get_num_vertices(), cudaMemcpyHostToHost);

    cudaMemcpy(sequential_exec_path_in_edges_->GetPtr() + 2,
               output_in_edges.data(),
               sizeof(VertexID) * output_in_edges.size(), cudaMemcpyHostToHost);

    sequential_exec_path_in_edges_->GetPtr()[0] = kMaxVertexID;
    sequential_exec_path_in_edges_->GetPtr()[1] =
        sequential_exec_path_->GetPtr()[0];

    for (VertexID _ = 0; _ < p.get_num_vertices(); _++) {
      inverted_index_of_sequential_exec_path_
          ->GetPtr()[sequential_exec_path_->GetPtr()[_]] = _;
    }

    exec_path_ = new VertexID[p.get_num_vertices()]();
    exec_path_in_edges_ = new VertexID[output_in_edges.size() + 2]();

    memcpy(exec_path_, output.data(), sizeof(VertexID) * p.get_num_vertices());
    memcpy(exec_path_in_edges_ + 2, output_in_edges.data(),
           sizeof(VertexID) * output_in_edges.size());
    exec_path_in_edges_[0] = kMaxVertexID;
    exec_path_in_edges_[1] = sequential_exec_path_->GetPtr()[0];
    delete[] visited_data;
  }

  VertexID* get_exec_path_ptr() const { return exec_path_; }

  VertexID* get_exec_path_in_edges_ptr() const { return exec_path_in_edges_; }

  UnifiedOwnedBufferVertexID* get_sequential_exec_path_ptr() const {
    return sequential_exec_path_;
  }

  UnifiedOwnedBufferVertexID* get_sequential_exec_path_in_edges_ptr() const {
    return sequential_exec_path_in_edges_;
  }

  UnifiedOwnedBufferVertexID* get_inverted_index_of_sequential_exec_path_ptr()
      const {
    return sequential_exec_path_in_edges_;
  }

  inline VertexID get_depth() const { return depth_; }

  inline VertexID get_n_vertices() const { return n_vertices_; }

 public:
  UnifiedOwnedBufferVertexID* sequential_exec_path_ = nullptr;
  UnifiedOwnedBufferVertexID* sequential_exec_path_in_edges_ = nullptr;
  UnifiedOwnedBufferVertexID* inverted_index_of_sequential_exec_path_ = nullptr;

  VertexID* exec_path_ = nullptr;
  VertexID* exec_path_in_edges_ = nullptr;

  VertexID n_vertices_ = 0;
  VertexID depth_ = 0;
};  // namespace kernel

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_KERNEL_DATA_STRUCTURES_EXEC_PLAN_CUH_