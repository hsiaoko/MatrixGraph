#ifndef MATRIXGRAPH_CORE_TASK_KERNEL_DATA_STRUCTURES_JOIN_PLAN_CUH_
#define MATRIXGRAPH_CORE_TASK_KERNEL_DATA_STRUCTURES_JOIN_PLAN_CUH_

#include "core/common/consts.h"
#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/host_buffer.cuh"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/metadata.h"
#include "core/data_structures/unified_buffer.cuh"
#include "core/util/bitmap.h"
#include <vector>

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

using sics::matrixgraph::core::common::kMaxVertexID;

class WOJExecutionPlan {
 private:
  using Bitmap = sics::matrixgraph::core::util::Bitmap;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using UnifiedOwnedBufferVertexID =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexID>;
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;

 public:
  WOJExecutionPlan() = default;

  __host__ void GenerateWOJExecutionPlan(const ImmutableCSR& p,
                                         const ImmutableCSR& g) {
    exec_path_in_edges_ = new VertexID[p.get_num_outgoing_edges() * 2]();

    EdgeIndex eid = 0;
    for (VertexID v_idx = 0; v_idx < p.get_num_vertices(); v_idx++) {
      auto u = p.GetVertexByLocalID(v_idx);
      for (EdgeIndex v_nbr_idx = 0; v_nbr_idx < u.outdegree; v_nbr_idx++) {
        exec_path_in_edges_[2 * eid] = u.vid;
        exec_path_in_edges_[2 * eid + 1] = u.outgoing_edges[v_nbr_idx];
        eid++;
      }
    }
    n_edges_p_ = p.get_num_outgoing_edges();
    n_edges_g_ = g.get_num_outgoing_edges();
    n_vertices_p_ = p.get_num_vertices();
    n_vertices_g_ = g.get_num_vertices();
  }

  void GenerateJoinPlan(const ExecutionPlan& exec_plan) {}

  inline void SetNDevices(VertexID n_devices) { n_devices_ = n_devices; }

  // Maps logical GPU index 0..n-1 to physical cuda device ids (for multi-GPU).
  // If unset, CudaDeviceId(i) returns i.
  inline void SetCudaDeviceIds(const std::vector<int>& ids) {
    cuda_device_ids_ = ids;
    if (!cuda_device_ids_.empty()) {
      n_devices_ = static_cast<VertexID>(cuda_device_ids_.size());
    }
  }

  inline int CudaDeviceId(int logical_index) const {
    const int n = n_devices_ > 0 ? static_cast<int>(n_devices_) : 1;
    const int lid = ((logical_index % n) + n) % n;
    if (!cuda_device_ids_.empty()) {
      return cuda_device_ids_[static_cast<size_t>(lid)];
    }
    return lid;
  }

  inline VertexID* get_exec_path_in_edges_ptr() const {
    return exec_path_in_edges_;
  }

  inline VertexID get_n_edges_p() const { return n_edges_p_; }

  inline VertexID get_n_vertices_p() const { return n_vertices_p_; }

  inline VertexID get_n_vertices_g() const { return n_vertices_g_; }

  inline VertexID get_n_edges_g() const { return n_edges_g_; }

  inline VertexID get_n_devices() const { return n_devices_; }

 private:
  VertexID* exec_path_in_edges_ = nullptr;

  VertexID n_vertices_p_ = 0;
  VertexID n_edges_p_ = 0;
  VertexID n_vertices_g_ = 0;
  VertexID n_edges_g_ = 0;

  int n_devices_ = 1;

  std::vector<int> cuda_device_ids_;

  std::vector<VertexID> join_key_uid_;
};

}  // namespace data_structures
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_KERNEL_DATA_STRUCTURES_JOIN_PLAN_CUH_