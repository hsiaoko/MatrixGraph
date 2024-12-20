#ifndef MATRIXGRAPH_CORE_TASK_KERNEL_KERNEL_WOJ_SUBISO_CUH_
#define MATRIXGRAPH_CORE_TASK_KERNEL_KERNEL_WOJ_SUBISO_CUH_

#include <vector>

#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/unified_buffer.cuh"
#include "core/task/kernel/data_structures/exec_plan.cuh"
#include "core/task/kernel/data_structures/hash_buckets.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

class WOJSubIsoKernelWrapper {
private:
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using UnifiedOwnedBufferVertexID =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexID>;
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;
  using Edges = sics::matrixgraph::core::data_structures::Edges;

public:
  // deleting copy constructor
  WOJSubIsoKernelWrapper(const WOJSubIsoKernelWrapper &obj) = delete;

  void operator=(const WOJSubIsoKernelWrapper &) = delete;

  // @Description: GetInstance() is a method that returns an instance
  // when it is invoked. It returns the same instance if it is invoked more
  // than once as an instance of Singleton class is already created.
  static WOJSubIsoKernelWrapper *GetInstance();

  static void
  Filter(const cudaStream_t &stream, VertexID u_idx, VertexID depth_p,
         const UnifiedOwnedBufferVertexID &exec_path,
         const data_structures::UnifiedOwnedBuffer<VertexID>
             &inverted_index_of_exec_path,
         const UnifiedOwnedBufferVertexID &exec_path_in_edges,
         VertexID n_vertices_p, EdgeIndex n_edges_p,
         const data_structures::UnifiedOwnedBuffer<uint8_t> &data_p,
         const data_structures::UnifiedOwnedBuffer<VertexLabel> &v_label_p,
         VertexID n_vertices_g, EdgeIndex n_edges_g,
         const data_structures::UnifiedOwnedBuffer<uint8_t> &data_g,
         const data_structures::UnifiedOwnedBuffer<VertexID> &edgelist_g,
         const data_structures::UnifiedOwnedBuffer<VertexLabel> &v_label_g,
         const HashBuckets &hash_buckets);

  static void Filter(const ImmutableCSR &p, const ImmutableCSR &g,
                     const Edges &e, const ExecutionPlan &exec_plan);

private:
  WOJSubIsoKernelWrapper() = default;

  inline static WOJSubIsoKernelWrapper *ptr_ = nullptr;
};

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // INC_51_11_GRAPH_COMPUTING_MATRIXGRAPH_CORE_TASK_KERNEL_KERNEL_SUBISO_CUH_