#include <vector>

#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/unified_buffer.cuh"

#ifndef MATRIXGRAPH_CORE_TASK_KERNEL_KERNEL_SUBISO_CUH_
#define MATRIXGRAPH_CORE_TASK_KERNEL_KERNEL_SUBISO_CUH_

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

class SubIsoKernelWrapper {
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using UnifiedOwnedBufferVertexID =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexID>;

public:
  // deleting copy constructor
  SubIsoKernelWrapper(const SubIsoKernelWrapper &obj) = delete;

  void operator=(const SubIsoKernelWrapper &) = delete;

  // @Description: GetInstance() is a method that returns an instance
  // when it is invoked. It returns the same instance if it is invoked more
  // than once as an instance of Singleton class is already created.
  static SubIsoKernelWrapper *GetInstance();

  static void
  SubIso(const cudaStream_t &stream, VertexID depth_p,
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
         const data_structures::UnifiedOwnedBuffer<VertexID> &weft_count_,
         const data_structures::UnifiedOwnedBuffer<EdgeIndex> &weft_offset,
         const data_structures::UnifiedOwnedBuffer<VertexID> &weft_size,
         const data_structures::UnifiedOwnedBuffer<VertexID>
             &v_candidate_offset_for_each_weft,
         const data_structures::UnifiedOwnedBuffer<VertexID> &matches_data);

private:
  SubIsoKernelWrapper() = default;

  inline static SubIsoKernelWrapper *ptr_ = nullptr;
};

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // INC_51_11_GRAPH_COMPUTING_MATRIXGRAPH_CORE_TASK_KERNEL_KERNEL_SUBISO_CUH_