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

public:
  // deleting copy constructor
  SubIsoKernelWrapper(const SubIsoKernelWrapper &obj) = delete;

  void operator=(const SubIsoKernelWrapper &) = delete;

  // @Description: GetInstance() is a method that returns an instance
  // when it is invoked. It returns the same instance if it is invoked more
  // than once as an instance of Singleton class is already created.
  static SubIsoKernelWrapper *GetInstance();

  static void
  SubIso(const cudaStream_t &stream,
         VertexID n_vertices_p,
         EdgeIndex n_edges_p,

         const data_structures::UnifiedOwnedBuffer<uint8_t> &data_p,
         const data_structures::UnifiedOwnedBuffer<VertexLabel> &v_label_p,
         size_t tile_size, size_t n_strips, size_t n_nz_tile_g,
         const data_structures::UnifiedOwnedBuffer<VertexID>
             &n_vertices_for_each_csr_g,
         const data_structures::UnifiedOwnedBuffer<EdgeIndex>
             &n_edges_for_each_csr_g,
         const data_structures::UnifiedOwnedBuffer<VertexID> &tile_offset_row_g,
         const data_structures::UnifiedOwnedBuffer<VertexID> &tile_row_idx_g,
         const data_structures::UnifiedOwnedBuffer<VertexID> &tile_col_idx_g,
         const data_structures::UnifiedOwnedBuffer<uint64_t> &csr_offset_g,
         const data_structures::UnifiedOwnedBuffer<uint8_t> &data_g,
         const std::vector<data_structures::UnifiedOwnedBuffer<VertexID>> &m,
         const std::vector<data_structures::UnifiedOwnedBuffer<EdgeIndex>>
             &m_offset);

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