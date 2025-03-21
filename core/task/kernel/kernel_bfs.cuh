#ifndef MATRIXGRAPH_KERNEL_BFS_CUH
#define MATRIXGRAPH_KERNEL_BFS_CUH

#include <cuda_runtime.h>

#include <vector>

#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/unified_buffer.cuh"
#include "core/task/kernel/data_structures/exec_plan.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

class BFSKernelWrapper {
 private:
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using UnifiedOwnedBufferVertexID =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexID>;
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;
  using Edges = sics::matrixgraph::core::data_structures::Edges;

 public:
  BFSKernelWrapper(const BFSKernelWrapper& obj) = delete;
  void operator=(const BFSKernelWrapper&) = delete;

  static BFSKernelWrapper* GetInstance();

  static void BFS(
      const cudaStream_t& stream, VertexID n_vertices_g, EdgeIndex n_edges_g,
      VertexID source_vertex,
      const data_structures::UnifiedOwnedBuffer<uint8_t>& data_g,
      const data_structures::UnifiedOwnedBuffer<VertexLabel>& v_level_g);

 private:
  BFSKernelWrapper() = default;
  inline static BFSKernelWrapper* ptr_ = nullptr;
};

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_KERNEL_BFS_CUH
