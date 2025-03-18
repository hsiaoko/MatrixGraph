//
// Created by hsiaoko on 3/18/25.
//

#ifndef MATRIXGRAPH_KERNEL_WCC_CUH
#define MATRIXGRAPH_KERNEL_WCC_CUH
#include <vector>

#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/unified_buffer.cuh"
#include "core/task/kernel/data_structures/exec_plan.cuh"
#include "core/task/kernel/data_structures/woj_exec_plan.cuh"
#include "core/task/kernel/data_structures/woj_matches.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

class WCCKernelWrapper {
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
  WCCKernelWrapper(const WCCKernelWrapper& obj) = delete;

  void operator=(const WCCKernelWrapper&) = delete;

  // @Description: GetInstance() is a method that returns an instance
  // when it is invoked. It returns the same instance if it is invoked more
  // than once as an instance of Singleton class is already created.
  static WCCKernelWrapper* GetInstance();

  static void WCC(
      const cudaStream_t& stream, VertexID n_vertices_g, EdgeIndex n_edges_g,
      const data_structures::UnifiedOwnedBuffer<uint8_t>& data_g,
      const data_structures::UnifiedOwnedBuffer<VertexLabel>& v_label_g);

 private:
  WCCKernelWrapper() = default;

  inline static WCCKernelWrapper* ptr_ = nullptr;
};

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_KERNEL_WCC_CUH
