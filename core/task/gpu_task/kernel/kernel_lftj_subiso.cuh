#ifndef MATRIXGRAPH_CORE_TASK_GPU_KERNEL_LFTJ_SUBISO_CUH_
#define MATRIXGRAPH_CORE_TASK_GPU_KERNEL_LFTJ_SUBISO_CUH_

#include <stdint.h>
#include <vector>

#include "core/common/types.h"
#include "core/data_structures/immutable_csr.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

class LFTJSubIsoKernelWrapper {
 public:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;

  // Single-GPU count-only LFTJ enumeration.
  // `num_threads` controls total CUDA threads launched (0 = default).
  static uint64_t Enumerate(
      const ImmutableCSR& pattern, const ImmutableCSR& data_graph,
      const std::vector<EdgeIndex>& data_offsets,
      const std::vector<VertexID>& data_neighbors,
      const std::vector<std::vector<VertexID>>& pattern_adj,
      const std::vector<std::vector<VertexID>>& candidates,
      const std::vector<VertexID>& order,
      const std::vector<std::vector<VertexID>>& bn_list,
      bool canonical, int num_threads);
};

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_GPU_KERNEL_LFTJ_SUBISO_CUH_
