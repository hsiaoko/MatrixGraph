#ifndef MATRIXGRAPH_CORE_TASK_WCC_CUH_
#define MATRIXGRAPH_CORE_TASK_WCC_CUH_

#include <string>

#include "core/common/types.h"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/grid_csr_tiled_matrix.cuh"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/unified_buffer.cuh"
#include "core/task/kernel/data_structures/exec_plan.cuh"
#include "core/task/kernel/data_structures/woj_exec_plan.cuh"
#include "core/task/task_base.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

class WCC : public TaskBase {
 private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using GraphID = sics::matrixgraph::core::common::GraphID;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using GridCSRTiledMatrix =
      sics::matrixgraph::core::data_structures::GridCSRTiledMatrix;
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;
  using Edges = sics::matrixgraph::core::data_structures::Edges;
  using GridGraphMetadata =
      sics::matrixgraph::core::data_structures::GridGraphMetadata;
  using UnifiedOwnedBufferUint32 =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint32_t>;
  using UnifiedOwnedBufferUint64 =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint64_t>;
  using UnifiedOwnedBufferVertexID =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexID>;

 public:
  WCC(const std::string& data_graph_path) : data_graph_path_(data_graph_path) {}

  __host__ void Run();

 private:
  __host__ void LoadData();
  __host__ void HashMin(const ImmutableCSR& g);

  ImmutableCSR g_;
  Edges e_;
  const std::string data_graph_path_;
  VertexLabel* label_p_ = nullptr;
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_COMPONENTS_WCC_CUH_