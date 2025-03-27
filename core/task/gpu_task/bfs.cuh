#ifndef MATRIXGRAPH_BFS_CUH
#define MATRIXGRAPH_BFS_CUH

#include <string>

#include "core/common/types.h"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/grid_csr_tiled_matrix.cuh"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/unified_buffer.cuh"
#include "core/task/gpu_task/task_base.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

class BFS : public TaskBase {
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
  BFS(const std::string& data_graph_path, VertexID src)
      : data_graph_path_(data_graph_path), src_(src) {}

  __host__ void Run();

 private:
  __host__ void LoadData();
  __host__ void ExecuteBFS(const ImmutableCSR& g, VertexID src = 0);

  ImmutableCSR g_;
  Edges e_;
  VertexID src_ = 0;

  const std::string data_graph_path_;
  VertexLabel* label_p_ = nullptr;
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_BFS_CUH
