#ifndef MATRIXGRAPH_CORE_TASK_SUBISO_CUH_
#define MATRIXGRAPH_CORE_TASK_SUBISO_CUH_

#include <string>

#include "core/common/types.h"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/grid_csr_tiled_matrix.cuh"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/unified_buffer.cuh"
#include "core/task/task_base.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

class SubIso : public TaskBase {
private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using GraphID = sics::matrixgraph::core::common::GraphID;
  using TileIndex = sics::matrixgraph::core::common::TileIndex;
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

public:
  SubIso(const std::string &pattern_path, const std::string &data_graph_path,
         const std::string &output_path)
      : pattern_path_(pattern_path), data_graph_path_(data_graph_path),
        output_path_(output_path) {}

  __host__ void Run();

private:
  __host__ void LoadData();

  __host__ void InitLabel();

  __host__ void AllocMappingBuf();

  __host__ void Matching(const ImmutableCSR &p, const GridCSRTiledMatrix &g);

  ImmutableCSR p_;
  GridCSRTiledMatrix *g_;

  VertexLabel *p_vlabel_;
  VertexLabel *g_vlabel_;

  UnifiedOwnedBufferUint32 m_;

  const std::string pattern_path_;
  const std::string data_graph_path_;
  const std::string output_path_;
};

} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_COMPONENTS_SubIso_CUH_