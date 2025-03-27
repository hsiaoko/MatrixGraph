#ifndef MATRIXGRAPH_CORE_TASK_GEMM_CUH_
#define MATRIXGRAPH_CORE_TASK_GEMM_CUH_

#include <string>

#include "core/common/types.h"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/grid_csr_tiled_matrix.cuh"
#include "core/task/gpu_task/task_base.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

class GEMM : public TaskBase {
 private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using GraphID = sics::matrixgraph::core::common::GraphID;
  using TileIndex = sics::matrixgraph::core::common::TileIndex;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using GridCSRTiledMatrix =
      sics::matrixgraph::core::data_structures::GridCSRTiledMatrix;
  using Edges = sics::matrixgraph::core::data_structures::Edges;
  using GridGraphMetadata =
      sics::matrixgraph::core::data_structures::GridGraphMetadata;

 public:
  GEMM(const std::string& input_path, const std::string& input_path_transposed,

       const std::string& output_path, size_t count)
      : input_path_(input_path),
        input_path_transposed_(input_path_transposed),
        output_path_(output_path),
        count_(count) {
    LoadData();
  }

  __host__ void Run();

 private:
  __host__ void LoadData();

  __host__ void SizePred();

  __host__ void InitResultMatrix();

  __host__ void InitC();

  __host__ void InitResultMatrixUnifiedMemory();

  __host__ Edges* Walks(const GridCSRTiledMatrix& A,
                        const GridCSRTiledMatrix& B, VertexID tile_size,
                        VertexID n_strips);

  __host__ std::vector<Edges>* GridPartitioning(const Edges& edges_blocks,
                                                GraphID n_partitions);

  __host__ GridCSRTiledMatrix* ConvertGridEdgelist2GridTiledMatrix(
      const std::vector<Edges>& edges_blocks,
      const GridGraphMetadata& grid_graph_metadata, VertexID tile_size);

  __host__ void FillTiles();

  __host__ void Count(const GridCSRTiledMatrix& G);

  GridCSRTiledMatrix* A_;
  GridCSRTiledMatrix* B_;
  GridCSRTiledMatrix* C_;

  const std::string input_path_;
  const std::string input_path_transposed_;

  const std::string output_path_;
  const size_t count_;
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_COMPONENTS_GEMM_CUH_