#ifndef MATRIXGRAPH_CORE_COMPONENTS_PPR_QUERY_CUH_
#define MATRIXGRAPH_CORE_COMPONENTS_PPR_QUERY_CUH_

#include <string>

#include "core/common/types.h"
#include "core/data_structures/grid_tiled_matrix.cuh"
#include "core/task/kernel/gemm.cuh"
#include "core/task/task_base.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

class PPRQuery : public TaskBase {
private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using TileIndex = sics::matrixgraph::core::common::TileIndex;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using GridTiledMatrix =
      sics::matrixgraph::core::data_structures::GridTiledMatrix;

public:
  PPRQuery(const std::string &input_path,
           const std::string &input_path_transposed,

           const std::string &output_path, size_t count)
      : input_path_(input_path), input_path_transposed_(input_path_transposed),
        output_path_(output_path), count_(count) {

    LoadData();
  }

  __host__ void Run();

private:
  __host__ void LoadData();

  __host__ void InitC();

  __host__ void FillTiles();

  GridTiledMatrix *A_;
  GridTiledMatrix *B_;

  GridTiledMatrix *C_;

  const std::string input_path_;
  const std::string input_path_transposed_;

  const std::string output_path_;
  const size_t count_;
};

} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_COMPONENTS_MATRIXMULTIPLIER_CUH_