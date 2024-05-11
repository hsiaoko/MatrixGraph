#ifndef MATRIXGRAPH_CORE_COMPONENTS_MATRIXMULTIPLIER_CUH_
#define MATRIXGRAPH_CORE_COMPONENTS_MATRIXMULTIPLIER_CUH_

#include <string>

#include "core/common/types.h"
#include "core/data_structures/tiled_matrix.cuh"
#include "core/task/kernel/gemm.cuh"
#include "core/task/task_base.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

class MatrixMultiplier : public TaskBase {
private:
  using TiledMatrix = sics::matrixgraph::core::data_structures::TiledMatrix;
  using Tile = sics::matrixgraph::core::data_structures::Tile;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using TileIndex = sics::matrixgraph::core::common::TileIndex;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;

public:
  MatrixMultiplier(const std::string &input_path,
                   const std::string &output_path, size_t gemm_count)
      : input_path_(input_path), output_path_(output_path),
        gemm_count_(gemm_count) {

    p_tiled_matrix_ = new TiledMatrix();
    p_tiled_matrix_t_ = new TiledMatrix();
    p_tiled_matrix_->Read(input_path_ + "origin/");
    p_tiled_matrix_t_->Read(input_path_ + "transposed/");

    p_tiled_matrix_->Show();
  }

  __host__ void Run();

  __host__ TiledMatrix *GEMM(const TiledMatrix &tiled_matrix,
                             const TiledMatrix &tiled_matrix_t);

private:
  __host__ cudaError_t TileGEMMWrapper(const cudaStream_t &stream,
                                       const Tile &tile, const Tile &tile_t,
                                       Tile *tile_c);

  size_t ConvertToTaskId(VertexID a, VertexID b, VertexID c, VertexID d,
                         VertexID range) const;

  void ConvertFromTaskId(VertexID k, VertexID *a, VertexID *b, VertexID *c,
                           VertexID *d, VertexID range);

  const std::string input_path_;
  const std::string output_path_;
  const size_t gemm_count_;

  TiledMatrix *p_tiled_matrix_t_;
  TiledMatrix *p_tiled_matrix_;

  std::unordered_map<size_t, Tile *> map_tile_ptr_;
};

} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_COMPONENTS_MATRIXMULTIPLIER_CUH_