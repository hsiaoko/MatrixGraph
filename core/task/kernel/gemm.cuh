#ifndef MATRIXGRAPH_CORE_TASK_KERNEL_GEMM_CUH_
#define MATRIXGRAPH_CORE_TASK_KERNEL_GEMM_CUH_

#include "core/common/types.h"
#include "core/data_structures/tiled_matrix.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

using TiledMatrix = sics::matrixgraph::core::data_structures::TiledMatrix;
using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
using VertexIndex = sics::matrixgraph::core::common::VertexIndex;
using VertexID = sics::matrixgraph::core::common::VertexID;
using TileIndex = sics::matrixgraph::core::common::TileIndex;
using Mask = sics::matrixgraph::core::data_structures::Mask;
using Tile = sics::matrixgraph::core::data_structures::Tile;

struct TileGEMMParams {
  VertexID n_nz_A;
  VertexID n_nz_B;
  TileIndex tile_size_x;
  TileIndex tile_size_y;
  int *offset;
  int *n_nz_for_each_row;
  TileIndex *bar_offset_A;
  TileIndex *bar_offset_B;
  TileIndex *bar_offset_C;
  TileIndex *row_idx_A;
  TileIndex *row_idx_B;
  TileIndex *row_idx_C;
  TileIndex *col_idx_A;
  TileIndex *col_idx_B;
  TileIndex *col_idx_C;
  VertexLabel *data_A;
  VertexLabel *data_B;
  VertexLabel *data_C;
  uint64_t *bit_mask_A;
  uint64_t *bit_mask_B;
  uint64_t *bit_mask_C;
};

__global__ void TileGEMM_kernel(TileGEMMParams params);

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_TASK_KERNEL_GEMM_CUH_