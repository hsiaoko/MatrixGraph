#include <algorithm>
#include <cmath>

#include "core/common/types.h"
#include "core/task/matrix_multiplier.cuh"
#include "core/util/cuda_check.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using TiledMatrix = sics::matrixgraph::core::data_structures::TiledMatrix;
using VertexID = sics::matrixgraph::core::common::VertexID;
using Mask = sics::matrixgraph::core::data_structures::Mask;
using Tile = sics::matrixgraph::core::data_structures::Tile;

__host__ TiledMatrix *
MatrixMultiplier::GEMM(const TiledMatrix &tiled_matrix,
                       const TiledMatrix &tiled_matrix_t) {

  std::cout << "TiledMatrix GEMM" << std::endl;

  auto max_val =
      tiled_matrix.get_metadata().n_cols > tiled_matrix_t.get_metadata().n_rows
          ? tiled_matrix.get_metadata().n_cols
          : tiled_matrix_t.get_metadata().n_rows;

  auto C = new TiledMatrix;

  VertexID n_rows_c = tiled_matrix.get_metadata().n_rows,
           n_cols_c = tiled_matrix_t.get_metadata().n_rows, n_nz_tile_c = 0;

  VertexID *tile_ptr_c, *tile_row_idx_c, *tile_col_idx_c, *tile_n_nz_c;
  auto tile_size = tiled_matrix.get_metadata().n_nz_tile;

  auto mask_c = new Mask(tile_size * tile_size);
  std::vector<Tile *> data_ptr_vec;

  for (VertexID i = 0; i < tiled_matrix.get_metadata().n_rows; i++) {
    // Create a new stream for each task.
    auto bin_id = 0;
    cudaSetDevice(bin_id);
    cudaStream_t *p_stream = new cudaStream_t;
    cudaStreamCreate(p_stream);

    auto n_nz_tile_A = tiled_matrix.get_tile_ptr_ptr()[i + 1] -
                       tiled_matrix.get_tile_ptr_ptr()[i];

    auto n_nz_tile_A_t = tiled_matrix.get_tile_ptr_ptr()[i + 1] -
                         tiled_matrix.get_tile_ptr_ptr()[i];

    if (n_nz_tile_A == 0)
      continue;

    // TODO (hsiaoko):
    //  1. Add the code to compute the matrix multiplication of the two tiles.
    //  2. Fix an error that two tiles might not be correctly compute w.r.t
    //  row_idx and col_idx.
    for (VertexID j = 0; j < tiled_matrix_t.get_metadata().n_rows; j++) {
      // If  tile and tile_t share the same row, we need to compute the
      // Matrix multiplication of the two tiles.
      auto offset = tiled_matrix.get_tile_ptr_ptr()[i];
      auto offset_t = tiled_matrix_t.get_tile_ptr_ptr()[j];

      while (offset < tiled_matrix.get_tile_ptr_ptr()[i + 1] ||
             offset_t < tiled_matrix_t.get_tile_ptr_ptr()[j + 1]) {

        if (tiled_matrix.get_tile_col_idx_ptr()[offset] ==
            tiled_matrix_t.get_tile_col_idx_ptr()[offset_t]) {
          auto p_tile = tiled_matrix.GetTilebyIdx(offset);
          auto p_tile_t = tiled_matrix_t.GetTilebyIdx(offset_t);

          auto p_tile_c = new Tile();
          TileGEMMWrapper(*p_stream, *p_tile, *p_tile_t, p_tile_c);
          p_tile->Show();
          p_tile_t->Show();
          // p_tile_c->Show();

          if (p_tile_c->get_n_nz() != 0) {
            n_nz_tile_c++;
            p_tile = p_tile_c;
            data_ptr_vec.emplace_back(p_tile_c);
          }
          offset++;
          offset_t++;
        } else if (tiled_matrix.get_tile_col_idx_ptr()[offset] <
                   tiled_matrix_t.get_tile_col_idx_ptr()[offset_t]) {
          offset++;
        } else {
          offset_t++;
        }
      }
    }
  }

  tile_ptr_c = new VertexID[n_rows_c + 1]();
  tile_row_idx_c = new VertexID[n_nz_tile_c]();
  tile_col_idx_c = new VertexID[n_nz_tile_c]();
  tile_n_nz_c = new VertexID[n_nz_tile_c]();

  auto line = 0;
  auto tile_row_offset = 0;
  for (size_t i = 0; i < data_ptr_vec.size(); i++) {
    tile_row_idx_c[i] = data_ptr_vec[i]->get_tile_x();
    tile_col_idx_c[i] = data_ptr_vec[i]->get_tile_y();

    tile_n_nz_c[i] = data_ptr_vec[i]->get_n_nz();
    if (data_ptr_vec[i]->get_tile_x() == line) {
      tile_row_offset++;
    } else {
      line = data_ptr_vec[i]->get_tile_x();
      tile_ptr_c[line] = i;
    }
  }
  tile_ptr_c[n_rows_c] = data_ptr_vec.size();

  C->Init(n_rows_c, n_cols_c, n_nz_tile_c, data_ptr_vec.data(), mask_c,
          tile_ptr_c, tile_row_idx_c, tile_col_idx_c, tile_n_nz_c);

  // Sync to Get the final result.
  // return C;
  // return nullptr;
}

__host__ void MatrixMultiplier::Run() {
  std::cout << "Blocking." << std::endl;

  cudaSetDevice(0);

  auto tiled_matrix = p_tiled_matrix_;
  auto tiled_matrix_t = p_tiled_matrix_t_;
  auto tiled_matrix_metadata = p_tiled_matrix_->get_metadata();
  auto tiled_matrix_t_metadata = p_tiled_matrix_t_->get_metadata();

  assert(tiled_matrix_metadata.n_rows == tiled_matrix_metadata.n_cols);
  assert(tiled_matrix_t_metadata.n_rows == tiled_matrix_t_metadata.n_cols);

  tiled_matrix->Show();
  tiled_matrix_t->Show();

  CUDA_LOG_INFO("Result: ");
  for (size_t i = 0; i < gemm_count_; i++) {

    for (VertexID nz_tile_x = 0;
         nz_tile_x < tiled_matrix->get_metadata().n_rows; nz_tile_x++) {
      auto n_nz_tile_A = tiled_matrix->get_tile_ptr_ptr()[nz_tile_x + 1] -
                         tiled_matrix->get_tile_ptr_ptr()[nz_tile_x];

      if (n_nz_tile_A == 0)
        continue;
      for (int nz_tile_y = 0; nz_tile_y < n_nz_tile_A; nz_tile_y++) {
        auto p_tile = tiled_matrix->GetTilebyIdx(
            tiled_matrix->get_tile_ptr_ptr()[nz_tile_x] + nz_tile_y);
        auto p_tile_y = p_tile->get_tile_y();

        auto n_nz_tile_A_t = tiled_matrix_t->get_tile_ptr_ptr()[p_tile_y + 1] -
                             tiled_matrix_t->get_tile_ptr_ptr()[p_tile_y];

        if (n_nz_tile_A_t == 0)
          continue;

        // There exist no-empty tile that should be processed.
        for (auto nz_tile_x_t = 0; nz_tile_x_t < n_nz_tile_A_t; nz_tile_x_t++) {
          auto p_tile_t = tiled_matrix_t->GetTilebyIdx(
              tiled_matrix_t->get_tile_ptr_ptr()[nz_tile_y] + nz_tile_x_t);

          auto task_id =
              ConvertToTaskId(nz_tile_x, nz_tile_y, nz_tile_x_t, nz_tile_y,
                              tiled_matrix_metadata.n_rows);
          auto stream = GetStream(task_id);

          auto p_tile_output_host = new Tile();

          p_tile_output_host->InitAsOutput(*p_tile, *p_tile_t);
          p_tile_output_host->Show();
          while(1);

          map_tile_ptr_.insert(std::make_pair(task_id, p_tile_output_host));

          TileGEMMWrapper(stream, *p_tile, *p_tile_t, p_tile_output_host);
        }
      }
    }
  }

  std::cout << "Blocking done." << std::endl;
}

__host__ cudaError_t MatrixMultiplier::TileGEMMWrapper(
    const cudaStream_t &stream, const Tile &tile_a, const Tile &tile_b,
    Tile *tile_c) {

  auto tile_a_device = new Tile();
  auto tile_b_device = new Tile();
  auto *tile_c_device = new Tile();

  tile_a_device->InitDevice(tile_a.get_tile_size(), tile_a.get_tile_x(),
                            tile_a.get_tile_y(), tile_a.get_n_nz());
  tile_b_device->InitDevice(tile_b.get_tile_size(), tile_b.get_tile_x(),
                            tile_b.get_tile_y(), tile_b.get_n_nz());
  tile_c_device->InitDevice(tile_c->get_tile_size(), tile_c->get_tile_x(),
                            tile_c->get_tile_y(), tile_c->get_n_nz());

  tile_a_device->MemcpyAsyncHost2Device(tile_a, stream);
  tile_b_device->MemcpyAsyncHost2Device(tile_b, stream);
  tile_c_device->MemcpyAsyncHost2Device(*tile_c, stream);

  int *offset_device;
  cudaMalloc(reinterpret_cast<void **>(&offset_device), sizeof(int));
  cudaMemsetAsync(offset_device, 0, sizeof(int), stream);

  int *n_nz_for_each_row_device;
  cudaMalloc(reinterpret_cast<void **>(&n_nz_for_each_row_device),
             sizeof(int) * tile_c->get_tile_size());
  cudaMemsetAsync(n_nz_for_each_row_device, 0,
                  sizeof(int) * tile_c->get_tile_size(), stream);

  dim3 dimBlock(4, 4);
  dim3 dimGrid(1);

  kernel::TileGEMMParams gemm_params;

  gemm_params.n_nz_A = tile_a_device->get_n_nz();
  gemm_params.n_nz_B = tile_b_device->get_n_nz();
  gemm_params.tile_size_x = tile_a_device->get_tile_size();
  gemm_params.tile_size_y = tile_b_device->get_tile_size();
  gemm_params.offset = offset_device;
  gemm_params.n_nz_for_each_row = n_nz_for_each_row_device;
  gemm_params.bar_offset_A = tile_a_device->GetBarOffsetPtr();
  gemm_params.bar_offset_B = tile_b_device->GetBarOffsetPtr();
  gemm_params.bar_offset_C = tile_c_device->GetBarOffsetPtr();
  gemm_params.row_idx_A = tile_a_device->GetRowIdxPtr();
  gemm_params.row_idx_B = tile_b_device->GetRowIdxPtr();
  gemm_params.row_idx_C = tile_c_device->GetRowIdxPtr();
  gemm_params.col_idx_A = tile_a_device->GetColIdxPtr();
  gemm_params.col_idx_B = tile_b_device->GetColIdxPtr();
  gemm_params.col_idx_C = tile_c_device->GetColIdxPtr();
  gemm_params.data_A = tile_a_device->GetDataPtr();
  gemm_params.data_B = tile_b_device->GetDataPtr();
  gemm_params.data_C = tile_c_device->GetDataPtr();
  gemm_params.bit_mask_A =
      tile_a_device->GetMaskPtr()->GetDataPtr()->GetDataBasePointer();
  gemm_params.bit_mask_B =
      tile_b_device->GetMaskPtr()->GetDataPtr()->GetDataBasePointer();
  gemm_params.bit_mask_C =
      tile_b_device->GetMaskPtr()->GetDataPtr()->GetDataBasePointer();

  kernel::TileGEMM_kernel<<<dimGrid, dimBlock, 48 * 1024, stream>>>(
      gemm_params);

  tile_c->MemcpyAsyncDevice2Host(*tile_c_device, stream);
  cudaStreamSynchronize(stream);

  tile_c->Show();
  std::cout << "GPU finished" << std::endl;
  return cudaSuccess;
}

size_t MatrixMultiplier::ConvertToTaskId(VertexID a, VertexID b, VertexID c,
                                         VertexID d, VertexID range) const {
  VertexID max_val = static_cast<VertexID>(std::pow(range + 1, 4)) - 1;

  VertexID k = (a * (range + 1) * (range + 1) * (range + 1)) +
               (b * (range + 1) * (range + 1)) + (c * (range + 1)) + d;
  return std::min(k, max_val);
}

void MatrixMultiplier::ConvertFromTaskId(VertexID k, VertexID *a, VertexID *b,
                                         VertexID *c, VertexID *d,
                                         VertexID range) {
  // Assuming a,b,c,d are all with in the range [0, range]
  VertexID max_val = static_cast<VertexID>(std::pow(range + 1, 4)) - 1;
  k = std::min(k, max_val);
  *d = k % (range + 1);
  *c = (k / (range + 1)) % (range + 1);
  *b = (k / ((range + 1) * (range + 1))) % (range + 1);
  *a = k / ((range + 1) * (range + 1) * (range + 1));
}

} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics