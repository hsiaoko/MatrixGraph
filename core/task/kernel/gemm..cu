#include "core/task/kernel/gemm.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

__global__ void TileGEMM_kernel(TileGEMMParams params) {

  unsigned int tid_x = blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.x;
  unsigned int tid_y = blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y;

  // GEMM on col_i of A and row_i of B.
  for (int row_i = tid_x; row_i < params.tile_size_x; row_i += blockDim.x) {
    for (int col_i = tid_y; col_i < params.tile_size_y; col_i += blockDim.y) {
      if (params.bit_mask_C[WORD_OFFSET(row_i * params.tile_size_x + col_i)] &
          1ul << BIT_OFFSET(params.tile_size_x * row_i + col_i)) {

        TileIndex n_nz_row_A, n_nz_row_B;
        if (row_i == params.tile_size_x - 1)
          n_nz_row_A = params.n_nz_A - params.bar_offset_A[row_i];
        else
          n_nz_row_A =
              params.bar_offset_A[row_i + 1] - params.bar_offset_A[row_i];

        if (col_i == params.tile_size_x - 1)
          n_nz_row_B = params.n_nz_B - params.bar_offset_B[col_i];
        else
          n_nz_row_B =
              params.bar_offset_B[col_i + 1] - params.bar_offset_B[col_i];

        TileIndex p_A = params.bar_offset_A[row_i];
        TileIndex p_B = params.bar_offset_B[col_i];
        TileIndex p = 0;

        while (p < n_nz_row_A || p < n_nz_row_B) {
          if (params.col_idx_A[p_A] < params.col_idx_B[p_B]) {
            p_A++;
          } else if (params.col_idx_A[p_A] > params.col_idx_B[p_B]) {
            p_B++;
          } else {
            int local_offset = atomicAdd(params.offset, 1);
            params.row_idx_C[local_offset] = row_i;
            params.col_idx_C[local_offset] = col_i;
            atomicAdd(
                params.data_C + local_offset,
                *(params.data_A + (params.bar_offset_A[row_i] + p_A)) *
                    *(params.data_B + (params.bar_offset_B[row_i] + p_B)));
            atomicAdd(params.n_nz_for_each_row + row_i, 1);
            p_A++;
            p_B++;
          }
          p = p_A > p_B ? p_A : p_B;
        }
      }
    }
  }
}

__global__ void TileGemm_kernel(
    VertexID n_nz_A, VertexID n_nz_B, TileIndex tile_size_x,
    TileIndex tile_size_y, int *offset, int *n_nz_for_each_row,
    TileIndex *bar_offset_A, TileIndex *bar_offset_B, TileIndex *bar_offset_C,
    TileIndex *row_idx_A, TileIndex *row_idx_B, TileIndex *row_idx_C,
    TileIndex *col_idx_A, TileIndex *col_idx_B, TileIndex *col_idx_C,
    VertexLabel *data_A, VertexLabel *data_B, VertexLabel *data_C,
    uint64_t *bit_mask_A, uint64_t *bit_mask_B, uint64_t *bit_mask_C) {

  unsigned int tid_x = blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.x;
  unsigned int tid_y = blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y;

  // GEMM on col_i of A and row_i of B.
  for (int row_i = tid_x; row_i < tile_size_x; row_i += blockDim.x) {
    for (int col_i = tid_y; col_i < tile_size_y; col_i += blockDim.y) {
      if (bit_mask_C[WORD_OFFSET(row_i * tile_size_x + col_i)] &
          1ul << BIT_OFFSET(tile_size_x * row_i + col_i)) {

        TileIndex n_nz_row_A, n_nz_row_B;
        if (row_i == tile_size_x - 1)
          n_nz_row_A = n_nz_A - bar_offset_A[row_i];
        else
          n_nz_row_A = bar_offset_A[row_i + 1] - bar_offset_A[row_i];

        if (col_i == tile_size_x - 1)
          n_nz_row_B = n_nz_B - bar_offset_B[col_i];
        else
          n_nz_row_B = bar_offset_B[col_i + 1] - bar_offset_B[col_i];

        TileIndex p_A = bar_offset_A[row_i];
        TileIndex p_B = bar_offset_B[col_i];
        TileIndex p = 0;

        while (p < n_nz_row_A || p < n_nz_row_B) {
          if (col_idx_A[p_A] < col_idx_B[p_B]) {
            p_A++;
          } else if (col_idx_A[p_A] > col_idx_B[p_B]) {
            p_B++;
          } else {
            int local_offset = atomicAdd(offset, 1);
            row_idx_C[local_offset] = row_i;
            col_idx_C[local_offset] = col_i;
            atomicAdd(data_C + local_offset,
                      *(data_A + (bar_offset_A[row_i] + p_A)) *
                          *(data_B + (bar_offset_B[row_i] + p_B)));
            atomicAdd(n_nz_for_each_row + row_i, 1);
            p_A++;
            p_B++;
          }
          p = p_A > p_B ? p_A : p_B;
        }
      }
    }
  }
}

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics