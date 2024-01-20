#ifndef MATRIXGRAPH_CORE_GPU_GLOBAL_FUNC_CUH_
#define MATRIXGRAPH_CORE_GPU_GLOBAL_FUNC_CUH_

#include "cublas_v2.h"
#include <assert.h>
#include <cuda.h>
#include <cutlass/gemm/device/gemm.h>
#include <mma.h>
#include <stdio.h>

#include "core/data_structures/tiled_matrix.cuh"
#include "core/gpu/device_func.cuh"
#include "core/gpu/kernel_data_structures/kernel_bitmap.cuh"
#include "core/util/bitmap.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace gpu {

using sics::matrixgraph::core::data_structures::Tile;
using VertexID = sics::matrixgraph::core::common::VertexID;
using TileIndex = sics::matrixgraph::core::common::TileIndex;
using VertexLabel = sics::matrixgraph::core::common::VertexLabel;

// Naive GEMM computation.
__global__ void NaiveGemm_kernel(int M, int N, int K, float alpha,
                                 float const *A, int lda, float const *B,
                                 int ldb, float beta, float *C, int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float accumulator = 0;

    for (int k = 0; k < K; ++k) {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }

    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}

// cuda Tensor core GEMM computation.
__global__ void TensorCoreGemm_kernel(int M, int N, int K, float alpha,
                                      half const *A, int lda, half const *B,
                                      int ldb, float beta, float *C, int ldc) {

  const int WMMA_M = 16;
  const int WMMA_N = 16;
  const int WMMA_K = 16;

  // int lda = M;
  // int ldb = K;
  // int ldc = M;

  int warpM = (blockIdx.x & blockDim.x + threadIdx.x) / 32;
  int warpN = (blockIdx.y & blockDim.y + threadIdx.y);

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                         nvcuda::wmma::col_major>
      a_frag;

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                         nvcuda::wmma::col_major>
      b_frag;

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                         float>
      c_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                         float>
      acc_frag;

  nvcuda::wmma::fill_fragment(c_frag, 0.0f);

  // Load the inputs
  nvcuda::wmma::load_matrix_sync(a_frag, A, 16);
  nvcuda::wmma::load_matrix_sync(b_frag, B, 16);

  for (int i = 0; i < K; i += WMMA_K) {
    int aRow = warpM * WMMA_M;
    int aCol = i;
    int bRow = i;
    int bCol = warpN * WMMA_N;

    if (aRow < M && aCol < K && bRow < K && bCol < N) {
      // Load the inputs
      nvcuda::wmma::load_matrix_sync(a_frag, A + aRow + aCol * lda, lda);
      nvcuda::wmma::load_matrix_sync(b_frag, A + bRow + bCol * ldb, ldb);

      // Perform the matrix multiplication
      nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }

  // Store the output
  nvcuda::wmma::store_matrix_sync(C, c_frag, 16, nvcuda::wmma::mem_row_major);
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
  for (int row_i = tid_y; row_i < tile_size_x; row_i += blockDim.x) {
    for (int col_i = tid_x; col_i < tile_size_y; col_i += blockDim.y) {
      if (bit_mask_C[WORD_OFFSET(col_i * tile_size_x + row_i)] &
          1ul << BIT_OFFSET(4 * row_i + col_i)) {

        TileIndex n_nz_row_A, n_nz_row_B;
        if (row_i == tile_size_x)
          n_nz_row_A = n_nz_A - bar_offset_A[row_i];
        else
          n_nz_row_A = bar_offset_A[row_i + 1] - bar_offset_A[row_i];

        if (row_i == tile_size_x)
          n_nz_row_B = n_nz_B - bar_offset_B[row_i];
        else
          n_nz_row_B = bar_offset_B[row_i + 1] - bar_offset_B[row_i];

        TileIndex p_A = 0;
        TileIndex p_B = 0;
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
                      *(data_A + (bar_offset_A[col_i] + p_A)) *
                          *(data_B + (bar_offset_B[col_i] + p_B)));
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

} // namespace gpu
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // MATRIXGRAPH_CORE_GPU_GLOBAL_FUNC_CUH_
