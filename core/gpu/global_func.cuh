#ifndef MATRIXGRAPH_CORE_GPU_GLOBAL_FUNC_CUH_
#define MATRIXGRAPH_CORE_GPU_GLOBAL_FUNC_CUH_

#include "cublas_v2.h"
#include <assert.h>
#include <cuda.h>
#include <cutlass/gemm/device/gemm.h>
#include <mma.h>
#include <stdio.h>

#include "core/components/execution_plan_generator.h"
#include "core/gpu/device_func.cuh"
#include "core/gpu/kernel_data_structures/kernel_bitmap.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace gpu {

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

} // namespace gpu
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // MATRIXGRAPH_CORE_GPU_GLOBAL_FUNC_CUH_
