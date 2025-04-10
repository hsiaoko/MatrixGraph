#include "core/common/consts.h"
#include "core/common/types.h"
#include "core/data_structures/kernel_bitmap.cuh"
#include "core/data_structures/kernel_bitmap_no_ownership.cuh"
#include "core/data_structures/mini_kernel_bitmap.cuh"
#include "core/task/gpu_task/kernel/kernel_matrix_ops.cuh"
#include "core/util/bitmap_no_ownership.h"
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
using VertexID = sics::matrixgraph::core::common::VertexID;
using sics::matrixgraph::core::common::kBlockDim;
using sics::matrixgraph::core::common::kGridDim;
using sics::matrixgraph::core::common::kMaxVertexID;
using sics::matrixgraph::core::task::kernel::KernelBitmapNoOwnership;
using BitmapNoOwnerShip = sics::matrixgraph::core::util::BitmapNoOwnerShip;

struct ParametersMatrix {
  float* A;
  float* B;
  float* C;
  int m;
  int k;
  int n;
};

MatrixOpsKernelWrapper* MatrixOpsKernelWrapper::GetInstance() {
  if (ptr_ == nullptr) {
    ptr_ = new MatrixOpsKernelWrapper();
  }
  return ptr_;
}

/**
 * @brief CUDA Kernel for ReLU activation function
 * @param input  Input array (device pointer)
 * @param n      Number of elements in the array
 */
static __global__ void ReluKernel(float* input, int n) {
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int step = blockDim.x * gridDim.x;
  for (VertexID idx = tid; idx < n; idx += step) {
    printf("%f ", input[idx]);
    input[idx] = input[idx] > 0 ? input[idx] : 0;
    printf("%f \n", input[idx]);
  }
}

static __global__ void MatrixMulSharedKernel(ParametersMatrix params) {
  __shared__ float s_A[16][16];  // the size of Tile = 16 x 16
  __shared__ float s_B[16][16];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;

  for (int tile = 0; tile < (params.k + 15) / 16; ++tile) {
    //  load the matrix from global memory to shared memory.
    int loadRow = row;
    int loadCol = tile * 16 + threadIdx.x;
    if (loadRow < params.m && loadCol < params.k) {
      s_A[threadIdx.y][threadIdx.x] = params.A[loadRow * params.k + loadCol];
    } else {
      s_A[threadIdx.y][threadIdx.x] = 0.0f;
    }

    loadRow = tile * 16 + threadIdx.y;
    loadCol = col;
    if (loadRow < params.k && loadCol < params.n) {
      s_B[threadIdx.y][threadIdx.x] = params.B[loadRow * params.n + loadCol];
    } else {
      s_B[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int k = 0; k < 16; ++k) {
      sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < params.m && col < params.n) {
    params.C[row * params.n + col] = sum;
  }
}

static __global__ void TransposedMatrixMulSharedKernel(
    ParametersMatrix params) {
  __shared__ float s_A[16][16];  // the size of Tile = 16 x 16
  __shared__ float s_B[16][16];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;

  for (int tile = 0; tile < (params.k + 15) / 16; ++tile) {
    //  load the matrix from global memory to shared memory.
    int loadRow = row;
    int loadCol = tile * 16 + threadIdx.x;
    if (loadRow < params.m && loadCol < params.k) {
      s_A[threadIdx.y][threadIdx.x] = params.A[loadRow * params.k + loadCol];
    } else {
      s_A[threadIdx.y][threadIdx.x] = 0.0f;
    }

    loadRow = tile * 16 + threadIdx.y;
    loadCol = col;
    if (loadRow < params.k && loadCol < params.n) {
      s_B[threadIdx.y][threadIdx.x] = params.B[loadRow + params.n * loadCol];
    } else {
      s_B[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int k = 0; k < 16; ++k) {
      sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < params.m && col < params.n) {
    params.C[row * params.n + col] = sum;
  }
}

static __global__ void MatrixAddKernel(float* A, float* B, int m, int n) {
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int step = blockDim.x * gridDim.x;

  for (VertexID idx = tid; idx < m * n; idx += step) B[idx] += A[idx];
}

void MatrixOpsKernelWrapper::MatMult(const cudaStream_t& stream, float* A,
                                     float* B, float* C, int m, int k, int n,
                                     bool transposed) {
  ParametersMatrix params{.A = A, .B = B, .C = C, .m = m, .k = k, .n = n};

  dim3 dimBlock(16, 16);
  dim3 dimGrid((n + 15) / 16, (m + 15) / 16);

  if (transposed) {
    TransposedMatrixMulSharedKernel<<<dimGrid, dimBlock, 0, stream>>>(params);
  } else {
    MatrixMulSharedKernel<<<dimGrid, dimBlock, 0, stream>>>(params);
  }
}

void MatrixOpsKernelWrapper::MatAdd(const cudaStream_t& stream, float* A,
                                    float* B, int m, int n) {
  dim3 dimBlock(kBlockDim);
  dim3 dimGrid(kGridDim);
  MatrixAddKernel<<<dimGrid, dimBlock, 0, stream>>>(A, B, m, n);
}
void MatrixOpsKernelWrapper::Activate(const cudaStream_t& stream, float* A,
                                      int m, int n) {
  dim3 dimBlock(kBlockDim);
  dim3 dimGrid(kGridDim);
  ReluKernel<<<dimGrid, dimBlock, 0, stream>>>(A, m * n);
}

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
