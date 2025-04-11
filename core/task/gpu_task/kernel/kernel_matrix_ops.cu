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
    input[idx] = input[idx] > 0 ? input[idx] : 0;
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

static __global__ void MatrixMulKernel(ParametersMatrix params) {
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int step = blockDim.x * gridDim.x;

  for (unsigned k_idx = tid; k_idx < params.k; k_idx += step) {
    for (unsigned m_idx = 0; m_idx < params.m; m_idx++) {
      for (unsigned n_idx = 0; n_idx < params.n; n_idx++) {
        atomicAdd(params.C + m_idx * params.n + n_idx,
                  params.A[m_idx * params.k + k_idx] *
                      params.B[n_idx * params.k + k_idx]);
      }
    }
  }
}

static __global__ void MatrixMulTransposedBKernel(ParametersMatrix params) {
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int step = blockDim.x * gridDim.x;

  for (unsigned k_idx = tid; k_idx < params.k; k_idx += step) {
    for (unsigned m_idx = 0; m_idx < params.m; m_idx++) {
      for (unsigned n_idx = 0; n_idx < params.n; n_idx++) {
        params.C[m_idx * params.n + n_idx] +=
            params.A[m_idx * params.k + k_idx] *
            params.B[n_idx * params.k + k_idx];
      }
    }
  }
}

static __global__ void InplaceRectangularTransposeKernel(float* input,
                                                         float* output,
                                                         int rows, int cols) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int col_step = blockDim.y * gridDim.y;
  const unsigned int row_step = blockDim.x * gridDim.x;

  for (int row = idx_x; row < rows; row += row_step) {
    for (int col = idx_y; col < cols; col += col_step) {
      output[row + rows * col] = input[row * cols + col];
    }
  }
}

static __global__ void MatrixAddKernel(float* A, float* B, int m, int n) {
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int step = blockDim.x * gridDim.x;

  for (VertexID idx = tid; idx < m * n; idx += step) atomicAdd(B + idx, A[idx]);
}

void MatrixOpsKernelWrapper::MatMult(const cudaStream_t& stream, float* A,
                                     float* B, float* C, int m, int k, int n,
                                     bool transposed) {
  ParametersMatrix params{.A = A, .B = B, .C = C, .m = m, .k = k, .n = n};

  // For MatrixMulSharedKernel();
  // dim3 dimBlock(16, 16);
  // dim3 dimGrid((n + 15) / 16, (m + 15) / 16);

  dim3 dimBlock(kBlockDim);
  dim3 dimGrid(kGridDim);
  MatrixMulKernel<<<dimGrid, dimBlock, 0, stream>>>(params);
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

void MatrixOpsKernelWrapper::Transpose(const cudaStream_t& stream, float* input,
                                       float* output, int rows, int cols) {
  dim3 dimBlock(32, 32);
  dim3 dimGrid(32, 32);

  InplaceRectangularTransposeKernel<<<dimGrid, dimBlock, 0, stream>>>(
      input, output, rows, cols);
}

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
