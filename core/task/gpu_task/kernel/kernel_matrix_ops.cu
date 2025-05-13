#include <cuda_runtime.h>

#include <chrono>
#include <iostream>

#include "core/common/consts.h"
#include "core/common/types.h"
#include "core/data_structures/kernel_bitmap.cuh"
#include "core/data_structures/kernel_bitmap_no_ownership.cuh"
#include "core/data_structures/mini_kernel_bitmap.cuh"
#include "core/task/gpu_task/kernel/kernel_matrix_ops.cuh"
#include "core/util/bitmap_no_ownership.h"

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

template <typename T>
struct ParametersMatrix {
  const T* A;
  const T* B;
  T* C;
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

inline __m256 fast_exp_avx(__m256 x) {
  // 限制输入范围 [-10, 10] 避免数值问题
  const __m256 max_x = _mm256_set1_ps(10.0f);
  const __m256 min_x = _mm256_set1_ps(-10.0f);
  x = _mm256_min_ps(x, max_x);
  x = _mm256_max_ps(x, min_x);

  // 使用多项式近似 exp(x) (5阶泰勒展开)
  const __m256 c0 = _mm256_set1_ps(1.0f);
  const __m256 c1 = _mm256_set1_ps(1.0f);
  const __m256 c2 = _mm256_set1_ps(0.5f);
  const __m256 c3 = _mm256_set1_ps(0.166666667f);  // 1/6
  const __m256 c4 = _mm256_set1_ps(0.041666667f);  // 1/24
  const __m256 c5 = _mm256_set1_ps(0.008333333f);  // 1/120

  __m256 x2 = _mm256_mul_ps(x, x);
  __m256 x3 = _mm256_mul_ps(x2, x);
  __m256 x4 = _mm256_mul_ps(x3, x);
  __m256 x5 = _mm256_mul_ps(x4, x);

  __m256 result = c0;
  result = _mm256_add_ps(result, _mm256_mul_ps(c1, x));
  result = _mm256_add_ps(result, _mm256_mul_ps(c2, x2));
  result = _mm256_add_ps(result, _mm256_mul_ps(c3, x3));
  result = _mm256_add_ps(result, _mm256_mul_ps(c4, x4));
  result = _mm256_add_ps(result, _mm256_mul_ps(c5, x5));

  return result;
}

static void SimdSquaredDifferenceSIMD(const float* v_a, const float* v_b,
                                      float* v_c, size_t n) {
  constexpr size_t simd_width = 8;  // AVX 一次处理 8 个 float
  size_t i = 0;

  // 主 SIMD 循环
  for (; i + simd_width <= n; i += simd_width) {
    // 加载向量数据
    __m256 vec_a = _mm256_loadu_ps(v_a + i);
    __m256 vec_b = _mm256_loadu_ps(v_b + i);

    // 计算差值 (a - b)
    __m256 diff = _mm256_sub_ps(vec_a, vec_b);

    // 计算平方 (a - b)^2
    __m256 squared = _mm256_mul_ps(diff, diff);

    // 存储结果
    _mm256_storeu_ps(v_c + i, squared);
  }

  // 处理剩余不足 8 的元素（标量处理）
  for (; i < n; ++i) {
    float diff = v_a[i] - v_b[i];
    v_c[i] = diff * diff;
  }
  // for (size_t i = 0; i < n; ++i) {
  //   v_c[i] = (v_a[i] - v_b[i]) * (v_a[i] - v_b[i]);
  // }
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

void ReluSIMD(float* input, int n) {
  constexpr int simd_width = 8;
  const __m256 zero = _mm256_set1_ps(0.0f);

  int i = 0;
  for (; i + simd_width <= n; i += simd_width) {
    __m256 vec = _mm256_loadu_ps(input + i);
    __m256 result = _mm256_max_ps(vec, zero);  // max(input, 0) 等价于 ReLU
    _mm256_storeu_ps(input + i, result);
  }

  // 剩余元素处理
  for (; i < n; ++i) {
    input[i] = input[i] > 0 ? input[i] : 0;
  }
}

/**
 * @brief CUDA Kernel for Sigmoid activation function
 * @param input  Input array (device pointer)
 * @param n      Number of elements in the array
 */
static __global__ void SigmoidKernel(float* input, int n) {
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int step = blockDim.x * gridDim.x;
  for (unsigned int idx = tid; idx < n; idx += step) {
    input[idx] = 1.0f / (1.0f + __expf(-input[idx]));
  }
}

void SigmoidSIMD(float* input, int n) {
  constexpr int simd_width = 8;  // AVX 处理 8 个 float
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 zero = _mm256_set1_ps(0.0f);

  int i = 0;
  // 主 SIMD 循环
  for (; i + simd_width <= n; i += simd_width) {
    __m256 vec = _mm256_loadu_ps(input + i);     // 加载数据
    __m256 neg_vec = _mm256_sub_ps(zero, vec);   // -input
    __m256 exp_neg = fast_exp_avx(neg_vec);      // exp(-input)
    __m256 denom = _mm256_add_ps(one, exp_neg);  // 1 + exp(-input)
    __m256 result = _mm256_div_ps(one, denom);   // 1 / (1 + exp(-input))
    _mm256_storeu_ps(input + i, result);         // 存回内存
  }

  // 处理剩余不足 8 的元素 (标量处理)
  for (; i < n; ++i) {
    input[i] = 1.0f / (1.0f + expf(-input[i]));
  }
}

static __global__ void MatrixMulSharedKernel(ParametersMatrix<float> params) {
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

template <typename T>
static __global__ void MatrixMulKernel(ParametersMatrix<T> params) {
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

template <typename T>
static void MatrixMulSIMD(ParametersMatrix<T> params) {
  for (int i = 0; i < params.m; i++) {
    for (int j = 0; j < params.n; j++) {
      __m256 sum = _mm256_setzero_ps();

      // 主计算部分 - 每次处理8个元素
      int k = 0;
      for (; k <= params.k - 8; k += 8) {
        __m256 a = _mm256_loadu_ps(&params.A[i * params.k + k]);
        __m256 b = _mm256_loadu_ps(&params.B[k * params.n + j]);
        sum = _mm256_fmadd_ps(a, b, sum);
      }

      // 处理剩余元素
      float s = 0.0f;
      for (; k < params.k; k++) {
        s += params.A[i * params.k + k] * params.B[k * params.n + j];
      }

      // 合并SIMD和标量结果
      alignas(32) float temp[8];
      _mm256_store_ps(temp, sum);
      params.C[i * params.n + j] = temp[0] + temp[1] + temp[2] + temp[3] +
                                   temp[4] + temp[5] + temp[6] + temp[7] + s;
    }
  }
}

template <typename T>
static __global__ void MatrixMulTransposedBKernel(ParametersMatrix<T> params) {
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int step = blockDim.x * gridDim.x;

  for (unsigned k_idx = tid; k_idx < params.k; k_idx += step) {
    for (unsigned m_idx = 0; m_idx < params.m; m_idx++) {
      for (unsigned n_idx = 0; n_idx < params.n; n_idx++) {
        atomicAdd(params.C + m_idx * params.n + n_idx,
                  params.A[m_idx * params.k + k_idx] *
                      params.B[k_idx * params.n + n_idx]);
      }
    }
  }
}

template <typename T>
static void MatrixTransposedBMulSIMD(ParametersMatrix<T> params) {
  for (int i = 0; i < params.m; i++) {
    for (int j = 0; j < params.n; j++) {
      __m256 sum = _mm256_setzero_ps();

      // 主计算部分 - 每次处理8个元素
      int k = 0;
      for (; k <= params.k - 8; k += 8) {
        __m256 a = _mm256_loadu_ps(&params.A[i * params.k + k]);
        __m256 b = _mm256_loadu_ps(&params.B[k * params.n + j]);
        sum = _mm256_fmadd_ps(a, b, sum);
      }

      // 处理剩余元素
      float s = 0.0f;
      for (; k < params.k; k++) {
        s += params.A[i * params.k + k] * params.B[k * params.n + j];
      }

      // 合并SIMD和标量结果
      alignas(32) float temp[8];
      _mm256_store_ps(temp, sum);
      params.C[i * params.n + j] = temp[0] + temp[1] + temp[2] + temp[3] +
                                   temp[4] + temp[5] + temp[6] + temp[7] + s;
    }
  }
}

template <typename T>
static __global__ void MatrixMulTransposedAKernel(ParametersMatrix<T> params) {
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int step = blockDim.x * gridDim.x;

  for (unsigned k_idx = tid; k_idx < params.k; k_idx += step) {
    for (unsigned m_idx = 0; m_idx < params.m; m_idx++) {
      for (unsigned n_idx = 0; n_idx < params.n; n_idx++) {
        atomicAdd(params.C + m_idx * params.n + n_idx,
                  params.A[k_idx * params.m + m_idx] *
                      params.B[n_idx * params.k + k_idx]);
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

  for (VertexID idx = tid; idx < m * n; idx += step) atomicAdd(A + idx, B[idx]);
}

void MatrixAddSIMD(float* A, const float* B, int m, int n) {
  const int total_elements = m * n;
  constexpr int simd_width = 8;

  for (int i = 0; i < total_elements; i += simd_width) {
    if (i + simd_width <= total_elements) {
      __m256 vec_a = _mm256_loadu_ps(A + i);
      __m256 vec_b = _mm256_loadu_ps(B + i);
      __m256 result = _mm256_add_ps(vec_a, vec_b);
      _mm256_storeu_ps(A + i, result);
    } else {
      for (int j = i; j < total_elements; ++j) {
        A[j] += B[j];
      }
    }
  }
}

void MatrixOpsKernelWrapper::MatMult(const cudaStream_t& stream, float* A,
                                     float* B, float* C, int m, int k, int n,
                                     bool transposed_a, bool transposed_b) {
  ParametersMatrix<float> params{
      .A = A, .B = B, .C = C, .m = m, .k = k, .n = n};

  // For MatrixMulSharedKernel();
  // dim3 dimBlock(16, 16);
  // dim3 dimGrid((n + 15) / 16, (m + 15) / 16);

  dim3 dimBlock(kBlockDim);
  dim3 dimGrid(kGridDim);
  if (transposed_b) {
    MatrixMulTransposedBKernel<<<dimGrid, dimBlock, 0, stream>>>(params);
  } else {
    MatrixMulKernel<<<dimGrid, dimBlock, 0, stream>>>(params);
  }
}

void MatrixOpsKernelWrapper::MatAdd(const cudaStream_t& stream, float* A,
                                    float* B, int m, int n) {
  dim3 dimBlock(kBlockDim);
  dim3 dimGrid(kGridDim);
  MatrixAddKernel<<<dimGrid, dimBlock, 0, stream>>>(A, B, m, n);
}

void MatrixOpsKernelWrapper::Relu(const cudaStream_t& stream, float* A, int m,
                                  int n) {
  dim3 dimBlock(kBlockDim);
  dim3 dimGrid(kGridDim);
  ReluKernel<<<dimGrid, dimBlock, 0, stream>>>(A, m * n);
}

void MatrixOpsKernelWrapper::Sigmoid(const cudaStream_t& stream, float* A,
                                     int m, int n) {
  dim3 dimBlock(kBlockDim);
  dim3 dimGrid(kGridDim);
  SigmoidKernel<<<dimGrid, dimBlock, 0, stream>>>(A, m * n);
}

void MatrixOpsKernelWrapper::Transpose(const cudaStream_t& stream, float* input,
                                       float* output, int rows, int cols) {
  dim3 dimBlock(32, 32);
  dim3 dimGrid(32, 32);

  InplaceRectangularTransposeKernel<<<dimGrid, dimBlock, 0, stream>>>(
      input, output, rows, cols);
}

void MatrixOpsKernelWrapper::CPUMatMult(const float* A, const float* B,
                                        float* C, int m, int k, int n,
                                        bool transposed_a, bool transposed_b) {
  ParametersMatrix<float> params{
      .A = A, .B = B, .C = C, .m = m, .k = k, .n = n};
  if (transposed_b) {
    MatrixTransposedBMulSIMD(params);
  } else {
    MatrixMulSIMD(params);
  }
}

void MatrixOpsKernelWrapper::CPUSigmoid(float* A, int m, int n) {
  SigmoidSIMD(A, m * n);
}

void MatrixOpsKernelWrapper::CPURelu(float* A, int m, int n) {
  ReluSIMD(A, m * n);
}

void MatrixOpsKernelWrapper::CPUMatAdd(float* A, float* B, int m, int n) {
  MatrixAddSIMD(A, B, m, n);
}

void MatrixOpsKernelWrapper::CPUSimdSquaredDifference(const float* v_a,
                                                      const float* v_b,
                                                      float* v_c, size_t n) {
  SimdSquaredDifferenceSIMD(v_a, v_b, v_c, n);
}

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
