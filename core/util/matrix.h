
#ifndef GRAPH_COMPUTING_MATRIXGRAPH_CORE_UTIL_MATRIX_H_
#define GRAPH_COMPUTING_MATRIXGRAPH_CORE_UTIL_MATRIX_H_

#include <cstdlib>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <sys/time.h>

namespace sics {
namespace matrixgraph {
namespace core {
namespace util {

// Kernel to initialize a matrix with small integers.
template <typename T>
__global__ void InitializeMatrix_kernel(T *matrix, int rows, int columns,
                                        int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i + j * rows;

    // Generate arbitrary elements.
    int const k = 16807;
    int const m = 16;
    T value = (((offset + seed) * k % m) - m / 2);

    matrix[offset] = 1;
  }
}

/// Simple function to initialize a matrix to arbitrary small integers.
template <typename T>
cudaError_t InitializeMatrix(T *matrix, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid((rows + block.x - 1) / block.x, (columns + block.y - 1) / block.y);

  InitializeMatrix_kernel<<<grid, block>>>(matrix, rows, columns, seed);

  return cudaGetLastError();
}

// Allocates device memory for a matrix then fills with arbitrary small
// integers.
template <typename T>
cudaError_t AllocateMatrix(T **matrix, int rows, int columns, int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(T) * rows * columns;

  // Allocate device memory.
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: " << cudaGetErrorString(result)
              << std::endl;
    return result;
  }

  // Clear the allocation.
  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
              << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Initialize matrix elements to arbitrary small integers.
  result = InitializeMatrix(*matrix, rows, columns, seed);

  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: " << cudaGetErrorString(result)
              << std::endl;
    return result;
  }

  return result;
}

void PrintMatrix(const float *A, int nr_rows_A, int nr_cols_A,
                  int scope = INT_MAX) {

  for (int i = 0; i < nr_rows_A; ++i) {
    if (i > scope)
      break;
    for (int j = 0; j < nr_cols_A; ++j) {
      if (j > scope)
        break;
      std::cout << A[j * nr_rows_A + i] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void cuRANDFill(float *A, int nr_rows_A, int nr_cols_A) {
  // Create a pseudo-random number generator
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

  // Set the seed for the random number generator using the system clock
  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

  // Fill the array with random numbers on the device
  curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

int8_t float2int8(float f, float scale) {
  int8_t i = int8_t(f * scale);
  if (i < -127)
    i = -127;
  if (i > 127)
    i = 127;
  return i;
}

template <typename T, typename S>
void AllocateDeviceMemory(int m, int n, int k, T **A, T **B, S **C) {
  cudaMallocManaged(A, m * k * sizeof(T));
  cudaMallocManaged(B, k * n * sizeof(T));
  cudaMallocManaged(C, m * n * sizeof(S));
}

template <typename T, typename S> void FreeDeviceMemory(T *A, T *B, S *C) {
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}

template <typename T, typename S>
int cuBLASGemmExWrapper(cublasHandle_t handle, cublasOperation_t transA,
                        cublasOperation_t transB, int m, int n, int k, T *A,
                        T *B, S *C, int lda, int ldb, int ldc, S *alpha,
                        S *beta, int algo) {
  cudaDataType_t AType, BType, CType, ComputeType;
  if (std::is_same<T, float>::value) {
    AType = BType = CType = ComputeType = CUDA_R_32F;
  } else if (std::is_same<T, __half>::value) {
    AType = BType = CType = ComputeType = CUDA_R_16F;
  } else if (std::is_same<T, int8_t>::value) {
    AType = BType = CUDA_R_8I;
    CType = ComputeType = CUDA_R_32I;
  } else {
    printf("Not supported data type.");
    return -1;
  }
  cublasStatus_t status;
  status = cublasGemmEx(handle, transA, transB, m, n, k, alpha, A, AType, lda,
                        B, BType, ldb, beta, C, CType, ldc, ComputeType,
                        static_cast<cublasGemmAlgo_t>(algo));

  if (status == CUBLAS_STATUS_SUCCESS)
    return 1;
  else
    return -1;
}

template <typename T, typename S>
void TestcuBLASGemm(cublasHandle_t handle, int m, int n, int k, T *A, T *B, S *C,
               S *alpha, S *beta, int algo, int iteration) {
  auto start_time = std::chrono::system_clock::now();
  int success = 0;
  for (int i = 0; i < iteration; ++i) {
    struct timeval start, end;
    cudaDeviceSynchronize();
    cudaProfilerStart();
    gettimeofday(&start, NULL);
    success = cuBLASGemmExWrapper(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, B,
                                  A, C, n, k, n, alpha, beta,
                                  static_cast<cublasGemmAlgo_t>(algo));
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    cudaProfilerStop();
  }
  auto end_time = std::chrono::system_clock::now();
  if (success > 0)
    std::cout << "Alg " << algo << " time cost: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     end_time - start_time)
                         .count() /
                     (double)CLOCKS_PER_SEC
              << std::endl;
}

} // namespace util
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // GRAPH_COMPUTING_MATRIXGRAPH_CORE_UTIL_MATRIX_H_
