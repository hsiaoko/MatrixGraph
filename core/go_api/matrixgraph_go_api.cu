#include "go_api/matrixgraph_go_api.h"

#include <cuda_runtime.h>

#include "task/gpu_task/kernel/kernel_matrix_ops.cuh"

namespace kernel = sics::matrixgraph::core::task::kernel;

namespace {

inline int cuda_err_to_int(cudaError_t e) {
  return (e == cudaSuccess) ? 0 : 1;
}

}  // namespace

extern "C" {

int matrixgraph_matmult(const float* A, const float* B, float* C, int m, int k, int n) {
  size_t sz_a = static_cast<size_t>(m) * k * sizeof(float);
  size_t sz_b = static_cast<size_t>(k) * n * sizeof(float);
  size_t sz_c = static_cast<size_t>(m) * n * sizeof(float);

  float* d_A = nullptr;
  float* d_B = nullptr;
  float* d_C = nullptr;
  cudaStream_t stream = nullptr;

  cudaError_t err = cudaMalloc(&d_A, sz_a);
  if (err != cudaSuccess) return cuda_err_to_int(err);
  err = cudaMalloc(&d_B, sz_b);
  if (err != cudaSuccess) { cudaFree(d_A); return cuda_err_to_int(err); }
  err = cudaMalloc(&d_C, sz_c);
  if (err != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); return cuda_err_to_int(err); }

  err = cudaMemcpy(d_A, A, sz_a, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) goto matmult_fail;
  err = cudaMemcpy(d_B, B, sz_b, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) goto matmult_fail;

  err = cudaStreamCreate(&stream);
  if (err != cudaSuccess) goto matmult_fail;

  kernel::MatrixOpsKernelWrapper::MatMult(stream, d_A, d_B, d_C, m, k, n, false, false);

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  stream = nullptr;

  err = cudaMemcpy(C, d_C, sz_c, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) goto matmult_fail;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;

matmult_fail:
  if (stream) cudaStreamDestroy(stream);
  if (d_A) cudaFree(d_A);
  if (d_B) cudaFree(d_B);
  if (d_C) cudaFree(d_C);
  return 1;
}

int matrixgraph_relu(float* A, int m, int n) {
  size_t sz = static_cast<size_t>(m) * n * sizeof(float);
  float* d_A = nullptr;

  cudaError_t err = cudaMalloc(&d_A, sz);
  if (err != cudaSuccess) return cuda_err_to_int(err);

  err = cudaMemcpy(d_A, A, sz, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) { cudaFree(d_A); return cuda_err_to_int(err); }

  cudaStream_t stream = nullptr;
  err = cudaStreamCreate(&stream);
  if (err != cudaSuccess) { cudaFree(d_A); return cuda_err_to_int(err); }

  kernel::MatrixOpsKernelWrapper::Relu(stream, d_A, m, n);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  err = cudaMemcpy(A, d_A, sz, cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  return cuda_err_to_int(err);
}

int matrixgraph_matadd(const float* A, float* B, int m, int n) {
  size_t sz = static_cast<size_t>(m) * n * sizeof(float);
  float* d_A = nullptr;
  float* d_B = nullptr;
  cudaStream_t stream = nullptr;

  cudaError_t err = cudaMalloc(&d_A, sz);
  if (err != cudaSuccess) return cuda_err_to_int(err);
  err = cudaMalloc(&d_B, sz);
  if (err != cudaSuccess) { cudaFree(d_A); return cuda_err_to_int(err); }

  err = cudaMemcpy(d_A, A, sz, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) goto matadd_fail;
  err = cudaMemcpy(d_B, B, sz, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) goto matadd_fail;

  err = cudaStreamCreate(&stream);
  if (err != cudaSuccess) goto matadd_fail;

  kernel::MatrixOpsKernelWrapper::MatAdd(stream, d_A, d_B, m, n);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  stream = nullptr;

  err = cudaMemcpy(B, d_B, sz, cudaMemcpyDeviceToHost);

matadd_fail:
  if (stream) cudaStreamDestroy(stream);
  if (d_A) cudaFree(d_A);
  if (d_B) cudaFree(d_B);
  return cuda_err_to_int(err);
}

int matrixgraph_transpose(const float* A, float* B, int m, int n) {
  size_t sz_a = static_cast<size_t>(m) * n * sizeof(float);
  size_t sz_b = static_cast<size_t>(n) * m * sizeof(float);
  float* d_A = nullptr;
  float* d_B = nullptr;
  cudaStream_t stream = nullptr;

  cudaError_t err = cudaMalloc(&d_A, sz_a);
  if (err != cudaSuccess) return cuda_err_to_int(err);
  err = cudaMalloc(&d_B, sz_b);
  if (err != cudaSuccess) { cudaFree(d_A); return cuda_err_to_int(err); }

  err = cudaMemcpy(d_A, A, sz_a, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) goto transpose_fail;

  err = cudaStreamCreate(&stream);
  if (err != cudaSuccess) goto transpose_fail;

  kernel::MatrixOpsKernelWrapper::Transpose(stream, d_A, d_B, m, n);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  stream = nullptr;

  err = cudaMemcpy(B, d_B, sz_b, cudaMemcpyDeviceToHost);

transpose_fail:
  if (stream) cudaStreamDestroy(stream);
  if (d_A) cudaFree(d_A);
  if (d_B) cudaFree(d_B);
  return cuda_err_to_int(err);
}

}  // extern "C"
