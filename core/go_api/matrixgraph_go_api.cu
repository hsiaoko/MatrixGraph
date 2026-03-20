#include "go_api/matrixgraph_go_api.h"
#include "task/gpu_task/GARMatch.cuh"
#include "task/gpu_task/kernel/kernel_matrix_ops.cuh"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace kernel = sics::matrixgraph::core::task::kernel;
namespace task = sics::matrixgraph::core::task;

namespace {

inline int cuda_err_to_int(cudaError_t e) { return (e == cudaSuccess) ? 0 : 1; }

inline void log_cuda_error(const char* op) {
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[matrixgraph] %s CUDA error: %s\n", op,
                 cudaGetErrorString(e));
  }
}

// If MATRIXGRAPH_CUDA_DEVICE is set (e.g. to "1"), use that GPU for subsequent
// CUDA calls.
inline void set_device_from_env() {
  const char* s = std::getenv("MATRIXGRAPH_CUDA_DEVICE");
  if (s == nullptr || s[0] == '\0') return;
  int device = std::atoi(s);
  if (device >= 0) {
    cudaError_t e = cudaSetDevice(device);
    if (e != cudaSuccess) {
      std::fprintf(stderr, "[matrixgraph] cudaSetDevice(%d) failed: %s\n",
                   device, cudaGetErrorString(e));
    }
  }
}

}  // namespace

extern "C" {

int matrixgraph_matmult(const float* A, const float* B, float* C, int m, int k,
                        int n) {
  set_device_from_env();
  size_t sz_a = static_cast<size_t>(m) * k * sizeof(float);
  size_t sz_b = static_cast<size_t>(k) * n * sizeof(float);
  size_t sz_c = static_cast<size_t>(m) * n * sizeof(float);

  float* d_A = nullptr;
  float* d_B = nullptr;
  float* d_C = nullptr;
  cudaStream_t stream = nullptr;

  cudaError_t err = cudaMalloc(&d_A, sz_a);
  if (err != cudaSuccess) {
    log_cuda_error("MatMult cudaMalloc");
    return cuda_err_to_int(err);
  }
  err = cudaMalloc(&d_B, sz_b);
  if (err != cudaSuccess) {
    cudaFree(d_A);
    log_cuda_error("MatMult cudaMalloc");
    return cuda_err_to_int(err);
  }
  err = cudaMalloc(&d_C, sz_c);
  if (err != cudaSuccess) {
    cudaFree(d_A);
    cudaFree(d_B);
    log_cuda_error("MatMult cudaMalloc");
    return cuda_err_to_int(err);
  }

  err = cudaMemcpy(d_A, A, sz_a, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) goto matmult_fail;
  err = cudaMemcpy(d_B, B, sz_b, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) goto matmult_fail;

  err = cudaStreamCreate(&stream);
  if (err != cudaSuccess) goto matmult_fail;

  kernel::MatrixOpsKernelWrapper::MatMult(stream, d_A, d_B, d_C, m, k, n, false,
                                          false);

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) goto matmult_fail;
  cudaStreamDestroy(stream);
  stream = nullptr;

  err = cudaMemcpy(C, d_C, sz_c, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) goto matmult_fail;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;

matmult_fail:
  log_cuda_error("MatMult");
  if (stream) cudaStreamDestroy(stream);
  if (d_A) cudaFree(d_A);
  if (d_B) cudaFree(d_B);
  if (d_C) cudaFree(d_C);
  return 1;
}

int matrixgraph_relu(float* A, int m, int n) {
  set_device_from_env();
  size_t sz = static_cast<size_t>(m) * n * sizeof(float);
  float* d_A = nullptr;

  cudaError_t err = cudaMalloc(&d_A, sz);
  if (err != cudaSuccess) return cuda_err_to_int(err);

  err = cudaMemcpy(d_A, A, sz, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(d_A);
    return cuda_err_to_int(err);
  }

  cudaStream_t stream = nullptr;
  err = cudaStreamCreate(&stream);
  if (err != cudaSuccess) {
    cudaFree(d_A);
    return cuda_err_to_int(err);
  }

  kernel::MatrixOpsKernelWrapper::Relu(stream, d_A, m, n);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  err = cudaMemcpy(A, d_A, sz, cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  return cuda_err_to_int(err);
}

int matrixgraph_matadd(const float* A, float* B, int m, int n) {
  set_device_from_env();
  size_t sz = static_cast<size_t>(m) * n * sizeof(float);
  float* d_A = nullptr;
  float* d_B = nullptr;
  cudaStream_t stream = nullptr;

  cudaError_t err = cudaMalloc(&d_A, sz);
  if (err != cudaSuccess) return cuda_err_to_int(err);
  err = cudaMalloc(&d_B, sz);
  if (err != cudaSuccess) {
    cudaFree(d_A);
    return cuda_err_to_int(err);
  }

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
  set_device_from_env();
  size_t sz_a = static_cast<size_t>(m) * n * sizeof(float);
  size_t sz_b = static_cast<size_t>(n) * m * sizeof(float);
  float* d_A = nullptr;
  float* d_B = nullptr;
  cudaStream_t stream = nullptr;

  cudaError_t err = cudaMalloc(&d_A, sz_a);
  if (err != cudaSuccess) return cuda_err_to_int(err);
  err = cudaMalloc(&d_B, sz_b);
  if (err != cudaSuccess) {
    cudaFree(d_A);
    return cuda_err_to_int(err);
  }

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

int matrixgraph_gar_match(
    const uint32_t* g_v_id, const int32_t* g_v_label_idx, int g_n_vertices,
    const uint32_t* g_e_src, const uint32_t* g_e_dst, const uint32_t* g_e_id,
    const int32_t* g_e_label_idx, int g_n_edges,
    const int32_t* p_node_label_idx, int p_n_nodes, const int32_t* p_edge_src,
    const int32_t* p_edge_dst, const int32_t* p_edge_label_idx, int p_n_edges,
    int* out_num_conditions, uint32_t* out_row_pivot_id,
    int32_t* out_row_cond_j, int32_t* out_row_pos, int32_t* out_row_offset,
    int32_t* out_row_count, int out_row_capacity, int* out_row_size,
    uint32_t* out_matched_v_ids, int out_match_capacity, int* out_match_size) {
  // Route to GARMatch::SubIso.
  return task::GARMatch::SubIso(
      g_v_id, g_v_label_idx, g_n_vertices, g_e_src, g_e_dst, g_e_id,
      g_e_label_idx, g_n_edges, p_node_label_idx, p_n_nodes, p_edge_src,
      p_edge_dst, p_edge_label_idx, p_n_edges, out_num_conditions,
      out_row_pivot_id, out_row_cond_j, out_row_pos, out_row_offset,
      out_row_count, out_row_capacity, out_row_size, out_matched_v_ids,
      out_match_capacity, out_match_size);
}

}  // extern "C"
