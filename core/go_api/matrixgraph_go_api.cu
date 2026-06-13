#include "go_api/matrixgraph_go_api.h"
#include "task/gpu_task/compute_features.cuh"
#include "task/gpu_task/gar_match.cuh"
#include "task/gpu_task/graph_aggregate.cuh"
#include "task/gpu_task/subiso.cuh"
#include "task/gpu_task/kernel/kernel_matrix_ops.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

namespace kernel = sics::matrixgraph::core::task::kernel;
namespace task = sics::matrixgraph::core::task;
using GraphAggregate = sics::matrixgraph::core::task::GraphAggregate;
using ComputeFeaturesTask = sics::matrixgraph::core::task::ComputeFeaturesTask;
using FeatureRequest = sics::matrixgraph::core::task::kernel::FeatureRequest;
using FeatureValue = sics::matrixgraph::core::task::kernel::FeatureValue;
using AllFeatures = sics::matrixgraph::core::task::kernel::AllFeatures;
using AggPrim = sics::matrixgraph::core::task::kernel::AggPrim;
using AttributeName = sics::matrixgraph::core::data_structures::AttributeName;
using SubIso = sics::matrixgraph::core::task::SubIso;

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
  return task::GARMatch::Run(
      g_v_id, g_v_label_idx, g_n_vertices, g_e_src, g_e_dst, g_e_id,
      g_e_label_idx, g_n_edges, p_node_label_idx, p_n_nodes, p_edge_src,
      p_edge_dst, p_edge_label_idx, p_n_edges, out_num_conditions,
      out_row_pivot_id, out_row_cond_j, out_row_pos, out_row_offset,
      out_row_count, out_row_capacity, out_row_size, out_matched_v_ids,
      out_match_capacity, out_match_size);
}

// ---------------------------------------------------------------------------
// GraphAggregate C API
// ---------------------------------------------------------------------------

void* matrixgraph_graph_aggregate_create(void) {
  set_device_from_env();
  try {
    return new GraphAggregate();
  } catch (...) {
    return nullptr;
  }
}

void matrixgraph_graph_aggregate_destroy(void* handle) {
  auto* task = static_cast<GraphAggregate*>(handle);
  delete task;
}

int matrixgraph_graph_aggregate_load_synthetic(void* handle,
                                               uint32_t n_vertices,
                                               uint32_t out_degree) {
  set_device_from_env();
  auto* task = static_cast<GraphAggregate*>(handle);
  if (!task) return 1;
  try {
    task->LoadSyntheticData(n_vertices, out_degree);
  } catch (...) {
    return 1;
  }
  return 0;
}

int matrixgraph_graph_aggregate_set_num_streams(void* handle,
                                                uint32_t n_streams) {
  set_device_from_env();
  auto* task = static_cast<GraphAggregate*>(handle);
  if (!task) return 1;
  try {
    task->SetNumStreams(n_streams);
  } catch (...) {
    return 1;
  }
  return 0;
}

int matrixgraph_graph_aggregate_compute_features(
    void* handle, const uint32_t* pivot_vertex_ids, uint32_t n_pivots,
    const MatrixGraphFeatureRequest* requests, uint32_t n_requests,
    MatrixGraphFeatureValue* out_values) {
  set_device_from_env();
  auto* task = static_cast<GraphAggregate*>(handle);
  if (!task || !pivot_vertex_ids || !requests || !out_values) {
    return 1;
  }

  std::vector<uint32_t> vids(pivot_vertex_ids, pivot_vertex_ids + n_pivots);

  std::vector<FeatureRequest> cpp_reqs;
  cpp_reqs.reserve(n_requests);
  for (uint32_t i = 0; i < n_requests; ++i) {
    FeatureRequest r;
    std::memcpy(r.attr_name.data, requests[i].attr_name,
                sizeof(r.attr_name.data));
    r.neighbor_label = requests[i].neighbor_label;
    r.use_outgoing = requests[i].use_outgoing != 0;
    r.prim = static_cast<AggPrim>(requests[i].prim);
    cpp_reqs.push_back(r);
  }

  std::vector<FeatureValue> cpp_out;
  try {
    cpp_out = task->ComputeFeatures(vids, cpp_reqs);
  } catch (...) {
    return 1;
  }

  if (cpp_out.size() != static_cast<size_t>(n_pivots) * n_requests) {
    return 1;
  }

  for (size_t i = 0; i < cpp_out.size(); ++i) {
    out_values[i].type = static_cast<int32_t>(cpp_out[i].type);
    out_values[i].i64 = cpp_out[i].i64;
    out_values[i].f64 = cpp_out[i].f64;
    out_values[i].b = cpp_out[i].b ? 1 : 0;
  }
  return 0;
}

namespace {

inline void CopyFeatureValueToC(const FeatureValue& src,
                                MatrixGraphFeatureValue* dst) {
  dst->type = static_cast<int32_t>(src.type);
  dst->i64 = src.i64;
  dst->f64 = src.f64;
  dst->b = src.b ? 1 : 0;
}

inline void CopyAllFeaturesToC(const AllFeatures& src,
                               MatrixGraphAllFeatures* dst) {
  CopyFeatureValueToC(src.count, &dst->count);
  CopyFeatureValueToC(src.count_greater_than_mean, &dst->count_greater_than_mean);
  CopyFeatureValueToC(src.num_unique, &dst->num_unique);
  CopyFeatureValueToC(src.sum, &dst->sum);
  CopyFeatureValueToC(src.mean, &dst->mean);
  CopyFeatureValueToC(src.variance, &dst->variance);
  CopyFeatureValueToC(src.std, &dst->std);
  CopyFeatureValueToC(src.mode, &dst->mode);
  CopyFeatureValueToC(src.min, &dst->min);
  CopyFeatureValueToC(src.max, &dst->max);
  CopyFeatureValueToC(src.median, &dst->median);
  CopyFeatureValueToC(src.quarter, &dst->quarter);
  CopyFeatureValueToC(src.quartile3, &dst->quartile3);
  CopyFeatureValueToC(src.entropy, &dst->entropy);
  CopyFeatureValueToC(src.percent_true, &dst->percent_true);
  CopyFeatureValueToC(src.skew, &dst->skew);
}

}  // namespace

int matrixgraph_graph_aggregate_compute_all(
    void* handle, const uint32_t* pivot_vertex_ids, uint32_t n_pivots,
    const char* attr_name, uint8_t use_outgoing,
    MatrixGraphAllFeatures* out_values) {
  set_device_from_env();
  auto* task = static_cast<GraphAggregate*>(handle);
  if (!task || !pivot_vertex_ids || !attr_name || !out_values) {
    return 1;
  }

  std::vector<uint32_t> vids(pivot_vertex_ids, pivot_vertex_ids + n_pivots);

  std::vector<AllFeatures> cpp_out;
  try {
    cpp_out = task->ComputeAll(vids, AttributeName(attr_name),
                               use_outgoing != 0);
  } catch (...) {
    return 1;
  }

  if (cpp_out.size() != n_pivots) {
    return 1;
  }

  for (size_t i = 0; i < cpp_out.size(); ++i) {
    CopyAllFeaturesToC(cpp_out[i], &out_values[i]);
  }
  return 0;
}

// ---------------------------------------------------------------------------
// ComputeFeatures C API
// ---------------------------------------------------------------------------

/**
 * @brief C API wrapper for ComputeFeaturesTask construction.
 *
 * Allocates a task with an empty graph path.  The caller must load a graph
 * before computing features.
 */
void* matrixgraph_compute_features_create(void) {
  set_device_from_env();
  try {
    return new ComputeFeaturesTask("");
  } catch (...) {
    return nullptr;
  }
}

/**
 * @brief C API wrapper for ComputeFeaturesTask destruction.
 */
void matrixgraph_compute_features_destroy(void* handle) {
  auto* task = static_cast<ComputeFeaturesTask*>(handle);
  delete task;
}

/**
 * @brief C API wrapper for ComputeFeaturesTask::LoadGraph().
 */
int matrixgraph_compute_features_load_graph(void* handle,
                                            const char* graph_path) {
  set_device_from_env();
  auto* task = static_cast<ComputeFeaturesTask*>(handle);
  if (!task || !graph_path) return 1;
  try {
    task->LoadGraph(graph_path);
  } catch (...) {
    return 1;
  }
  return 0;
}

/**
 * @brief C API wrapper for ComputeFeaturesTask::LoadAttributes().
 */
int matrixgraph_compute_features_load_attributes(
    void* handle, uint32_t n_columns,
    const ComputeFeaturesAttributeColumn* columns) {
  set_device_from_env();
  auto* task = static_cast<ComputeFeaturesTask*>(handle);
  if (!task || (n_columns > 0 && !columns)) return 1;
  try {
    task->LoadAttributes(n_columns, columns);
  } catch (...) {
    return 1;
  }
  return 0;
}

/**
 * @brief C API wrapper for ComputeFeaturesTask::LoadLabels().
 */
int matrixgraph_compute_features_load_labels(void* handle,
                                             const uint32_t* labels,
                                             uint32_t n) {
  set_device_from_env();
  auto* task = static_cast<ComputeFeaturesTask*>(handle);
  if (!task || (n > 0 && !labels)) return 1;
  try {
    task->LoadLabels(labels, n);
  } catch (...) {
    return 1;
  }
  return 0;
}

/**
 * @brief C API wrapper for ComputeFeaturesTask::Compute().
 *
 * Copies the plan, navigators, conditions and pivots into std::vectors, runs
 * the computation, and copies the results into the caller-provided
 * @p out_values buffer.
 */
int matrixgraph_compute_features_compute(
    void* handle, const uint32_t* pivot_vertex_ids, uint32_t n_pivots,
    const MatrixGraphPlanNode* plan, uint32_t n_plan_nodes,
    const MatrixGraphPlanNode* navs, uint32_t n_navs,
    const MatrixGraphCondNode* conds, uint32_t n_conds,
    const int32_t* output_expr_indices, uint32_t n_outputs,
    MatrixGraphFeatureValue* out_values) {
  set_device_from_env();
  auto* task = static_cast<ComputeFeaturesTask*>(handle);
  if (!task || !pivot_vertex_ids || !plan || !output_expr_indices ||
      !out_values) {
    return 1;
  }

  std::vector<uint32_t> vids(pivot_vertex_ids, pivot_vertex_ids + n_pivots);
  std::vector<MatrixGraphPlanNode> cpp_plan(plan, plan + n_plan_nodes);
  std::vector<MatrixGraphPlanNode> cpp_navs;
  if (navs != nullptr) {
    cpp_navs.assign(navs, navs + n_navs);
  }
  std::vector<int32_t> cpp_outputs(output_expr_indices,
                                   output_expr_indices + n_outputs);
  std::vector<MatrixGraphCondNode> cpp_conds;
  if (conds != nullptr && n_conds > 0) {
    cpp_conds.assign(conds, conds + n_conds);
  }

  std::vector<MatrixGraphFeatureValue> cpp_out;
  try {
    cpp_out = task->Compute(vids, cpp_plan, cpp_navs, cpp_conds, cpp_outputs);
  } catch (...) {
    return 1;
  }

  if (cpp_out.size() != static_cast<size_t>(n_pivots) * n_outputs) {
    return 1;
  }
  std::memcpy(out_values, cpp_out.data(),
              sizeof(MatrixGraphFeatureValue) * cpp_out.size());
  return 0;
}

int matrixgraph_subiso(
    uint32_t p_num_vertices, uint32_t p_num_in_edges, uint32_t p_num_out_edges,
    uint32_t p_max_vid, uint32_t p_min_vid, const uint8_t* p_csr_data,
    uint64_t p_csr_data_size, const uint32_t* p_labels,
    uint32_t g_num_vertices, uint32_t g_num_in_edges, uint32_t g_num_out_edges,
    uint32_t g_max_vid, uint32_t g_min_vid, const uint8_t* g_csr_data,
    uint64_t g_csr_data_size, const uint32_t* g_labels,
    int max_result_tables, int max_result_rows, int max_result_cols,
    uint32_t* out_table_cols, uint32_t* out_table_rows,
    uint32_t* out_headers_flat, uint32_t* out_data_flat,
    int* out_num_tables) {
  set_device_from_env();
  return SubIso::Run(p_num_vertices, p_num_in_edges, p_num_out_edges,
                     p_max_vid, p_min_vid, p_csr_data, p_csr_data_size,
                     p_labels, g_num_vertices, g_num_in_edges,
                     g_num_out_edges, g_max_vid, g_min_vid, g_csr_data,
                     g_csr_data_size, g_labels, max_result_tables,
                     max_result_rows, max_result_cols, out_table_cols,
                     out_table_rows, out_headers_flat, out_data_flat,
                     out_num_tables);
}

}  // extern "C"
