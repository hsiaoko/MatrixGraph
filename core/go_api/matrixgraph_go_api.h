#ifndef MATRIXGRAPH_GO_API_H_
#define MATRIXGRAPH_GO_API_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file matrixgraph_go_api.h
 * @brief C API exported for Go CGO and other foreign-language callers.
 *
 * All pointers passed to these functions are host memory unless otherwise
 * noted.  Functions generally return 0 on success and non-zero on error.
 */
#include <stdint.h>
#include "core/task/gpu_task/compute_features_types.h"

// ---------------------------------------------------------------------------
// GraphAggregate feature computation
// ---------------------------------------------------------------------------

typedef struct {
  char attr_name[64];
  uint32_t neighbor_label;
  uint8_t use_outgoing;
  int32_t prim;  // AggPrim enum value
} MatrixGraphFeatureRequest;

/** Create a GraphAggregate handle. Returns opaque pointer (or NULL on failure). */
void* matrixgraph_graph_aggregate_create(void);

/** Destroy a GraphAggregate handle. */
void matrixgraph_graph_aggregate_destroy(void* handle);

/** Load synthetic ring graph with per-vertex attributes "score" and "flag". */
int matrixgraph_graph_aggregate_load_synthetic(void* handle, uint32_t n_vertices,
                                                 uint32_t out_degree);

/** Configure the number of CUDA streams used for compute parallelism.
 *  Must be called before the first compute call.  n_streams=0 leaves the
 *  default (MATRIXGRAPH_CUDA_STREAMS env, or 2).
 */
int matrixgraph_graph_aggregate_set_num_streams(void* handle,
                                                  uint32_t n_streams);

/** Compute features for given pivots and requests.
 *  out_values must be pre-allocated with size n_pivots * n_requests.
 */
int matrixgraph_graph_aggregate_compute_features(
    void* handle, const uint32_t* pivot_vertex_ids, uint32_t n_pivots,
    const MatrixGraphFeatureRequest* requests, uint32_t n_requests,
    MatrixGraphFeatureValue* out_values);

/** Output container for the fused compute-all kernel.
 *  Mirrors kernel::AllFeatures field-for-field.
 */
typedef struct {
  MatrixGraphFeatureValue count;
  MatrixGraphFeatureValue count_greater_than_mean;
  MatrixGraphFeatureValue num_unique;
  MatrixGraphFeatureValue sum;
  MatrixGraphFeatureValue mean;
  MatrixGraphFeatureValue variance;
  MatrixGraphFeatureValue std;
  MatrixGraphFeatureValue mode;
  MatrixGraphFeatureValue min;
  MatrixGraphFeatureValue max;
  MatrixGraphFeatureValue median;
  MatrixGraphFeatureValue quarter;
  MatrixGraphFeatureValue quartile3;
  MatrixGraphFeatureValue entropy;
  MatrixGraphFeatureValue percent_true;
  MatrixGraphFeatureValue skew;
} MatrixGraphAllFeatures;

/** Fused compute-all: produce every aggregation primitive for each pivot
 *  in a single kernel launch.  out_values must be pre-allocated with size
 *  n_pivots.  attr_name is a null-terminated attribute name.
 */
int matrixgraph_graph_aggregate_compute_all(
    void* handle, const uint32_t* pivot_vertex_ids, uint32_t n_pivots,
    const char* attr_name, uint8_t use_outgoing,
    MatrixGraphAllFeatures* out_values);

/** C = A * B (row-major). A: m×k, B: k×n, C: m×n. */
int matrixgraph_matmult(const float* A, const float* B, float* C, int m, int k, int n);

/** ReLU in-place on A (m×n). */
int matrixgraph_relu(float* A, int m, int n);

/** B = A + B in-place (m×n). */
int matrixgraph_matadd(const float* A, float* B, int m, int n);

/** B = A^T. A: m×n, B: n×m. */
int matrixgraph_transpose(const float* A, float* B, int m, int n);

/**
 * GAR match placeholder API.
 * Input: serialized graph g arrays + serialized pattern p arrays.
 * Output: flattened match arrays.
 *
 * NOTE: current implementation is a stub and returns empty output.
 */
int matrixgraph_gar_match(
    const uint32_t* g_v_id,
    const int32_t* g_v_label_idx,
    int g_n_vertices,
    const uint32_t* g_e_src,
    const uint32_t* g_e_dst,
    const uint32_t* g_e_id,
    const int32_t* g_e_label_idx,
    int g_n_edges,
    const int32_t* p_node_label_idx,
    int p_n_nodes,
    const int32_t* p_edge_src,
    const int32_t* p_edge_dst,
    const int32_t* p_edge_label_idx,
    int p_n_edges,
    int* out_num_conditions,
    uint32_t* out_row_pivot_id,
    int32_t* out_row_cond_j,
    int32_t* out_row_pos,
    int32_t* out_row_offset,
    int32_t* out_row_count,
    int out_row_capacity,
    int* out_row_size,
    uint32_t* out_matched_v_ids,
    int out_match_capacity,
    int* out_match_size);

/**
 * SubIso (GPU WOJ) matching from flat CSR buffers.
 *
 * CSR buffer layout (each field is uint32_t, contiguous):
 *   [global_id * n_vertices]
 *   [in_degree * n_vertices]
 *   [out_degree * n_vertices]
 *   [in_offset * (n_vertices+1)]
 *   [out_offset * (n_vertices+1)]
 *   [incoming_edges * n_in_edges]
 *   [outgoing_edges * n_out_edges]
 *   [edges_globalid * (max_vid+1)]
 *   [localid * (max_vid+1)]
 *
 * labels: uint32_t[n_vertices] vertex labels.
 *
 * Output buffers must be pre-allocated by caller:
 *   out_table_cols  [max_result_tables]
 *   out_table_rows  [max_result_tables]
 *   out_headers_flat[max_result_tables * max_result_cols]
 *   out_data_flat   [max_result_tables * max_result_rows * max_result_cols]
 */
// ---------------------------------------------------------------------------
// ComputeFeatures task C API
// ---------------------------------------------------------------------------

/**
 * @brief Create a ComputeFeaturesTask handle.
 *
 * The returned opaque pointer must be released with
 * matrixgraph_compute_features_destroy().  Returns NULL if the handle could
 * not be allocated.
 */
void* matrixgraph_compute_features_create(void);

/**
 * @brief Destroy a ComputeFeaturesTask handle and free all associated GPU
 *        memory.
 *
 * @param handle Handle returned by matrixgraph_compute_features_create().
 *               Passing NULL is safe and is a no-op.
 */
void matrixgraph_compute_features_destroy(void* handle);

/**
 * @brief Load a graph from a MatrixGraph CSR directory path.
 *
 * The path is expected to contain the files written by ImmutableCSR::Write().
 * Must be called before loading attributes or computing features.
 *
 * @return 0 on success, non-zero on error.
 */
int matrixgraph_compute_features_load_graph(void* handle, const char* graph_path);

/**
 * @brief Load columnar per-vertex attributes.
 *
 * Each column must have exactly one value per graph vertex.  Calls are
 * cumulative: attributes from earlier calls are preserved.  The `values`
 * pointers only need to remain valid for the duration of this call.
 *
 * @return 0 on success, non-zero on error.
 */
int matrixgraph_compute_features_load_attributes(
    void* handle, uint32_t n_columns,
    const ComputeFeaturesAttributeColumn* columns);

/**
 * @brief Load optional per-vertex labels.
 *
 * Labels enable label filtering in NeighborNav (nav.target_label) and will be
 * required by pattern navigators in future phases.
 *
 * @param labels uint32_t array of length @p n.
 * @param n      Number of vertices (must match the graph).
 * @return 0 on success, non-zero on error.
 */
int matrixgraph_compute_features_load_labels(void* handle,
                                             const uint32_t* labels,
                                             uint32_t n);

/**
 * @brief Evaluate the flat expression plan for the given pivots.
 *
 * All plan/navigator/condition/pivot data are copied to temporary device
 * buffers for this call.  The graph, attributes and labels loaded earlier are
 * reused.
 *
 * @param handle              Task handle.
 * @param pivot_vertex_ids    Array of pivot vertex ids (host memory).
 * @param n_pivots            Length of @p pivot_vertex_ids.
 * @param plan                Flat expression plan (host memory).
 * @param n_plan_nodes        Length of @p plan.
 * @param navs                Flat navigator plan (host memory).  May be NULL
 *                            if @p n_navs == 0.
 * @param n_navs              Length of @p navs.
 * @param conds               Flat condition array (host memory).  May be NULL
 *                            if @p n_conds == 0.
 * @param n_conds             Length of @p conds.
 * @param output_expr_indices Indices into @p plan that should be emitted.
 * @param n_outputs           Length of @p output_expr_indices.
 * @param out_values          Pre-allocated output buffer of size
 *                            n_pivots * n_outputs (host memory).
 * @return 0 on success, non-zero on error.
 */
int matrixgraph_compute_features_compute(
    void* handle, const uint32_t* pivot_vertex_ids, uint32_t n_pivots,
    const MatrixGraphPlanNode* plan, uint32_t n_plan_nodes,
    const MatrixGraphPlanNode* navs, uint32_t n_navs,
    const MatrixGraphCondNode* conds, uint32_t n_conds,
    const int32_t* output_expr_indices, uint32_t n_outputs,
    MatrixGraphFeatureValue* out_values);

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
    int* out_num_tables);

#ifdef __cplusplus
}
#endif

#endif /* MATRIXGRAPH_GO_API_H_ */
