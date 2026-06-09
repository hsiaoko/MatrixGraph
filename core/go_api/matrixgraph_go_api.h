#ifndef MATRIXGRAPH_GO_API_H_
#define MATRIXGRAPH_GO_API_H_

#ifdef __cplusplus
extern "C" {
#endif

/** C API for Go CGO: all pointers are host memory. Returns 0 on success, non-zero on error. */
#include <stdint.h>

// ---------------------------------------------------------------------------
// GraphAggregate feature computation
// ---------------------------------------------------------------------------

typedef struct {
  char attr_name[64];
  uint32_t neighbor_label;
  uint8_t use_outgoing;
  int32_t prim;  // AggPrim enum value
} MatrixGraphFeatureRequest;

typedef struct {
  int32_t type;  // ValueType
  union {
    int64_t i64;
    double f64;
    uint8_t b;
  };
} MatrixGraphFeatureValue;

/** Create a GraphAggregate handle. Returns opaque pointer (or NULL on failure). */
void* matrixgraph_graph_aggregate_create(void);

/** Destroy a GraphAggregate handle. */
void matrixgraph_graph_aggregate_destroy(void* handle);

/** Load synthetic ring graph with per-vertex attributes "score" and "flag". */
int matrixgraph_graph_aggregate_load_synthetic(void* handle, uint32_t n_vertices,
                                                 uint32_t out_degree);

/** Compute features for given pivots and requests.
 *  out_values must be pre-allocated with size n_pivots * n_requests.
 */
int matrixgraph_graph_aggregate_compute_features(
    void* handle, const uint32_t* pivot_graph_ids,
    const uint32_t* pivot_vertex_ids, uint32_t n_pivots,
    const MatrixGraphFeatureRequest* requests, uint32_t n_requests,
    MatrixGraphFeatureValue* out_values);

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
