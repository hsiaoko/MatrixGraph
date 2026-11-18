#ifndef MATRIXGRAPH_CORE_GO_API_MATRIXGRAPH_GO_API_H_
#define MATRIXGRAPH_CORE_GO_API_MATRIXGRAPH_GO_API_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file matrixgraph_go_api.h
 * @brief C API exported for Go CGO and other foreign-language callers.
 *
 * Self-contained on purpose: this header is an ABI contract, so it includes
 * nothing from the C++ library. The wire structs below are frozen -- field
 * order, widths and the MG_VALUE_* numbering must not change without a new
 * API version, because the Go side lays these out itself.
 *
 * All pointers passed to these functions are host memory unless otherwise
 * noted. Functions return 0 on success and non-zero on error.
 */

#include <stdint.h>

/* ------------------------------------------------------------------------- */
/* Wire value representation                                                  */
/* ------------------------------------------------------------------------- */

/**
 * @brief Value type tags carried in MatrixGraphFeatureValue::type.
 *
 * The numbering is the wire contract and is unrelated to the library's
 * internal ValueType enum; the C API converts at the boundary.
 */
enum MatrixGraphValueType {
  MG_VALUE_INVALID = 0,
  MG_VALUE_INT = 1,
  MG_VALUE_FLOAT64 = 2,
  MG_VALUE_BOOL = 3,
  MG_VALUE_STRING = 4,
  MG_VALUE_TIME = 5,
  MG_VALUE_FLOAT32 = 6,
};

/**
 * @brief A typed feature value (16 bytes).
 *
 * Discriminated union: only the member matching `type` is valid. The members
 * share one 8-byte slot, so assigning several of them in sequence overwrites
 * the earlier ones.
 */
struct MatrixGraphFeatureValue {
  int32_t type; /**< MatrixGraphValueType. */
  union {
    int64_t i64; /**< Valid when type == MG_VALUE_INT or MG_VALUE_TIME. */
    double f64; /**< Valid when type == MG_VALUE_FLOAT64 or MG_VALUE_FLOAT32. */
    int32_t b;  /**< Valid when type == MG_VALUE_BOOL. */
  };
};

#ifdef __cplusplus
static_assert(sizeof(MatrixGraphFeatureValue) == 16,
              "MatrixGraphFeatureValue is part of the C ABI");
#endif

/**
 * @brief Aggregation primitive ids for the feature APIs.
 *
 * The numbering is the featurelib order (count=0, sum=1, mean=2, ...): the
 * external Go library emits these constants, and the library's internal
 * AggPrim uses the same order, so the C API passes them through unchanged.
 */
enum MatrixGraphAggPrim {
  MG_AGG_COUNT = 0,
  MG_AGG_SUM = 1,
  MG_AGG_MEAN = 2,
  MG_AGG_MEDIAN = 3,
  MG_AGG_MODE = 4,
  MG_AGG_MAX = 5,
  MG_AGG_MIN = 6,
  MG_AGG_VARIANCE = 7,
  MG_AGG_STD = 8,
  MG_AGG_SKEW = 9,
  MG_AGG_ENTROPY = 10,
  MG_AGG_NUM_UNIQUE = 11,
  MG_AGG_PERCENT_TRUE = 12,
  MG_AGG_QUARTER = 13,
  MG_AGG_QUARTILE3 = 14,
  MG_AGG_COUNT_GREATER_THAN_MEAN = 15,
  MG_AGG_DFEAT = 16,
};

/* ------------------------------------------------------------------------- */
/* GraphAggregate: per-pivot aggregation over neighbor attributes             */
/* ------------------------------------------------------------------------- */

/**
 * @brief One aggregation request.
 *
 * attr_name is the per-vertex attribute key, NUL-terminated. neighbor_label
 * filters edges by label (0 = any). use_outgoing selects the adjacency.
 */
typedef struct {
  char attr_name[64];
  uint32_t neighbor_label;
  uint8_t use_outgoing;
  int32_t prim; /**< MatrixGraphAggPrim. */
} MatrixGraphFeatureRequest;

/** Create a GraphAggregate handle. Returns NULL on failure. */
void* matrixgraph_graph_aggregate_create(void);

/** Destroy a handle returned by matrixgraph_graph_aggregate_create(). */
void matrixgraph_graph_aggregate_destroy(void* handle);

/** Load a synthetic ring graph with per-vertex "score" and "flag" columns. */
int matrixgraph_graph_aggregate_load_synthetic(void* handle,
                                               uint32_t n_vertices,
                                               uint32_t out_degree);

/**
 * @brief Set the number of CUDA streams used per GPU.
 *
 * Must be called before the first compute. n_streams = 0 keeps the default
 * (MATRIXGRAPH_CUDA_STREAMS, or 2).
 */
int matrixgraph_graph_aggregate_set_num_streams(void* handle,
                                                uint32_t n_streams);

/**
 * @brief Compute one feature per (pivot, request) pair.
 *
 * out_values must hold n_pivots * n_requests entries and is written
 * row-major: out_values[i * n_requests + j] belongs to pivot i, request j.
 */
int matrixgraph_graph_aggregate_compute_features(
    void* handle, const uint32_t* pivot_vertex_ids, uint32_t n_pivots,
    const MatrixGraphFeatureRequest* requests, uint32_t n_requests,
    MatrixGraphFeatureValue* out_values);

/**
 * @brief Output of the fused compute-all kernel.
 *
 * Mirrors the library's AllFeatures struct field for field, so callers can
 * read a whole feature vector from one launch.
 */
typedef struct {
  MatrixGraphFeatureValue count;
  MatrixGraphFeatureValue sum;
  MatrixGraphFeatureValue mean;
  MatrixGraphFeatureValue median;
  MatrixGraphFeatureValue mode;
  MatrixGraphFeatureValue max;
  MatrixGraphFeatureValue min;
  MatrixGraphFeatureValue variance;
  MatrixGraphFeatureValue std;
  MatrixGraphFeatureValue skew;
  MatrixGraphFeatureValue entropy;
  MatrixGraphFeatureValue num_unique;
  MatrixGraphFeatureValue percent_true;
  MatrixGraphFeatureValue quarter;
  MatrixGraphFeatureValue quartile3;
  MatrixGraphFeatureValue count_greater_than_mean;
  MatrixGraphFeatureValue dfeat;
} MatrixGraphAllFeatures;

/**
 * @brief Compute every primitive for every pivot in one launch.
 *
 * attr_name is a NUL-terminated attribute key; out_values must hold n_pivots
 * entries.
 */
int matrixgraph_graph_aggregate_compute_all(void* handle,
                                            const uint32_t* pivot_vertex_ids,
                                            uint32_t n_pivots,
                                            const char* attr_name,
                                            uint8_t use_outgoing,
                                            MatrixGraphAllFeatures* out_values);

/* ------------------------------------------------------------------------- */
/* ExecuteAggPrim: primitives over a caller-supplied value list               */
/* ------------------------------------------------------------------------- */

/**
 * @brief Primitive ids for the value-list API.
 *
 * Same featurelib numbering as MatrixGraphAggPrim; kept as a separate enum so
 * the two surfaces can diverge in a future version without ambiguity.
 */
enum MatrixGraphExecuteAggPrim {
  MG_EXEC_AGG_COUNT = 0,
  MG_EXEC_AGG_SUM = 1,
  MG_EXEC_AGG_MEAN = 2,
  MG_EXEC_AGG_MEDIAN = 3,
  MG_EXEC_AGG_MODE = 4,
  MG_EXEC_AGG_MAX = 5,
  MG_EXEC_AGG_MIN = 6,
  MG_EXEC_AGG_VARIANCE = 7,
  MG_EXEC_AGG_STD = 8,
  MG_EXEC_AGG_SKEW = 9,
  MG_EXEC_AGG_ENTROPY = 10,
  MG_EXEC_AGG_NUM_UNIQUE = 11,
  MG_EXEC_AGG_PERCENT_TRUE = 12,
  MG_EXEC_AGG_QUARTER = 13,
  MG_EXEC_AGG_QUARTILE3 = 14,
  MG_EXEC_AGG_COUNT_GREATER_THAN_MEAN = 15,
  MG_EXEC_AGG_DFEAT = 16,
};

/** Output of the value-list compute-all call (same field set as above). */
typedef struct {
  MatrixGraphFeatureValue count;
  MatrixGraphFeatureValue sum;
  MatrixGraphFeatureValue mean;
  MatrixGraphFeatureValue median;
  MatrixGraphFeatureValue mode;
  MatrixGraphFeatureValue max;
  MatrixGraphFeatureValue min;
  MatrixGraphFeatureValue variance;
  MatrixGraphFeatureValue std;
  MatrixGraphFeatureValue skew;
  MatrixGraphFeatureValue entropy;
  MatrixGraphFeatureValue num_unique;
  MatrixGraphFeatureValue percent_true;
  MatrixGraphFeatureValue quarter;
  MatrixGraphFeatureValue quartile3;
  MatrixGraphFeatureValue count_greater_than_mean;
  MatrixGraphFeatureValue dfeat;
} MatrixGraphExecuteAllFeatures;

/** Create an ExecuteAggPrim handle. */
void* matrixgraph_execute_agg_prim_create(void);

/** Destroy a handle returned by matrixgraph_execute_agg_prim_create(). */
void matrixgraph_execute_agg_prim_destroy(void* handle);

/** Set CUDA streams per GPU; n_streams = 0 keeps the default. */
int matrixgraph_execute_agg_prim_set_num_streams(void* handle,
                                                 uint32_t n_streams);

/**
 * @brief Apply one primitive to one value list.
 *
 * prim is a MatrixGraphExecuteAggPrim id; values holds n entries.
 */
int matrixgraph_execute_agg_prim_compute(void* handle, int32_t prim,
                                         const MatrixGraphFeatureValue* values,
                                         uint32_t n,
                                         MatrixGraphFeatureValue* out);

/** Apply every primitive to one value list. */
int matrixgraph_execute_agg_prim_compute_all(
    void* handle, const MatrixGraphFeatureValue* values, uint32_t n,
    MatrixGraphExecuteAllFeatures* out);

/**
 * @brief Apply one primitive to a batch of lists.
 *
 * flat_values holds all entries concatenated; offsets has n_lists + 1 entries
 * and list i spans flat_values[offsets[i] .. offsets[i + 1]). out has n_lists
 * entries.
 */
int matrixgraph_execute_agg_prim_compute_batch(
    void* handle, int32_t prim, const MatrixGraphFeatureValue* flat_values,
    const uint32_t* offsets, uint32_t n_lists, MatrixGraphFeatureValue* out);

/**
 * @brief Apply several primitives to a batch of lists.
 *
 * prims holds n_primitives ids. The output is row-major:
 * out[i * n_primitives + j] is primitive j applied to list i.
 */
int matrixgraph_execute_agg_prim_compute_batch_multi_prim(
    void* handle, const MatrixGraphFeatureValue* flat_values,
    const uint32_t* offsets, uint32_t n_lists, const int32_t* prims,
    uint32_t n_primitives, MatrixGraphFeatureValue* out);

/** Apply every primitive to a batch of lists; out has n_lists entries. */
int matrixgraph_execute_agg_prim_compute_all_batch(
    void* handle, const MatrixGraphFeatureValue* flat_values,
    const uint32_t* offsets, uint32_t n_lists,
    MatrixGraphExecuteAllFeatures* out);

/* ------------------------------------------------------------------------- */
/* Matrix helpers                                                            */
/* ------------------------------------------------------------------------- */

/** C = A * B (row-major). A: m x k, B: k x n, C: m x n. */
int matrixgraph_matmult(const float* A, const float* B, float* C, int m, int k,
                        int n);

/** ReLU in place on A (m x n). */
int matrixgraph_relu(float* A, int m, int n);

/** B = A + B in place (m x n). */
int matrixgraph_matadd(const float* A, float* B, int m, int n);

/** B = A^T. A: m x n, B: n x m. */
int matrixgraph_transpose(const float* A, float* B, int m, int n);

/* ------------------------------------------------------------------------- */
/* Serialized-graph entry points                                             */
/* ------------------------------------------------------------------------- */

/**
 * @brief Serialized-graph GAR match.
 *
 * Both graphs arrive as flat arrays; the match rows and matched vertex ids are
 * written into caller-provided arrays.
 */
int matrixgraph_gar_match(
    const uint32_t* g_v_id, const int32_t* g_v_label_idx, int g_n_vertices,
    const uint32_t* g_e_src, const uint32_t* g_e_dst, const uint32_t* g_e_id,
    const int32_t* g_e_label_idx, int g_n_edges,
    const int32_t* p_node_label_idx, int p_n_nodes, const int32_t* p_edge_src,
    const int32_t* p_edge_dst, const int32_t* p_edge_label_idx, int p_n_edges,
    int* out_num_conditions, uint32_t* out_row_pivot_id,
    int32_t* out_row_cond_j, int32_t* out_row_pos, int32_t* out_row_offset,
    int32_t* out_row_count, int out_row_capacity, int* out_row_size,
    uint32_t* out_matched_v_ids, int out_match_capacity, int* out_match_size);

/**
 * @brief Serialized-graph subgraph isomorphism.
 *
 * Both graphs arrive as CSR buffers plus per-vertex labels; the result tables
 * are written into caller-provided arrays. See docs/go_api/ for the layout.
 */
int matrixgraph_subiso(
    uint32_t p_num_vertices, uint32_t p_num_in_edges, uint32_t p_num_out_edges,
    uint32_t p_max_vid, uint32_t p_min_vid, const uint8_t* p_csr_data,
    uint64_t p_csr_data_size, const uint32_t* p_labels, uint32_t g_num_vertices,
    uint32_t g_num_in_edges, uint32_t g_num_out_edges, uint32_t g_max_vid,
    uint32_t g_min_vid, const uint8_t* g_csr_data, uint64_t g_csr_data_size,
    const uint32_t* g_labels, int max_result_tables, int max_result_rows,
    int max_result_cols, uint32_t* out_table_cols, uint32_t* out_table_rows,
    uint32_t* out_headers_flat, uint32_t* out_data_flat, int* out_num_tables);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MATRIXGRAPH_CORE_GO_API_MATRIXGRAPH_GO_API_H_ */