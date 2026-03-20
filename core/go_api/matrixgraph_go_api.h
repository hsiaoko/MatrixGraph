#ifndef MATRIXGRAPH_GO_API_H_
#define MATRIXGRAPH_GO_API_H_

#ifdef __cplusplus
extern "C" {
#endif

/** C API for Go CGO: all pointers are host memory. Returns 0 on success, non-zero on error. */
#include <stdint.h>

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

#ifdef __cplusplus
}
#endif

#endif /* MATRIXGRAPH_GO_API_H_ */
