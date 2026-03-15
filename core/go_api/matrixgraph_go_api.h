#ifndef MATRIXGRAPH_GO_API_H_
#define MATRIXGRAPH_GO_API_H_

#ifdef __cplusplus
extern "C" {
#endif

/** C API for Go CGO: all pointers are host memory. Returns 0 on success, non-zero on error. */

/** C = A * B (row-major). A: m×k, B: k×n, C: m×n. */
int matrixgraph_matmult(const float* A, const float* B, float* C, int m, int k, int n);

/** ReLU in-place on A (m×n). */
int matrixgraph_relu(float* A, int m, int n);

/** B = A + B in-place (m×n). */
int matrixgraph_matadd(const float* A, float* B, int m, int n);

/** B = A^T. A: m×n, B: n×m. */
int matrixgraph_transpose(const float* A, float* B, int m, int n);

#ifdef __cplusplus
}
#endif

#endif /* MATRIXGRAPH_GO_API_H_ */
