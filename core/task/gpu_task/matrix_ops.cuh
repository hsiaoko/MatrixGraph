#ifndef MATRIXGRAPH_CORE_TASK_MatrixOps_CUH_
#define MATRIXGRAPH_CORE_TASK_MatrixOps_CUH_

#include "core/common/types.h"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/unified_buffer.cuh"
#include "core/task/gpu_task/kernel/kernel_matrix_ops.cuh"
#include "core/task/gpu_task/task_base.cuh"
#include <cublas_v2.h>
#include <string>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

class MatrixOps : public TaskBase {
 private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using GraphID = sics::matrixgraph::core::common::GraphID;
  using UnifiedOwnedBufferFloat =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<float>;

 public:
  MatrixOps() = default;

  __host__ void Run();

  template <typename T>
  __host__ void Matmult(const data_structures::UnifiedOwnedBuffer<T>& unified_a,
                        const data_structures::UnifiedOwnedBuffer<T>& unified_b,
                        data_structures::UnifiedOwnedBuffer<T>* unified_c,
                        int m, int n, int k) {}

  void cuBLASMatmult(float* A, float* B, float* C, int m, int n, int k,
                     bool transa_tag = false, bool transb_tag = false);

  /**
   * @brif: Optimized matrix multiplication kernel using shared memory
   *  tiling for GPU acceleration. Computes `C = A × B` where:
   *
   * - Input matrix `A`: shape `(m × k)`
   * - Input matrix `B`: shape `(k × n)`
   * - Output matrix `C`: shape `(m × n)`
   *
   * @pram:
   *    - A: pointer to matrix A (row-major).
   *    - B: pointer to matrix A (row-major).
   *    - C: pointer to output matrix C (row-major).
   *    - m: rows of A and C
   *    - k: columns of  A and rows of B
   *    - n: columns of B and C
   * @note: nput/Output Example
   *  **Matrix A (4×2):**
   *
   *   [0.0, 1.0,
   *   2.0, 3.0,
   *   4.0, 5.0,
   *   6.0, 7.0]
   *
   * **Matrix B (1×1):**
   *
   *  [-1.0,
   *  -1.0]
   *
   * **Output C (2×2):**
   *
   *  [-1.0,
   *  -5.0,
   *  -9.0,
   *  -13.0]
   */
  void MatMult(float* A, float* B, float* C, int m, int k, int n);

  /**
   * @brif: Applies ReLU activation in-place on a GPU array.
   * @pram:
   *    - A: pointer to matrix A (row-major).
   *    - m: number of row of A
   *    - n: number of column of A
   */
  void Activate(float* A, int m, int n);

  /**
   * @brif: Performs element-wise matrix addition on GPU.
   *        Computes `C = A + B` where `A` and `B` are matrices of shape `(m ×
   * n)`. The result is stored in-place in matrix `B`.
   *
   * @pram:
   *    - A:    Pointer to input matrix A (row-major, size `m×n`).
   *    - B:    Pointer to input/output matrix B (row-major, size `m×n`).
   *            After execution, contains `A + B`.
   *    - m:    Number of rows in matrices A and B.
   *    - n:    Number of columns in matrices A and B.
   *
   * @note: Input/Output Example
   *  **Matrix A (2×3):**
   *   [1.0, 2.0, 3.0,
   *    4.0, 5.0, 6.0]
   *
   *  **Matrix B (2×3):**
   *   [0.5, 1.5, 2.5,
   *    3.5, 4.5, 5.5]
   *
   *  **After Matadd(A, B, 2, 3):**
   *   [1.5, 3.5, 5.5,
   *    7.5, 9.5, 11.5] (stored in B)
   */
  void MatAdd(float* A, float* B, int m, int n);

  /**
   * @brif: Performs in-place matrix transposition on GPU.
   *        Computes B = A^T where A is a matrix of shape (m × n)
   *        and B will be of shape (n × m). Supports both square
   *        and rectangular matrices.
   *
   * @pram:
   *    - A:    Pointer to input matrix (row-major, size m×n).
   *    - B:    Pointer to output matrix (row-major, size n×m).
   *            After execution, contains transposed matrix A^T.
   *    - m:    Number of rows in input matrix A.
   *    - n:    Number of columns in input matrix A.
   *
   * @impl_note:
   *   - Uses 1 thread per matrix element
   *   - For square matrices (m == n), employs optimized
   *     diagonal-swapping algorithm
   *   - For rectangular matrices, uses coalesced memory access
   *     pattern to maximize bandwidth utilization
   *
   * @note: Input/Output Example
   *  **Matrix A (2×3):**
   *   [1.0, 2.0, 3.0,
   *    4.0, 5.0, 6.0]
   *
   *  **Output B (3×2):**
   *   [1.0, 4.0,
   *    2.0, 5.0,
   *    3.0, 6.0]
   */
  void Transpose(float* A, float* B, int m, int n);

  void cuBLASRelu(float* A, size_t n);
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // USE_MATRIXGRAPH_MatrixOps_CUH
