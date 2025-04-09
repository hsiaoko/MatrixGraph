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
  void Matmult(float* A, float* B, float* C, int m, int n, int k);

  // @Description: Applies ReLU activation in-place on a GPU array
  // @Parameters:
  //   - A: pointer to matrix A (row-major).
  //   - n: number of elements of A
  // @Notes: formula: A[i] = max(0, A[i])
  void Activate(float* A, int n);

  void cuBLASRelu(float* A, size_t n);
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // USE_MATRIXGRAPH_MatrixOps_CUH
