#ifndef MATRIXGRAPH_KERNEL_MatrixOps_CUH
#define MATRIXGRAPH_KERNEL_MatrixOps_CUH

#include <cuda_runtime.h>

#include <vector>

#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/unified_buffer.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

class MatrixOpsKernelWrapper {
 private:
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using UnifiedOwnedBufferVertexID =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexID>;
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;
  using Edges = sics::matrixgraph::core::data_structures::Edges;

 public:
  MatrixOpsKernelWrapper(const MatrixOpsKernelWrapper& obj) = delete;
  void operator=(const MatrixOpsKernelWrapper&) = delete;

  static MatrixOpsKernelWrapper* GetInstance();

  static void MatMult(const cudaStream_t& stream, float* A, float* B, float* C,
                      int m, int k, int n, bool transposed_a = false,
                      bool transposed_b = false);

  static void CPUMatMult(const float* A, const float* B, float* C, int m, int k,
                         int n, bool transposed_a = false,
                         bool transposed_b = false);

  static void Relu(const cudaStream_t& stream, float* A, int m, int n);

  static void CPURelu(float* A, int m, int n);

  static void Sigmoid(const cudaStream_t& stream, float* A, int m, int n);

  static void CPUSigmoid(float* A, int m, int n);

  static void MatAdd(const cudaStream_t& stream, float* A, float* B, int m,
                     int n);

  static void CPUMatAdd(float* A, float* B, int m, int n);

  static void Transpose(const cudaStream_t& stream, float* A, float* B, int m,
                        int n);

  static void CPUTranspose(const float* A, const float* B, int m, int n);

  static void CPUSimdSquaredDifference(const float* v_a, const float* v_b,
                                       float* v_c, size_t n);

 private:
  MatrixOpsKernelWrapper() = default;
  inline static MatrixOpsKernelWrapper* ptr_ = nullptr;
};

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_KERNEL_MatrixOps_CUH
