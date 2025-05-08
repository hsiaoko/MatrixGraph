#ifndef MATRIXGRAPH_KERNEL_PAGERANK_CUH
#define MATRIXGRAPH_KERNEL_PAGERANK_CUH

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

class GemmKernelWrapper {
 private:
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using UnifiedOwnedBufferVertexID =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexID>;
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;
  using Edges = sics::matrixgraph::core::data_structures::Edges;

 public:
  // deleting copy constructor
  GemmKernelWrapper(const GemmKernelWrapper& obj) = delete;
  void operator=(const GemmKernelWrapper&) = delete;

  static GemmKernelWrapper* GetInstance();

  typedef<class T> static void Gemm(const cudaStream_t& stream, const T* A,
                                    const* B, const* C, int m, int k, int n);

 private:
  GemmKernelWrapper() = default;
  inline static GemmKernelWrapper* ptr_ = nullptr;
};

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_KERNEL_PAGERANK_CUH
