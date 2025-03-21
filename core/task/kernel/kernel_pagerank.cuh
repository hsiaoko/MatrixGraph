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

class PageRankKernelWrapper {
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
  PageRankKernelWrapper(const PageRankKernelWrapper& obj) = delete;
  void operator=(const PageRankKernelWrapper&) = delete;

  static PageRankKernelWrapper* GetInstance();

  static void PageRank(
      const cudaStream_t& stream, VertexID n_vertices_g, EdgeIndex n_edges_g,
      const data_structures::UnifiedOwnedBuffer<uint8_t>& data_g,
      data_structures::UnifiedOwnedBuffer<float>& page_ranks,
      float damping_factor = 0.85f, float epsilon = 1e-6f,
      int max_iterations = 10);

 private:
  PageRankKernelWrapper() = default;
  inline static PageRankKernelWrapper* ptr_ = nullptr;
};

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_KERNEL_PAGERANK_CUH
