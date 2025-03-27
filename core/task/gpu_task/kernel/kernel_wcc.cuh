#ifndef MATRIXGRAPH_KERNEL_WCC_CUH
#define MATRIXGRAPH_KERNEL_WCC_CUH

#include <cuda_runtime.h>

#include <vector>

#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/exec_plan.cuh"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/unified_buffer.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

/**
 * @brief CUDA kernel wrapper for Weakly Connected Components (WCC) computation
 *
 * This class implements a singleton pattern to manage CUDA kernels for
 * computing Weakly Connected Components in a graph. It uses the HashMin
 * algorithm for parallel WCC computation on GPU.
 */
class WCCKernelWrapper {
 private:
  // Type aliases for improved readability
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using UnifiedOwnedBufferVertexID =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexID>;
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;
  using Edges = sics::matrixgraph::core::data_structures::Edges;

 public:
  /**
   * @brief Delete copy constructor to enforce singleton pattern
   */
  WCCKernelWrapper(const WCCKernelWrapper& obj) = delete;

  /**
   * @brief Delete assignment operator to enforce singleton pattern
   */
  void operator=(const WCCKernelWrapper&) = delete;

  /**
   * @brief Get the singleton instance of WCCKernelWrapper
   * @return Pointer to the singleton instance
   */
  static WCCKernelWrapper* GetInstance();

  /**
   * @brief Execute WCC computation on GPU using HashMin algorithm
   * @param stream CUDA stream for asynchronous execution
   * @param n_vertices_g Number of vertices in the graph
   * @param n_edges_g Number of edges in the graph
   * @param data_g Unified buffer containing graph data in CSR format
   * @param v_label_g Unified buffer for vertex labels (component IDs)
   */
  static void WCC(
      const cudaStream_t& stream, VertexID n_vertices_g, EdgeIndex n_edges_g,
      const data_structures::UnifiedOwnedBuffer<uint8_t>& data_g,
      const data_structures::UnifiedOwnedBuffer<VertexLabel>& v_label_g);

 private:
  /**
   * @brief Private constructor for singleton pattern
   */
  WCCKernelWrapper() = default;

  /**
   * @brief Singleton instance pointer
   */
  inline static WCCKernelWrapper* ptr_ = nullptr;
};

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_KERNEL_WCC_CUH
