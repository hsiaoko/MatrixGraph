#ifndef MATRIXGRAPH_CORE_TASK_GPU_TASK_KERNEL_KERNEL_WCC_CUH_
#define MATRIXGRAPH_CORE_TASK_GPU_TASK_KERNEL_KERNEL_WCC_CUH_

// HashMin WCC launch on the caller's stream: iterates label propagation
// and leaves one component label per vertex in v_label_g.
#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/exec_plan.cuh"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/unified_buffer.cuh"

#include <cuda_runtime.h>
#include <vector>

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

  /**
   * @brief Delete assignment operator to enforce singleton pattern
   */

  /**
   * @brief Get the singleton instance of WCCKernelWrapper
   * @return Pointer to the singleton instance
   */
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

  /**
   * @brief Singleton instance pointer
   */
};

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_GPU_TASK_KERNEL_KERNEL_WCC_CUH_
