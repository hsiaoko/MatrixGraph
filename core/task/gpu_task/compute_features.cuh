#ifndef MATRIXGRAPH_CORE_TASK_GPU_TASK_COMPUTE_FEATURES_CUH_
#define MATRIXGRAPH_CORE_TASK_GPU_TASK_COMPUTE_FEATURES_CUH_

#include <memory>
#include <string>
#include <vector>

#include "core/data_structures/attributes.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/task/gpu_task/compute_features_types.h"
#include "core/task/gpu_task/kernel/kernel_compute_features.cuh"
#include "core/task/gpu_task/task_base.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

/**
 * @file compute_features.cuh
 * @brief Host-side GPU task for evaluating feature-expression plans.
 *
 * ComputeFeaturesTask is a standalone task (it does not depend on
 * GraphAggregate).  It loads a MatrixGraph CSR graph, optional per-vertex
 * attributes and labels, and evaluates a flat expression plan on a set of
 * pivot vertices.
 *
 * Execution model:
 *   - One CUDA block is launched per pivot vertex.
 *   - All threads in the block cooperate to collect navigator bindings,
 *     evaluate sub-expressions and run aggregation primitives.
 *   - The result is a row-major array of MatrixGraphFeatureValue:
 *     result[i * n_outputs + j] is the j-th output feature for the i-th pivot.
 *
 * Thread-safety and lifetime:
 *   - The task object is not thread-safe; external callers must serialize
 *     concurrent access.
 *   - Attribute column pointers passed to LoadAttributes() must remain valid
 *     only for the duration of that call; the task allocates its own GPU copy.
 *   - The destructor releases all device buffers.
 */
class ComputeFeaturesTask : public TaskBase {
 private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;
  using Attributes = sics::matrixgraph::core::data_structures::Attributes;
  using Attribute = sics::matrixgraph::core::data_structures::Attribute;
  using AttributeName = sics::matrixgraph::core::data_structures::AttributeName;
  using DeviceAttributes =
      sics::matrixgraph::core::data_structures::DeviceAttributes;

 public:
  explicit ComputeFeaturesTask(const std::string& graph_path)
      : graph_path_(graph_path) {}

  ~ComputeFeaturesTask() { FreeDeviceBuffers(); }

  /**
   * @brief Load graph topology from a MatrixGraph CSR directory.
   *
   * This reads the CSR metadata and the contiguous graph buffer from disk and
   * uploads them to the GPU.  Must be called before LoadAttributes() or
   * Compute().
   */
  __host__ void LoadGraph(const std::string& graph_path);

  /**
   * @brief Load one or more columnar vertex attributes.
   *
   * Each column must contain exactly n_vertices entries, where n_vertices is
   * the number of vertices loaded by LoadGraph().  Calls are cumulative: new
   * columns are appended to the existing set, and the per-vertex attribute maps
   * are rebuilt.
   *
   * @param n_columns Number of columns in the @p columns array.
   * @param columns   Host-owned array of column descriptors.  The underlying
   *                  `values` pointers only need to remain valid for the
   *                  duration of this call.
   */
  __host__ void LoadAttributes(
      uint32_t n_columns,
      const ComputeFeaturesAttributeColumn* columns);

  /**
   * @brief Load optional per-vertex labels.
   *
   * Labels are used by NeighborNav label filters (target_label) and will be
   * used by pattern matching in later phases.
   *
   * @param labels uint32_t array of length @p n.
   * @param n      Number of vertices (must match the graph).
   */
  __host__ void LoadLabels(const uint32_t* labels, uint32_t n);

  /**
   * @brief Evaluate the plan for the given pivots.
   *
   * The plan, navigators, conditions and output indices are uploaded to the
   * GPU for this call and freed before the function returns.
   *
   * @param pivot_vertex_ids    List of pivot vertex ids to evaluate.
   * @param plan                Flat expression plan.
   * @param navs                Flat navigator plan.
   * @param conds               Flat condition array (may be empty).
   * @param output_expr_indices Indices into @p plan that should be emitted.
   * @return Row-major vector of size pivot_vertex_ids.size() *
   *         output_expr_indices.size().
   */
  __host__ std::vector<MatrixGraphFeatureValue> Compute(
      const std::vector<uint32_t>& pivot_vertex_ids,
      const std::vector<MatrixGraphPlanNode>& plan,
      const std::vector<MatrixGraphPlanNode>& navs,
      const std::vector<MatrixGraphCondNode>& conds,
      const std::vector<int32_t>& output_expr_indices);

  /** @brief TaskBase entry point; loads the graph if not already loaded. */
  __host__ void Run();

 private:
  /** @brief Upload the CSR graph buffer read by LoadGraph() to the GPU. */
  __host__ void TransferGraphDataToDevice();

  /** @brief Release all device buffers owned by this task. */
  __host__ void FreeDeviceBuffers();

  /** @brief Size in bytes of one element of the given MatrixGraph value type. */
  __host__ static size_t ValueTypeSize(MatrixGraphValueType type);

  /** @brief Convert an int32_t value-type code to the typed enum. */
  __host__ static MatrixGraphValueType ValueTypeFromMG(int32_t type);

  std::string graph_path_;
  std::unique_ptr<ImmutableCSR> graph_;

  // Device-side CSR buffer.
  uint8_t* d_graph_data_ = nullptr;
  size_t d_graph_data_size_ = 0;

  // One DeviceAttributes object per vertex, built from d_columns_.
  std::vector<DeviceAttributes> per_vertex_attrs_;

  // Device-side array of Attributes views (one per vertex).
  Attributes* d_per_vertex_attrs_ = nullptr;

  // Optional device-side per-vertex labels.
  uint32_t* d_labels_ = nullptr;
  uint32_t n_labels_ = 0;

  // Owned device buffers backing the attribute columns.
  struct DeviceColumnBuffer {
    char key[64];
    void* d_values = nullptr;
    MatrixGraphValueType type = MG_VALUE_INVALID;
    size_t bytes = 0;
  };
  std::vector<DeviceColumnBuffer> d_columns_;
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_GPU_TASK_COMPUTE_FEATURES_CUH_
