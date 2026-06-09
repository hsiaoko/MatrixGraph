#ifndef MATRIXGRAPH_CORE_TASK_GPU_TASK_GRAPH_AGGREGATE_CUH_
#define MATRIXGRAPH_CORE_TASK_GPU_TASK_GRAPH_AGGREGATE_CUH_

#include <memory>
#include <string>
#include <vector>

#include "core/common/types.h"
#include "core/data_structures/attributes.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/task/gpu_task/kernel/kernel_graph_aggregate.cuh"
#include "core/task/gpu_task/task_base.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

// GraphAggregate task: processes multiple graphs simultaneously.
// Each graph has an array of per-vertex Attributes (attribute-name -> Attribute).
class GraphAggregate : public TaskBase {
 private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;
  using Attributes = sics::matrixgraph::core::data_structures::Attributes;
  using Attribute = sics::matrixgraph::core::data_structures::Attribute;
  using AttributeName = sics::matrixgraph::core::data_structures::AttributeName;

 public:
  // ---------------------------------------------------------------------------
  // Helper: manages device memory for per-vertex Attributes of a single graph.
  // ---------------------------------------------------------------------------
  class DevicePerVertexAttributes {
   public:
    DevicePerVertexAttributes() = default;
    ~DevicePerVertexAttributes() { Free(); }

    DevicePerVertexAttributes(const DevicePerVertexAttributes&) = delete;
    DevicePerVertexAttributes& operator=(const DevicePerVertexAttributes&) = delete;

    DevicePerVertexAttributes(DevicePerVertexAttributes&& other) noexcept;
    DevicePerVertexAttributes& operator=(DevicePerVertexAttributes&& other) noexcept;

    // Build device-side per-vertex Attributes from a host-side array.
    // h_attrs[i].attr_map must already be built on host (pointing to host temp
    // memory). This function copies the Attributes structs and their HashMap
    // bucket data to device memory, then fixes up the pointers.
    __host__ void Build(const Attributes* h_attrs, uint32_t n_vertices);

    __host__ Attributes* GetDevicePtr() const { return d_attrs_; }
    __host__ uint32_t GetNumVertices() const { return n_vertices_; }

    __host__ void Free();

   private:
    Attributes* d_attrs_ = nullptr;
    uint32_t n_vertices_ = 0;

    // Pooled device memory for all vertices' HashMap buckets.
    AttributeName* d_hash_keys_ = nullptr;
    Attribute* d_hash_values_ = nullptr;
    uint8_t* d_hash_occupied_ = nullptr;
    uint32_t total_hash_capacity_ = 0;
  };

  explicit GraphAggregate(const std::vector<std::string>& data_graph_paths)
      : data_graph_paths_(data_graph_paths) {}

  ~GraphAggregate() { FreeDeviceBuffers(); }

  __host__ void Run();

  // Compute features for a list of pivots.
  // pivot_graph_ids[i] / pivot_vertex_ids[i] identify the i-th pivot.
  // requests describe which aggregation to perform.
  // If out_values is non-null, results are copied back to host.
  __host__ void ComputeFeatures(
      const std::vector<uint32_t>& pivot_graph_ids,
      const std::vector<uint32_t>& pivot_vertex_ids,
      const std::vector<kernel::FeatureRequest>& requests,
      std::vector<kernel::FeatureValue>* out_values = nullptr);

  // Accessors
  __host__ size_t GetNumGraphs() const { return graphs_.size(); }
  __host__ const ImmutableCSR& GetGraph(size_t idx) const { return *graphs_[idx]; }
  __host__ uint32_t GetNumVertices(size_t graph_idx) const {
    return graph_idx < per_graph_vertex_attrs_.size()
               ? per_graph_vertex_attrs_[graph_idx].GetNumVertices()
               : 0;
  }
  __host__ const Attributes* GetDeviceVertexAttributes(size_t graph_idx) const {
    return graph_idx < per_graph_vertex_attrs_.size()
               ? per_graph_vertex_attrs_[graph_idx].GetDevicePtr()
               : nullptr;
  }

  // Test helper: create a synthetic ring graph with n_vertices and
  // out_degree_per_vertex random outgoing edges.  Also fills per-vertex
  // Attributes with two attributes: "score" (float) and "flag" (bool).
  // Backward-compatible wrapper that clears any existing graphs first.
  __host__ void LoadSyntheticData(uint32_t n_vertices,
                                  uint32_t out_degree_per_vertex);

  // Add a new synthetic graph to the internal multi-graph list.
  // Device buffers are invalidated so the next ComputeFeatures will re-transfer.
  __host__ void AddSyntheticGraph(uint32_t n_vertices,
                                  uint32_t out_degree_per_vertex);

 private:
  __host__ void LoadData();
  __host__ void BuildDeviceAttributes();
  __host__ void TransferGraphDataToDevice();
  __host__ void FreeDeviceBuffers();

  // Array of graph topologies.
  std::vector<std::unique_ptr<ImmutableCSR>> graphs_;

  // Per-graph per-vertex Attributes stored in device memory.
  std::vector<DevicePerVertexAttributes> per_graph_vertex_attrs_;

  // Device-side graph data buffers (CSR raw buffers).
  std::vector<uint8_t*> d_graph_data_buffers_;
  std::vector<uint32_t> graph_buffer_sizes_;

  // Device-side pointer arrays (double-indirection for kernel).
  uint8_t** d_graph_data_array_ = nullptr;
  Attributes** d_vertex_attrs_array_ = nullptr;
  uint32_t* d_graph_n_vertices_ = nullptr;
  uint32_t* d_graph_n_in_edges_ = nullptr;
  uint32_t* d_graph_n_out_edges_ = nullptr;

  std::vector<std::string> data_graph_paths_;
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_GPU_TASK_GRAPH_AGGREGATE_CUH_
