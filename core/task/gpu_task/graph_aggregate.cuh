#ifndef MATRIXGRAPH_CORE_TASK_GPU_TASK_GRAPH_AGGREGATE_CUH_
#define MATRIXGRAPH_CORE_TASK_GPU_TASK_GRAPH_AGGREGATE_CUH_

#include <cuda_runtime.h>

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

// Columnar host attribute description used by LoadAttributes.
struct GraphAggregateAttributeColumn {
  char key[64];                 // attribute name
  int32_t value_type;           // ValueType enum value
  uint32_t n_values;            // must equal number of vertices
  const void* values = nullptr; // host pointer; n_values contiguous entries
  const uint8_t* valid = nullptr; // host pointer; n_values bytes (1=valid) or null
};

// GraphAggregate task: per-vertex feature aggregation over a single graph.
//
// The task owns one ImmutableCSR graph and its per-vertex Attributes.  Callers
// supply a list of pivot vertices and a list of aggregation requests; the task
// executes them on one or more GPUs, using one or more CUDA streams per GPU.
class GraphAggregate : public TaskBase {
 private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;
  using Attributes = sics::matrixgraph::core::data_structures::Attributes;
  using Attribute = sics::matrixgraph::core::data_structures::Attribute;
  using AttributeName = sics::matrixgraph::core::data_structures::AttributeName;

 public:
  // ---------------------------------------------------------------------------
  // Helper: manages device memory for per-vertex Attributes of a single graph
  // on a single GPU.
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

    // Clone these attributes to another GPU.  The source must have been built
    // on its home device.  Scalar per-vertex attributes sharing a column buffer
    // are supported; list attributes are not yet cloned.
    __host__ DevicePerVertexAttributes CloneToDevice(int target_device) const;

    __host__ Attributes* GetDevicePtr() const { return d_attrs_; }
    __host__ uint32_t GetNumVertices() const { return n_vertices_; }
    __host__ int HomeDevice() const { return home_device_; }

    __host__ void Free();

    // Take ownership of per-attribute-key device column buffers.
    __host__ void TakeColumnBuffers(std::vector<uint8_t*>&& buffers) {
      column_buffers_ = std::move(buffers);
    }

   private:
    Attributes* d_attrs_ = nullptr;
    uint32_t n_vertices_ = 0;
    int home_device_ = -1;

    // Pooled device memory for all vertices' HashMap buckets.
    AttributeName* d_hash_keys_ = nullptr;
    Attribute* d_hash_values_ = nullptr;
    uint8_t* d_hash_occupied_ = nullptr;
    uint32_t total_hash_capacity_ = 0;

    // Per-attribute-key device column buffers referenced by the HashMap values.
    // Kept here so Free() can release them.
    std::vector<uint8_t*> column_buffers_;
  };

  GraphAggregate();
  explicit GraphAggregate(const std::string& graph_path);

  ~GraphAggregate() { FreeDeviceBuffers(); }

  // Set the active GPU device indices.  Must be called before loading data.
  // Empty vector means auto-detect via MATRIXGRAPH_CUDA_DEVICES env.
  __host__ void SetDevices(const std::vector<int>& device_ids);

  // Get the device indices currently in use.
  __host__ const std::vector<int>& GetDevices() const { return device_ids_; }

  // Load graph topology from a MatrixGraph CSR directory.
  // The CSR buffer is replicated to every selected GPU.
  __host__ void LoadGraph(const std::string& graph_path);

  // Load columnar per-vertex attributes from host memory.  The columns are
  // replicated to every selected GPU and a per-vertex attribute map is built
  // on each GPU.
  __host__ void LoadAttributes(
      uint32_t n_columns,
      const GraphAggregateAttributeColumn* columns);

  // Set the per-vertex Attributes for the loaded graph.  Ownership of the
  // device memory is transferred into the task for the primary GPU; for
  // multi-GPU configurations the attributes are cloned to each GPU.
  __host__ void SetVertexAttributes(DevicePerVertexAttributes attrs);

  // Configure the number of CUDA streams used per GPU.
  // Must be called before loading data (default from MATRIXGRAPH_CUDA_STREAMS
  // env, or 2 if unset).
  __host__ void SetNumStreams(uint32_t n_streams);

  // Compute features for a list of pivots.
  // requests describe which aggregation to perform.
  __host__ std::vector<kernel::FeatureValue> ComputeFeatures(
      const std::vector<uint32_t>& pivot_vertex_ids,
      const std::vector<kernel::FeatureRequest>& requests);

  // Fused compute-all: produce all aggregation primitives for each pivot in a
  // single kernel launch.  Much faster than calling ComputeFeatures with one
  // request per primitive because shared work (sum, mean, sort) is done once.
  __host__ std::vector<kernel::AllFeatures> ComputeAll(
      const std::vector<uint32_t>& pivot_vertex_ids,
      const kernel::AttributeName& attr_name,
      bool use_outgoing);

  // Accessors
  __host__ bool HasGraph() const { return graph_ != nullptr; }
  __host__ const ImmutableCSR& GetGraph() const { return *graph_; }
  __host__ uint32_t GetNumVertices() const {
    return per_gpu_states_.empty() ? 0
                                   : per_gpu_states_[0].vertex_attrs.GetNumVertices();
  }
  __host__ uint32_t GetNumStreams() const { return n_streams_; }
  __host__ const Attributes* GetDeviceVertexAttributes(size_t gpu_idx = 0) const {
    return gpu_idx < per_gpu_states_.size()
               ? per_gpu_states_[gpu_idx].vertex_attrs.GetDevicePtr()
               : nullptr;
  }

  // Test helper: create a synthetic ring graph with n_vertices and
  // out_degree_per_vertex outgoing edges.  Also fills per-vertex
  // Attributes with two attributes: "score" (float64) and "flag" (bool).
  __host__ void LoadSyntheticData(uint32_t n_vertices,
                                  uint32_t out_degree_per_vertex);

  // Generate and load synthetic per-vertex "score" (float64) and "flag" (bool)
  // attributes for the graph that has already been loaded.  Useful for demos
  // on real topologies where attributes are not available.
  __host__ void GenerateSyntheticAttributes();

  __host__ void Run();

 private:
  // Per-stream reusable device buffers on a single GPU.
  struct StreamBuffers {
    uint32_t* d_pivot_vids = nullptr;
    uint32_t pivot_capacity = 0;

    // For ComputeFeatures.
    kernel::FeatureValue* d_outputs = nullptr;
    uint32_t output_request_capacity = 0;

    // For ComputeAll.
    kernel::AllFeatures* d_all_outputs = nullptr;
    uint32_t all_output_pivot_capacity = 0;

    __host__ void Free();
    __host__ void EnsureCapacity(
        uint32_t n_pivots,
        uint32_t n_requests,
        uint32_t max_neighbors);
  };

  // Per-GPU state.
  struct PerGpuState {
    int device_id = -1;

    // Device-side graph buffer on this GPU.
    uint8_t* d_graph_data = nullptr;
    size_t d_graph_data_size = 0;

    // Device-side attributes on this GPU.
    DevicePerVertexAttributes vertex_attrs;

    // Streams and per-stream buffers on this GPU.
    std::vector<cudaStream_t> streams;
    std::vector<StreamBuffers> stream_buffers;
  };

  __host__ void DetectDevices();
  __host__ void ValidateDevices();
  __host__ void FreeDeviceBuffers();
  __host__ void ComputeNumStreams();
  __host__ void BuildPerGpuStates();
  __host__ void TransferGraphToAllGpus();
  __host__ void ReplicateAttributesToAllGpus();

  __host__ static size_t ValueTypeSize(int32_t value_type);
  __host__ uint32_t ComputeMaxDegree() const;

  // Determine the per-pivot neighbor cap for this launch from max_degree_,
  // the 4096 correctness cap, and the available shared memory on the selected
  // GPUs.  Returns max_neighbors and writes the required dynamic shared bytes.
  __host__ uint32_t ComputeLaunchMaxNeighbors(size_t* shared_mem_size) const;

  // Single graph topology (host master copy).
  std::unique_ptr<ImmutableCSR> graph_;

  // Cached maximum vertex degree (max of in/out) for workspace sizing.
  uint32_t max_degree_ = 0;

  // Active GPUs and their state.
  std::vector<int> device_ids_;
  std::vector<PerGpuState> per_gpu_states_;

  // Streams per GPU.
  uint32_t n_streams_ = 2;

  // Primary device used to build attributes passed via SetVertexAttributes.
  int primary_device_ = -1;

  std::string graph_path_;
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_GPU_TASK_GRAPH_AGGREGATE_CUH_
