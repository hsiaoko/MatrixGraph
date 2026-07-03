#include "core/task/gpu_task/graph_aggregate.cuh"

#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <unordered_map>

#include "core/common/consts.h"
#include "core/util/cuda_check.cuh"
#include "core/util/cuda_device.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using VertexID = sics::matrixgraph::core::common::VertexID;
using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using Attributes = sics::matrixgraph::core::data_structures::Attributes;
using Attribute = sics::matrixgraph::core::data_structures::Attribute;
using AttributeName = sics::matrixgraph::core::data_structures::AttributeName;
using ValueType = sics::matrixgraph::core::data_structures::ValueType;
template <typename T>
using DefaultHash = sics::matrixgraph::core::data_structures::DefaultHash<T>;

namespace {

// RAII helper to restore the current CUDA device on scope exit.
class CudaDeviceGuard {
 public:
  explicit CudaDeviceGuard(int device) : original_device_(-1) {
    cudaGetDevice(&original_device_);
    cudaSetDevice(device);
  }
  ~CudaDeviceGuard() {
    if (original_device_ >= 0) cudaSetDevice(original_device_);
  }
 private:
  int original_device_;
};

inline size_t ValueTypeElementSize(ValueType type) {
  switch (type) {
    case ValueType::kInt:
    case ValueType::kFloat64:
    case ValueType::kTime:
      return 8;
    case ValueType::kBool:
      return 1;
    case ValueType::kFloat32:
      return 4;
    case ValueType::kString:
      return sizeof(sics::matrixgraph::core::data_structures::StringView);
    default:
      return 0;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// DevicePerVertexAttributes
// ---------------------------------------------------------------------------
GraphAggregate::DevicePerVertexAttributes::DevicePerVertexAttributes(
    DevicePerVertexAttributes&& other) noexcept {
  d_attrs_ = other.d_attrs_;
  n_vertices_ = other.n_vertices_;
  home_device_ = other.home_device_;
  d_hash_keys_ = other.d_hash_keys_;
  d_hash_values_ = other.d_hash_values_;
  d_hash_occupied_ = other.d_hash_occupied_;
  total_hash_capacity_ = other.total_hash_capacity_;
  other.d_attrs_ = nullptr;
  other.d_hash_keys_ = nullptr;
  other.d_hash_values_ = nullptr;
  other.d_hash_occupied_ = nullptr;
  other.n_vertices_ = 0;
  other.home_device_ = -1;
  other.total_hash_capacity_ = 0;
}

GraphAggregate::DevicePerVertexAttributes&
GraphAggregate::DevicePerVertexAttributes::operator=(
    DevicePerVertexAttributes&& other) noexcept {
  if (this != &other) {
    Free();
    d_attrs_ = other.d_attrs_;
    n_vertices_ = other.n_vertices_;
    home_device_ = other.home_device_;
    d_hash_keys_ = other.d_hash_keys_;
    d_hash_values_ = other.d_hash_values_;
    d_hash_occupied_ = other.d_hash_occupied_;
    total_hash_capacity_ = other.total_hash_capacity_;
    other.d_attrs_ = nullptr;
    other.d_hash_keys_ = nullptr;
    other.d_hash_values_ = nullptr;
    other.d_hash_occupied_ = nullptr;
    other.n_vertices_ = 0;
    other.home_device_ = -1;
    other.total_hash_capacity_ = 0;
  }
  return *this;
}

__host__ void GraphAggregate::DevicePerVertexAttributes::Build(
    const Attributes* h_attrs, uint32_t n_vertices) {
  if (n_vertices == 0) return;

  int current_device = -1;
  CUDA_CHECK(cudaGetDevice(&current_device));
  home_device_ = current_device;

  n_vertices_ = n_vertices;

  // 1. Allocate device memory for the Attributes array.
  CUDA_CHECK(cudaMalloc(&d_attrs_, sizeof(Attributes) * n_vertices));

  // 2. Compute total HashMap capacity across all vertices.
  total_hash_capacity_ = 0;
  std::vector<uint32_t> capacities(n_vertices);
  for (uint32_t i = 0; i < n_vertices; ++i) {
    capacities[i] = h_attrs[i].attr_map.capacity;
    total_hash_capacity_ += capacities[i];
  }

  if (total_hash_capacity_ > 0) {
    CUDA_CHECK(
        cudaMalloc(&d_hash_keys_, sizeof(AttributeName) * total_hash_capacity_));
    CUDA_CHECK(
        cudaMalloc(&d_hash_values_, sizeof(Attribute) * total_hash_capacity_));
    CUDA_CHECK(
        cudaMalloc(&d_hash_occupied_, sizeof(uint8_t) * total_hash_capacity_));
  }

  // 3. Build a host-side temporary Attributes array with device pointers.
  Attributes* h_device_attrs = new Attributes[n_vertices];
  uint32_t offset = 0;
  for (uint32_t i = 0; i < n_vertices; ++i) {
    h_device_attrs[i] = h_attrs[i];
    uint32_t cap = capacities[i];
    if (cap > 0) {
      h_device_attrs[i].attr_map.keys = d_hash_keys_ + offset;
      h_device_attrs[i].attr_map.values = d_hash_values_ + offset;
      h_device_attrs[i].attr_map.occupied = d_hash_occupied_ + offset;
    } else {
      h_device_attrs[i].attr_map.keys = nullptr;
      h_device_attrs[i].attr_map.values = nullptr;
      h_device_attrs[i].attr_map.occupied = nullptr;
    }
    offset += cap;
  }

  // 4. Copy the Attributes structs to device.
  CUDA_CHECK(cudaMemcpy(d_attrs_, h_device_attrs,
                        sizeof(Attributes) * n_vertices,
                        cudaMemcpyHostToDevice));

  // 5. Copy HashMap bucket data to device.
  // Detect whether the host pointers form one contiguous pool (the common
  // case when built by LoadAttributes / CloneToDevice).  If so, use a single
  // H2D copy per array instead of millions of tiny per-vertex copies.
  bool contiguous = false;
  if (n_vertices > 0 && h_attrs[0].attr_map.keys != nullptr) {
    contiguous = true;
    const uint8_t* expected_keys =
        reinterpret_cast<const uint8_t*>(h_attrs[0].attr_map.keys);
    const uint8_t* expected_values =
        reinterpret_cast<const uint8_t*>(h_attrs[0].attr_map.values);
    const uint8_t* expected_occupied = h_attrs[0].attr_map.occupied;
    for (uint32_t i = 0; i < n_vertices; ++i) {
      uint32_t cap = capacities[i];
      if (cap == 0) continue;
      if (reinterpret_cast<const uint8_t*>(h_attrs[i].attr_map.keys) !=
              expected_keys ||
          reinterpret_cast<const uint8_t*>(h_attrs[i].attr_map.values) !=
              expected_values ||
          h_attrs[i].attr_map.occupied != expected_occupied) {
        contiguous = false;
        break;
      }
      expected_keys += sizeof(AttributeName) * cap;
      expected_values += sizeof(Attribute) * cap;
      expected_occupied += sizeof(uint8_t) * cap;
    }
  }

  if (contiguous && total_hash_capacity_ > 0) {
    CUDA_CHECK(cudaMemcpy(d_hash_keys_, h_attrs[0].attr_map.keys,
                          sizeof(AttributeName) * total_hash_capacity_,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hash_values_, h_attrs[0].attr_map.values,
                          sizeof(Attribute) * total_hash_capacity_,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hash_occupied_, h_attrs[0].attr_map.occupied,
                          sizeof(uint8_t) * total_hash_capacity_,
                          cudaMemcpyHostToDevice));
  } else {
    // Fallback: per-vertex copy for non-contiguous host layouts.
    offset = 0;
    for (uint32_t i = 0; i < n_vertices; ++i) {
      uint32_t cap = capacities[i];
      if (cap == 0) continue;
      if (h_attrs[i].attr_map.keys) {
        CUDA_CHECK(cudaMemcpy(d_hash_keys_ + offset, h_attrs[i].attr_map.keys,
                              sizeof(AttributeName) * cap,
                              cudaMemcpyHostToDevice));
      }
      if (h_attrs[i].attr_map.values) {
        CUDA_CHECK(cudaMemcpy(d_hash_values_ + offset,
                              h_attrs[i].attr_map.values,
                              sizeof(Attribute) * cap,
                              cudaMemcpyHostToDevice));
      }
      if (h_attrs[i].attr_map.occupied) {
        CUDA_CHECK(cudaMemcpy(d_hash_occupied_ + offset,
                              h_attrs[i].attr_map.occupied,
                              sizeof(uint8_t) * cap,
                              cudaMemcpyHostToDevice));
      }
      offset += cap;
    }
  }

  delete[] h_device_attrs;
}

__host__ GraphAggregate::DevicePerVertexAttributes
GraphAggregate::DevicePerVertexAttributes::CloneToDevice(int target_device) const {
  DevicePerVertexAttributes cloned;
  if (n_vertices_ == 0) return cloned;

  // Read the Attributes array from the source device.
  std::vector<Attributes> h_attrs(n_vertices_);
  {
    CudaDeviceGuard guard(home_device_);
    CUDA_CHECK(cudaMemcpy(h_attrs.data(), d_attrs_,
                          sizeof(Attributes) * n_vertices_,
                          cudaMemcpyDeviceToHost));
  }

  // Read all hash map bucket data from the source device.
  std::vector<AttributeName> h_keys(total_hash_capacity_);
  std::vector<Attribute> h_values(total_hash_capacity_);
  std::vector<uint8_t> h_occupied(total_hash_capacity_);
  {
    CudaDeviceGuard guard(home_device_);
    if (d_hash_keys_) {
      CUDA_CHECK(cudaMemcpy(h_keys.data(), d_hash_keys_,
                            sizeof(AttributeName) * total_hash_capacity_,
                            cudaMemcpyDeviceToHost));
    }
    if (d_hash_values_) {
      CUDA_CHECK(cudaMemcpy(h_values.data(), d_hash_values_,
                            sizeof(Attribute) * total_hash_capacity_,
                            cudaMemcpyDeviceToHost));
    }
    if (d_hash_occupied_) {
      CUDA_CHECK(cudaMemcpy(h_occupied.data(), d_hash_occupied_,
                            sizeof(uint8_t) * total_hash_capacity_,
                            cudaMemcpyDeviceToHost));
    }
  }

  // Restore per-vertex attr_map host pointers so h_attrs can be traversed.
  uint32_t offset = 0;
  for (uint32_t i = 0; i < n_vertices_; ++i) {
    uint32_t cap = h_attrs[i].attr_map.capacity;
    if (cap > 0) {
      h_attrs[i].attr_map.keys = h_keys.data() + offset;
      h_attrs[i].attr_map.values = h_values.data() + offset;
      h_attrs[i].attr_map.occupied = h_occupied.data() + offset;
    }
    offset += cap;
  }

  // Identify unique per-attribute-key column buffers and clone them.
  // For each key, find the min/max data pointer across vertices to determine
  // the underlying column buffer size.
  struct BufferInfo {
    const uint8_t* src_min = nullptr;
    const uint8_t* src_max = nullptr;
    size_t elem_size = 0;
    ValueType type = ValueType::kInvalid;
  };
  std::unordered_map<std::string, BufferInfo> buffer_info_by_key;

  for (uint32_t v = 0; v < n_vertices_; ++v) {
    Attributes& attrs = h_attrs[v];
    for (uint32_t s = 0; s < attrs.attr_map.capacity; ++s) {
      if (!h_occupied[s]) continue;
      AttributeName& name = h_keys[s];
      Attribute& attr = h_values[s];
      size_t elem_size = ValueTypeElementSize(attr.type);
      if (elem_size == 0 || attr.data == nullptr) continue;

      std::string key(name.data);
      auto& info = buffer_info_by_key[key];
      if (info.elem_size == 0) info.elem_size = elem_size;
      if (info.type == ValueType::kInvalid) info.type = attr.type;
      const uint8_t* ptr = static_cast<const uint8_t*>(attr.data);
      if (info.src_min == nullptr || ptr < info.src_min) info.src_min = ptr;
      if (info.src_max == nullptr || ptr > info.src_max) info.src_max = ptr;
    }
  }

  // Allocate target buffers and copy data.  We use cudaMemcpyPeerAsync when
  // possible; otherwise fall back to a host staging buffer.
  std::unordered_map<std::string, uint8_t*> target_buffers;
  std::vector<std::unique_ptr<uint8_t[]>> staging_buffers;
  cloned.column_buffers_.reserve(buffer_info_by_key.size());

  for (auto& kv : buffer_info_by_key) {
    const std::string& key = kv.first;
    BufferInfo& info = kv.second;
    size_t bytes = (info.src_max - info.src_min) + info.elem_size;

    CudaDeviceGuard guard(target_device);
    uint8_t* d_target = nullptr;
    CUDA_CHECK(cudaMalloc(&d_target, bytes));
    target_buffers[key] = d_target;
    cloned.column_buffers_.push_back(d_target);

    int can_peer = 0;
    {
      CudaDeviceGuard src_guard(home_device_);
      cudaDeviceCanAccessPeer(&can_peer, home_device_, target_device);
    }

    if (can_peer) {
      CudaDeviceGuard src_guard(home_device_);
      CUDA_CHECK(cudaMemcpyPeer(d_target, target_device,
                                const_cast<uint8_t*>(info.src_min),
                                home_device_, bytes));
    } else {
      std::unique_ptr<uint8_t[]> staging(new uint8_t[bytes]);
      {
        CudaDeviceGuard src_guard(home_device_);
        CUDA_CHECK(cudaMemcpy(staging.get(), info.src_min, bytes,
                              cudaMemcpyDeviceToHost));
      }
      {
        CudaDeviceGuard dst_guard(target_device);
        CUDA_CHECK(cudaMemcpy(d_target, staging.get(), bytes,
                              cudaMemcpyHostToDevice));
      }
      staging_buffers.push_back(std::move(staging));
    }
  }

  // Fix up attribute data pointers in h_values to point into target buffers.
  for (uint32_t v = 0; v < n_vertices_; ++v) {
    Attributes& attrs = h_attrs[v];
    for (uint32_t s = 0; s < attrs.attr_map.capacity; ++s) {
      if (!h_occupied[s]) continue;
      AttributeName& name = h_keys[s];
      Attribute& attr = h_values[s];
      size_t elem_size = ValueTypeElementSize(attr.type);
      if (elem_size == 0 || attr.data == nullptr) continue;

      std::string key(name.data);
      const uint8_t* src_ptr = static_cast<const uint8_t*>(attr.data);
      uint8_t* target_base = target_buffers[key];
      attr.data = target_base + (src_ptr - buffer_info_by_key[key].src_min);
    }
  }

  // Build the cloned attributes on the target device.
  cloned.Build(h_attrs.data(), n_vertices_);
  return cloned;
}

__host__ void GraphAggregate::DevicePerVertexAttributes::Free() {
  if (d_attrs_) {
    cudaFree(d_attrs_);
  }
  if (d_hash_keys_) cudaFree(d_hash_keys_);
  if (d_hash_values_) cudaFree(d_hash_values_);
  if (d_hash_occupied_) cudaFree(d_hash_occupied_);
  for (uint8_t* buf : column_buffers_) {
    if (buf) cudaFree(buf);
  }
  d_attrs_ = nullptr;
  d_hash_keys_ = nullptr;
  d_hash_values_ = nullptr;
  d_hash_occupied_ = nullptr;
  n_vertices_ = 0;
  home_device_ = -1;
  total_hash_capacity_ = 0;
  column_buffers_.clear();
}

// ---------------------------------------------------------------------------
// StreamBuffers
// ---------------------------------------------------------------------------
__host__ void GraphAggregate::StreamBuffers::Free() {
  if (d_pivot_vids) cudaFree(d_pivot_vids);
  if (d_outputs) cudaFree(d_outputs);
  if (d_all_outputs) cudaFree(d_all_outputs);
  d_pivot_vids = nullptr;
  d_outputs = nullptr;
  d_all_outputs = nullptr;
  pivot_capacity = 0;
  output_request_capacity = 0;
  all_output_pivot_capacity = 0;
}

__host__ void GraphAggregate::StreamBuffers::EnsureCapacity(
    uint32_t n_pivots,
    uint32_t n_requests,
    uint32_t /*max_neighbors*/) {
  // Pivot ids.
  if (pivot_capacity < n_pivots) {
    if (d_pivot_vids) cudaFree(d_pivot_vids);
    CUDA_CHECK(cudaMalloc(&d_pivot_vids, sizeof(uint32_t) * n_pivots));
    pivot_capacity = n_pivots;
  }

  // Per-request outputs.
  if (output_request_capacity < n_requests || pivot_capacity < n_pivots) {
    if (d_outputs) cudaFree(d_outputs);
    CUDA_CHECK(cudaMalloc(
        &d_outputs,
        sizeof(kernel::FeatureValue) * n_pivots * n_requests));
    output_request_capacity = n_requests;
  }

  // All-features outputs.
  if (all_output_pivot_capacity < n_pivots) {
    if (d_all_outputs) cudaFree(d_all_outputs);
    CUDA_CHECK(cudaMalloc(
        &d_all_outputs,
        sizeof(kernel::AllFeatures) * n_pivots));
    all_output_pivot_capacity = n_pivots;
  }
}

// ---------------------------------------------------------------------------
// GraphAggregate
// ---------------------------------------------------------------------------
GraphAggregate::GraphAggregate() {
  ComputeNumStreams();
  DetectDevices();
  BuildPerGpuStates();
}

GraphAggregate::GraphAggregate(const std::string& graph_path)
    : graph_path_(graph_path) {
  ComputeNumStreams();
  DetectDevices();
  BuildPerGpuStates();
  LoadGraph(graph_path);
}

__host__ void GraphAggregate::ComputeNumStreams() {
  int s = sics::matrixgraph::core::util::MatrixGraphCudaStreamsPerGpu();
  n_streams_ = static_cast<uint32_t>(std::max(1, s));
}

__host__ void GraphAggregate::DetectDevices() {
  device_ids_ = sics::matrixgraph::core::util::MatrixGraphCudaDeviceList();
  if (device_ids_.empty()) {
    device_ids_.push_back(
        sics::matrixgraph::core::util::MatrixGraphCudaPrimaryDeviceIndex());
  }
  ValidateDevices();
}

__host__ void GraphAggregate::ValidateDevices() {
  int n_devices = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n_devices));
  std::vector<int> valid;
  for (int id : device_ids_) {
    if (id >= 0 && id < n_devices) {
      valid.push_back(id);
    } else {
      std::cerr << "[GraphAggregate] Ignoring invalid device " << id
                << std::endl;
    }
  }
  if (valid.empty()) {
    valid.push_back(
        sics::matrixgraph::core::util::MatrixGraphCudaPrimaryDeviceIndex());
  }
  device_ids_ = std::move(valid);
}

__host__ void GraphAggregate::SetDevices(const std::vector<int>& device_ids) {
  if (graph_ != nullptr) {
    std::cerr << "[GraphAggregate] SetDevices must be called before loading data."
              << std::endl;
    return;
  }
  device_ids_ = device_ids;
  ValidateDevices();
  BuildPerGpuStates();
}

__host__ void GraphAggregate::SetNumStreams(uint32_t n_streams) {
  if (n_streams == 0) n_streams = 1;
  if (!per_gpu_states_.empty() &&
      !per_gpu_states_[0].streams.empty()) {
    std::cerr << "[GraphAggregate] SetNumStreams called after streams are in "
                 "use; change will take effect on next construction."
              << std::endl;
    return;
  }
  n_streams_ = n_streams;
}

__host__ void GraphAggregate::BuildPerGpuStates() {
  FreeDeviceBuffers();
  per_gpu_states_.clear();
  per_gpu_states_.reserve(device_ids_.size());
  for (int dev : device_ids_) {
    PerGpuState state;
    state.device_id = dev;
    per_gpu_states_.push_back(std::move(state));
  }
}

__host__ void GraphAggregate::LoadGraph(const std::string& graph_path) {
  graph_path_ = graph_path;
  graph_ = std::make_unique<ImmutableCSR>();
  graph_->Read(graph_path);
  max_degree_ = ComputeMaxDegree();
  std::cout << "[GraphAggregate] Loaded graph: "
            << graph_->get_num_vertices() << " vertices, "
            << graph_->get_num_outgoing_edges() << " outgoing edges, "
            << "max degree=" << max_degree_ << std::endl;
  TransferGraphToAllGpus();
}

__host__ void GraphAggregate::TransferGraphToAllGpus() {
  if (!graph_) return;

  const ImmutableCSR& g = *graph_;
  const uint8_t* h_buf = g.GetGraphBuffer();
  const uint8_t* h_end =
      reinterpret_cast<const uint8_t*>(g.GetLocalIDBasePointer()) +
      sizeof(VertexID) * (g.get_max_vid() + 1);
  size_t buf_size = h_end - h_buf;

  for (auto& gpu : per_gpu_states_) {
    CudaDeviceGuard guard(gpu.device_id);
    if (gpu.d_graph_data) cudaFree(gpu.d_graph_data);
    gpu.d_graph_data_size = buf_size;
    CUDA_CHECK(cudaMalloc(&gpu.d_graph_data, buf_size));
    CUDA_CHECK(cudaMemcpy(gpu.d_graph_data, h_buf, buf_size,
                          cudaMemcpyHostToDevice));
  }

  std::cout << "[GraphAggregate] Graph data replicated to "
            << per_gpu_states_.size() << " GPU(s) (" << buf_size
            << " bytes each)" << std::endl;
}

__host__ size_t GraphAggregate::ValueTypeSize(int32_t value_type) {
  return ValueTypeElementSize(static_cast<ValueType>(value_type));
}

__host__ uint32_t GraphAggregate::ComputeMaxDegree() const {
  if (!graph_) return 0;
  const uint32_t n = graph_->get_num_vertices();
  const VertexID* indegree = graph_->GetInDegreeBasePointer();
  const VertexID* outdegree = graph_->GetOutDegreeBasePointer();
  if (!indegree || !outdegree) return 0;

  uint32_t max_deg = 0;
  for (uint32_t v = 0; v < n; ++v) {
    max_deg = std::max(max_deg, static_cast<uint32_t>(indegree[v]));
    max_deg = std::max(max_deg, static_cast<uint32_t>(outdegree[v]));
  }
  return max_deg;
}

__host__ uint32_t GraphAggregate::ComputeLaunchMaxNeighbors(
    size_t* shared_mem_size) const {
  constexpr uint32_t kMaxNeighborsCap = 4096;
  uint32_t max_neighbors = std::min(max_degree_, kMaxNeighborsCap);

  size_t bytes = kernel::ComputeGraphAggregateSharedMemSize(max_neighbors);

  int min_device_max_shmem = std::numeric_limits<int>::max();
  for (const auto& gpu : per_gpu_states_) {
    CudaDeviceGuard guard(gpu.device_id);
    int dev_max = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &dev_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, gpu.device_id));
    min_device_max_shmem = std::min(min_device_max_shmem, dev_max);
  }

  while (max_neighbors > 0 &&
         bytes > static_cast<size_t>(min_device_max_shmem)) {
    max_neighbors = (max_neighbors > 256) ? max_neighbors - 256 : 0;
    bytes = kernel::ComputeGraphAggregateSharedMemSize(max_neighbors);
  }
  if (max_neighbors == 0) {
    max_neighbors = 1;
    bytes = kernel::ComputeGraphAggregateSharedMemSize(max_neighbors);
  }

  *shared_mem_size = bytes;
  return max_neighbors;
}

__host__ void GraphAggregate::LoadAttributes(
    uint32_t n_columns,
    const GraphAggregateAttributeColumn* columns) {
  if (!graph_) {
    std::cerr << "[GraphAggregate::LoadAttributes] Graph not loaded"
              << std::endl;
    return;
  }
  if (n_columns == 0 || columns == nullptr) return;

  const uint32_t n_vertices = graph_->get_num_vertices();

  // Validate columns.
  for (uint32_t c = 0; c < n_columns; ++c) {
    if (columns[c].n_values != n_vertices) {
      std::cerr << "[GraphAggregate::LoadAttributes] Column '" << columns[c].key
                << "' has " << columns[c].n_values << " values, expected "
                << n_vertices << std::endl;
      return;
    }
    size_t elem_size = ValueTypeSize(columns[c].value_type);
    if (elem_size == 0 || columns[c].values == nullptr) {
      std::cerr << "[GraphAggregate::LoadAttributes] Unsupported or empty column"
                << std::endl;
      return;
    }
  }

  // Build per-vertex attribute descriptors on the host once.
  uint32_t capacity = 1;
  while (capacity < static_cast<uint32_t>(n_columns / 0.7f) + 1) capacity <<= 1;

  std::vector<Attributes> h_attrs(n_vertices);
  std::vector<AttributeName> h_keys(n_vertices * capacity);
  std::vector<Attribute> h_values(n_vertices * capacity);
  std::vector<uint8_t> h_occupied(n_vertices * capacity, 0);

  // Compute the hash slot for each column once (same for every vertex).
  std::vector<uint32_t> column_slots(n_columns);

  for (uint32_t v = 0; v < n_vertices; ++v) {
    h_attrs[v].vertex_id = v;
    h_attrs[v].attr_map.size = n_columns;
    h_attrs[v].attr_map.capacity = capacity;
    h_attrs[v].attr_map.keys = h_keys.data() + v * capacity;
    h_attrs[v].attr_map.values = h_values.data() + v * capacity;
    h_attrs[v].attr_map.occupied = h_occupied.data() + v * capacity;

    for (uint32_t c = 0; c < n_columns; ++c) {
      AttributeName name(columns[c].key);
      uint32_t h = DefaultHash<AttributeName>{}(name);
      uint32_t idx = h & (capacity - 1);
      for (uint32_t probe = 0; probe < capacity; ++probe) {
        uint32_t slot = (idx + probe) & (capacity - 1);
        if (!h_occupied[v * capacity + slot]) {
          h_keys[v * capacity + slot] = name;
          Attribute& attr = h_values[v * capacity + slot];
          std::memset(&attr, 0, sizeof(Attribute));
          std::strncpy(attr.name, columns[c].key, sizeof(attr.name) - 1);
          attr.type = static_cast<ValueType>(columns[c].value_type);
          attr.n_rows = 1;
          attr.n_elements = 1;
          attr.offsets = nullptr;
          h_occupied[v * capacity + slot] = 1;
          if (v == 0) column_slots[c] = slot;
          break;
        }
      }
    }
  }

  // For each GPU: allocate column buffers, copy data, fix pointers, build.
  for (auto& gpu : per_gpu_states_) {
    CudaDeviceGuard guard(gpu.device_id);

    std::vector<uint8_t*> d_columns(n_columns);
    for (uint32_t c = 0; c < n_columns; ++c) {
      size_t elem_size = ValueTypeSize(columns[c].value_type);
      size_t bytes = elem_size * n_vertices;
      CUDA_CHECK(cudaMalloc(&d_columns[c], bytes));
      CUDA_CHECK(cudaMemcpy(d_columns[c], columns[c].values, bytes,
                            cudaMemcpyHostToDevice));
    }

    for (uint32_t v = 0; v < n_vertices; ++v) {
      for (uint32_t c = 0; c < n_columns; ++c) {
        size_t elem_size = ValueTypeSize(columns[c].value_type);
        h_values[v * capacity + column_slots[c]].data =
            d_columns[c] + v * elem_size;
      }
    }

    gpu.vertex_attrs.Build(h_attrs.data(), n_vertices);
    gpu.vertex_attrs.TakeColumnBuffers(std::move(d_columns));
  }

  std::cout << "[GraphAggregate] Loaded " << n_columns
            << " attribute column(s) on " << per_gpu_states_.size()
            << " GPU(s)" << std::endl;
}

__host__ void GraphAggregate::SetVertexAttributes(
    DevicePerVertexAttributes attrs) {
  if (per_gpu_states_.empty()) {
    std::cerr << "[GraphAggregate::SetVertexAttributes] No GPUs available"
              << std::endl;
    return;
  }

  // Store on the primary GPU.
  int target_device = attrs.HomeDevice();
  if (target_device < 0) {
    CUDA_CHECK(cudaGetDevice(&target_device));
  }
  primary_device_ = target_device;

  per_gpu_states_[0].vertex_attrs = std::move(attrs);

  // Clone to remaining GPUs.
  for (size_t i = 1; i < per_gpu_states_.size(); ++i) {
    per_gpu_states_[i].vertex_attrs =
        per_gpu_states_[0].vertex_attrs.CloneToDevice(per_gpu_states_[i].device_id);
  }
}

__host__ void GraphAggregate::Run() {
  std::cout << "[GraphAggregate] Run()" << std::endl;
  if (!graph_) {
    LoadGraph(graph_path_);
  }
}

__host__ void GraphAggregate::FreeDeviceBuffers() {
  for (auto& gpu : per_gpu_states_) {
    CudaDeviceGuard guard(gpu.device_id);
    if (gpu.d_graph_data) {
      cudaFree(gpu.d_graph_data);
      gpu.d_graph_data = nullptr;
      gpu.d_graph_data_size = 0;
    }
    gpu.vertex_attrs.Free();
    for (auto& sb : gpu.stream_buffers) {
      CudaDeviceGuard sb_guard(gpu.device_id);
      sb.Free();
    }
    gpu.stream_buffers.clear();
    for (cudaStream_t stream : gpu.streams) {
      if (stream) cudaStreamDestroy(stream);
    }
    gpu.streams.clear();
  }
}

// ---------------------------------------------------------------------------
// ComputeFeatures with multi-GPU + per-GPU stream parallelism.
// ---------------------------------------------------------------------------
__host__ std::vector<kernel::FeatureValue> GraphAggregate::ComputeFeatures(
    const std::vector<uint32_t>& pivot_vertex_ids,
    const std::vector<kernel::FeatureRequest>& requests) {
  using kernel::FeatureRequest;
  using kernel::FeatureValue;
  using kernel::ComputeFeaturesKernel;

  std::vector<FeatureValue> host_results;

  if (pivot_vertex_ids.empty() || requests.empty()) {
    std::cout << "[GraphAggregate::ComputeFeatures] empty input, skipping."
              << std::endl;
    return host_results;
  }

  if (!graph_) {
    std::cerr << "[GraphAggregate::ComputeFeatures] Graph not loaded"
              << std::endl;
    return host_results;
  }

  if (per_gpu_states_.empty() ||
      per_gpu_states_[0].vertex_attrs.GetNumVertices() !=
          graph_->get_num_vertices()) {
    std::cerr << "[GraphAggregate::ComputeFeatures] Vertex attributes not set"
              << std::endl;
    return host_results;
  }

  const uint32_t n_pivots = static_cast<uint32_t>(pivot_vertex_ids.size());
  const uint32_t n_requests = static_cast<uint32_t>(requests.size());
  host_results.resize(n_pivots * n_requests);

  // Pinned host buffer for async D2H copies.
  FeatureValue* pinned_results = nullptr;
  CUDA_CHECK(cudaMallocHost(&pinned_results,
                            sizeof(FeatureValue) * n_pivots * n_requests));

  // Copy requests once per GPU (they live in that GPU's memory).
  std::vector<FeatureRequest*> d_requests_per_gpu(per_gpu_states_.size());
  for (size_t g = 0; g < per_gpu_states_.size(); ++g) {
    CudaDeviceGuard guard(per_gpu_states_[g].device_id);
    FeatureRequest* d_requests = nullptr;
    CUDA_CHECK(cudaMalloc(&d_requests, sizeof(FeatureRequest) * n_requests));
    CUDA_CHECK(cudaMemcpy(d_requests, requests.data(),
                          sizeof(FeatureRequest) * n_requests,
                          cudaMemcpyHostToDevice));
    d_requests_per_gpu[g] = d_requests;
  }

  size_t shared_mem_size = 0;
  const uint32_t max_neighbors = ComputeLaunchMaxNeighbors(&shared_mem_size);

  // Opt-in to larger dynamic shared memory if the default 48 KB is insufficient.
  constexpr int kDefaultSharedMem = 48 * 1024;
  if (shared_mem_size > static_cast<size_t>(kDefaultSharedMem)) {
    for (auto& gpu : per_gpu_states_) {
      CudaDeviceGuard guard(gpu.device_id);
      CUDA_CHECK(cudaFuncSetAttribute(
          ComputeFeaturesKernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(shared_mem_size)));
    }
  }

  // Partition pivots across GPUs.
  const uint32_t n_gpus = static_cast<uint32_t>(per_gpu_states_.size());
  std::vector<uint32_t> gpu_offsets(n_gpus + 1, 0);
  for (uint32_t g = 0; g < n_gpus; ++g) {
    gpu_offsets[g + 1] =
        std::min(n_pivots,
                 gpu_offsets[g] + (n_pivots + n_gpus - 1) / n_gpus);
  }

  // Launch each GPU asynchronously.
  for (uint32_t g = 0; g < n_gpus; ++g) {
    const uint32_t begin = gpu_offsets[g];
    const uint32_t end = gpu_offsets[g + 1];
    if (begin >= end) continue;
    const uint32_t chunk_n = end - begin;

    PerGpuState& gpu_state = per_gpu_states_[g];
    CudaDeviceGuard guard(gpu_state.device_id);

    // Ensure streams and buffers.
    if (gpu_state.streams.empty()) {
      gpu_state.streams.resize(n_streams_);
      for (uint32_t s = 0; s < n_streams_; ++s) {
        CUDA_CHECK(cudaStreamCreate(&gpu_state.streams[s]));
      }
      gpu_state.stream_buffers.resize(n_streams_);
    }

    // Partition chunk across streams.
    std::vector<uint32_t> stream_offsets(n_streams_ + 1, 0);
    for (uint32_t s = 0; s < n_streams_; ++s) {
      stream_offsets[s + 1] =
          std::min(chunk_n,
                   stream_offsets[s] + (chunk_n + n_streams_ - 1) / n_streams_);
    }

    for (uint32_t s = 0; s < n_streams_; ++s) {
      const uint32_t s_begin = stream_offsets[s];
      const uint32_t s_end = stream_offsets[s + 1];
      if (s_begin >= s_end) continue;
      const uint32_t sub_chunk_n = s_end - s_begin;
      const uint32_t global_begin = begin + s_begin;

      StreamBuffers& sb = gpu_state.stream_buffers[s];
      sb.EnsureCapacity(sub_chunk_n, n_requests, max_neighbors);

      CUDA_CHECK(cudaMemcpyAsync(
          sb.d_pivot_vids,
          pivot_vertex_ids.data() + global_begin,
          sizeof(uint32_t) * sub_chunk_n,
          cudaMemcpyHostToDevice,
          gpu_state.streams[s]));

      const uint32_t grid_size = sub_chunk_n;
      ComputeFeaturesKernel<<<grid_size, common::kBlockDim,
                              shared_mem_size, gpu_state.streams[s]>>>(
          gpu_state.d_graph_data,
          graph_->get_num_vertices(),
          graph_->get_num_incoming_edges(),
          graph_->get_num_outgoing_edges(),
          gpu_state.vertex_attrs.GetDevicePtr(),
          sb.d_pivot_vids,
          sub_chunk_n,
          d_requests_per_gpu[g],
          n_requests,
          max_neighbors,
          sb.d_outputs);

      CUDA_CHECK(cudaMemcpyAsync(
          pinned_results + global_begin * n_requests,
          sb.d_outputs,
          sizeof(FeatureValue) * sub_chunk_n * n_requests,
          cudaMemcpyDeviceToHost,
          gpu_state.streams[s]));
    }
  }

  // Synchronize all streams on all GPUs.
  for (auto& gpu_state : per_gpu_states_) {
    CudaDeviceGuard guard(gpu_state.device_id);
    for (cudaStream_t stream : gpu_state.streams) {
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }

  for (void* ptr : d_requests_per_gpu) {
    if (ptr) cudaFree(ptr);
  }

  std::memcpy(host_results.data(), pinned_results,
              sizeof(FeatureValue) * n_pivots * n_requests);
  CUDA_CHECK(cudaFreeHost(pinned_results));

  return host_results;
}

// ---------------------------------------------------------------------------
// ComputeAll with multi-GPU + per-GPU stream parallelism.
// ---------------------------------------------------------------------------
__host__ std::vector<kernel::AllFeatures> GraphAggregate::ComputeAll(
    const std::vector<uint32_t>& pivot_vertex_ids,
    const kernel::AttributeName& attr_name,
    bool use_outgoing) {
  using kernel::AllFeatures;
  using kernel::ComputeAllFeaturesKernel;

  std::vector<AllFeatures> host_results;

  if (pivot_vertex_ids.empty()) {
    std::cout << "[GraphAggregate::ComputeAll] empty input, skipping."
              << std::endl;
    return host_results;
  }

  if (!graph_) {
    std::cerr << "[GraphAggregate::ComputeAll] Graph not loaded" << std::endl;
    return host_results;
  }

  if (per_gpu_states_.empty() ||
      per_gpu_states_[0].vertex_attrs.GetNumVertices() !=
          graph_->get_num_vertices()) {
    std::cerr << "[GraphAggregate::ComputeAll] Vertex attributes not set"
              << std::endl;
    return host_results;
  }

  const uint32_t n_pivots = static_cast<uint32_t>(pivot_vertex_ids.size());
  host_results.resize(n_pivots);

  // Pinned host buffer for async D2H copies.
  AllFeatures* pinned_results = nullptr;
  CUDA_CHECK(cudaMallocHost(&pinned_results,
                            sizeof(AllFeatures) * n_pivots));

  size_t shared_mem_size = 0;
  const uint32_t max_neighbors = ComputeLaunchMaxNeighbors(&shared_mem_size);
  std::cout << "[GraphAggregate::ComputeAll] max_neighbors=" << max_neighbors
            << " shared_mem=" << shared_mem_size << " bytes" << std::endl;

  constexpr int kDefaultSharedMem = 48 * 1024;
  if (shared_mem_size > static_cast<size_t>(kDefaultSharedMem)) {
    for (auto& gpu : per_gpu_states_) {
      CudaDeviceGuard guard(gpu.device_id);
      CUDA_CHECK(cudaFuncSetAttribute(
          ComputeAllFeaturesKernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(shared_mem_size)));
    }
  }

  cudaEvent_t timer_start, timer_stop;
  CUDA_CHECK(cudaEventCreate(&timer_start));
  CUDA_CHECK(cudaEventCreate(&timer_stop));
  CUDA_CHECK(cudaEventRecord(timer_start));

  const uint32_t n_gpus = static_cast<uint32_t>(per_gpu_states_.size());
  std::vector<uint32_t> gpu_offsets(n_gpus + 1, 0);
  for (uint32_t g = 0; g < n_gpus; ++g) {
    gpu_offsets[g + 1] =
        std::min(n_pivots,
                 gpu_offsets[g] + (n_pivots + n_gpus - 1) / n_gpus);
  }

  for (uint32_t g = 0; g < n_gpus; ++g) {
    const uint32_t begin = gpu_offsets[g];
    const uint32_t end = gpu_offsets[g + 1];
    if (begin >= end) continue;
    const uint32_t chunk_n = end - begin;

    PerGpuState& gpu_state = per_gpu_states_[g];
    CudaDeviceGuard guard(gpu_state.device_id);

    if (gpu_state.streams.empty()) {
      gpu_state.streams.resize(n_streams_);
      for (uint32_t s = 0; s < n_streams_; ++s) {
        CUDA_CHECK(cudaStreamCreate(&gpu_state.streams[s]));
      }
      gpu_state.stream_buffers.resize(n_streams_);
    }

    std::vector<uint32_t> stream_offsets(n_streams_ + 1, 0);
    for (uint32_t s = 0; s < n_streams_; ++s) {
      stream_offsets[s + 1] =
          std::min(chunk_n,
                   stream_offsets[s] + (chunk_n + n_streams_ - 1) / n_streams_);
    }

    for (uint32_t s = 0; s < n_streams_; ++s) {
      const uint32_t s_begin = stream_offsets[s];
      const uint32_t s_end = stream_offsets[s + 1];
      if (s_begin >= s_end) continue;
      const uint32_t sub_chunk_n = s_end - s_begin;
      const uint32_t global_begin = begin + s_begin;

      StreamBuffers& sb = gpu_state.stream_buffers[s];
      sb.EnsureCapacity(sub_chunk_n, 1, max_neighbors);

      CUDA_CHECK(cudaMemcpyAsync(
          sb.d_pivot_vids,
          pivot_vertex_ids.data() + global_begin,
          sizeof(uint32_t) * sub_chunk_n,
          cudaMemcpyHostToDevice,
          gpu_state.streams[s]));

      const uint32_t grid_size = sub_chunk_n;
      ComputeAllFeaturesKernel<<<grid_size, common::kAllFeaturesBlockDim,
                                 shared_mem_size, gpu_state.streams[s]>>>(
          gpu_state.d_graph_data,
          graph_->get_num_vertices(),
          graph_->get_num_incoming_edges(),
          graph_->get_num_outgoing_edges(),
          gpu_state.vertex_attrs.GetDevicePtr(),
          sb.d_pivot_vids,
          sub_chunk_n,
          attr_name,
          use_outgoing,
          max_neighbors,
          sb.d_all_outputs);

      CUDA_CHECK(cudaMemcpyAsync(
          pinned_results + global_begin,
          sb.d_all_outputs,
          sizeof(AllFeatures) * sub_chunk_n,
          cudaMemcpyDeviceToHost,
          gpu_state.streams[s]));
    }
  }

  for (auto& gpu_state : per_gpu_states_) {
    CudaDeviceGuard guard(gpu_state.device_id);
    for (cudaStream_t stream : gpu_state.streams) {
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }

  CUDA_CHECK(cudaEventRecord(timer_stop));
  CUDA_CHECK(cudaEventSynchronize(timer_stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
  cudaEventDestroy(timer_start);
  cudaEventDestroy(timer_stop);
  std::cout << "[GraphAggregate::ComputeAll] GPU kernel time: "
            << elapsed_ms << " ms" << std::endl;

  std::memcpy(host_results.data(), pinned_results,
              sizeof(AllFeatures) * n_pivots);
  CUDA_CHECK(cudaFreeHost(pinned_results));

  return host_results;
}

// ---------------------------------------------------------------------------
// Synthetic data generation (for testing)
// ---------------------------------------------------------------------------
__host__ void GraphAggregate::LoadSyntheticData(
    uint32_t n_vertices, uint32_t out_degree_per_vertex) {
  FreeDeviceBuffers();
  BuildPerGpuStates();

  std::cout << "[GraphAggregate] Adding synthetic graph: " << n_vertices
            << " vertices, out-degree=" << out_degree_per_vertex << std::endl;

  graph_ = std::make_unique<ImmutableCSR>();
  ImmutableCSR* csr = graph_.get();

  uint32_t n_edges = n_vertices * out_degree_per_vertex;
  uint32_t max_vid = n_vertices - 1;

  size_t buf_size =
      sizeof(VertexID) * n_vertices +                    // globalid
      sizeof(VertexID) * n_vertices +                    // indegree
      sizeof(VertexID) * n_vertices +                    // outdegree
      sizeof(EdgeIndex) * (n_vertices + 1) +             // in_offset
      sizeof(EdgeIndex) * (n_vertices + 1) +             // out_offset
      sizeof(VertexID) * n_edges +                       // incoming_edges
      sizeof(VertexID) * n_edges +                       // outgoing_edges
      sizeof(VertexID) * (max_vid + 1) +                 // edges_globalid
      sizeof(VertexID) * (max_vid + 1);                  // localid_by_globalid

  uint8_t* buf = new uint8_t[buf_size]();
  csr->SetNumVertices(n_vertices);
  csr->SetNumIncomingEdges(n_edges);
  csr->SetNumOutgoingEdges(n_edges);
  csr->SetMaxVid(max_vid);
  csr->SetMinVid(0);
  csr->SetGraphBuffer(buf);
  csr->ParseBasePtr(buf);

  VertexID* globalid = csr->GetGloablIDBasePointer();
  VertexID* indegree = csr->GetInDegreeBasePointer();
  VertexID* outdegree = csr->GetOutDegreeBasePointer();
  EdgeIndex* in_offset = csr->GetInOffsetBasePointer();
  EdgeIndex* out_offset = csr->GetOutOffsetBasePointer();
  VertexID* incoming_edges = csr->GetIncomingEdgesBasePointer();
  VertexID* outgoing_edges = csr->GetOutgoingEdgesBasePointer();

  std::vector<std::vector<VertexID>> out_edges_host(n_vertices);
  std::vector<std::vector<VertexID>> in_edges_host(n_vertices);

  for (uint32_t v = 0; v < n_vertices; ++v) {
    globalid[v] = v;
    for (uint32_t e = 0; e < out_degree_per_vertex; ++e) {
      VertexID dst = (v + 1 + e) % n_vertices;
      out_edges_host[v].push_back(dst);
      in_edges_host[dst].push_back(v);
    }
  }

  out_offset[0] = 0;
  for (uint32_t v = 0; v < n_vertices; ++v) {
    outdegree[v] = static_cast<VertexID>(out_edges_host[v].size());
    out_offset[v + 1] = out_offset[v] + outdegree[v];
    for (uint32_t i = 0; i < outdegree[v]; ++i) {
      outgoing_edges[out_offset[v] + i] = out_edges_host[v][i];
    }
  }

  in_offset[0] = 0;
  for (uint32_t v = 0; v < n_vertices; ++v) {
    indegree[v] = static_cast<VertexID>(in_edges_host[v].size());
    in_offset[v + 1] = in_offset[v] + indegree[v];
    for (uint32_t i = 0; i < indegree[v]; ++i) {
      incoming_edges[in_offset[v] + i] = in_edges_host[v][i];
    }
  }

  VertexID* edges_globalid = csr->GetEdgesGloablIDBasePointer();
  VertexID* localid = csr->GetLocalIDBasePointer();
  for (uint32_t i = 0; i <= max_vid; ++i) {
    edges_globalid[i] = i;
    localid[i] = i;
  }

  max_degree_ = out_degree_per_vertex;
  TransferGraphToAllGpus();
  GenerateSyntheticAttributes();

  std::cout << "[GraphAggregate] Synthetic graph ready." << std::endl;
}

__host__ void GraphAggregate::GenerateSyntheticAttributes() {
  if (!graph_) {
    std::cerr << "[GraphAggregate::GenerateSyntheticAttributes] Graph not loaded"
              << std::endl;
    return;
  }

  const uint32_t n_vertices = graph_->get_num_vertices();
  std::vector<double> h_scores(n_vertices);
  std::vector<uint8_t> h_flags(n_vertices);
  for (uint32_t v = 0; v < n_vertices; ++v) {
    h_scores[v] = v * 0.5;
    h_flags[v] = (v % 2 == 0) ? 1 : 0;
  }

  GraphAggregateAttributeColumn cols[2];
  std::strncpy(cols[0].key, "score", sizeof(cols[0].key) - 1);
  cols[0].value_type = static_cast<int32_t>(ValueType::kFloat64);
  cols[0].n_values = n_vertices;
  cols[0].values = h_scores.data();

  std::strncpy(cols[1].key, "flag", sizeof(cols[1].key) - 1);
  cols[1].value_type = static_cast<int32_t>(ValueType::kBool);
  cols[1].n_values = n_vertices;
  cols[1].values = h_flags.data();

  LoadAttributes(2, cols);
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
