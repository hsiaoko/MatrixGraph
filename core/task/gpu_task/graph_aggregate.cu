#include "core/task/gpu_task/graph_aggregate.cuh"

#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>

#include "core/util/cuda_check.cuh"

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

// ---------------------------------------------------------------------------
// DevicePerVertexAttributes
// ---------------------------------------------------------------------------
GraphAggregate::DevicePerVertexAttributes::DevicePerVertexAttributes(
    DevicePerVertexAttributes&& other) noexcept {
  d_attrs_ = other.d_attrs_;
  n_vertices_ = other.n_vertices_;
  d_hash_keys_ = other.d_hash_keys_;
  d_hash_values_ = other.d_hash_values_;
  d_hash_occupied_ = other.d_hash_occupied_;
  total_hash_capacity_ = other.total_hash_capacity_;
  other.d_attrs_ = nullptr;
  other.d_hash_keys_ = nullptr;
  other.d_hash_values_ = nullptr;
  other.d_hash_occupied_ = nullptr;
  other.n_vertices_ = 0;
  other.total_hash_capacity_ = 0;
}

GraphAggregate::DevicePerVertexAttributes&
GraphAggregate::DevicePerVertexAttributes::operator=(
    DevicePerVertexAttributes&& other) noexcept {
  if (this != &other) {
    Free();
    d_attrs_ = other.d_attrs_;
    n_vertices_ = other.n_vertices_;
    d_hash_keys_ = other.d_hash_keys_;
    d_hash_values_ = other.d_hash_values_;
    d_hash_occupied_ = other.d_hash_occupied_;
    total_hash_capacity_ = other.total_hash_capacity_;
    other.d_attrs_ = nullptr;
    other.d_hash_keys_ = nullptr;
    other.d_hash_values_ = nullptr;
    other.d_hash_occupied_ = nullptr;
    other.n_vertices_ = 0;
    other.total_hash_capacity_ = 0;
  }
  return *this;
}

__host__ void GraphAggregate::DevicePerVertexAttributes::Build(
    const Attributes* h_attrs, uint32_t n_vertices) {
  if (n_vertices == 0) return;
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

  // 5. Copy each vertex's HashMap bucket data to device.
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
      CUDA_CHECK(cudaMemcpy(d_hash_values_ + offset, h_attrs[i].attr_map.values,
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

  delete[] h_device_attrs;
}

__host__ void GraphAggregate::DevicePerVertexAttributes::Free() {
  if (d_attrs_) cudaFree(d_attrs_);
  if (d_hash_keys_) cudaFree(d_hash_keys_);
  if (d_hash_values_) cudaFree(d_hash_values_);
  if (d_hash_occupied_) cudaFree(d_hash_occupied_);
  d_attrs_ = nullptr;
  d_hash_keys_ = nullptr;
  d_hash_values_ = nullptr;
  d_hash_occupied_ = nullptr;
  n_vertices_ = 0;
  total_hash_capacity_ = 0;
}

// ---------------------------------------------------------------------------
// GraphAggregate
// ---------------------------------------------------------------------------
__host__ void GraphAggregate::Run() {
  std::cout << "[GraphAggregate] Run()" << std::endl;
  LoadData();
  BuildDeviceAttributes();
  // TODO: launch kernels based on user requirements.
}

__host__ void GraphAggregate::LoadData() {
  std::cout << "[GraphAggregate] Loading " << data_graph_paths_.size()
            << " graph(s)" << std::endl;
  graphs_.resize(data_graph_paths_.size());
  for (size_t i = 0; i < data_graph_paths_.size(); ++i) {
    graphs_[i] = std::make_unique<ImmutableCSR>();
    graphs_[i]->Read(data_graph_paths_[i]);
    std::cout << "[GraphAggregate] Graph " << i << " loaded: "
              << graphs_[i]->get_num_vertices() << " vertices" << std::endl;
  }
}

__host__ void GraphAggregate::BuildDeviceAttributes() {
  per_graph_vertex_attrs_.resize(graphs_.size());
  for (size_t i = 0; i < graphs_.size(); ++i) {
    uint32_t n = graphs_[i]->get_num_vertices();
    // TODO: replace this placeholder with real per-vertex Attributes data.
    // For now we allocate empty Attributes arrays (zero-initialized).
    Attributes* h_empty = new Attributes[n]();
    per_graph_vertex_attrs_[i].Build(h_empty, n);
    delete[] h_empty;
  }
  std::cout << "[GraphAggregate] Device attributes allocated for "
            << graphs_.size() << " graph(s)" << std::endl;
}

__host__ void GraphAggregate::TransferGraphDataToDevice() {
  size_t n = graphs_.size();
  d_graph_data_buffers_.resize(n);
  graph_buffer_sizes_.resize(n);

  for (size_t i = 0; i < n; ++i) {
    const ImmutableCSR& g = *graphs_[i];
    const uint8_t* h_buf = g.GetGraphBuffer();
    const uint8_t* h_end = reinterpret_cast<const uint8_t*>(g.GetLocalIDBasePointer())
                           + sizeof(VertexID) * (g.get_max_vid() + 1);
    size_t buf_size = h_end - h_buf;
    graph_buffer_sizes_[i] = buf_size;

    CUDA_CHECK(cudaMalloc(&d_graph_data_buffers_[i], buf_size));
    CUDA_CHECK(cudaMemcpy(d_graph_data_buffers_[i], h_buf, buf_size,
                          cudaMemcpyHostToDevice));
  }

  // Build pointer arrays on device.
  std::vector<uint8_t*> h_data_ptrs(n);
  std::vector<Attributes*> h_attr_ptrs(n);
  std::vector<uint32_t> h_n_vertices(n);
  std::vector<uint32_t> h_n_in_edges(n);
  std::vector<uint32_t> h_n_out_edges(n);

  for (size_t i = 0; i < n; ++i) {
    h_data_ptrs[i] = d_graph_data_buffers_[i];
    h_attr_ptrs[i] = per_graph_vertex_attrs_[i].GetDevicePtr();
    h_n_vertices[i] = graphs_[i]->get_num_vertices();
    h_n_in_edges[i] = graphs_[i]->get_num_incoming_edges();
    h_n_out_edges[i] = graphs_[i]->get_num_outgoing_edges();
  }

  CUDA_CHECK(cudaMalloc(&d_graph_data_array_, sizeof(uint8_t*) * n));
  CUDA_CHECK(cudaMemcpy(d_graph_data_array_, h_data_ptrs.data(),
                        sizeof(uint8_t*) * n, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_vertex_attrs_array_, sizeof(Attributes*) * n));
  CUDA_CHECK(cudaMemcpy(d_vertex_attrs_array_, h_attr_ptrs.data(),
                        sizeof(Attributes*) * n, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_graph_n_vertices_, sizeof(uint32_t) * n));
  CUDA_CHECK(cudaMemcpy(d_graph_n_vertices_, h_n_vertices.data(),
                        sizeof(uint32_t) * n, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_graph_n_in_edges_, sizeof(uint32_t) * n));
  CUDA_CHECK(cudaMemcpy(d_graph_n_in_edges_, h_n_in_edges.data(),
                        sizeof(uint32_t) * n, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_graph_n_out_edges_, sizeof(uint32_t) * n));
  CUDA_CHECK(cudaMemcpy(d_graph_n_out_edges_, h_n_out_edges.data(),
                        sizeof(uint32_t) * n, cudaMemcpyHostToDevice));

  std::cout << "[GraphAggregate] Graph data transferred to device ("
            << n << " graph(s))" << std::endl;
}

__host__ void GraphAggregate::FreeDeviceBuffers() {
  for (auto& ptr : d_graph_data_buffers_) {
    if (ptr) cudaFree(ptr);
  }
  d_graph_data_buffers_.clear();

  if (d_graph_data_array_) cudaFree(d_graph_data_array_);
  if (d_vertex_attrs_array_) cudaFree(d_vertex_attrs_array_);
  if (d_graph_n_vertices_) cudaFree(d_graph_n_vertices_);
  if (d_graph_n_in_edges_) cudaFree(d_graph_n_in_edges_);
  if (d_graph_n_out_edges_) cudaFree(d_graph_n_out_edges_);

  d_graph_data_array_ = nullptr;
  d_vertex_attrs_array_ = nullptr;
  d_graph_n_vertices_ = nullptr;
  d_graph_n_in_edges_ = nullptr;
  d_graph_n_out_edges_ = nullptr;
}

__host__ void GraphAggregate::ComputeFeatures(
    const std::vector<uint32_t>& pivot_graph_ids,
    const std::vector<uint32_t>& pivot_vertex_ids,
    const std::vector<kernel::FeatureRequest>& requests,
    std::vector<kernel::FeatureValue>* out_values) {
  using kernel::FeatureRequest;
  using kernel::FeatureValue;
  using kernel::ComputeFeaturesKernel;

  if (pivot_graph_ids.empty() || pivot_vertex_ids.empty() || requests.empty()) {
    std::cout << "[GraphAggregate::ComputeFeatures] empty input, skipping."
              << std::endl;
    return;
  }

  // Ensure graph data is on device.
  if (d_graph_data_array_ == nullptr) {
    TransferGraphDataToDevice();
  }

  uint32_t n_pivots = static_cast<uint32_t>(pivot_graph_ids.size());
  uint32_t n_requests = static_cast<uint32_t>(requests.size());

  // Copy pivots to device.
  uint32_t* d_pivot_graph_id = nullptr;
  uint32_t* d_pivot_vertex_id = nullptr;
  CUDA_CHECK(cudaMalloc(&d_pivot_graph_id, sizeof(uint32_t) * n_pivots));
  CUDA_CHECK(cudaMalloc(&d_pivot_vertex_id, sizeof(uint32_t) * n_pivots));
  CUDA_CHECK(cudaMemcpy(d_pivot_graph_id, pivot_graph_ids.data(),
                        sizeof(uint32_t) * n_pivots, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pivot_vertex_id, pivot_vertex_ids.data(),
                        sizeof(uint32_t) * n_pivots, cudaMemcpyHostToDevice));

  // Copy requests to device.
  FeatureRequest* d_requests = nullptr;
  CUDA_CHECK(cudaMalloc(&d_requests, sizeof(FeatureRequest) * n_requests));
  CUDA_CHECK(cudaMemcpy(d_requests, requests.data(),
                        sizeof(FeatureRequest) * n_requests, cudaMemcpyHostToDevice));

  // Determine max_neighbors. For now, use a configurable limit.
  // TODO: scan actual degrees for tighter bound.
  uint32_t max_neighbors = 4096;

  // Allocate workspace and outputs.
  FeatureValue* d_workspace = nullptr;
  FeatureValue* d_outputs = nullptr;
  CUDA_CHECK(cudaMalloc(&d_workspace,
                        sizeof(FeatureValue) * n_pivots * max_neighbors));
  CUDA_CHECK(cudaMalloc(&d_outputs,
                        sizeof(FeatureValue) * n_pivots * n_requests));

  // Launch kernel.
  const uint32_t block_size = 256;
  const uint32_t grid_size = (n_pivots + block_size - 1) / block_size;
  ComputeFeaturesKernel<<<grid_size, block_size>>>(
      d_graph_data_array_,
      d_graph_n_vertices_,
      d_graph_n_in_edges_,
      d_graph_n_out_edges_,
      d_vertex_attrs_array_,
      d_pivot_graph_id,
      d_pivot_vertex_id,
      n_pivots,
      d_requests,
      n_requests,
      d_workspace,
      max_neighbors,
      d_outputs);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy back if requested.
  if (out_values != nullptr) {
    out_values->resize(n_pivots * n_requests);
    CUDA_CHECK(cudaMemcpy(out_values->data(), d_outputs,
                          sizeof(FeatureValue) * n_pivots * n_requests,
                          cudaMemcpyDeviceToHost));
  }

  // Cleanup temporary device memory.
  cudaFree(d_pivot_graph_id);
  cudaFree(d_pivot_vertex_id);
  cudaFree(d_requests);
  cudaFree(d_workspace);
  cudaFree(d_outputs);
}

// ---------------------------------------------------------------------------
// Synthetic data generation (for testing)
// ---------------------------------------------------------------------------
__host__ void GraphAggregate::LoadSyntheticData(uint32_t n_vertices,
                                                uint32_t out_degree_per_vertex) {
  std::cout << "[GraphAggregate] Loading synthetic data: " << n_vertices
            << " vertices, out-degree=" << out_degree_per_vertex << std::endl;

  // 1. Build a synthetic directed ring-like CSR graph.
  graphs_.resize(1);
  graphs_[0] = std::make_unique<ImmutableCSR>();
  ImmutableCSR* csr = graphs_[0].get();

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

  // 2. Build synthetic per-vertex Attributes.
  //    Each vertex has: "score" (double) = v * 0.5,  "flag" (bool) = v%2==0.
  std::vector<double> h_scores(n_vertices);
  std::vector<uint8_t> h_flags(n_vertices);
  for (uint32_t v = 0; v < n_vertices; ++v) {
    h_scores[v] = v * 0.5;
    h_flags[v] = (v % 2 == 0) ? 1 : 0;
  }

  double* d_scores = nullptr;
  uint8_t* d_flags = nullptr;
  CUDA_CHECK(cudaMalloc(&d_scores, sizeof(double) * n_vertices));
  CUDA_CHECK(cudaMalloc(&d_flags, sizeof(uint8_t) * n_vertices));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), sizeof(double) * n_vertices,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_flags, h_flags.data(), sizeof(uint8_t) * n_vertices,
                        cudaMemcpyHostToDevice));

  // We intentionally leak d_scores / d_flags for the synthetic test lifetime.
  // In production code these would be owned by a proper buffer manager.

  per_graph_vertex_attrs_.resize(1);
  uint32_t n_attrs = 2;
  uint32_t capacity = 1;
  while (capacity < static_cast<uint32_t>(n_attrs / 0.7f) + 1) capacity <<= 1;

  std::vector<Attributes> h_attrs(n_vertices);
  std::vector<AttributeName> h_keys(n_vertices * capacity);
  std::vector<Attribute> h_values(n_vertices * capacity);
  std::vector<uint8_t> h_occupied(n_vertices * capacity, 0);

  for (uint32_t v = 0; v < n_vertices; ++v) {
    h_attrs[v].vertex_id = v;
    h_attrs[v].attr_map.size = n_attrs;
    h_attrs[v].attr_map.capacity = capacity;
    h_attrs[v].attr_map.keys = h_keys.data() + v * capacity;
    h_attrs[v].attr_map.values = h_values.data() + v * capacity;
    h_attrs[v].attr_map.occupied = h_occupied.data() + v * capacity;

    // Insert "score"
    uint32_t h = DefaultHash<AttributeName>{}(AttributeName("score"));
    uint32_t idx = h & (capacity - 1);
    for (uint32_t probe = 0; probe < capacity; ++probe) {
      uint32_t slot = (idx + probe) & (capacity - 1);
      if (!h_occupied[v * capacity + slot]) {
        h_keys[v * capacity + slot] = AttributeName("score");
        h_values[v * capacity + slot].type = ValueType::kFloat64;
        h_values[v * capacity + slot].n_rows = 1;
        h_values[v * capacity + slot].n_elements = 1;
        h_values[v * capacity + slot].data = d_scores + v;  // device pointer
        h_values[v * capacity + slot].offsets = nullptr;
        h_occupied[v * capacity + slot] = 1;
        break;
      }
    }

    // Insert "flag"
    h = DefaultHash<AttributeName>{}(AttributeName("flag"));
    idx = h & (capacity - 1);
    for (uint32_t probe = 0; probe < capacity; ++probe) {
      uint32_t slot = (idx + probe) & (capacity - 1);
      if (!h_occupied[v * capacity + slot]) {
        h_keys[v * capacity + slot] = AttributeName("flag");
        h_values[v * capacity + slot].type = ValueType::kBool;
        h_values[v * capacity + slot].n_rows = 1;
        h_values[v * capacity + slot].n_elements = 1;
        h_values[v * capacity + slot].data = d_flags + v;  // device pointer
        h_values[v * capacity + slot].offsets = nullptr;
        h_occupied[v * capacity + slot] = 1;
        break;
      }
    }
  }

  per_graph_vertex_attrs_[0].Build(h_attrs.data(), n_vertices);
  std::cout << "[GraphAggregate] Synthetic data ready." << std::endl;
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
