#ifndef MATRIXGRAPH_CORE_TASK_KERNEL_DATA_STRUCTURES_IMMUTABLE_CSR_GPU_CUH_
#define MATRIXGRAPH_CORE_TASK_KERNEL_DATA_STRUCTURES_IMMUTABLE_CSR_GPU_CUH_

#include "core/common/types.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/metadata.h"
#include "core/util/cuda_check.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

class ImmutableCSRGPU {
private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using GraphID = sics::matrixgraph::core::common::GraphID;
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using SubGraphMetadata =
      sics::matrixgraph::core::data_structures::SubGraphMetadata;
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;
  using ImmutableCSRVertex =
      sics::matrixgraph::core::data_structures::ImmutableCSRVertex;

public:
  ImmutableCSRGPU(const ImmutableCSR &csr_cpu);

  ImmutableCSRGPU() = default;

  ~ImmutableCSRGPU() = default;

  void Init(const ImmutableCSR &csr_cpu);

  void Free() { CUDA_CHECK(cudaFree(graph_base_pointer_)); }

  void ParseBasePtr(uint8_t *graph_base_pointer);

  __host__ __device__ void SetGraphBuffer(uint8_t *buffer) {
    graph_base_pointer_ = buffer;
  }

  __host__ __device__ void SetGlobalIDBuffer(VertexID *buffer) {
    globalid_by_localid_base_pointer_ = buffer;
  }

  __host__ __device__ void SetEdgesGlobalIDBuffer(VertexID *buffer) {
    edges_globalid_by_localid_base_pointer_ = buffer;
  }

  __host__ __device__ void SetInDegreeBuffer(VertexID *buffer) {
    indegree_base_pointer_ = buffer;
  }

  __host__ __device__ void SetOutDegreeBuffer(VertexID *buffer) {
    outdegree_base_pointer_ = buffer;
  }

  __host__ __device__ void SetInOffsetBuffer(EdgeIndex *buffer) {
    in_offset_base_pointer_ = buffer;
  }

  __host__ __device__ void SetOutOffsetBuffer(EdgeIndex *buffer) {
    out_offset_base_pointer_ = buffer;
  }

  __host__ __device__ void SetIncomingEdgesBuffer(VertexID *buffer) {
    incoming_edges_base_pointer_ = buffer;
  }

  __host__ __device__ void SetOutgoingEdgesBuffer(VertexID *buffer) {
    outgoing_edges_base_pointer_ = buffer;
  }

  __host__ __device__ void SetNumVertices(VertexID num_vertices) {
    metadata_.num_vertices = num_vertices;
  }

  __host__ __device__ void SetNumIncomingEdges(EdgeIndex num_incoming_edges) {
    metadata_.num_incoming_edges = num_incoming_edges;
  }

  __host__ __device__ void SetNumOutgoingEdges(EdgeIndex num_outgoing_edges) {
    metadata_.num_outgoing_edges = num_outgoing_edges;
  }

  __host__ __device__ void SetMaxVid(VertexID max_vid) {
    metadata_.max_vid = max_vid;
  }

  __host__ __device__ void SetMinVid(VertexID min_vid) {
    metadata_.min_vid = min_vid;
  }

  __host__ __device__ VertexID get_num_vertices() const {
    return metadata_.num_vertices;
  }

  __host__ __device__ EdgeIndex get_num_incoming_edges() const {
    return metadata_.num_incoming_edges;
  }

  __host__ __device__ EdgeIndex get_num_outgoing_edges() const {
    return metadata_.num_outgoing_edges;
  }

  __host__ __device__ VertexID get_max_vid() const { return metadata_.max_vid; }

  __host__ __device__ VertexID get_min_vid() const { return metadata_.min_vid; }

  __host__ __device__ uint8_t *GetGraphBuffer() const {
    return graph_base_pointer_;
  }

  __host__ __device__ VertexID *GetGloablIDBasePointer() const {
    return globalid_by_localid_base_pointer_;
  }

  __host__ __device__ VertexID *GetEdgesGloablIDBasePointer() const {
    return edges_globalid_by_localid_base_pointer_;
  }

  __host__ __device__ VertexID *GetInDegreeBasePointer() const {
    return indegree_base_pointer_;
  }

  __host__ __device__ VertexID *GetOutDegreeBasePointer() const {
    return outdegree_base_pointer_;
  }

  __host__ __device__ EdgeIndex *GetInOffsetBasePointer() const {
    return in_offset_base_pointer_;
  }

  __host__ __device__ EdgeIndex *GetOutOffsetBasePointer() const {
    return out_offset_base_pointer_;
  }

  __host__ __device__ VertexID *GetIncomingEdgesBasePointer() const {
    return incoming_edges_base_pointer_;
  }

  __host__ __device__ VertexID *GetOutgoingEdgesBasePointer() const {
    return outgoing_edges_base_pointer_;
  }

  __host__ __device__ VertexLabel *GetVLabelBasePointer() const {
    return vertex_label_base_pointer_;
  }

  __host__ __device__ VertexID GetGlobalIDByLocalID(VertexID i) const {
    return globalid_by_localid_base_pointer_[i];
  }

  __host__ __device__ VertexID GetEdgeGlobalIDByLocalID(VertexID i) const {
    return edges_globalid_by_localid_base_pointer_[i];
  }

  __host__ __device__ VertexID GetInOffsetByLocalID(VertexID i) const {
    return in_offset_base_pointer_[i];
  }

  __host__ __device__ VertexID GetOutOffsetByLocalID(VertexID i) const {
    return out_offset_base_pointer_[i];
  }

  __host__ __device__ VertexID *GetIncomingEdgesByLocalID(VertexID i) const {
    return incoming_edges_base_pointer_ + in_offset_base_pointer_[i];
  }

  __host__ __device__ VertexID *GetOutgoingEdgesByLocalID(VertexID i) const {
    return outgoing_edges_base_pointer_ + out_offset_base_pointer_[i];
  }

  __host__ __device__ VertexID GetInDegreeByLocalID(VertexID i) const {
    return indegree_base_pointer_[i];
  }

  __host__ __device__ VertexID GetOutDegreeByLocalID(VertexID i) const {
    return outdegree_base_pointer_[i];
  }

  __host__ __device__ ImmutableCSRVertex GetVertexByLocalID(VertexID i) const;

private:
  // Metadata to build the CSR.
  SubGraphMetadata metadata_;

  // Serialized data in CSR format.
  uint8_t *graph_base_pointer_ = nullptr;

  VertexID *globalid_by_localid_base_pointer_ = nullptr;
  VertexID *edges_globalid_by_localid_base_pointer_ = nullptr;

  VertexID *local_vid_by_edges_globalid_base_pointer_ = nullptr;

  VertexID *incoming_edges_base_pointer_ = nullptr;
  VertexID *outgoing_edges_base_pointer_ = nullptr;
  VertexID *indegree_base_pointer_ = nullptr;
  VertexID *outdegree_base_pointer_ = nullptr;
  EdgeIndex *in_offset_base_pointer_ = nullptr;
  EdgeIndex *out_offset_base_pointer_ = nullptr;

  VertexLabel *vertex_label_base_pointer_ = nullptr;
}; // namespace kernel

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_TASK_KERNEL_DATA_STRUCTURES_IMMUTABLE_CSR_GPU_CUH_