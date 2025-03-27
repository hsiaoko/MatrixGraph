#include "core/data_structures/immutable_csr_gpu.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

ImmutableCSRGPU::ImmutableCSRGPU(const ImmutableCSR& csr) { Init(csr); }

void ImmutableCSRGPU::Init(const ImmutableCSR& csr) {
  metadata_ = csr.GetMetadata();

  size_t data_size = sizeof(VertexID) * get_num_vertices() +
                     sizeof(VertexID) * get_num_vertices() +
                     sizeof(VertexID) * get_num_vertices() +
                     sizeof(EdgeIndex) * (get_num_vertices() + 1) +
                     sizeof(EdgeIndex) * (get_num_vertices() + 1) +
                     sizeof(VertexID) * get_num_incoming_edges() +
                     sizeof(VertexID) * get_num_outgoing_edges() +
                     sizeof(VertexID) * (get_max_vid() + 1);

  CUDA_CHECK(cudaMallocManaged(&graph_base_pointer_, data_size));
  CUDA_CHECK(cudaMemcpy(graph_base_pointer_, csr.GetGraphBuffer(), data_size,
                        cudaMemcpyDefault));

  ParseBasePtr(graph_base_pointer_);
}

void ImmutableCSRGPU::ParseBasePtr(uint8_t* graph_base_pointer) {
  SetGlobalIDBuffer(reinterpret_cast<VertexID*>(graph_base_pointer));
  SetInDegreeBuffer(
      reinterpret_cast<VertexID*>(globalid_by_localid_base_pointer_) +
      metadata_.num_vertices);
  SetOutDegreeBuffer(reinterpret_cast<VertexID*>(indegree_base_pointer_) +
                     metadata_.num_vertices);
  SetInOffsetBuffer(reinterpret_cast<EdgeIndex*>(
      reinterpret_cast<VertexID*>(outdegree_base_pointer_) +
      metadata_.num_vertices));
  SetOutOffsetBuffer(reinterpret_cast<EdgeIndex*>(
      reinterpret_cast<VertexID*>(in_offset_base_pointer_) +
      metadata_.num_vertices + 1));
  SetIncomingEdgesBuffer(reinterpret_cast<VertexID*>(
      reinterpret_cast<EdgeIndex*>(out_offset_base_pointer_) +
      metadata_.num_vertices + 1));
  SetOutgoingEdgesBuffer(reinterpret_cast<VertexID*>(
      incoming_edges_base_pointer_ + metadata_.num_incoming_edges));
  SetEdgesGlobalIDBuffer(reinterpret_cast<VertexID*>(
      outgoing_edges_base_pointer_ + metadata_.num_outgoing_edges));
}

}  // namespace data_structures
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics