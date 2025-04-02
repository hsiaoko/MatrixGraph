#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_IMMUTABLE_CSR_H_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_IMMUTABLE_CSR_H_

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "core/common/yaml_config.h"
#include "core/data_structures/metadata.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

struct ImmutableCSRVertex {
 private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;

 public:
  VertexID vid;
  VertexID indegree = 0;
  VertexID outdegree = 0;
  VertexID* incoming_edges;
  VertexID* outgoing_edges;
  VertexLabel vlabel = 0;
};

class ImmutableCSR {
 private:
  using GraphID = sics::matrixgraph::core::common::GraphID;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;

  struct VidCountPair {
    bool operator<(const VidCountPair& other) const {
      return count > other.count;
    }

    VertexID vid = 0;
    VertexID count = 0;
  };

 public:
  explicit ImmutableCSR(SubGraphMetadata metadata) : metadata_(metadata) {}

  ImmutableCSR() = default;

  ~ImmutableCSR() = default;

  void PrintGraph(VertexID display_num = 0) const;

  void PrintGraphAbs(VertexID display_num = 0) const;

  void Write(const std::string& root_path, GraphID gid = 0);

  void Read(const std::string& root_path);

  void ParseBasePtr(uint8_t* graph_base_pointer);

  void SortByDegree();

  void SortByDistance(VertexID sim_granularity);

  void GenerateVLabel(VertexID range = 5);

  void SetGraphBuffer(uint8_t* buffer) { graph_base_pointer_.reset(buffer); }

  void SetVertexLabelBuffer(size_t n_vertices) {
    vertex_label_base_pointer_ = std::make_unique<VertexID[]>(n_vertices);
  }

  void SetGlobalIDBuffer(VertexID* buffer) {
    globalid_by_localid_base_pointer_ = buffer;
  }

  void SetLocalIDBuffer(VertexID* buffer) {
    localid_by_globalid_base_pointer_ = buffer;
  }

  void SetEdgesGlobalIDBuffer(VertexID* buffer) {
    edges_globalid_by_localid_base_pointer_ = buffer;
  }

  void SetInDegreeBuffer(VertexID* buffer) { indegree_base_pointer_ = buffer; }

  void SetOutDegreeBuffer(VertexID* buffer) {
    outdegree_base_pointer_ = buffer;
  }

  void SetInOffsetBuffer(EdgeIndex* buffer) {
    in_offset_base_pointer_ = buffer;
  }

  void SetOutOffsetBuffer(EdgeIndex* buffer) {
    out_offset_base_pointer_ = buffer;
  }

  void SetIncomingEdgesBuffer(VertexID* buffer) {
    incoming_edges_base_pointer_ = buffer;
  }

  void SetOutgoingEdgesBuffer(VertexID* buffer) {
    outgoing_edges_base_pointer_ = buffer;
  }

  void SetNumVertices(VertexID num_vertices) {
    metadata_.num_vertices = num_vertices;
  }

  void SetNumIncomingEdges(EdgeIndex num_incoming_edges) {
    metadata_.num_incoming_edges = num_incoming_edges;
  }

  void SetNumOutgoingEdges(EdgeIndex num_outgoing_edges) {
    metadata_.num_outgoing_edges = num_outgoing_edges;
  }

  void SetMaxVid(VertexID max_vid) { metadata_.max_vid = max_vid; }

  void SetMinVid(VertexID min_vid) { metadata_.min_vid = min_vid; }

  void SetGid(VertexID gid) { metadata_.gid = gid; }

  void ReOrder(VertexID tile_size);

  SubGraphMetadata GetMetadata() const { return metadata_; }

  GraphID get_gid() const { return metadata_.gid; }

  VertexID get_num_vertices() const { return metadata_.num_vertices; }

  EdgeIndex get_num_incoming_edges() const {
    return metadata_.num_incoming_edges;
  }

  EdgeIndex get_num_outgoing_edges() const {
    return metadata_.num_outgoing_edges;
  }

  VertexID get_max_vid() const { return metadata_.max_vid; }

  VertexID get_min_vid() const { return metadata_.min_vid; }

  uint8_t* GetGraphBuffer() const { return graph_base_pointer_.get(); }

  inline VertexID* GetGloablIDBasePointer() const {
    return globalid_by_localid_base_pointer_;
  }

  inline VertexID* GetLocalIDBasePointer() const {
    return localid_by_globalid_base_pointer_;
  }

  inline VertexID* GetEdgesGloablIDBasePointer() const {
    return edges_globalid_by_localid_base_pointer_;
  }

  inline VertexID* GetInDegreeBasePointer() const {
    return indegree_base_pointer_;
  }

  inline VertexID* GetOutDegreeBasePointer() const {
    return outdegree_base_pointer_;
  }

  inline EdgeIndex* GetInOffsetBasePointer() const {
    return in_offset_base_pointer_;
  }

  inline EdgeIndex* GetOutOffsetBasePointer() const {
    return out_offset_base_pointer_;
  }

  inline VertexID* GetIncomingEdgesBasePointer() const {
    return incoming_edges_base_pointer_;
  }

  inline VertexID* GetOutgoingEdgesBasePointer() const {
    return outgoing_edges_base_pointer_;
  }

  inline VertexLabel* GetVLabelBasePointer() const {
    return vertex_label_base_pointer_.get();
  }

  inline VertexID GetGlobalIDByLocalID(VertexID i) const {
    return globalid_by_localid_base_pointer_[i];
  }

  inline VertexID GetLocalIDByGlobalID(VertexID i) const {
    return localid_by_globalid_base_pointer_[i];
  }

  inline VertexID GetEdgeGlobalIDByLocalID(VertexID i) const {
    return edges_globalid_by_localid_base_pointer_[i];
  }

  inline VertexID GetInOffsetByLocalID(VertexID i) const {
    return in_offset_base_pointer_[i];
  }

  inline VertexID GetOutOffsetByLocalID(VertexID i) const {
    return out_offset_base_pointer_[i];
  }

  inline VertexID* GetIncomingEdgesByLocalID(VertexID i) const {
    return incoming_edges_base_pointer_ + in_offset_base_pointer_[i];
  }

  inline VertexID* GetOutgoingEdgesByLocalID(VertexID i) const {
    return outgoing_edges_base_pointer_ + out_offset_base_pointer_[i];
  }

  inline VertexID GetInDegreeByLocalID(VertexID i) const {
    return indegree_base_pointer_[i];
  }

  inline VertexID GetOutDegreeByLocalID(VertexID i) const {
    return outdegree_base_pointer_[i];
  }

  inline ImmutableCSRVertex GetVertexByLocalID(VertexID i) const;

 protected:
  // Metadata to build the CSR.
  SubGraphMetadata metadata_;

  // Serialized data in CSR format.
  std::unique_ptr<uint8_t[]> graph_base_pointer_ = nullptr;

  VertexID* globalid_by_localid_base_pointer_ = nullptr;
  VertexID* localid_by_globalid_base_pointer_ = nullptr;

  VertexID* edges_globalid_by_localid_base_pointer_ = nullptr;

  VertexID* local_vid_by_edges_globalid_base_pointer_ = nullptr;

  VertexID* incoming_edges_base_pointer_ = nullptr;
  VertexID* outgoing_edges_base_pointer_ = nullptr;
  VertexID* indegree_base_pointer_ = nullptr;
  VertexID* outdegree_base_pointer_ = nullptr;
  EdgeIndex* in_offset_base_pointer_ = nullptr;
  EdgeIndex* out_offset_base_pointer_ = nullptr;

  std::unique_ptr<VertexLabel[]> vertex_label_base_pointer_ = nullptr;
};

}  // namespace data_structures
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // INC_51_11_GRAPH_COMPUTING_MATRIXGRAPH_CORE_DATA_STRUCTURES_IMMUTABLE_CSR_H_
