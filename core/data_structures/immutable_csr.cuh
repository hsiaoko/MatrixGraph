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

public:
  VertexID vid;
  VertexID indegree = 0;
  VertexID outdegree = 0;
  VertexID *incoming_edges;
  VertexID *outgoing_edges;
};

class ImmutableCSR {
private:
  using GraphID = sics::matrixgraph::core::common::GraphID;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;

  struct VidCountPair {
    bool operator<(const VidCountPair &other) const {
      return count > other.count;
    }

    VertexID vid = 0;
    VertexID count = 0;
  };

public:
  explicit ImmutableCSR(SubGraphMetadata metadata) : metadata_(metadata) {}

  ImmutableCSR() = default;

  void PrintGraph(VertexID display_num = 0) const;

  void PrintGraphAbs(VertexID display_num = 0) const;

  void Write(const std::string &root_path, GraphID gid = 0);

  void Read(const std::string &root_path);

  void SortByDegree();

  void SortByDistance(VertexID sim_granularity);

  void ReOrder(VertexID tile_size);

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

  void SetGraphBuffer(uint8_t *buffer) { graph_base_pointer_.reset(buffer); }

  void SetGlobalIDBuffer(VertexID *buffer) {
    globalid_by_localid_base_pointer_ = buffer;
  }
  void SetInDegreeBuffer(VertexID *buffer) { indegree_base_pointer_ = buffer; }
  void SetOutDegreeBuffer(VertexID *buffer) {
    outdegree_base_pointer_ = buffer;
  }
  void SetInOffsetBuffer(EdgeIndex *buffer) {
    in_offset_base_pointer_ = buffer;
  }
  void SetOutOffsetBuffer(EdgeIndex *buffer) {
    out_offset_base_pointer_ = buffer;
  }
  void SetIncomingEdgesBuffer(VertexID *buffer) {
    incoming_edges_base_pointer_ = buffer;
  }
  void SetOutgoingEdgesBuffer(VertexID *buffer) {
    outgoing_edges_base_pointer_ = buffer;
  }

  void SetVertexLabelBuffer(VertexLabel *buffer) {
    vertex_label_base_pointer_.reset(buffer);
  }

  uint8_t *GetGraphBuffer() { return graph_base_pointer_.get(); }
  VertexID *GetGloablIDBasePointer() const {
    return globalid_by_localid_base_pointer_;
  }
  VertexID *GetInDegreeBasePointer() const { return indegree_base_pointer_; }
  VertexID *GetOutDegreeBasePointer() const { return outdegree_base_pointer_; }
  EdgeIndex *GetInOffsetBasePointer() const { return in_offset_base_pointer_; }
  EdgeIndex *GetOutOffsetBasePointer() const {
    return out_offset_base_pointer_;
  }
  VertexID *GetIncomingEdgesBasePointer() const {
    return incoming_edges_base_pointer_;
  }
  VertexID *GetOutgoingEdgesBasePointer() const {
    return outgoing_edges_base_pointer_;
  }

  VertexID GetGlobalIDByLocalID(VertexID i) const {
    return globalid_by_localid_base_pointer_[i];
  }
  VertexID GetInOffsetByLocalID(VertexID i) const {
    return in_offset_base_pointer_[i];
  }
  VertexID GetOutOffsetByLocalID(VertexID i) const {
    return out_offset_base_pointer_[i];
  }
  VertexID *GetIncomingEdgesByLocalID(VertexID i) const {
    return incoming_edges_base_pointer_ + in_offset_base_pointer_[i];
  }
  VertexID *GetOutgoingEdgesByLocalID(VertexID i) const {
    return outgoing_edges_base_pointer_ + out_offset_base_pointer_[i];
  }
  VertexID GetInDegreeByLocalID(VertexID i) const {
    return indegree_base_pointer_[i];
  }
  VertexID GetOutDegreeByLocalID(VertexID i) const {
    return outdegree_base_pointer_[i];
  }

  ImmutableCSRVertex GetVertexByLocalID(VertexID i) const;

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

protected:
  // Metadata to build the CSR.
  SubGraphMetadata metadata_;

  // Serialized data in CSR format.
  std::unique_ptr<uint8_t[]> graph_base_pointer_;

  VertexID *globalid_by_localid_base_pointer_;
  VertexID *incoming_edges_base_pointer_;
  VertexID *outgoing_edges_base_pointer_;
  VertexID *indegree_base_pointer_;
  VertexID *outdegree_base_pointer_;
  EdgeIndex *in_offset_base_pointer_;
  EdgeIndex *out_offset_base_pointer_;

  std::unique_ptr<VertexLabel[]> vertex_label_base_pointer_;
};

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // INC_51_11_GRAPH_COMPUTING_MATRIXGRAPH_CORE_DATA_STRUCTURES_IMMUTABLE_CSR_H_
