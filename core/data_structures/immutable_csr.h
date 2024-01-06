#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_IMMUTABLE_CSR_H_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_IMMUTABLE_CSR_H_

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

struct ImmutableCSRMetadata {
  using GraphID = sics::matrixgraph::core::common::GraphID;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;

  // Subgraph Metadata
  GraphID gid = 0;
  VertexID num_vertices;
  EdgeIndex num_incoming_edges;
  EdgeIndex num_outgoing_edges;
  VertexID max_vid;
  VertexID min_vid;
};

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

public:
  explicit ImmutableCSR(ImmutableCSRMetadata metadata) : metadata_(metadata) {}

  ImmutableCSR() {}

  void ShowGraph(VertexID display_num = 0) const {
    std::cout << "### GID: " << metadata_.gid
              << ",  num_vertices: " << metadata_.num_vertices
              << ", num_incoming_edges: " << metadata_.num_incoming_edges
              << ", num_outgoing_edges: " << metadata_.num_outgoing_edges
              << " Show top " << display_num << " ###" << std::endl;
    for (VertexID i = 0; i < metadata_.num_vertices; i++) {
      if (i > display_num)
        break;
      auto u = GetVertexByLocalID(i);

      std::stringstream ss;

      ss << "  ===vid: " << u.vid << ", indegree: " << u.indegree
         << ", outdegree: " << u.outdegree << "===" << std::endl;
      if (u.indegree != 0) {
        ss << "    Incoming edges: ";
        for (VertexID i = 0; i < u.indegree; i++)
          ss << u.incoming_edges[i] << ",";
        ss << std::endl << std::endl;
      }
      if (u.outdegree != 0) {
        ss << "    Outgoing edges: ";
        for (VertexID i = 0; i < u.outdegree; i++)
          ss << u.outgoing_edges[i] << ",";
        ss << std::endl << std::endl;
      }
      ss << "****************************************" << std::endl;
      std::string s = ss.str();
      std::cout << s << std::endl;
    }
  }

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

  void SetGraphBuffer(uint8_t *buffer) { graph_base_pointer_ = buffer; }

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
    vertex_label_base_pointer_ = buffer;
  }

  uint8_t *GetGraphBuffer() { return graph_base_pointer_; }
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

  ImmutableCSRVertex GetVertexByLocalID(VertexID i) const {
    ImmutableCSRVertex v;
    v.vid = GetGlobalIDByLocalID(i);
    if (get_num_incoming_edges() > 0) {
      v.indegree = GetInDegreeByLocalID(i);
      v.incoming_edges = incoming_edges_base_pointer_ + GetInOffsetByLocalID(i);
    }
    if (get_num_outgoing_edges() > 0) {
      v.outdegree = GetOutDegreeByLocalID(i);
      v.outgoing_edges =
          outgoing_edges_base_pointer_ + GetOutOffsetByLocalID(i);
    }
    return v;
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

private:
  void ParseSubgraphCSR() {}

protected:
  // Metadata to build the CSR.
  ImmutableCSRMetadata metadata_;

  // Serialized data in CSR format.
  uint8_t *graph_base_pointer_;
  VertexID *globalid_by_localid_base_pointer_;
  VertexID *incoming_edges_base_pointer_;
  VertexID *outgoing_edges_base_pointer_;
  VertexID *indegree_base_pointer_;
  VertexID *outdegree_base_pointer_;
  EdgeIndex *in_offset_base_pointer_;
  EdgeIndex *out_offset_base_pointer_;
  VertexLabel *vertex_label_base_pointer_;
};

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // INC_51_11_GRAPH_COMPUTING_MATRIXGRAPH_CORE_DATA_STRUCTURES_IMMUTABLE_CSR_H_
