#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_IMMUTABLE_CSR_H_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_IMMUTABLE_CSR_H_

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

public:
  explicit ImmutableCSR(SubGraphMetadata metadata) : metadata_(metadata) {}

  ImmutableCSR() = default;

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

  void Write(const std::string &root_path, GraphID gid = 0) {
    std::cout << "Write" << std::endl;

    if (!std::filesystem::exists(root_path))
      std::filesystem::create_directory(root_path);
    if (!std::filesystem::exists(root_path + "graphs"))
      std::filesystem::create_directory(root_path + "graphs");
    if (!std::filesystem::exists(root_path + "label"))
      std::filesystem::create_directory(root_path + "label");

    std::ofstream out_meta_file(root_path + "meta.yaml");
    auto aligned_max_vid = (((metadata_.max_vid + 1) >> 6) << 6) + 64;

    std::ofstream out_data_file(root_path + "graphs/" + std::to_string(gid) +
                                ".bin");

    // Write topology of graph.
    out_data_file.write(reinterpret_cast<char *>(GetGloablIDBasePointer()),
                        sizeof(VertexID) * get_num_vertices());
    out_data_file.write(reinterpret_cast<char *>(GetInDegreeBasePointer()),
                        sizeof(VertexID) * get_num_vertices());
    out_data_file.write(reinterpret_cast<char *>(GetOutDegreeBasePointer()),
                        sizeof(VertexID) * get_num_vertices());
    out_data_file.write(reinterpret_cast<char *>(GetInOffsetBasePointer()),
                        sizeof(EdgeIndex) * get_num_vertices());
    out_data_file.write(reinterpret_cast<char *>(GetOutOffsetBasePointer()),
                        sizeof(EdgeIndex) * get_num_vertices());
    out_data_file.write(reinterpret_cast<char *>(GetIncomingEdgesBasePointer()),
                        sizeof(VertexID) * get_num_outgoing_edges());
    out_data_file.write(reinterpret_cast<char *>(GetOutgoingEdgesBasePointer()),
                        sizeof(VertexID) * get_num_outgoing_edges());

    // Write label data with all 0.
    std::ofstream out_label_file(root_path + "label/" + std::to_string(gid) +
                                 ".bin");
    auto buffer_label = new VertexLabel[get_num_vertices()]();
    out_label_file.write(reinterpret_cast<char *>(buffer_label),
                         sizeof(VertexLabel) * get_num_vertices());
    delete[] buffer_label;

    out_data_file.close();
    out_label_file.close();

    // Write graph metadata.
    std::vector<SubGraphMetadata> subgraph_metadata_vec;
    subgraph_metadata_vec.push_back(
        {gid, get_num_vertices(), get_num_incoming_edges(),
         get_num_outgoing_edges(), get_max_vid(), get_min_vid()});

    YAML::Node out_node;
    out_node["GraphMetadata"]["num_vertices"] = get_num_vertices();
    out_node["GraphMetadata"]["num_edges"] =
        get_num_incoming_edges() + get_num_outgoing_edges();
    out_node["GraphMetadata"]["max_vid"] = get_max_vid();
    out_node["GraphMetadata"]["min_vid"] = get_min_vid();
    out_node["GraphMetadata"]["count_border_vertices"] = 0;
    out_node["GraphMetadata"]["num_subgraphs"] = 1;
    out_node["GraphMetadata"]["subgraphs"] = subgraph_metadata_vec;
    out_meta_file << out_node << std::endl;
    out_meta_file.close();
  }

  void Read(const std::string &root_path) {
    std::cout << "Read: " << root_path << std::endl;

    std::cout << root_path + "meta.yaml" << std::endl;
    YAML::Node metadata_node;
    GraphMetadata metadata;
    try {
      metadata_node = YAML::LoadFile(root_path + "meta.yaml");
      metadata = metadata_node.as<GraphMetadata>();
    } catch (YAML::BadFile &e) {
      std::cout << "meta.yaml file read failed! " << e.msg << std::endl;
    }

    assert(metadata.num_subgraphs == 1);

    SetMaxVid(metadata.subgraphs[0].max_vid);
    SetMinVid(metadata.subgraphs[0].min_vid);
    SetNumIncomingEdges(metadata.subgraphs[0].num_incoming_edges);
    SetNumOutgoingEdges(metadata.subgraphs[0].num_outgoing_edges);
    SetNumVertices(metadata.subgraphs[0].num_vertices);

    std::ifstream data_file(root_path + "graphs/" + "0.bin", std::ios::binary);
    std::ifstream label_file(root_path + "label/" + "0.bin", std::ios::binary);

    data_file.seekg(0, std::ios::end);
    size_t file_size = data_file.tellg();
    data_file.seekg(0, std::ios::beg);

    graph_base_pointer_ = std::make_unique<uint8_t[]>(file_size);

    // Read the file data.
    data_file.read(reinterpret_cast<char *>(graph_base_pointer_.get()),
                   file_size);

    if (!data_file)
      throw std::runtime_error("Error reading file: " + root_path + "graphs/" +
                               "0.bin");

    SetGlobalIDBuffer(reinterpret_cast<VertexID *>(graph_base_pointer_.get()));

    SetInDegreeBuffer(
        reinterpret_cast<VertexID *>(globalid_by_localid_base_pointer_) +
        metadata_.num_vertices);
    SetOutDegreeBuffer(reinterpret_cast<VertexID *>(indegree_base_pointer_ +
                                                    metadata_.num_vertices));
    SetInOffsetBuffer(reinterpret_cast<EdgeIndex *>(outdegree_base_pointer_ +
                                                    metadata_.num_vertices));
    SetOutOffsetBuffer(reinterpret_cast<EdgeIndex *>(in_offset_base_pointer_ +
                                                     metadata_.num_vertices));
    SetIncomingEdgesBuffer(reinterpret_cast<VertexID *>(
        out_offset_base_pointer_ + metadata_.num_vertices));
    SetOutgoingEdgesBuffer(reinterpret_cast<VertexID *>(
        incoming_edges_base_pointer_ + metadata_.num_incoming_edges));

    label_file.seekg(0, std::ios::end);
    file_size = label_file.tellg();
    label_file.seekg(0, std::ios::beg);
    vertex_label_base_pointer_ = std::make_unique<VertexLabel[]>(file_size);

    // Read the label.
    label_file.read(reinterpret_cast<char *>(vertex_label_base_pointer_.get()),
                    file_size);
    ShowGraph(100);
    data_file.close();
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
