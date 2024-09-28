#ifndef SICS_GRAPH_SYSTEMS_TOOLS_COMMON_YAML_CONFIG_H_
#define SICS_GRAPH_SYSTEMS_TOOLS_COMMON_YAML_CONFIG_H_

#include <iostream>
#include <yaml-cpp/yaml.h>

#include "core/data_structures/metadata.h"

namespace YAML {

using sics::matrixgraph::core::data_structures::GraphMetadata;
using sics::matrixgraph::core::data_structures::SubGraphMetadata;
using sics::matrixgraph::core::data_structures::TiledMatrixMetadata;
using VertexID = sics::matrixgraph::core::common::VertexID;
using GraphID = sics::matrixgraph::core::common::GraphID;
using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;

template <> struct convert<TiledMatrixMetadata> {
  static Node encode(TiledMatrixMetadata &metadata) {
    Node node;
    node["n_cols"] = metadata.n_cols;
    node["n_rows"] = metadata.n_rows;
    node["n_nz_tile"] = metadata.n_nz_tile;

    return node;
  }

  static bool decode(const Node &node, TiledMatrixMetadata &metadata) {
    if (node["TiledMatrix"].size() != 3) {
      return false;
    }

    metadata.n_cols = node["TiledMatrix"]["n_cols"].as<VertexID>();
    metadata.n_rows = node["TiledMatrix"]["n_rows"].as<EdgeIndex>();
    metadata.n_nz_tile = node["TiledMatrix"]["n_nz_tile"].as<VertexID>();

    return true;
  }
};

// template is needed for this function
template <>
struct convert<sics::matrixgraph::core::data_structures::SubGraphMetadata> {
  static Node
  encode(const sics::matrixgraph::core::data_structures::SubGraphMetadata
             &subgraph_metadata) {
    Node node;
    node["gid"] = subgraph_metadata.gid;
    node["num_vertices"] = subgraph_metadata.num_vertices;
    node["num_incoming_edges"] = subgraph_metadata.num_incoming_edges;
    node["num_outgoing_edges"] = subgraph_metadata.num_outgoing_edges;
    node["max_vid"] = subgraph_metadata.max_vid;
    node["min_vid"] = subgraph_metadata.min_vid;
    return node;
  }
  static bool decode(const Node &node,
                     sics::matrixgraph::core::data_structures::SubGraphMetadata
                         &subgraph_metadata) {
    if (node.size() != 6)
      return false;

    subgraph_metadata.gid = node["gid"].as<GraphID>();
    subgraph_metadata.num_vertices = node["num_vertices"].as<EdgeIndex>();
    subgraph_metadata.num_incoming_edges =
        node["num_incoming_edges"].as<EdgeIndex>();
    subgraph_metadata.num_outgoing_edges =
        node["num_outgoing_edges"].as<EdgeIndex>();
    subgraph_metadata.max_vid = node["max_vid"].as<EdgeIndex>();
    subgraph_metadata.min_vid = node["min_vid"].as<EdgeIndex>();
    return true;
  }
};

// template is needed for this function
template <>
struct convert<sics::matrixgraph::core::data_structures::GraphMetadata> {
  static Node encode(
      const sics::matrixgraph::core::data_structures::GraphMetadata &metadata) {
    Node node;
    node["num_vertices"] = metadata.num_vertices;
    node["num_edges"] = metadata.num_edges;
    node["max_vid"] = metadata.max_vid;
    node["min_vid"] = metadata.min_vid;
    node["count_border_vertices"] = metadata.count_border_vertices;
    node["num_subgraphs"] = metadata.num_subgraphs;
    std::vector<sics::matrixgraph::core::data_structures::SubGraphMetadata> tmp;
    // for (size_t i = 0; i < metadata.num_subgraphs; i++) {
    //   tmp.push_back(metadata.GetSubGraphMetadata(i));
    // }
    // node["subgraphs"] = tmp;

    return node;
  }

  static bool
  decode(const Node &node,
         sics::matrixgraph::core::data_structures::GraphMetadata &metadata) {

    metadata.num_vertices =
        node["GraphMetadata"]["num_vertices"].as<VertexID>();
    metadata.num_edges = node["GraphMetadata"]["num_edges"].as<EdgeIndex>();
    metadata.max_vid = node["GraphMetadata"]["max_vid"].as<EdgeIndex>();
    metadata.min_vid = node["GraphMetadata"]["min_vid"].as<EdgeIndex>();
    metadata.num_subgraphs =
        node["GraphMetadata"]["num_subgraphs"].as<EdgeIndex>();
    metadata.subgraphs =
        node["GraphMetadata"]["subgraphs"].as<std::vector<SubGraphMetadata>>();
    return true;
  }
};

} // namespace YAML
#endif // SICS_GRAPH_SYSTEMS_TOOLS_COMMON_YAML_CONFIG_H_
