#ifndef SICS_GRAPH_SYSTEMS_TOOLS_COMMON_YAML_CONFIG_H_
#define SICS_GRAPH_SYSTEMS_TOOLS_COMMON_YAML_CONFIG_H_

#include <yaml-cpp/yaml.h>

#include "core/data_structures/metadata.h"

namespace YAML {

using sics::matrixgraph::core::data_structures::TiledMatrixMetadata;
using VertexID = sics::matrixgraph::core::common::VertexID;
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

} // namespace YAML
#endif // SICS_GRAPH_SYSTEMS_TOOLS_COMMON_YAML_CONFIG_H_
