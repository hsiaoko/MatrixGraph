#ifndef MATRIXGRAPH_TOOLS_GRAPH_CONVERTER_CONVERTER_GRID_EDGELIST_2_GIRD_CSR_TILED_MATRIX_CUH_
#define MATRIXGRAPH_TOOLS_GRAPH_CONVERTER_CONVERTER_GRID_EDGELIST_2_GIRD_CSR_TILED_MATRIX_CUH_

#include <cmath>

#include "core/common/types.h"
#include "core/common/yaml_config.h"
#include "core/data_structures/bit_tiled_matrix.cuh"
#include "core/data_structures/csr_tiled_matrix.cuh"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/metadata.h"
#include "core/io/csr_tiled_matrix_io.cuh"
#include "core/util/format_converter.cuh"
#include "tools/common/edgelist_subgraphs_io.cuh"

namespace sics {
namespace matrixgraph {
namespace tools {
namespace converter {

using GraphID = sics::matrixgraph::core::common::GraphID;
using VertexID = sics::matrixgraph::core::common::VertexID;
using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using GraphMetadata = sics::matrixgraph::core::data_structures::GraphMetadata;
using EdgelistMetadata =
    sics::matrixgraph::core::data_structures::EdgelistMetadata;
using Edges = sics::matrixgraph::core::data_structures::Edges;
using Edge = sics::matrixgraph::core::data_structures::Edge;
using CSRTiledMatrixIO = sics::matrixgraph::core::io::CSRTiledMatrixIO;
using EdgelistSubGraphsIO =
    sics::matrixgraph::tools::common::EdgelistSubGraphsIO;
using CSRTiledMatrix = sics::matrixgraph::core::data_structures::CSRTiledMatrix;
using sics::matrixgraph::core::util::format_converter::Edgelist2BitTiledMatrix;
using sics::matrixgraph::core::util::format_converter::Edgelist2CSRTiledMatrix;

static void ConvertGridGraph2CSRTiledMatrix(const std::string &input_path,
                                            const std::string &output_path,
                                            size_t tile_size) {
  if (!std::filesystem::exists(output_path))
    std::filesystem::create_directory(output_path);

  auto node = YAML::LoadFile(input_path + "meta.yaml");
  auto meta = node.as<GraphMetadata>();

  GraphMetadata graph_metadata;
  std::vector<Edges> edges_blocks;

  EdgelistSubGraphsIO::Read(input_path, &edges_blocks, &graph_metadata);

  GraphID n_chunks = sqrt(graph_metadata.num_subgraphs);

  size_t block_scope =
      ceil((float)(graph_metadata.max_vid + 1) / (float)n_chunks);

  for (GraphID gid = 0; gid < graph_metadata.num_subgraphs; ++gid) {
    std::string block_dir = output_path + "block" + std::to_string(gid) + "/";

    auto *csr_tile_matrix =
        Edgelist2CSRTiledMatrix(edges_blocks[gid], tile_size, block_scope);

    CSRTiledMatrixIO::Write(block_dir, *csr_tile_matrix);
  }

  // Write VLabel of GridCSRTiledMatrix.
  VertexLabel *buffer_vlabel = new VertexLabel[meta.num_vertices]();
  std::ofstream out_label_file(output_path + "label.bin", std::ios::binary);
  out_label_file.write(reinterpret_cast<char *>(buffer_vlabel),
                       sizeof(VertexLabel) * meta.num_vertices);
  delete[] buffer_vlabel;

  // Write Meta.yaml
  std::ofstream out_meta_file(output_path + "meta.yaml");
  YAML::Node out_node;
  out_node["GridGraphMetadata"]["n_vertices"] = meta.num_vertices;
  out_node["GridGraphMetadata"]["n_edges"] = meta.num_edges;
  out_node["GridGraphMetadata"]["n_chunks"] = n_chunks;
  out_node["GridGraphMetadata"]["max_vid"] = graph_metadata.max_vid;

  out_meta_file << out_node << std::endl;
  out_meta_file.close();
}

} // namespace converter
} // namespace tools
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_TOOLS_GRAPH_CONVERTER_CONVERTER_GRID_EDGELIST_2_CSR_TILED_MATRIX_CUH_