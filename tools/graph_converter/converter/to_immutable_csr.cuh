#ifndef MATRIXGRAPH_TOOLS_GRAPH_CONVERTER_CONVERTER_TO_IMMUTABLE_CSR_CUH_
#define MATRIXGRAPH_TOOLS_GRAPH_CONVERTER_CONVERTER_TO_IMMUTABLE_CSR_CUH_

#include "core/data_structures/edgelist.h"
#include "core/data_structures/immutable_csr.cuh"

namespace sics {
namespace matrixgraph {
namespace tools {
namespace converter {

using Vertex = sics::matrixgraph::core::data_structures::ImmutableCSRVertex;
using GraphID = sics::matrixgraph::core::common::GraphID;
using VertexID = sics::matrixgraph::core::common::VertexID;

// @DESCRIPTION: convert a edgelist graph from csv file to binary file. Here the
// compression operations is default in ConvertEdgelist.
// @PARAMETER: input_path and output_path indicates the input and output path
// respectively, sep determines the separator for the csv file, read_head
// indicates whether to read head.
static void ConvertEdgelistCSV2ImmutableCSR(const std::string &input_path,
                                            const std::string &output_path,
                                            const std::string &sep) {
  if (!std::filesystem::exists(output_path))
    std::filesystem::create_directory(output_path);

  sics::matrixgraph::core::data_structures::Edges edgelist;
  edgelist.ReadFromCSV(input_path, sep);
  auto p_immutable_csr =
      sics::matrixgraph::tools::format_converter::Edgelist2ImmutableCSR(
          edgelist);

  p_immutable_csr->Write(output_path);
  delete p_immutable_csr;
}

static void ConvertEdgelistBin2CSRBin(const std::string &input_path,
                                      const std::string &output_path,
                                      size_t tile_size) {
  YAML::Node node = YAML::LoadFile(input_path + "meta.yaml");

  sics::matrixgraph::core::data_structures::EdgelistMetadata edgelist_metadata =
      {node["EdgelistBin"]["num_vertices"].as<VertexID>(),
       node["EdgelistBin"]["num_edges"].as<VertexID>(),
       node["EdgelistBin"]["max_vid"].as<VertexID>()};

  auto buffer_edges =
      new sics::matrixgraph::core::data_structures::Edge[edgelist_metadata
                                                             .num_edges]();

  std::ifstream in_file(input_path + "edgelist.bin");
  if (!in_file) {

    std::cout << "Open file failed: " + input_path + "edgelist.bin"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  in_file.read(reinterpret_cast<char *>(buffer_edges),
               sizeof(sics::matrixgraph::core::data_structures::Edge) *
                   edgelist_metadata.num_edges);

  sics::matrixgraph::core::data_structures::Edges edgelist(edgelist_metadata,
                                                           buffer_edges);

  auto p_immutable_csr =
      sics::matrixgraph::tools::format_converter::Edgelist2ImmutableCSR(
          edgelist);
  p_immutable_csr->SortByDegree();
  p_immutable_csr->PrintGraph(3);
  p_immutable_csr->Write(output_path);
  delete p_immutable_csr;
}

} // namespace converter
} // namespace tools
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_TOOLS_GRAPH_CONVERTER_CONVERTER_TO_IMMUTABLE_CSR_CUH_