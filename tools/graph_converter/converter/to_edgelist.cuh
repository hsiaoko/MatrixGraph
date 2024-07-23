#ifndef MATRIXGRAPH_TOOLS_GRAPH_CONVERTER_CONVERTER_TO_EDGELIST_CUH_
#define MATRIXGRAPH_TOOLS_GRAPH_CONVERTER_CONVERTER_TO_EDGELIST_CUH_

#include "core/data_structures/edgelist.h"
#include "core/data_structures/immutable_csr.cuh"
#include "tools/common/format_converter.h"

namespace sics {
namespace matrixgraph {
namespace tools {
namespace converter {

using sics::matrixgraph::core::data_structures::ImmutableCSR;

static void ConvertEdgelistCSV2EdgelistBin(const std::string &input_path,
                                           const std::string &output_path,
                                           const std::string &sep,
                                           bool read_head = false) {
  std::cout << "ConvertEdgelistCSV2EdgelistBin" << std::endl;
  if (!std::filesystem::exists(output_path))
    std::filesystem::create_directory(output_path);

  sics::matrixgraph::core::data_structures::Edges edgelist;
  edgelist.ReadFromCSV(input_path, sep);
  edgelist.ShowGraph();
  edgelist.WriteToBinary(output_path);
}

static void ConvertImmutableCSR2EdgelistBin(const std::string &input_path,
                                            const std::string &output_path) {

  ImmutableCSR csr;
  csr.Read(input_path);

  csr.SortByDegree();
  std::cout << "[ConvertImmutableCSR2EdgelistBin]" << std::endl;

  EdgelistMetadata edgelist_metadata = {.num_vertices = csr.get_num_vertices(),
                                        .num_edges =
                                            csr.get_num_outgoing_edges(),
                                        .max_vid = csr.get_max_vid(),
                                        .min_vid = 0};

  auto *edges_ptr =
      sics::matrixgraph::tools::format_converter::ImmutableCSR2Edgelist(csr);

  edges_ptr->ShowGraph(3);
  std::cout << "[ConvertImmutableCSR2EdgelistBin] Done!" << std::endl;

  edges_ptr->WriteToBinary(output_path);
}

} // namespace converter
} // namespace tools
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_TOOLS_GRAPH_CONVERTER_CONVERTER_TO_EDGELIST_CUH_