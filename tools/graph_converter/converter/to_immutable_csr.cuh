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
  edgelist.ShowGraph(100);
  auto p_immutable_csr =
      sics::matrixgraph::core::util::format_converter::Edgelist2ImmutableCSR(
          edgelist);
  p_immutable_csr->PrintGraph(100);
  p_immutable_csr->Write(output_path);
  delete p_immutable_csr;
}

static void ConvertEdgelistBin2CSRBin(const std::string &input_path,
                                      const std::string &output_path) {
  std::cout << "ConvertEdgelistBin2CSRBin" << std::endl;

  sics::matrixgraph::core::data_structures::Edges edgelist;
  edgelist.ReadFromBin(input_path);
  edgelist.ShowGraph(10);

  auto p_immutable_csr =
      sics::matrixgraph::core::util::format_converter::Edgelist2ImmutableCSR(
          edgelist);
  // p_immutable_csr->SortByDegree();
  p_immutable_csr->GenerateVLabel(6);
  p_immutable_csr->PrintGraph(100);
  p_immutable_csr->Write(output_path);
  delete p_immutable_csr;
}

static void ConvertEdgelistCSV2CGGraphCSR(const std::string &input_path,
                                          const std::string &output_path,
                                          const std::string &sep) {
  if (!std::filesystem::exists(output_path))
    std::filesystem::create_directory(output_path);

  sics::matrixgraph::core::data_structures::Edges edgelist;
  edgelist.ReadFromCSV(input_path, sep);
  auto p_immutable_csr =
      sics::matrixgraph::core::util::format_converter::Edgelist2ImmutableCSR(
          edgelist);

  auto *offset = p_immutable_csr->GetOutOffsetBasePointer();
  auto *out_edges = p_immutable_csr->GetOutgoingEdgesBasePointer();
  VertexID *e_label = new VertexID[p_immutable_csr->get_num_outgoing_edges()]();

  std::ofstream offset_file(output_path + "/native_csrOffset_u32.bin");
  std::ofstream out_edges_file(output_path + "/native_csrDest_u32.bin");
  std::ofstream weight_file(output_path + "/native_csrWeight_u32.bin");

  p_immutable_csr->PrintGraph(10);

  offset_file.write((char *)offset,
                    sizeof(VertexID) *
                        (p_immutable_csr->get_num_vertices() + 1));
  out_edges_file.write((char *)out_edges,
                       sizeof(VertexID) *
                           p_immutable_csr->get_num_outgoing_edges());
  weight_file.write((char *)e_label,
                    sizeof(VertexID) *
                        p_immutable_csr->get_num_outgoing_edges());

  offset_file.close();
  out_edges_file.close();
  weight_file.close();

  delete p_immutable_csr;
}

static void ConvertEdgelistBin2CGGraphCSR(const std::string &input_path,
                                          const std::string &output_path) {

  sics::matrixgraph::core::data_structures::Edges edgelist;
  edgelist.ReadFromBin(input_path);
  edgelist.ShowGraph(3);

  auto p_immutable_csr =
      sics::matrixgraph::core::util::format_converter::Edgelist2ImmutableCSR(
          edgelist);

  auto *offset = p_immutable_csr->GetOutOffsetBasePointer();
  auto *out_edges = p_immutable_csr->GetOutgoingEdgesBasePointer();
  VertexID *e_label = new VertexID[p_immutable_csr->get_num_outgoing_edges()]();

  std::ofstream offset_file(output_path + "/native_csrOffset_u32.bin");
  std::ofstream out_edges_file(output_path + "/native_csrDest_u32.bin");
  std::ofstream weight_file(output_path + "/native_csrWeight_u32.bin");

  p_immutable_csr->PrintGraph(2);

  offset_file.write((char *)offset,
                    sizeof(VertexID) *
                        (p_immutable_csr->get_num_vertices() + 1));
  out_edges_file.write((char *)out_edges,
                       sizeof(VertexID) *
                           p_immutable_csr->get_num_outgoing_edges());
  weight_file.write((char *)e_label,
                    sizeof(VertexID) *
                        p_immutable_csr->get_num_outgoing_edges());

  offset_file.close();
  out_edges_file.close();
  weight_file.close();

  delete p_immutable_csr;
}

} // namespace converter
} // namespace tools
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_TOOLS_GRAPH_CONVERTER_CONVERTER_TO_IMMUTABLE_CSR_CUH_