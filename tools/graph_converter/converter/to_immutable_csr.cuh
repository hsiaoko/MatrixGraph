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
static void ConvertEdgelistCSV2ImmutableCSR(const std::string& input_path,
                                            const std::string& output_path,
                                            const std::string& sep,
                                            bool compressed = false,
                                            unsigned label_range = 1) {
  if (!std::filesystem::exists(output_path))
    std::filesystem::create_directory(output_path);

  sics::matrixgraph::core::data_structures::Edges edgelist;
  edgelist.ReadFromCSV(input_path, sep, compressed);

  edgelist.GenerateLocalID2GlobalID();
  if (compressed) {
    edgelist.Compacted();
  }

  edgelist.ShowGraph(10);

  auto p_immutable_csr =
      sics::matrixgraph::core::util::format_converter::Edgelist2ImmutableCSR(
          edgelist);
  // p_immutable_csr->SortByDegree();
  p_immutable_csr->GenerateVLabel(label_range);

  p_immutable_csr->PrintGraph(1);
  p_immutable_csr->Write(output_path);
  delete p_immutable_csr;
}

static void ConvertEdgelistBin2CSRBin(const std::string& input_path,
                                      const std::string& output_path,
                                      bool compressed = false,
                                      unsigned label_range = 1) {
  std::cout << "ConvertEdgelistBin2CSRBin" << std::endl;

  sics::matrixgraph::core::data_structures::Edges edgelist;
  edgelist.ReadFromBin(input_path);

  edgelist.GenerateLocalID2GlobalID();
  if (compressed) {
    edgelist.Compacted();
  }
  edgelist.ShowGraph(10);

  auto p_immutable_csr =
      sics::matrixgraph::core::util::format_converter::Edgelist2ImmutableCSR(
          edgelist);
  // p_immutable_csr->SortByDegree();
  p_immutable_csr->GenerateVLabel(label_range);
  p_immutable_csr->PrintGraph(1);
  p_immutable_csr->Write(output_path);
  delete p_immutable_csr;
}

static void ConvertEdgelistCSV2CGGraphCSR(const std::string& input_path,
                                          const std::string& output_path,
                                          const std::string& sep,
                                          unsigned label_range = 1) {
  if (!std::filesystem::exists(output_path))
    std::filesystem::create_directory(output_path);

  sics::matrixgraph::core::data_structures::Edges edgelist;
  edgelist.ReadFromCSV(input_path, sep);
  edgelist.GenerateVLabel(1);
  auto p_immutable_csr =
      sics::matrixgraph::core::util::format_converter::Edgelist2ImmutableCSR(
          edgelist);

  auto* offset = p_immutable_csr->GetOutOffsetBasePointer();
  auto* out_edges = p_immutable_csr->GetOutgoingEdgesBasePointer();
  VertexID* e_label = new VertexID[p_immutable_csr->get_num_outgoing_edges()]();

  std::ofstream offset_file(output_path + "/native_csrOffset_u32.bin");
  std::ofstream out_edges_file(output_path + "/native_csrDest_u32.bin");
  std::ofstream weight_file(output_path + "/native_csrWeight_u32.bin");

  p_immutable_csr->PrintGraph(10);

  offset_file.write(
      (char*)offset,
      sizeof(VertexID) * (p_immutable_csr->get_num_vertices() + 1));
  out_edges_file.write(
      (char*)out_edges,
      sizeof(VertexID) * p_immutable_csr->get_num_outgoing_edges());
  weight_file.write(
      (char*)e_label,
      sizeof(VertexID) * p_immutable_csr->get_num_outgoing_edges());

  offset_file.close();
  out_edges_file.close();
  weight_file.close();

  delete p_immutable_csr;
}

static void ConvertEdgelistBin2CGGraphCSR(const std::string& input_path,
                                          const std::string& output_path) {
  sics::matrixgraph::core::data_structures::Edges edgelist;
  edgelist.ReadFromBin(input_path);
  edgelist.ShowGraph(3);

  auto p_immutable_csr =
      sics::matrixgraph::core::util::format_converter::Edgelist2ImmutableCSR(
          edgelist);

  auto* offset = p_immutable_csr->GetOutOffsetBasePointer();
  auto* out_edges = p_immutable_csr->GetOutgoingEdgesBasePointer();
  VertexID* e_label = new VertexID[p_immutable_csr->get_num_outgoing_edges()]();

  std::ofstream offset_file(output_path + "/native_csrOffset_u32.bin");
  std::ofstream out_edges_file(output_path + "/native_csrDest_u32.bin");
  std::ofstream weight_file(output_path + "/native_csrWeight_u32.bin");

  p_immutable_csr->PrintGraph(3);

  offset_file.write(
      (char*)offset,
      sizeof(VertexID) * (p_immutable_csr->get_num_vertices() + 1));
  out_edges_file.write(
      (char*)out_edges,
      sizeof(VertexID) * p_immutable_csr->get_num_outgoing_edges());
  weight_file.write(
      (char*)e_label,
      sizeof(VertexID) * p_immutable_csr->get_num_outgoing_edges());

  offset_file.close();
  out_edges_file.close();
  weight_file.close();

  delete p_immutable_csr;
}

static void ConvertEGSMGraph2CSRBin(const std::string& input_path,
                                    const std::string& output_path) {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  std::ifstream ifs(input_path);
  if (ifs.fail()) {
    std::cout << "File not exist!\n";
    exit(-1);
  }

  std::cout << "ConvertEGSMGraph2EdgelistBin" << std::endl;
  // true for the query graph, false for the data graph

  sics::matrixgraph::core::data_structures::EdgelistMetadata edgelist_metadata;

  edgelist_metadata.num_edges;
  edgelist_metadata.num_vertices;

  char type;
  ifs >> type >> edgelist_metadata.num_vertices >> edgelist_metadata.num_edges;

  std::cout << " - num_vertices: " << edgelist_metadata.num_vertices
            << " num_edges: " << edgelist_metadata.num_edges << std::endl;

  auto buffer_edges =
      new sics::matrixgraph::core::data_structures::Edge[edgelist_metadata
                                                             .num_edges]();

  VertexID* localid2globalid = new VertexID[edgelist_metadata.num_vertices]();

  // assume that vertex id are compacted in Rapids Graph format.
  VertexLabel* v_label = new VertexLabel[edgelist_metadata.num_vertices]();

  EdgeIndex eid = 0;
  while (ifs >> type) {
    if (type == 'v') {
      VertexID vid, degree;
      VertexLabel label;
      ifs >> vid >> label >> degree;
      sics::matrixgraph::core::util::atomic::WriteMax(
          &edgelist_metadata.max_vid, vid);
      sics::matrixgraph::core::util::atomic::WriteMin(
          &edgelist_metadata.min_vid, vid);
      v_label[vid] = label;
      localid2globalid[vid] = vid;
    } else {
      VertexID src, dst;
      ifs >> buffer_edges[eid].src >> buffer_edges[eid].dst;
      eid++;
    }
  }

  sics::matrixgraph::core::data_structures::Edges edgelist(
      edgelist_metadata, buffer_edges, localid2globalid, v_label);

  edgelist.GenerateLocalID2GlobalID();
  edgelist.Compacted();

  edgelist.ShowGraph();

  auto p_immutable_csr =
      sics::matrixgraph::core::util::format_converter::Edgelist2ImmutableCSR(
          edgelist);
  // p_immutable_csr->SortByDegree();
  //  p_immutable_csr->GenerateVLabel(15);
  p_immutable_csr->PrintGraph(1);
  p_immutable_csr->Write(output_path);
  delete p_immutable_csr;

  std::cout << "[ConvertEGSMGraph2CSRBin] Done!" << std::endl;
  ifs.close();
}

}  // namespace converter
}  // namespace tools
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_TOOLS_GRAPH_CONVERTER_CONVERTER_TO_IMMUTABLE_CSR_CUH_