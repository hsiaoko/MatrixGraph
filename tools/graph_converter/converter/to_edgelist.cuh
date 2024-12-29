#ifndef MATRIXGRAPH_TOOLS_GRAPH_CONVERTER_CONVERTER_TO_EDGELIST_CUH_
#define MATRIXGRAPH_TOOLS_GRAPH_CONVERTER_CONVERTER_TO_EDGELIST_CUH_

#include "core/common/types.h"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/util/atomic.h"
#include "core/util/format_converter.cuh"

namespace sics {
namespace matrixgraph {
namespace tools {
namespace converter {

using sics::matrixgraph::core::data_structures::ImmutableCSR;
using VertexLabel = sics::matrixgraph::core::common::VertexLabel;

static void ConvertEdgelistCSV2EdgelistBin(const std::string &input_path,
                                           const std::string &output_path,
                                           const std::string &sep,
                                           bool compressed = false) {
  std::cout << "ConvertEdgelistCSV2EdgelistBin" << std::endl;
  if (!std::filesystem::exists(output_path))
    std::filesystem::create_directory(output_path);

  sics::matrixgraph::core::data_structures::Edges edgelist;
  edgelist.ReadFromCSV(input_path, sep, compressed);
  edgelist.ShowGraph();
  edgelist.WriteToBinary(output_path);
}

static void ConvertImmutableCSR2EdgelistBin(const std::string &input_path,
                                            const std::string &output_path,
                                            bool compressed = false) {

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
      sics::matrixgraph::core::util::format_converter::ImmutableCSR2Edgelist(
          csr);

  edges_ptr->GenerateLocalID2GlobalID();
  if (compressed) {
    edges_ptr->Compacted();
  }

  edges_ptr->WriteToBinary(output_path);
  std::cout << "[ConvertImmutableCSR2EdgelistBin] Done!" << std::endl;
}

static void
ConvertEdgelistBin2TransposedEdgelistBin(const std::string &input_path,
                                         const std::string &output_path) {
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
  edgelist.Transpose();
  edgelist.WriteToBinary(output_path);
  std::cout << "[ConvertEdgelistBin2TransposedEdgelistBin] Done!" << std::endl;
}

static void ConvertEGSMGraph2EdgelistBin(const std::string &input_path,
                                         const std::string &output_path) {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  std::ifstream ifs(input_path);
  if (ifs.fail()) {
    std::cout << "File not exist!\n";
    exit(-1);
  }

  // true for the query graph, false for the data graph

  sics::matrixgraph::core::data_structures::EdgelistMetadata edgelist_metadata;

  edgelist_metadata.num_edges;
  edgelist_metadata.num_vertices;

  char type;
  ifs >> type >> edgelist_metadata.num_vertices >> edgelist_metadata.num_edges;

  auto buffer_edges =
      new sics::matrixgraph::core::data_structures::Edge[edgelist_metadata
                                                             .num_edges]();

  VertexLabel *v_label = new VertexLabel[edgelist_metadata.num_vertices]();

  VertexID *localid2globalid = new VertexID[edgelist_metadata.num_vertices]();

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

  edgelist.ShowGraph();

  edgelist.WriteToBinary(output_path);
  std::cout << "[ConvertEGSMGraph2EdgelistBin] Done!" << std::endl;

  ifs.close();
}

} // namespace converter
} // namespace tools
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_TOOLS_GRAPH_CONVERTER_CONVERTER_TO_EDGELIST_CUH_