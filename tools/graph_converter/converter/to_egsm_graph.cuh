#ifndef MATRIXGRAPH_TOOLS_GRAPH_CONVERTER_CONVERTER_TO_EGSM_CUH_
#define MATRIXGRAPH_TOOLS_GRAPH_CONVERTER_CONVERTER_TO_EGSM_CUH_

#include "core/data_structures/edgelist.h"
#include "core/data_structures/immutable_csr.cuh"

namespace sics {
namespace matrixgraph {
namespace tools {
namespace converter {

using Vertex = sics::matrixgraph::core::data_structures::ImmutableCSRVertex;
using GraphID = sics::matrixgraph::core::common::GraphID;
using VertexID = sics::matrixgraph::core::common::VertexID;

static void ConvertCSRBin2EGSMGraph(const std::string& input_path,
                                    const std::string& output_path) {
  ImmutableCSR csr;
  csr.Read(input_path);

  std::ofstream file(output_path);
  if (!file.is_open()) {
    std::cerr << "Read error: " << output_path << std::endl;
    return;
  }

  file << "t " << csr.get_num_vertices() << " " << csr.get_num_outgoing_edges()
       << std::endl;

  for (VertexID v_idx = 0; v_idx < csr.get_num_vertices(); v_idx++) {
    auto u = csr.GetVertexByLocalID(v_idx);
    file << "v " << u.vid << " " << u.vlabel << " " << u.outdegree + u.indegree
         << std::endl;
  }

  for (VertexID v_idx = 0; v_idx < csr.get_num_vertices(); v_idx++) {
    auto u = csr.GetVertexByLocalID(v_idx);

    for (EdgeIndex e_idx = 0; e_idx < u.outdegree; e_idx++) {
      file << "e " << u.vid << " " << u.outgoing_edges[e_idx] << std::endl;
    }
  }

  file.close();
}

static void ConvertCSRBin2GNNPEGraph(const std::string& input_path,
                                     const std::string& output_path) {
  ImmutableCSR csr;
  csr.Read(input_path);

  std::ofstream file(output_path);
  if (!file.is_open()) {
    std::cerr << "Read error: " << output_path << std::endl;
    return;
  }

  for (VertexID v_idx = 0; v_idx < csr.get_num_vertices(); v_idx++) {
    auto u = csr.GetVertexByLocalID(v_idx);
    file << "v " << u.vid << " " << u.vlabel << " " << u.outdegree + u.indegree
         << std::endl;
  }

  for (VertexID v_idx = 0; v_idx < csr.get_num_vertices(); v_idx++) {
    auto u = csr.GetVertexByLocalID(v_idx);

    for (EdgeIndex e_idx = 0; e_idx < u.outdegree; e_idx++) {
      auto v_label = 0;
      file << "e " << u.vid << " " << u.outgoing_edges[e_idx] << " " << v_label
           << std::endl;
    }
  }

  file.close();
}

static void ConvertCSRBin2VF3Graph(const std::string& input_path,
                                   const std::string& output_path) {
  ImmutableCSR csr;
  csr.Read(input_path);

  std::ofstream file(output_path);
  if (!file.is_open()) {
    std::cerr << "Read error: " << output_path << std::endl;
    return;
  }

  file << csr.get_num_vertices() << std::endl;

  for (VertexID v_idx = 0; v_idx < csr.get_num_vertices(); v_idx++) {
    auto u = csr.GetVertexByLocalID(v_idx);
    file << u.vid << " " << u.vlabel << std::endl;
  }

  for (VertexID v_idx = 0; v_idx < csr.get_num_vertices(); v_idx++) {
    auto u = csr.GetVertexByLocalID(v_idx);
    file << u.outdegree << std::endl;

    for (EdgeIndex e_idx = 0; e_idx < u.outdegree; e_idx++) {
      auto v_label = 0;
      file << u.vid << " " << u.outgoing_edges[e_idx] << std::endl;
    }
  }

  file.close();
}

static void ConvertCSRBin2CECIGraph(const std::string& input_path,
                                    const std::string& output_path) {
  ImmutableCSR csr;
  csr.Read(input_path);

  std::ofstream file(output_path);
  if (!file.is_open()) {
    std::cerr << "Read error: " << output_path << std::endl;
    return;
  }
  file << "t # 0" << std::endl;
  for (VertexID v_idx = 0; v_idx < csr.get_num_vertices(); v_idx++) {
    auto u = csr.GetVertexByLocalID(v_idx);
    file << "v " << u.vid << " " << u.vlabel << " " << u.outdegree + u.indegree
         << std::endl;
  }

  for (VertexID v_idx = 0; v_idx < csr.get_num_vertices(); v_idx++) {
    auto u = csr.GetVertexByLocalID(v_idx);

    for (EdgeIndex e_idx = 0; e_idx < u.outdegree; e_idx++) {
      auto v_label = 0;
      file << "e " << u.vid << " " << u.outgoing_edges[e_idx] << " " << v_label
           << std::endl;
    }
  }

  file.close();
}

}  // namespace converter
}  // namespace tools
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_TOOLS_GRAPH_CONVERTER_CONVERTER_TO_IMMUTABLE_CSR_CUH_