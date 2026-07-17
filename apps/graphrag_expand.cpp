// GraphRAG expand: given seed vertices, do k-hop outgoing expansion on a
// MatrixGraph CSR and return the indices of expanded edges (into the CSR
// outgoing_edges array). Python can then map these indices back to
// edge Attributes / chunk_text.
#include <gflags/gflags.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <queue>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/common/types.h"
#include "core/data_structures/immutable_csr.cuh"

DEFINE_string(g, "", "Path to the input CSR graph directory (required)");
DEFINE_string(s, "", "Path to the seeds file: one local vertex id per line (required)");
DEFINE_string(o, "", "Path to the output directory (required)");
DEFINE_int32(k, 2, "Maximum hop depth (default: 2)");
DEFINE_int32(max_degree, 10,
             "Max outgoing edges to expand per vertex per hop; 0 = unlimited (default: 10)");

using sics::matrixgraph::core::common::EdgeIndex;
using sics::matrixgraph::core::common::VertexID;
using sics::matrixgraph::core::data_structures::ImmutableCSR;

bool ValidateParameters() {
  bool is_valid = true;
  if (FLAGS_g.empty()) {
    std::cerr << "Error: Input graph path (-g) is required" << std::endl;
    is_valid = false;
  }
  if (FLAGS_s.empty()) {
    std::cerr << "Error: Seeds file (-s) is required" << std::endl;
    is_valid = false;
  }
  if (FLAGS_o.empty()) {
    std::cerr << "Error: Output path (-o) is required" << std::endl;
    is_valid = false;
  }
  if (FLAGS_k < 0) {
    std::cerr << "Error: k must be >= 0" << std::endl;
    is_valid = false;
  }
  if (FLAGS_max_degree < 0) {
    std::cerr << "Error: max_degree must be >= 0" << std::endl;
    is_valid = false;
  }
  return is_valid;
}

std::vector<VertexID> LoadSeeds(const std::string& path) {
  std::vector<VertexID> seeds;
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Cannot open seeds file: " + path);
  }
  VertexID v;
  while (in >> v) {
    seeds.push_back(v);
  }
  return seeds;
}

// k-hop outgoing expansion on CPU using MatrixGraph ImmutableCSR.
// Returns:
//   - expanded_edge_indices: offsets into CSR outgoing_edges (sorted, unique)
//   - reached_vertices: local vertex ids that are reachable within k hops
void Expand(
    const ImmutableCSR& g,
    const std::vector<VertexID>& seeds,
    int max_hop,
    int max_degree,
    std::vector<EdgeIndex>* expanded_edge_indices,
    std::vector<VertexID>* reached_vertices) {
  std::set<EdgeIndex> edge_set;
  std::unordered_set<VertexID> visited;
  std::queue<std::pair<VertexID, int>> q;

  for (VertexID s : seeds) {
    if (s < g.get_num_vertices() && visited.insert(s).second) {
      q.emplace(s, 0);
    }
  }

  while (!q.empty()) {
    auto [u, hop] = q.front();
    q.pop();

    if (hop >= max_hop) continue;

    EdgeIndex start = g.GetOutOffsetByLocalID(u);
    EdgeIndex end = g.GetOutOffsetByLocalID(u + 1);
    EdgeIndex count = end - start;
    EdgeIndex limit = count;
    if (max_degree > 0 && count > static_cast<EdgeIndex>(max_degree)) {
      limit = static_cast<EdgeIndex>(max_degree);
    }

    for (EdgeIndex i = 0; i < limit; ++i) {
      EdgeIndex eidx = start + i;
      edge_set.insert(eidx);
      VertexID v = g.GetOutgoingEdgesByLocalID(u)[i];
      if (visited.insert(v).second) {
        q.emplace(v, hop + 1);
      }
    }
  }

  expanded_edge_indices->assign(edge_set.begin(), edge_set.end());
  reached_vertices->assign(visited.begin(), visited.end());
  std::sort(expanded_edge_indices->begin(), expanded_edge_indices->end());
  std::sort(reached_vertices->begin(), reached_vertices->end());
}

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "GraphRAG expand: k-hop outgoing expansion on MatrixGraph CSR\n"
      "Usage: " +
      std::string(argv[0]) +
      " -g <csr_dir> -s <seeds.txt> -o <output_dir> [-k <hop>] "
      "[-max_degree <n>]");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!ValidateParameters()) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "apps/graphrag_expand.cpp");
    return EXIT_FAILURE;
  }

  try {
    ImmutableCSR g;
    g.Read(FLAGS_g);

    auto seeds = LoadSeeds(FLAGS_s);

    std::vector<EdgeIndex> edge_indices;
    std::vector<VertexID> reached_vertices;
    Expand(g, seeds, FLAGS_k, FLAGS_max_degree, &edge_indices,
           &reached_vertices);

    std::filesystem::create_directories(FLAGS_o);

    // Write expanded edge indices: each is an offset into CSR outgoing_edges.
    std::ofstream edge_file(FLAGS_o + "/expanded_edges.bin", std::ios::binary);
    edge_file.write(reinterpret_cast<const char*>(edge_indices.data()),
                    sizeof(EdgeIndex) * edge_indices.size());
    edge_file.close();

    // Write reached vertices.
    std::ofstream vertex_file(FLAGS_o + "/reached_vertices.bin",
                              std::ios::binary);
    vertex_file.write(reinterpret_cast<const char*>(reached_vertices.data()),
                      sizeof(VertexID) * reached_vertices.size());
    vertex_file.close();

    // Write summary.
    std::ofstream summary_file(FLAGS_o + "/expand_summary.txt");
    summary_file << "num_vertices " << g.get_num_vertices() << "\n";
    summary_file << "num_seeds " << seeds.size() << "\n";
    summary_file << "max_hop " << FLAGS_k << "\n";
    summary_file << "max_degree " << FLAGS_max_degree << "\n";
    summary_file << "num_edges_expanded " << edge_indices.size() << "\n";
    summary_file << "num_vertices_reached " << reached_vertices.size() << "\n";
    summary_file.close();

    std::cout << "[GraphRAG Expand] seeds=" << seeds.size()
              << ", hop=" << FLAGS_k
              << ", max_degree=" << FLAGS_max_degree
              << ", edges=" << edge_indices.size()
              << ", vertices=" << reached_vertices.size()
              << std::endl;
    std::cout << "[GraphRAG Expand] Wrote " << edge_indices.size()
              << " edge indices to " << FLAGS_o << "/expanded_edges.bin"
              << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
