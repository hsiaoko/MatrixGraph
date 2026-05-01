#include <filesystem>
#include <fstream>
#include <vector>

#include "core/data_structures/metadata.h"
#include "tools/common/edgelist_subgraphs_io.cuh"

namespace sics {
namespace matrixgraph {
namespace tools {
namespace common {

using std::filesystem::create_directory;
using std::filesystem::exists;
using VertexID = sics::matrixgraph::core::common::VertexID;
using GraphID = sics::matrixgraph::core::common::GraphID;
using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using EdgelistMetadata =
    sics::matrixgraph::core::data_structures::EdgelistMetadata;
using GraphMetadata = sics::matrixgraph::core::data_structures::GraphMetadata;
using SubGraphMetadata =
    sics::matrixgraph::core::data_structures::SubGraphMetadata;
using Bitmap = sics::matrixgraph::core::util::Bitmap;
using Edge = sics::matrixgraph::core::data_structures::Edge;
using Edges = sics::matrixgraph::core::data_structures::Edges;
using sics::matrixgraph::core::util::atomic::WriteAdd;
using sics::matrixgraph::core::util::atomic::WriteMax;

void EdgelistSubGraphsIO::Write(const std::string &output_root_path,
                                const std::vector<Edges> &edge_buckets,
                                const GraphMetadata &graph_metadata) {

  std::cout << "Writing subgraphs" << std::endl;
  if (!std::filesystem::exists(output_root_path))
    std::filesystem::create_directory(output_root_path);
  if (!std::filesystem::exists(output_root_path + "graphs/"))
    std::filesystem::create_directory(output_root_path + "graphs/");
  if (!std::filesystem::exists(output_root_path + "local_id_to_global_id/"))
    std::filesystem::create_directory(output_root_path +
                                      "local_id_to_global_id/");

  std::vector<SubGraphMetadata> subgraph_metadata_vec;
  for (GraphID gid = 0; gid < graph_metadata.num_subgraphs; gid++) {

    std::ofstream out_data_file(output_root_path + "graphs/" +
                                std::to_string(gid) + ".bin");
    std::ofstream out_local_id_to_global_id(output_root_path +
                                            "local_id_to_global_id/" +
                                            std::to_string(gid) + ".bin");

    out_data_file.write(
        reinterpret_cast<char *>(edge_buckets[gid].get_base_ptr()),
        sizeof(Edge) * edge_buckets[gid].get_metadata().num_edges);

    out_local_id_to_global_id.write(
        reinterpret_cast<char *>(
            edge_buckets[gid].get_localid_to_globalid_ptr()),
        sizeof(VertexID) * edge_buckets[gid].get_metadata().num_vertices);

    subgraph_metadata_vec.push_back(
        {gid, edge_buckets[gid].get_metadata().num_vertices,
         edge_buckets[gid].get_metadata().num_edges,
         edge_buckets[gid].get_metadata().num_edges,
         edge_buckets[gid].get_metadata().max_vid,
         edge_buckets[gid].get_metadata().min_vid});

    out_data_file.close();
    out_local_id_to_global_id.close();
  }

  std::ofstream out_meta_file(output_root_path + "meta.yaml");
  YAML::Node out_node;
  out_node["GraphMetadata"]["num_vertices"] = graph_metadata.num_vertices;
  out_node["GraphMetadata"]["num_edges"] = graph_metadata.num_edges;
  out_node["GraphMetadata"]["max_vid"] = graph_metadata.max_vid;
  out_node["GraphMetadata"]["min_vid"] = graph_metadata.min_vid;
  out_node["GraphMetadata"]["count_border_vertices"] = 0;
  out_node["GraphMetadata"]["num_subgraphs"] = graph_metadata.num_subgraphs;
  out_node["GraphMetadata"]["subgraphs"] = subgraph_metadata_vec;

  out_meta_file << out_node << std::endl;
  out_meta_file.close();
}

void EdgelistSubGraphsIO::Read(const std::string &input_root_path,
                               std::vector<Edges> *edges_blocks,
                               GraphMetadata *graph_metadata) {

  auto node = YAML::LoadFile(input_root_path + "meta.yaml");

  *graph_metadata = node.as<GraphMetadata>();

  edges_blocks->reserve(graph_metadata->num_subgraphs);

  for (GraphID _ = 0; _ < graph_metadata->num_subgraphs; _++) {

    EdgelistMetadata edgelist_metadata = {
        .num_vertices = graph_metadata->subgraphs[_].num_vertices,
        .num_edges = graph_metadata->subgraphs[_].num_outgoing_edges,
        .max_vid = graph_metadata->subgraphs[_].max_vid,
        .min_vid = graph_metadata->subgraphs[_].min_vid};

    std::ifstream in_file(input_root_path + "graphs/" + std::to_string(_) +
                          ".bin");
    std::ifstream id_map_file(input_root_path + "local_id_to_global_id/" +
                              std::to_string(_) + ".bin");

    auto buffer_edges = new Edge[edgelist_metadata.num_edges]();

    in_file.read(reinterpret_cast<char *>(buffer_edges),
                 sizeof(Edge) * edgelist_metadata.num_edges);

    edges_blocks->emplace_back(Edges(edgelist_metadata, buffer_edges));

    auto *localid_to_globalid = new VertexID[edgelist_metadata.num_vertices]();

    id_map_file.read(reinterpret_cast<char *>(localid_to_globalid),
                     sizeof(VertexID) * edgelist_metadata.num_vertices);
    edges_blocks->back().SetLocalIDToGlobalID(localid_to_globalid);

    in_file.close();
  }
}

} // namespace common
} // namespace tools
} // namespace matrixgraph
} // namespace sics