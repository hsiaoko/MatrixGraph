#ifndef MATRIXGRAPH_TOOLS_COMMON_EDGELIST_SUBGRAPHS_WRITER_CUH_
#define MATRIXGRAPH_TOOLS_COMMON_EDGELIST_SUBGRAPHS_WRITER_CUH_

#include <string>

#include "core/data_structures/edgelist.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/metadata.h"
#include "tools/common/types.h"

namespace sics {
namespace matrixgraph {
namespace tools {
namespace common {

class EdgelistSubGraphsIO {
private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
  using Edges = sics::matrixgraph::core::data_structures::Edges;
  using GraphMetadata = sics::matrixgraph::core::data_structures::GraphMetadata;
  using StoreStrategy = sics::matrixgraph::tools::common::StoreStrategy;

public:
  EdgelistSubGraphsIO() = default;

  void Init(const std::string &root_path) {
    if (!std::filesystem::exists(root_path))
      std::filesystem::create_directory(root_path);
    if (!std::filesystem::exists(root_path + "/graphs"))
      std::filesystem::create_directory(root_path + "/graphs");
    if (!std::filesystem::exists(root_path + "/local_id_to_global_id"))
      std::filesystem::create_directory(root_path + "/local_id_to_global_id");
  }

  // @DESCRIPTION Write a set of edges to disk. It first converts edges to
  // Edgelist graphs, one Edgelist for each bucket, and then store the Edgelist
  // graphs in output_root_path/graphs.
  // @PARAMETERS
  //  edge_bucket: a set of edges.
  //  graph_metadata: metadata of the graph.
  static void Write(const std::string &output_root_path,
                    const std::vector<Edges> &edge_buckets,
                    const GraphMetadata &graph_metadata);

  // @DESCRIPTION Read a set of edges from disk. It first reads the
  // Edgelist graphs from output_root_path/graphs, and then converts them to
  // a set of edges.
  // @PARAMETERS
  //  input_root_path: the root path of the input data.
  //  edge_buckets: a set of edges.
  //  graph_metadata: metadata of the graph.
  static void Read(const std::string &input_root_path,
                   std::vector<Edges> *edges_blocks,
                   GraphMetadata *graph_metadata);
};

} // namespace common
} // namespace tools
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_TOOLS_COMMON_EDGELIST_SUBGRAPHS_WRITER_CUH_