#ifndef MATRIXGRAPH_TOOLS_COMMON_BI_TILED_MATRIX_IO_CUH_
#define MATRIXGRAPH_TOOLS_COMMON_BI_TILED_MATRIX_IO_CUH_

#include <string>

#include "core/data_structures/bit_tiled_matrix.cuh"
#include "core/data_structures/metadata.h"
#include "tools/common/types.h"

namespace sics {
namespace matrixgraph {
namespace tools {
namespace common {

class BitTiledMatrixIO {
private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
  using GraphMetadata = sics::matrixgraph::core::data_structures::GraphMetadata;
  using BitTiledMatrix =
      sics::matrixgraph::core::data_structures::BitTiledMatrix;
  using TiledMatrixMetadata =
      sics::matrixgraph::core::data_structures::TiledMatrixMetadata;
  using StoreStrategy = sics::matrixgraph::tools::common::StoreStrategy;

public:
  BitTiledMatrixIO() = default;

  static void Init(const std::string &root_path) {
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
  static void Write(const std::string &output_path,
                    const BitTiledMatrix &bit_tiled_matrix);

  // @DESCRIPTION Read a set of edges from disk. It first reads the
  // Edgelist graphs from output_root_path/graphs, and then converts them to
  // a set of edges.
  // @PARAMETERS
  //  input_root_path: the root path of the input data.
  //  edge_buckets: a set of edges.
  //  graph_metadata: metadata of the graph.
  static void Read(const std::string &input_path, BitTiledMatrix *edges_blocks);
};

} // namespace common
} // namespace tools
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_TOOLS_COMMON_BI_TILED_MATRIX_IO_CUH_