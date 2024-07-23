#ifndef MATRIXGRAPH_TOOLS_COMMON_GRID_TILED_MATRIX_IO_CUH_
#define MATRIXGRAPH_TOOLS_COMMON_GRID_TILED_MATRIX_IO_CUH_

#include <string>

#include "core/data_structures/bit_tiled_matrix.cuh"
#include "core/data_structures/grid_tiled_matrix.cuh"
#include "core/data_structures/metadata.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace io {

class GridTiledMatrixIO {
private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
  using GraphMetadata = sics::matrixgraph::core::data_structures::GraphMetadata;
  using BitTiledMatrix =
      sics::matrixgraph::core::data_structures::BitTiledMatrix;
  using GridTiledMatrix =
      sics::matrixgraph::core::data_structures::GridTiledMatrix;
  using TiledMatrixMetadata =
      sics::matrixgraph::core::data_structures::TiledMatrixMetadata;

public:
  GridTiledMatrixIO() = default;

  // @DESCRIPTION Write a set of edges to disk. It first converts edges to
  // Edgelist graphs, one Edgelist for each bucket, and then store the Edgelist
  // graphs in output_root_path/graphs.
  // @PARAMETERS
  static void Write(const std::string &output_path,
                    const GridTiledMatrix &grid_tiled_matrix);

  // @DESCRIPTION Read a set of edges from disk. It first reads the
  // Edgelist graphs from output_root_path/graphs, and then converts them to
  // a set of edges.
  // @PARAMETERS
  //  input_root_path: the root path of the input data.
  static void Read(const std::string &input_path,
                   GridTiledMatrix *grid_tiled_matrix);
};

} // namespace io
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_TOOLS_COMMON_GRID_TILED_MATRIX_IO_CUH_