#include "core/io/grid_tiled_matrix_io.cuh"

#include <cmath>
#include <yaml-cpp/yaml.h>

#include "core/data_structures/metadata.h"
#include "core/io/bit_tiled_matrix_io.cuh"
#include "tools/common/types.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace io {
using GridGraphMetadata =
    sics::matrixgraph::core::data_structures::GridGraphMetadata;
using BitTiledMatrixIO = sics::matrixgraph::core::io::BitTiledMatrixIO;
using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using GraphID = sics::matrixgraph::core::common::GraphID;

void GridTiledMatrixIO::Read(const std::string &input_path,
                             GridTiledMatrix *grid_tiled_matrix) {
  YAML::Node node = YAML::LoadFile(input_path + "meta.yaml");

  GridGraphMetadata grid_tiled_matrix_metadata = {
      .n_chunks = node["GridGraphMetadata"]["n_chunks"].as<GraphID>(),
      .n_vertices = node["GridGraphMetadata"]["n_vertices"].as<VertexID>(),
      .n_edges = node["GridGraphMetadata"]["n_edges"].as<EdgeIndex>()};

  // std::cout << grid_tiled_matrix_metadata.n_vertices << std::endl;

  grid_tiled_matrix = new GridTiledMatrix(grid_tiled_matrix_metadata);

  BitTiledMatrixIO bit_tiled_matrix_io;

  for (GraphID gid = 0; gid < pow(grid_tiled_matrix_metadata.n_chunks, 2);
       gid++) {
    std::string block_dir = input_path + "block" + std::to_string(gid) + "/";

    std::cout << block_dir << std::endl;
    auto *bit_tiled_matrix_ptr =
        grid_tiled_matrix->GetBitTileMatrixPtrByIdx(gid);

    std::cout << "out 1|" << bit_tiled_matrix_ptr << std::endl;
    bit_tiled_matrix_io.Read(block_dir, bit_tiled_matrix_ptr);
    std::cout << "out 2|" << bit_tiled_matrix_ptr << std::endl;

    bit_tiled_matrix_ptr->Print();
  }

  grid_tiled_matrix->Print();
}

} // namespace io
} // namespace core
} // namespace matrixgraph
} // namespace sics