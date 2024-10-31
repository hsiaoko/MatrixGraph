#include "core/io/grid_csr_tiled_matrix_io.cuh"

#include <cmath>
#include <fstream>
#include <yaml-cpp/yaml.h>

#include "core/common/types.h"
#include "core/data_structures/metadata.h"
#include "core/io/csr_tiled_matrix_io.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace io {
using GridGraphMetadata =
    sics::matrixgraph::core::data_structures::GridGraphMetadata;
using CSRTiledMatrixIO = sics::matrixgraph::core::io::CSRTiledMatrixIO;
using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using GraphID = sics::matrixgraph::core::common::GraphID;

void GridCSRTiledMatrixIO::Read(const std::string &input_path,
                                GridCSRTiledMatrix **grid_tiled_matrix) {
  YAML::Node node = YAML::LoadFile(input_path + "meta.yaml");

  GridGraphMetadata grid_tiled_matrix_metadata = {
      .n_chunks = node["GridGraphMetadata"]["n_chunks"].as<GraphID>(),
      .n_vertices = node["GridGraphMetadata"]["n_vertices"].as<VertexID>(),
      .n_edges = node["GridGraphMetadata"]["n_edges"].as<EdgeIndex>()};

  *grid_tiled_matrix = new GridCSRTiledMatrix(grid_tiled_matrix_metadata);

  CSRTiledMatrixIO csr_tiled_matrix_io;

  for (GraphID gid = 0; gid < pow(grid_tiled_matrix_metadata.n_chunks, 2);
       gid++) {
    std::string block_dir = input_path + "block" + std::to_string(gid) + "/";

    auto *csr_tiled_matrix_ptr =
        (*grid_tiled_matrix)->GetTiledMatrixPtrByIdx(gid);

    csr_tiled_matrix_io.Read(block_dir, csr_tiled_matrix_ptr);
  }
}

void GridCSRTiledMatrixIO::Write(const std::string &output_path,
                                 const GridCSRTiledMatrix &grid_tiled_matrix) {

  auto meta = grid_tiled_matrix.get_metadata();

  CSRTiledMatrixIO csr_tiled_matrix_io;
  for (size_t _ = 0; _ < pow(meta.n_chunks, 2); _++) {
    std::string block_dir = output_path + "block" + std::to_string(_) + "/";
    CSRTiledMatrixIO::Write(block_dir,
                            *grid_tiled_matrix.GetTiledMatrixPtrByIdx(_));
  }

  std::ofstream out_meta_file(output_path + "meta.yaml");
  YAML::Node out_node;
  out_node["GridGraphMetadata"]["n_vertices"] = meta.n_vertices;
  out_node["GridGraphMetadata"]["n_edges"] = meta.n_edges;
  out_node["GridGraphMetadata"]["n_chunks"] = meta.n_chunks;

  out_meta_file << out_node << std::endl;
  out_meta_file.close();
}

} // namespace io
} // namespace core
} // namespace matrixgraph
} // namespace sics