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
using VertexLabel = sics::matrixgraph::core::common::VertexLabel;

void GridCSRTiledMatrixIO::Read(const std::string &input_path,
                                GridCSRTiledMatrix **grid_tiled_matrix) {
  YAML::Node node = YAML::LoadFile(input_path + "meta.yaml");

  std::ifstream label_file(input_path + "label.bin", std::ios::binary);
  if (!label_file)
    throw std::runtime_error("Error reading file: " + input_path + "label/" +
                             "0.bin");

  GridGraphMetadata grid_tiled_matrix_metadata = {
      .n_chunks = node["GridGraphMetadata"]["n_chunks"].as<GraphID>(),
      .n_vertices = node["GridGraphMetadata"]["n_vertices"].as<VertexID>(),
      .n_edges = node["GridGraphMetadata"]["n_edges"].as<EdgeIndex>(),
      .max_vid = node["GridGraphMetadata"]["max_vid"].as<VertexID>()};

  *grid_tiled_matrix = new GridCSRTiledMatrix(grid_tiled_matrix_metadata);

  CSRTiledMatrixIO csr_tiled_matrix_io;

  for (GraphID gid = 0; gid < pow(grid_tiled_matrix_metadata.n_chunks, 2);
       gid++) {
    std::string block_dir = input_path + "block" + std::to_string(gid) + "/";

    auto *csr_tiled_matrix_ptr =
        (*grid_tiled_matrix)->GetTiledMatrixPtrByIdx(gid);

    csr_tiled_matrix_io.Read(block_dir, csr_tiled_matrix_ptr);
  }

  // Read VLabel.
  label_file.seekg(0, std::ios::end);
  auto file_size = label_file.tellg();
  label_file.seekg(0, std::ios::beg);

  (*grid_tiled_matrix)
      ->SetVLabelBasePointer(
          new VertexLabel[grid_tiled_matrix_metadata.n_vertices]());
  auto *buffer_vlabel = (*grid_tiled_matrix)->GetVLabelBasePointer();
  label_file.read(reinterpret_cast<char *>(buffer_vlabel), file_size);
}

void GridCSRTiledMatrixIO::Write(const std::string &output_path,
                                 const GridCSRTiledMatrix &grid_tiled_matrix) {
  if (!std::filesystem::exists(output_path))
    std::filesystem::create_directory(output_path);

  auto meta = grid_tiled_matrix.get_metadata();

  CSRTiledMatrixIO csr_tiled_matrix_io;
  for (size_t _ = 0; _ < pow(meta.n_chunks, 2); _++) {
    std::string block_dir = output_path + "block" + std::to_string(_) + "/";
    CSRTiledMatrixIO::Write(block_dir,
                            *grid_tiled_matrix.GetTiledMatrixPtrByIdx(_));
  }

  // Write VLabel.
  VertexLabel *buffer_vlabel = grid_tiled_matrix.GetVLabelBasePointer();
  std::ofstream out_label_file(output_path + "label.bin", std::ios::binary);
  out_label_file.write(reinterpret_cast<char *>(buffer_vlabel),
                       sizeof(VertexLabel) * meta.n_vertices);
  delete[] buffer_vlabel;

  // Write metadata.
  std::ofstream out_meta_file(output_path + "meta.yaml");
  YAML::Node out_node;
  out_node["GridGraphMetadata"]["n_vertices"] = meta.n_vertices;
  out_node["GridGraphMetadata"]["n_edges"] = meta.n_edges;
  out_node["GridGraphMetadata"]["n_chunks"] = meta.n_chunks;
  out_node["GridGraphMetadata"]["max_vid"] = meta.max_vid;

  out_meta_file << out_node << std::endl;
  out_meta_file.close();
}

} // namespace io
} // namespace core
} // namespace matrixgraph
} // namespace sics