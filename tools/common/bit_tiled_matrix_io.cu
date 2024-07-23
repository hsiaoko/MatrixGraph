#include <filesystem>
#include <fstream>
#include <vector>

#include "core/common/yaml_config.h"
#include "core/data_structures/metadata.h"
#include "core/util/atomic.h"
#include "tools/common/bit_tiled_matrix_io.cuh"

namespace sics {
namespace matrixgraph {
namespace tools {
namespace common {

using std::filesystem::create_directory;
using std::filesystem::exists;
using VertexID = sics::matrixgraph::core::common::VertexID;
using GraphID = sics::matrixgraph::core::common::GraphID;
using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using GraphMetadata = sics::matrixgraph::core::data_structures::GraphMetadata;
using SubGraphMetadata =
    sics::matrixgraph::core::data_structures::SubGraphMetadata;
using Bitmap = sics::matrixgraph::core::util::Bitmap;
using BitTiledMatrix = sics::matrixgraph::core::data_structures::BitTiledMatrix;
using sics::matrixgraph::core::util::atomic::WriteAdd;
using sics::matrixgraph::core::util::atomic::WriteMax;

void BitTiledMatrixIO::Write(const std::string &output_path,
                             const BitTiledMatrix &bit_tiled_matrix) {

  std::cout << "[Write BitTiledMatrix] root_dir: " << output_path << std::endl;
  // Create dir of grid of gid.
  if (!std::filesystem::exists(output_path))
    std::filesystem::create_directory(output_path);
  if (!std::filesystem::exists(output_path + "/tiles"))
    std::filesystem::create_directory(output_path + "/tiles");

  auto metadata = bit_tiled_matrix.GetMetadata();

  for (size_t _ = 0; _ < metadata.n_nz_tile; _++) {
    std::ofstream out_data_file(output_path + "tiles/" + std::to_string(_) +
                                ".bin");

    out_data_file.close();
  }

  std::ofstream out_meta_file(output_path + "meta.yaml");
  YAML::Node out_node;

  out_node["BitTiledMatrixMetadata"]["n_strips"] = metadata.n_strips;
  out_node["BitTiledMatrixMetadata"]["n_nz_tile"] = metadata.n_nz_tile;

  out_meta_file << out_node << std::endl;

  out_meta_file.close();
}

void BitTiledMatrixIO::Read(const std::string &input_path,
                            BitTiledMatrix *edges_blocks) {

  std::cout << "[Read BitTiledMatrix] root_dir: " << input_path << std::endl;

  // Read meta.yaml
  YAML::Node grid_graph_meta = YAML::LoadFile(input_path + "meta.yaml");
}

} // namespace common
} // namespace tools
} // namespace matrixgraph
} // namespace sics