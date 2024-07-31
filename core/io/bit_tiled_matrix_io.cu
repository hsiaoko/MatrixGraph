#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

#include "core/common/yaml_config.h"
#include "core/data_structures/bit_tile.cuh"
#include "core/data_structures/metadata.h"
#include "core/io/bit_tiled_matrix_io.cuh"
#include "core/util/atomic.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace io {

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
using sics::matrixgraph::core::data_structures::BitTile;
using sics::matrixgraph::core::util::atomic::WriteAdd;
using sics::matrixgraph::core::util::atomic::WriteMax;

void BitTiledMatrixIO::Write(const std::string &output_path,
                             const BitTiledMatrix &bit_tiled_matrix) {

  // Create dir of grid of gid.
  if (!std::filesystem::exists(output_path))
    std::filesystem::create_directory(output_path);
  if (!std::filesystem::exists(output_path + "/tiles"))
    std::filesystem::create_directory(output_path + "/tiles");
  if (!std::filesystem::exists(output_path + "/meta_buf"))
    std::filesystem::create_directory(output_path + "/meta_buf");

  auto metadata = bit_tiled_matrix.GetMetadata();

  std::cout << "[Write BitTiledMatrix] root_dir: " << output_path
            << " n_nz_tile: " << metadata.n_nz_tile << std::endl;
  for (VertexID _ = 0; _ < metadata.n_nz_tile; _++) {
    std::ofstream out_data_file(output_path + "tiles/" + std::to_string(_) +
                                ".bin");

    auto *bit_tile = bit_tiled_matrix.GetTileByIdx(_);

    auto bm = bit_tile->GetBM();
    out_data_file.write((char *)bm->data(), bm->GetBufferSize());

    out_data_file.close();
  }

  std::ofstream out_row_idx_file(output_path + "meta_buf/row_idx.bin");
  std::ofstream out_col_idx_file(output_path + "meta_buf/col_idx.bin");
  std::ofstream out_tile_offset_row_file(output_path +
                                         "meta_buf/tile_offset_row.bin");
  std::ofstream out_nz_tile_bm(output_path + "meta_buf/nz_tile_bm.bin");

  out_row_idx_file.write(
      reinterpret_cast<char *>(bit_tiled_matrix.GetTileRowIdxPtr()),
      sizeof(VertexID) * metadata.n_nz_tile);

  out_col_idx_file.write(
      reinterpret_cast<char *>(bit_tiled_matrix.GetTileColIdxPtr()),
      sizeof(VertexID) * metadata.n_nz_tile);

  out_tile_offset_row_file.write(
      reinterpret_cast<char *>(bit_tiled_matrix.GetTileOffsetRowPtr()),
      sizeof(VertexID) * (metadata.n_strips + 1));

  out_nz_tile_bm.write(
      reinterpret_cast<char *>(bit_tiled_matrix.GetNzTileBitmapPtr()->data()),
      bit_tiled_matrix.GetNzTileBitmapPtr()->GetBufferSize());

  std::ofstream out_meta_file(output_path + "meta.yaml");
  YAML::Node out_node;

  out_node["BitTiledMatrixMetadata"]["n_strips"] = metadata.n_strips;
  out_node["BitTiledMatrixMetadata"]["n_nz_tile"] = metadata.n_nz_tile;
  out_node["BitTiledMatrixMetadata"]["tile_size"] = metadata.tile_size;

  out_meta_file << out_node << std::endl;

  out_row_idx_file.close();
  out_col_idx_file.close();
  out_tile_offset_row_file.close();
  out_nz_tile_bm.close();
  out_meta_file.close();
  std::cout << "[Write BitTiledMatrix] Done!" << std::endl;
}

void BitTiledMatrixIO::Read(const std::string &input_path,
                            BitTiledMatrix *bit_tiled_matrix) {

  // Read meta.yaml
  YAML::Node in_node = YAML::LoadFile(input_path + "meta.yaml");

  TiledMatrixMetadata metadata{
      .n_strips = in_node["BitTiledMatrixMetadata"]["n_strips"].as<VertexID>(),
      .n_nz_tile =
          in_node["BitTiledMatrixMetadata"]["n_nz_tile"].as<VertexID>(),
      .tile_size =
          in_node["BitTiledMatrixMetadata"]["tile_size"].as<VertexID>()};

  if (metadata.n_nz_tile == 0) {
    bit_tiled_matrix = nullptr;
    return;
  }

  bit_tiled_matrix->Init(metadata);

  auto *nz_tile_bm = new util::Bitmap(pow(metadata.n_strips, 2));

  for (VertexID _ = 0; _ < metadata.n_nz_tile; _++) {
    std::ifstream in_data_file(input_path + "tiles/" + std::to_string(_) +
                               ".bin");

    auto *bit_tile_ptr = bit_tiled_matrix->GetTileByIdx(_);
    in_data_file.read(reinterpret_cast<char *>(bit_tile_ptr->GetBM()->data()),
                      bit_tile_ptr->GetBM()->GetBufferSize());

    in_data_file.close();
  }

  std::ifstream in_row_idx_file(input_path + "meta_buf/row_idx.bin");
  std::ifstream in_col_idx_file(input_path + "meta_buf/col_idx.bin");
  std::ifstream in_tile_offset_row_file(input_path +
                                        "meta_buf/tile_offset_row.bin");
  std::ifstream in_nz_tile_bm(input_path + "meta_buf/nz_tile_bm.bin");

  in_row_idx_file.read(
      reinterpret_cast<char *>(bit_tiled_matrix->GetTileRowIdxPtr()),
      sizeof(VertexID) * metadata.n_nz_tile);

  in_col_idx_file.read(
      reinterpret_cast<char *>(bit_tiled_matrix->GetTileColIdxPtr()),
      sizeof(VertexID) * metadata.n_nz_tile);

  in_tile_offset_row_file.read(
      reinterpret_cast<char *>(bit_tiled_matrix->GetTileOffsetRowPtr()),
      sizeof(VertexID) * (metadata.n_strips + 1));

  in_nz_tile_bm.read(
      reinterpret_cast<char *>(bit_tiled_matrix->GetNzTileBitmapPtr()->data()),
      bit_tiled_matrix->GetNzTileBitmapPtr()->GetBufferSize());

  in_row_idx_file.close();
  in_col_idx_file.close();
  in_tile_offset_row_file.close();
  in_nz_tile_bm.close();
}

} // namespace io
} // namespace core
} // namespace matrixgraph
} // namespace sics