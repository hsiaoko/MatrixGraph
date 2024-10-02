#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

#include "core/common/yaml_config.h"
#include "core/data_structures/csr_tiled_matrix.cuh"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/metadata.h"
#include "core/io/csr_tiled_matrix_io.cuh"
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
using CSRTiledMatrix = sics::matrixgraph::core::data_structures::CSRTiledMatrix;
using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;
using sics::matrixgraph::core::util::atomic::WriteAdd;
using sics::matrixgraph::core::util::atomic::WriteMax;

void CSRTiledMatrixIO::Write(const std::string &output_path,
                             const CSRTiledMatrix &csr_tiled_matrix) {
  // Create dir of grid of gid.
  if (!std::filesystem::exists(output_path))
    std::filesystem::create_directory(output_path);
  if (!std::filesystem::exists(output_path + "/meta_buf"))
    std::filesystem::create_directory(output_path + "/meta_buf");

  auto metadata = csr_tiled_matrix.GetMetadata();

  std::cout << "[Write BitTiledMatrix] root_dir: " << output_path
            << " n_nz_tile: " << metadata.n_nz_tile << std::endl;

  std::ofstream out_meta_file(output_path + "meta.yaml");
  YAML::Node out_node;

  out_node["CSRTiledMatrixMetadata"]["n_strips"] = metadata.n_strips;
  out_node["CSRTiledMatrixMetadata"]["n_nz_tile"] = metadata.n_nz_tile;
  out_node["CSRTiledMatrixMetadata"]["tile_size"] = metadata.tile_size;
  out_node["CSRTiledMatrixMetadata"]["subgraphs"] =
      csr_tiled_matrix.GetCSRMetadata();

  out_meta_file << out_node << std::endl;

  if (metadata.n_nz_tile != 0) {
    std::ofstream out_row_idx_file(output_path + "meta_buf/row_idx.bin");
    std::ofstream out_col_idx_file(output_path + "meta_buf/col_idx.bin");
    std::ofstream out_tile_offset_row_file(output_path +
                                           "meta_buf/tile_offset_row.bin");
    std::ofstream out_nz_tile_bm(output_path + "meta_buf/nz_tile_bm.bin");
    std::ofstream csr_offset(output_path + "meta_buf/csr_offset.bin");
    std::ofstream out_data(output_path + "meta_buf/data.bin");

    std::cout << output_path << std::endl;
    out_row_idx_file.write(
        reinterpret_cast<char *>(csr_tiled_matrix.GetTileRowIdxPtr()),
        sizeof(VertexID) * metadata.n_nz_tile);

    out_col_idx_file.write(
        reinterpret_cast<char *>(csr_tiled_matrix.GetTileColIdxPtr()),
        sizeof(VertexID) * metadata.n_nz_tile);

    out_tile_offset_row_file.write(
        reinterpret_cast<char *>(csr_tiled_matrix.GetTileOffsetRowPtr()),
        sizeof(VertexID) * (metadata.n_strips + 1));

    out_nz_tile_bm.write(
        reinterpret_cast<char *>(csr_tiled_matrix.GetNzTileBitmapPtr()->data()),
        csr_tiled_matrix.GetNzTileBitmapPtr()->GetBufferSize());

    csr_offset.write(
        reinterpret_cast<char *>(csr_tiled_matrix.GetCSROffsetPtr()),
        sizeof(uint64_t) * (metadata.n_nz_tile + 1));

    out_data.write(reinterpret_cast<char *>(csr_tiled_matrix.GetDataPtr()),
                   csr_tiled_matrix.GetDataBufferSize());

    // for (int i = 0; i < csr_tiled_matrix.GetMetadata().n_nz_tile; i++) {
    //   auto data = csr_tiled_matrix.GetCSRBasePtrByIdx(i);
    //   ImmutableCSR csr(csr_tiled_matrix.GetCSRMetadataByIdx(i));
    //   csr.ParseBasePtr(data);
    //   csr.PrintGraph();
    // }

    out_row_idx_file.close();
    out_col_idx_file.close();
    out_tile_offset_row_file.close();
    out_nz_tile_bm.close();
    out_meta_file.close();
  }
  std::cout << "[Write BitTiledMatrix] Done!" << std::endl;
}

void CSRTiledMatrixIO::Read(const std::string &input_path,
                            CSRTiledMatrix *csr_tiled_matrix) {

  // Read meta.yaml
  YAML::Node in_node = YAML::LoadFile(input_path + "meta.yaml");

  TiledMatrixMetadata metadata{
      .n_strips = in_node["CSRTiledMatrixMetadata"]["n_strips"].as<VertexID>(),
      .n_nz_tile =
          in_node["CSRTiledMatrixMetadata"]["n_nz_tile"].as<VertexID>(),
      .tile_size =
          in_node["CSRTiledMatrixMetadata"]["tile_size"].as<VertexID>()};

  if (metadata.n_nz_tile == 0) {
    csr_tiled_matrix = nullptr;
    return;
  }

  csr_tiled_matrix->Init(metadata);

  auto csr_metadata = in_node["CSRTiledMatrixMetadata"]["subgraphs"]
                          .as<std::vector<SubGraphMetadata>>();

  csr_tiled_matrix->SetCSRMetadata(csr_metadata);

  auto *nz_tile_bm = new util::Bitmap(pow(metadata.n_strips, 2));

  std::ifstream in_row_idx_file(input_path + "meta_buf/row_idx.bin");
  std::ifstream in_col_idx_file(input_path + "meta_buf/col_idx.bin");
  std::ifstream in_tile_offset_row_file(input_path +
                                        "meta_buf/tile_offset_row.bin");
  std::ifstream in_nz_tile_bm(input_path + "meta_buf/nz_tile_bm.bin");
  std::ifstream in_data(input_path + "meta_buf/data.bin");
  std::ifstream csr_offset(input_path + "meta_buf/csr_offset.bin");

  in_row_idx_file.read(
      reinterpret_cast<char *>(csr_tiled_matrix->GetTileRowIdxPtr()),
      sizeof(VertexID) * metadata.n_nz_tile);

  in_col_idx_file.read(
      reinterpret_cast<char *>(csr_tiled_matrix->GetTileColIdxPtr()),
      sizeof(VertexID) * metadata.n_nz_tile);

  in_tile_offset_row_file.read(
      reinterpret_cast<char *>(csr_tiled_matrix->GetTileOffsetRowPtr()),
      sizeof(VertexID) * (metadata.n_strips + 1));

  in_nz_tile_bm.read(
      reinterpret_cast<char *>(csr_tiled_matrix->GetNzTileBitmapPtr()->data()),
      csr_tiled_matrix->GetNzTileBitmapPtr()->GetBufferSize());

  csr_offset.read(reinterpret_cast<char *>(csr_tiled_matrix->GetCSROffsetPtr()),
                  sizeof(uint64_t) * (metadata.n_nz_tile + 1));

  in_data.read(reinterpret_cast<char *>(csr_tiled_matrix->GetDataPtr()),
               csr_tiled_matrix->GetDataBufferSize());

  in_row_idx_file.close();
  in_col_idx_file.close();
  in_tile_offset_row_file.close();
  in_nz_tile_bm.close();
}

} // namespace io
} // namespace core
} // namespace matrixgraph
} // namespace sics