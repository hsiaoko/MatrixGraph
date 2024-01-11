#ifndef SICS_MATRIXGRAPH_CORE_DATA_STRUCTURES_TILED_MATRIX_H_
#define SICS_MATRIXGRAPH_CORE_DATA_STRUCTURES_TILED_MATRIX_H_

#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>

#ifdef TBB_FOUND
#include <execution>
#endif

#include "core/common/types.h"
#include "core/common/yaml_config.h"
#include "core/data_structures/metadata.h"
#include "core/util/atomic.h"
#include "core/util/bitmap.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

class Mask {
private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using TileIndex = sics::matrixgraph::core::common::TileIndex;
  using Bitmap = sics::matrixgraph::core::util::Bitmap;

public:
  Mask() = default;
  ~Mask() { delete bm_; }

  Mask(VertexID x, VertexID y) { Init(x, y); }

  void Init(VertexID x, VertexID y) {
    mask_size_x_ = x;
    mask_size_y_ = x;

    bm_ = new Bitmap(x * y);
  }

  TileIndex get_mask_size_x() const { return mask_size_x_; }
  TileIndex get_mask_size_y() const { return mask_size_y_; }

  bool GetBit(VertexID x, VertexID y) {
    return bm_->GetBit(x * mask_size_x_ + y);
  }

  void SetBit(VertexID x, VertexID y) { bm_->SetBit(x * mask_size_x_ + y); }

private:
  TileIndex mask_size_x_ = 4;
  TileIndex mask_size_y_ = 4;

  Bitmap *bm_ = nullptr;
};

class Tile {
private:
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using TileIndex = sics::matrixgraph::core::common::TileIndex;

public:
  Tile() = default;

  Tile(TileIndex tile_size_x, TileIndex tile_size_y, VertexID tile_x,
       VertexID tile_y, VertexID n_nz) {
    Init_host(tile_size_x, tile_size_y, tile_x, tile_y, n_nz);
  }

  void operator=(const Tile &other) {
    row_idx_ = other.row_idx_;
    col_idx_ = other.col_idx_;
    row_ptr_ = other.row_ptr_;
  }

  void operator=(const Tile &&other) {
    row_idx_ = other.row_idx_;
    col_idx_ = other.col_idx_;
    row_ptr_ = other.row_ptr_;
  }


  cudaError_t Init_device(TileIndex tile_size_x, TileIndex tile_size_y,
                          VertexID tile_x, VertexID tile_y, VertexID n_nz) {
    tile_size_x_ = tile_size_x;
    tile_size_y_ = tile_size_y;
    tile_x_ = tile_x;
    tile_y_ = tile_y;
    n_nz_ = n_nz;

    if (row_ptr_ != nullptr)
      cudaFree(row_ptr_);
    if (mask_ != nullptr)
      cudaFree(mask_);

    if (row_idx_ != nullptr)
      cudaFree(row_idx_);
    if (col_idx_ != nullptr)
      cudaFree(col_idx_);
    if (data_ != nullptr)
      cudaFree(data_);

    cudaMalloc(reinterpret_cast<void **>(&row_ptr_),
               tile_size_y_ * sizeof(TileIndex));
    cudaMalloc(reinterpret_cast<void **>(&row_idx_),
               tile_size_y_ * sizeof(TileIndex));
    cudaMalloc(reinterpret_cast<void **>(&col_idx_), n_nz_ * sizeof(TileIndex));
    cudaMalloc(reinterpret_cast<void **>(&data_), n_nz_ * sizeof(VertexLabel));

    return cudaSuccess;
  }

  void Init_host(TileIndex tile_size_x, TileIndex tile_size_y, VertexID tile_x,
                 VertexID tile_y, VertexID n_nz) {
    tile_size_x_ = tile_size_x;
    tile_size_y_ = tile_size_y;
    tile_x_ = tile_x;
    tile_y_ = tile_y;
    n_nz_ = n_nz;

    if (row_ptr_ != nullptr)
      delete[] row_ptr_;
    if (mask_ != nullptr)
      delete mask_;
    if (row_idx_ != nullptr)
      delete[] row_idx_;
    if (col_idx_ != nullptr)
      delete[] col_idx_;
    if (data_ != nullptr)
      delete[] data_;

    row_ptr_ = new TileIndex[tile_size_y]();
    mask_ = new Mask(tile_size_x, tile_size_y);
    row_idx_ = new TileIndex[tile_size_y_]();
    col_idx_ = new TileIndex[n_nz_]();
    data_ = new VertexLabel[n_nz_]();
  }

  cudaError_t MemcpyAsyncHost2Device(const Tile &tile,
                                     const cudaStream_t &stream) {
    assert(tile_x_ == tile.tile_x_);
    assert(tile_y_ == tile.tile_y_);
    assert(n_nz_ == tile.n_nz_);
    assert(tile_size_x_ == tile.tile_size_x_);
    assert(tile_size_y_ == tile.tile_size_y_);

    cudaMemcpyAsync(row_ptr_, tile.row_ptr_, sizeof(TileIndex) * (tile_size_y_),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(row_idx_, tile.row_idx_, sizeof(TileIndex) * n_nz_,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(col_idx_, tile.col_idx_, sizeof(TileIndex) * n_nz_,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(data_, tile.data_, sizeof(TileIndex) * n_nz_,
                    cudaMemcpyHostToDevice, stream);

    return cudaSuccess;
  }

  cudaError_t MemcpyAsyncDevice2Host(const Tile &tile,
                                     const cudaStream_t &stream) {
    assert(tile_x_ == tile.tile_x_);
    assert(tile_y_ == tile.tile_y_);
    assert(n_nz_ == tile.n_nz_);
    assert(tile_size_x_ == tile.tile_size_x_);
    assert(tile_size_y_ == tile.tile_size_y_);

    cudaMemcpyAsync(row_ptr_, tile.row_ptr_, sizeof(TileIndex) * (tile_size_y_),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(row_idx_, tile.row_idx_, sizeof(TileIndex) * n_nz_,
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(col_idx_, tile.col_idx_, sizeof(TileIndex) * n_nz_,
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(data_, tile.data_, sizeof(TileIndex) * n_nz_,
                    cudaMemcpyDeviceToHost, stream);

    return cudaSuccess;
  }

  void Show(bool is_transposed = false) {
    std::cout << "**********  Tile: (" << (int)tile_x_ << ", " << (int)tile_y_
              << ")"
              << " n_nz: " << n_nz_ << " ************" << std::endl;
    if (is_transposed) {
      std::cout << " col_ptr: ";
    } else {
      std::cout << " row_ptr: ";
    }
    for (int i = 0; i < tile_size_y_; i++) {
      std::cout << (int)row_ptr_[i] << " ";
    }

    std::cout << std::endl << " row_idx: ";
    for (VertexID i = 0; i < n_nz_; i++) {
      std::cout << (int)row_idx_[i] << " ";
    }

    std::cout << std::endl << " col_idx: ";
    for (VertexID i = 0; i < n_nz_; i++) {
      std::cout << (int)col_idx_[i] << " ";
    }
    std::cout << std::endl << std::endl;
  }

  void SetBit(VertexID x, VertexID y) { mask_->SetBit(x, y); }
  bool GetBit(VertexID x, VertexID y) { return mask_->GetBit(x, y); }

  VertexID n_nz_ = 0;

  VertexID tile_x_;
  VertexID tile_y_;
  // store the index of tile (x,y)

  TileIndex tile_size_x_;
  TileIndex tile_size_y_;
  // store the size of tile, tile_size_x times tile_size_y

  TileIndex *row_ptr_ = nullptr;
  TileIndex *row_idx_ = nullptr;
  TileIndex *col_idx_ = nullptr;
  VertexLabel *data_ = nullptr;
  Mask *mask_ = nullptr;
};

class TiledMatrix {
private:
  using TiledMatrixMetadata =
      sics::matrixgraph::core::data_structures::TiledMatrixMetadata;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using TileIndex = sics::matrixgraph::core::common::TileIndex;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;

public:
  TiledMatrix() = default;

  void Init(VertexID n_rows, VertexID n_cols, VertexID n_nz_tile,
            Tile **data_ptr, Mask *mask, VertexID *tile_ptr,
            VertexID *tile_col_idx, VertexID *tile_n_nz) {
    metadata_.n_rows = n_rows;
    metadata_.n_cols = n_cols;
    metadata_.n_nz_tile = n_nz_tile;

    data_ptr_ = new Tile *[n_nz_tile];
    memcpy(data_ptr_, data_ptr, sizeof(Tile *) * n_nz_tile);
    mask_ = mask;

    tile_ptr_ = new VertexID[n_rows + 1]();
    memcpy(tile_ptr_, tile_ptr, sizeof(VertexID) * (n_rows + 1));

    tile_col_idx_ = new VertexID[n_nz_tile]();
    memcpy(tile_col_idx_, tile_col_idx, sizeof(VertexID) * (n_nz_tile));

    tile_n_nz_ = new VertexID[n_nz_tile]();
    memcpy(tile_n_nz_, tile_n_nz, sizeof(VertexID) * (n_nz_tile));
  }

  void Show() {
    std::cout << "TiledMatrix: " << std::endl;
    std::cout << "  n_rows: " << metadata_.n_rows << std::endl;
    std::cout << "  n_cols: " << metadata_.n_cols << std::endl;
    std::cout << "  n_nz_tile: " << metadata_.n_nz_tile << std::endl;

    std::cout << "  tile_ptr: ";
    for (VertexID i = 0; i < metadata_.n_rows + 1; i++) {
      std::cout << tile_ptr_[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "  tile_col_idx: ";
    for (VertexID i = 0; i < metadata_.n_nz_tile; i++) {
      std::cout << tile_col_idx_[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "  tile_n_nz: ";
    for (VertexID i = 0; i < metadata_.n_nz_tile; i++) {
      std::cout << tile_n_nz_[i] << " ";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < metadata_.n_nz_tile; i++) {
      data_ptr_[i]->Show();
    }
  }

  void Read(const std::string &root_path) {

    std::cout << "Read: " << root_path << std::endl;

    YAML::Node metadata_node;
    try {
      metadata_node = YAML::LoadFile(root_path + "meta.yaml");
      metadata_ = metadata_node.as<TiledMatrixMetadata>();
    } catch (YAML::BadFile &e) {
      std::cout << "meta.yaml file read failed! " << e.msg << std::endl;
    }

    tile_ptr_ = new VertexID[metadata_.n_rows + 1]();
    tile_n_nz_ = new VertexID[metadata_.n_nz_tile]();
    tile_col_idx_ = new VertexID[metadata_.n_nz_tile]();

    // Read tile_ptr, tile_col_idx tile_n_nz
    std::ifstream tile_ptr_file(root_path + "tile_ptr.bin");
    std::ifstream col_idx_file(root_path + "tile_col_idx.bin");
    std::ifstream tile_n_nz_file(root_path + "tile_n_nz.bin");

    tile_ptr_file.read(reinterpret_cast<char *>(tile_ptr_),
                       sizeof(VertexID) * (metadata_.n_rows + 1));

    col_idx_file.read(reinterpret_cast<char *>(tile_col_idx_),
                      sizeof(VertexID) * metadata_.n_nz_tile);
    tile_n_nz_file.read(reinterpret_cast<char *>(tile_n_nz_),
                        sizeof(VertexID) * metadata_.n_nz_tile);

    // Read tiles.
    data_ptr_ = new Tile *[metadata_.n_nz_tile]();
    for (auto i = 0; i < metadata_.n_nz_tile; i++) {
      std::ifstream data_file(root_path + "tiles/" + std::to_string(i) + ".bin",
                              std::ios::binary);

      TileIndex tile_size_x, tile_size_y;
      VertexID n_nz, tile_x, tile_y;

      data_file.read(reinterpret_cast<char *>(&n_nz), sizeof(VertexID));
      data_file.read(reinterpret_cast<char *>(&tile_x), sizeof(VertexID));
      data_file.read(reinterpret_cast<char *>(&tile_y), sizeof(VertexID));
      data_file.read(reinterpret_cast<char *>(&tile_size_x), sizeof(TileIndex));
      data_file.read(reinterpret_cast<char *>(&tile_size_y), sizeof(TileIndex));

      data_ptr_[i] = new Tile(tile_size_x, tile_size_y, tile_x, tile_y, n_nz);

      data_file.read(reinterpret_cast<char *>(data_ptr_[i]->row_ptr_),
                     sizeof(TileIndex) * data_ptr_[i]->tile_size_y_);
      data_file.read(reinterpret_cast<char *>(data_ptr_[i]->row_idx_),
                     sizeof(TileIndex) * data_ptr_[i]->n_nz_);
      data_file.read(reinterpret_cast<char *>(data_ptr_[i]->col_idx_),
                     sizeof(TileIndex) * data_ptr_[i]->n_nz_);
      data_file.read(reinterpret_cast<char *>(data_ptr_[i]->data_),
                     sizeof(VertexLabel) * data_ptr_[i]->n_nz_);

      data_file.close();
    }
  }

  void Write(const std::string &root_path) {

    // Init path
    if (!std::filesystem::exists(root_path))
      std::filesystem::create_directory(root_path);

    if (!std::filesystem::exists(root_path + "tiles"))
      std::filesystem::create_directory(root_path + "tiles");

    // Write tile_ptr, tile_col_idx tile_n_nz
    std::ofstream tile_ptr_file(root_path + "tile_ptr.bin");
    std::ofstream col_idx_file(root_path + "tile_col_idx.bin");
    std::ofstream tile_n_nz_file(root_path + "tile_n_nz.bin");

    tile_ptr_file.write(reinterpret_cast<char *>(tile_ptr_),
                        sizeof(VertexID) * (metadata_.n_rows + 1));
    col_idx_file.write(reinterpret_cast<char *>(tile_col_idx_),
                       sizeof(VertexID) * metadata_.n_nz_tile);
    tile_n_nz_file.write(reinterpret_cast<char *>(tile_n_nz_),
                         sizeof(VertexID) * metadata_.n_nz_tile);

    // Write tile
    for (auto i = 0; i < metadata_.n_nz_tile; i++) {
      std::ofstream data_file(root_path + "tiles/" + std::to_string(i) +
                              ".bin");
      data_file.write(reinterpret_cast<char *>(&data_ptr_[i]->n_nz_),
                      sizeof(VertexID));
      data_file.write(reinterpret_cast<char *>(&data_ptr_[i]->tile_x_),
                      sizeof(VertexID));
      data_file.write(reinterpret_cast<char *>(&data_ptr_[i]->tile_y_),
                      sizeof(VertexID));
      data_file.write(reinterpret_cast<char *>(&data_ptr_[i]->tile_size_x_),
                      sizeof(TileIndex));
      data_file.write(reinterpret_cast<char *>(&data_ptr_[i]->tile_size_y_),
                      sizeof(TileIndex));
      data_file.write(reinterpret_cast<char *>(data_ptr_[i]->row_ptr_),
                      sizeof(TileIndex) * data_ptr_[i]->tile_size_y_);
      data_file.write(reinterpret_cast<char *>(data_ptr_[i]->row_idx_),
                      sizeof(TileIndex) * data_ptr_[i]->n_nz_);
      data_file.write(reinterpret_cast<char *>(data_ptr_[i]->col_idx_),
                      sizeof(TileIndex) * data_ptr_[i]->n_nz_);
      data_file.write(reinterpret_cast<char *>(data_ptr_[i]->data_),
                      sizeof(VertexLabel) * data_ptr_[i]->n_nz_);

      data_file.close();
    }

    // Write Metadata
    std::ofstream out_meta_file(root_path + "meta.yaml");
    YAML::Node out_node;
    out_node["TiledMatrix"]["n_rows"] = metadata_.n_rows;
    out_node["TiledMatrix"]["n_cols"] = metadata_.n_cols;
    out_node["TiledMatrix"]["n_nz_tile"] = metadata_.n_nz_tile;

    out_meta_file << out_node << std::endl;
    out_meta_file.close();
  }

  Tile *GetTilebyIdx(VertexID idx) const { return data_ptr_[idx]; }

  void MarkasTransposed() { is_transposed_ = true; }

  void *get_data_ptr() const { return data_ptr_; }

  Mask *get_mask() const { return mask_; }

  VertexID get_tile_ptr_by_id(VertexID idx) const { return tile_ptr_[idx]; }

  TiledMatrixMetadata get_metadata() const { return metadata_; };

  VertexID *get_tile_ptr_ptr() const { return tile_ptr_; }
  VertexID *get_tile_col_idx_ptr() const { return tile_col_idx_; }
  VertexID *get_tile_n_nz_ptr() const { return tile_n_nz_; }

private:
  bool is_transposed_ = false;

  TiledMatrixMetadata metadata_;

  Tile **data_ptr_;

  Mask *mask_;

  VertexID *tile_ptr_;
  VertexID *tile_col_idx_;
  VertexID *tile_n_nz_;
};

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // SICS_MATRIXGRAPH_CORE_DATA_STRUCTURES_TILED_MATRIX_H_
