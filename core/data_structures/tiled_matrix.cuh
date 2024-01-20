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

  Mask(TileIndex mask_size) { Init(mask_size); }

  void Init(TileIndex mask_size) {
    mask_size_ = mask_size;

    bm_ = new Bitmap(mask_size * mask_size);
  }

  void InitDevice(TileIndex mask_size) {
    mask_size_ = mask_size;

    uint64_t *init_val;
    cudaMalloc((void **)&init_val,
               sizeof(uint64_t) * (WORD_OFFSET(mask_size_ * mask_size_) + 1));
    bm_ = new Bitmap(mask_size_ * mask_size_, init_val);
  }

  void FreeDevice() { bm_->FreeDevice(); }

  TileIndex get_mask_size_x() const { return mask_size_; }
  TileIndex get_mask_size_y() const { return mask_size_; }

  bool GetBit(VertexID x, VertexID y) {
    return bm_->GetBit(x * mask_size_ + y);
  }

  void SetBit(VertexID x, VertexID y) { bm_->SetBit(x * mask_size_ + y); }

  void Show() {
    std::cout << "     MASK     " << (int)mask_size_ << "X" << (int)mask_size_
              << std::endl;
    for (size_t i = 0; i < mask_size_; i++) {
      for (size_t j = 0; j < mask_size_; j++) {
        std::cout << GetBit(i, j) << " ";
      }
      std::cout << std::endl;
    }
  }

  Bitmap *GetDataPtr() const { return bm_; }

  // mask_size_ and mask_size_y should satisfy that 64 % (mask_size_ *
  // mask_size_) == 0.
  TileIndex mask_size_ = 4;

  Bitmap *bm_ = nullptr;
};

class Tile {
private:
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using TileIndex = sics::matrixgraph::core::common::TileIndex;

public:
  Tile() = default;

  Tile(TileIndex tile_size, VertexID tile_x, VertexID tile_y, VertexID n_nz) {
    assert(tile_size == 4 || tile_size == 8);
    InitHost(tile_size, tile_x, tile_y, n_nz);
  }

  void InitAsOutput(const Tile &A, const Tile &A_t) {
    auto mask = GetOutputMask(*(A.GetMaskPtr()), *(A_t.GetMaskPtr()));

    InitHost(A.get_tile_size(), A.get_tile_x(), A_t.get_tile_y(),
             mask->GetDataPtr()->Count(), nullptr, nullptr, nullptr, nullptr,
             mask);
  }

  cudaError_t InitDevice(TileIndex tile_size, VertexID tile_x, VertexID tile_y,
                         VertexID n_nz) {
    assert(tile_size == 4 || tile_size == 8);
    tile_size_ = tile_size;
    tile_x_ = tile_x;
    tile_y_ = tile_y;
    n_nz_ = n_nz;

    if (bar_offset_ != nullptr)
      cudaFree(bar_offset_);
    if (mask_ != nullptr)
      cudaFree(mask_);

    if (row_idx_ != nullptr)
      cudaFree(row_idx_);
    if (col_idx_ != nullptr)
      cudaFree(col_idx_);
    if (data_ != nullptr)
      cudaFree(data_);

    cudaMalloc(reinterpret_cast<void **>(&bar_offset_),
               tile_size_ * sizeof(TileIndex));
    cudaMalloc(reinterpret_cast<void **>(&row_idx_), n_nz_ * sizeof(TileIndex));
    cudaMalloc(reinterpret_cast<void **>(&col_idx_), n_nz_ * sizeof(TileIndex));
    cudaMalloc(reinterpret_cast<void **>(&data_), n_nz_ * sizeof(VertexLabel));
    mask_ = new Mask();
    mask_->InitDevice(tile_size_);
    return cudaSuccess;
  }

  void InitHost(TileIndex tile_size, VertexID tile_x, VertexID tile_y,
                VertexID n_nz, TileIndex *bar_offset = nullptr,
                TileIndex *row_idx = nullptr, TileIndex *col_idx = nullptr,
                VertexLabel *data = nullptr, Mask *mask = nullptr) {
    tile_size_ = tile_size;
    tile_x_ = tile_x;
    tile_y_ = tile_y;
    n_nz_ = n_nz;

    if (bar_offset_ != nullptr)
      delete[] bar_offset_;
    if (mask_ != nullptr)
      delete mask_;
    if (row_idx_ != nullptr)
      delete[] row_idx_;
    if (col_idx_ != nullptr)
      delete[] col_idx_;
    if (data_ != nullptr)
      delete[] data_;

    if (bar_offset == nullptr)
      bar_offset_ = new TileIndex[tile_size_]();
    else
      bar_offset_ = bar_offset;
    if (mask == nullptr)
      mask_ = new Mask(tile_size_);
    else
      mask_ = mask;
    if (row_idx == nullptr)
      row_idx_ = new TileIndex[tile_size_]();
    else
      row_idx_ = row_idx;
    if (col_idx == nullptr)
      col_idx_ = new TileIndex[n_nz_]();
    else
      col_idx_ = col_idx;
    if (data_ == nullptr) {
      data_ = new VertexLabel[n_nz_]();
    } else
      data_ = data;
  }

  void FreeDevice() {
    cudaFree(bar_offset_);
    cudaFree(row_idx_);
    cudaFree(col_idx_);
    cudaFree(data_);
    mask_->FreeDevice();
  }

  cudaError_t MemcpyAsyncHost2Device(const Tile &tile,
                                     const cudaStream_t &stream) {
    assert(tile_x_ == tile.tile_x_);
    assert(tile_y_ == tile.tile_y_);
    assert(n_nz_ == tile.n_nz_);
    assert(tile_size_ == tile.tile_size_);

    cudaMemcpyAsync(bar_offset_, tile.bar_offset_,
                    sizeof(TileIndex) * (tile_size_), cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(row_idx_, tile.row_idx_, sizeof(TileIndex) * n_nz_,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(col_idx_, tile.col_idx_, sizeof(TileIndex) * n_nz_,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(data_, tile.data_, sizeof(VertexLabel) * n_nz_,
                    cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(mask_->GetDataPtr()->GetDataBasePointer(),
                    tile.mask_->GetDataPtr()->GetDataBasePointer(),
                    sizeof(uint64_t) *
                        (WORD_OFFSET(tile.mask_->GetDataPtr()->size()) + 1),
                    cudaMemcpyHostToDevice, stream);
    return cudaSuccess;
  }

  cudaError_t MemcpyAsyncDevice2Host(const Tile &tile,
                                     const cudaStream_t &stream) {
    assert(tile_x_ == tile.tile_x_);
    assert(tile_y_ == tile.tile_y_);
    assert(n_nz_ == tile.n_nz_);
    assert(tile_size_ == tile.tile_size_);

    cudaMemcpyAsync(bar_offset_, tile.bar_offset_,
                    sizeof(TileIndex) * (tile_size_), cudaMemcpyDeviceToHost,
                    stream);
    cudaMemcpyAsync(row_idx_, tile.row_idx_, sizeof(TileIndex) * n_nz_,
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(col_idx_, tile.col_idx_, sizeof(TileIndex) * n_nz_,
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(data_, tile.data_, sizeof(VertexLabel) * n_nz_,
                    cudaMemcpyDeviceToHost, stream);

    cudaMemcpyAsync(mask_->GetDataPtr()->GetDataBasePointer(),
                    tile.mask_->GetDataPtr()->GetDataBasePointer(),
                    sizeof(uint64_t) *
                        (WORD_OFFSET(tile.mask_->GetDataPtr()->size()) + 1),
                    cudaMemcpyDeviceToHost, stream);
    return cudaSuccess;
  }

  void Show(bool is_transposed = false) const {
    std::cout << "**********  Tile: (" << (int)tile_x_ << ", " << (int)tile_y_
              << ")"
              << " n_nz: " << n_nz_ << " ************" << std::endl;
    if (n_nz_ == 0) {
      std::cout << "   empty" << std::endl
                << "****************************" << std::endl;
      return;
    }
    std::cout << " bar_offset: ";
    for (int i = 0; i < tile_size_; i++) {
      std::cout << (int)bar_offset_[i] << " ";
    }
    std::cout << std::endl << " row_idx: ";
    for (VertexID i = 0; i < n_nz_; i++) {
      std::cout << (int)row_idx_[i] << " ";
    }
    std::cout << std::endl << " col_idx: ";
    for (VertexID i = 0; i < n_nz_; i++) {
      std::cout << (int)col_idx_[i] << " ";
    }
    std::cout << std::endl << " data: ";
    for (VertexID i = 0; i < n_nz_; i++) {
      std::cout << data_[i] << " ";
    }
    std::cout << std::endl << std::endl;
    mask_->Show();
    std::cout << "****************************" << std::endl;
  }

  void SetBit(VertexID x, VertexID y) { mask_->SetBit(x, y); }
  bool GetBit(VertexID x, VertexID y) { return mask_->GetBit(x, y); }

  void SetMaskPtr(Mask *ptr) { mask_ = ptr; }
  Mask *GetMaskPtr() const { return mask_; }

  void SetBarOffsetPtr(TileIndex *ptr) { bar_offset_ = ptr; }
  void SetRowIdxPtr(TileIndex *ptr) { row_idx_ = ptr; }
  void SetColIdxPtr(TileIndex *ptr) { col_idx_ = ptr; }
  TileIndex *GetBarOffsetPtr() const { return bar_offset_; }
  TileIndex *GetRowIdxPtr() const { return row_idx_; }
  TileIndex *GetColIdxPtr() const { return col_idx_; }
  VertexLabel *GetDataPtr() const { return data_; }

  void set_n_nz(VertexID val) { n_nz_ = val; }
  void set_tile_x(VertexID val) { tile_x_ = val; }
  void set_tile_y(VertexID val) { tile_y_ = val; }
  void set_tile_size(TileIndex val) { tile_size_ = val; }

  TileIndex get_n_nz() const { return n_nz_; }
  VertexID get_tile_x() const { return tile_x_; }
  VertexID get_tile_y() const { return tile_y_; }
  TileIndex get_tile_size() const { return tile_size_; }

private:
  // @description: take as input the two masks of two Matrix and return the mask
  // that corresponds to the output matrix.
  // @param: mask_a, the first mask, mask_b, the second mask.
  Mask *GetOutputMask(const Mask &mask_a, const Mask &mask_b) {

    assert(mask_a.mask_size_ == mask_b.mask_size_);
    Mask *output = new Mask(mask_a.mask_size_);

    for (TileIndex i = 0; i < mask_a.mask_size_; i++) {
      uint64_t intersection = 0;

      uint64_t scope_A = ~(0xffffffffffffffff << (mask_a.mask_size_ * (i + 1)) |
                           ~(0xffffffffffffffff << (mask_a.mask_size_ * i)));

      uint64_t a_f =
          (mask_a.GetDataPtr()->GetFragment(i * mask_a.mask_size_) & scope_A) >>
          (mask_a.mask_size_ * i);

      for (TileIndex j = 0; j < mask_b.mask_size_; j++) {
        uint64_t scope_B =
            ~(0xffffffffffffffff << (mask_b.mask_size_ * (j + 1)) |
              ~(0xffffffffffffffff << (mask_b.mask_size_ * j)));
        uint64_t b_f =
            (mask_b.GetDataPtr()->GetFragment(j * mask_b.mask_size_) &
             scope_B) >>
            (mask_b.mask_size_ * j);
        if ((a_f & b_f) != 0)
          intersection |= (1 << j);
      }
      *(output->GetDataPtr()->GetPFragment(i * mask_a.mask_size_)) |=
          intersection << (mask_a.mask_size_ * i);
    }
    return output;
  };

  VertexID n_nz_ = 0;

  VertexID tile_x_;
  VertexID tile_y_;
  // store the index of tile (x,y)

  TileIndex tile_size_;
  // store the size of tile, tile_size_x times tile_size_

  TileIndex *bar_offset_ = nullptr;
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

      TileIndex tile_size;
      VertexID n_nz, tile_x, tile_y;

      data_file.read(reinterpret_cast<char *>(&n_nz), sizeof(VertexID));
      data_file.read(reinterpret_cast<char *>(&tile_x), sizeof(VertexID));
      data_file.read(reinterpret_cast<char *>(&tile_y), sizeof(VertexID));
      data_file.read(reinterpret_cast<char *>(&tile_size), sizeof(TileIndex));

      data_ptr_[i] = new Tile(tile_size, tile_x, tile_y, n_nz);

      data_file.read(reinterpret_cast<char *>(data_ptr_[i]->GetBarOffsetPtr()),
                     sizeof(TileIndex) * data_ptr_[i]->get_tile_size());
      data_file.read(reinterpret_cast<char *>(data_ptr_[i]->GetRowIdxPtr()),
                     sizeof(TileIndex) * data_ptr_[i]->get_n_nz());
      data_file.read(reinterpret_cast<char *>(data_ptr_[i]->GetColIdxPtr()),
                     sizeof(TileIndex) * data_ptr_[i]->get_n_nz());
      data_file.read(reinterpret_cast<char *>(data_ptr_[i]->GetDataPtr()),
                     sizeof(VertexLabel) * data_ptr_[i]->get_n_nz());
      data_file.close();

      std::ifstream mask_file(root_path + "mask/" + std::to_string(i) + ".bin");

      mask_file.read(
          reinterpret_cast<char *>(
              data_ptr_[i]->GetMaskPtr()->GetDataPtr()->GetDataBasePointer()),
          sizeof(uint64_t) *
              (data_ptr_[i]->GetMaskPtr()->GetDataPtr()->GetMaxWordOffset() +
               1));
      mask_file.close();
    }
  }

  void Write(const std::string &root_path) {

    // Init path
    if (!std::filesystem::exists(root_path))
      std::filesystem::create_directory(root_path);

    if (!std::filesystem::exists(root_path + "tiles"))
      std::filesystem::create_directory(root_path + "tiles");

    if (!std::filesystem::exists(root_path + "mask"))
      std::filesystem::create_directory(root_path + "mask");

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

      // Init Data
      auto tile_data = data_ptr_[i]->GetDataPtr();
      for (int j = 0; j < data_ptr_[i]->get_n_nz(); j++) {
        tile_data[j] |= 1;
      }

      VertexID n_nz = data_ptr_[i]->get_n_nz();
      VertexID tile_x = data_ptr_[i]->get_tile_x();
      VertexID tile_y = data_ptr_[i]->get_tile_y();
      data_file.write(reinterpret_cast<char *>(&n_nz), sizeof(VertexID));
      data_file.write(reinterpret_cast<char *>(&tile_x), sizeof(VertexID));
      data_file.write(reinterpret_cast<char *>(&tile_y), sizeof(VertexID));
      TileIndex tile_size = data_ptr_[i]->get_tile_size();
      data_file.write(reinterpret_cast<char *>(&tile_size), sizeof(TileIndex));

      data_file.write(reinterpret_cast<char *>(data_ptr_[i]->GetBarOffsetPtr()),
                      sizeof(TileIndex) * data_ptr_[i]->get_tile_size());
      data_file.write(reinterpret_cast<char *>(data_ptr_[i]->GetRowIdxPtr()),
                      sizeof(TileIndex) * data_ptr_[i]->get_n_nz());
      data_file.write(reinterpret_cast<char *>(data_ptr_[i]->GetColIdxPtr()),
                      sizeof(TileIndex) * data_ptr_[i]->get_n_nz());

      data_file.write(reinterpret_cast<char *>(data_ptr_[i]->GetDataPtr()),
                      sizeof(VertexLabel) * data_ptr_[i]->get_n_nz());

      // Write mask.
      std::ofstream mask_file(root_path + "mask/" + std::to_string(i) + ".bin");
      mask_file.write(
          reinterpret_cast<char *>(
              data_ptr_[i]->GetMaskPtr()->GetDataPtr()->GetDataBasePointer()),
          sizeof(uint64_t) *
              (data_ptr_[i]->GetMaskPtr()->GetDataPtr()->GetMaxWordOffset() +
               1));

      data_file.close();
      mask_file.close();
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
