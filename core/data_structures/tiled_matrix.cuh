#ifndef SICS_MATRIXGRAPH_CORE_DATA_STRUCTURES_TILED_MATRIX_H_
#define SICS_MATRIXGRAPH_CORE_DATA_STRUCTURES_TILED_MATRIX_H_

#include <cuda_runtime.h>
#include <stdint.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "core/data_structures/immutable_csr.cuh"
#include "core/util/bitmap.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

static uint8_t kDefaultTileSize = 64;

enum TiledMatrixType {
  kTransposed,
  kOrigin // default
};

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

    if (bm_ != nullptr)
      delete bm_;
    bm_ = new Bitmap(mask_size * mask_size);
  }

  void InitDevice(TileIndex mask_size);

  void FreeDevice();

  TileIndex get_mask_size_x() const { return mask_size_; }
  TileIndex get_mask_size_y() const { return mask_size_; }

  bool GetBit(VertexID x, VertexID y) const {
    return bm_->GetBit(x * mask_size_ + y);
  }

  void SetBit(VertexID x, VertexID y) { bm_->SetBit(x * mask_size_ + y); }

  void Show() const;

  Bitmap *GetDataPtr() const { return bm_; }

  // mask_size_ and mask_size_y should satisfy that 64 % (mask_size_ *
  // mask_size_) == 0.
  TileIndex mask_size_ = kDefaultTileSize;

  Bitmap *bm_ = nullptr;
};

class Tile {
private:
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using TileIndex = sics::matrixgraph::core::common::TileIndex;
  using Bitmap = sics::matrixgraph::core::util::Bitmap;

public:
  Tile() = default;

  Tile(TileIndex tile_size, VertexID tile_x, VertexID tile_y, VertexID n_nz) {
    InitHost(tile_size, tile_x, tile_y, n_nz);
  }

  ~Tile() {
    delete[] bar_offset_;
    delete[] row_idx_;
    delete[] col_idx_;
    delete[] data_;
    delete mask_;
  }

  void InitAsOutput(const Tile &A, const Tile &A_t) {
    auto mask = GetOutputMask(*(A.GetMaskPtr()), *(A_t.GetMaskPtr()));

    InitHost(A.get_tile_size(), A.get_tile_x(), A_t.get_tile_x(),
             mask->GetDataPtr()->Count(), nullptr, nullptr, nullptr, nullptr,
             mask);
  }

  void InitDevice(TileIndex tile_size, VertexID tile_x, VertexID tile_y,
                  VertexID n_nz);

  void InitHost(TileIndex tile_size, VertexID tile_x, VertexID tile_y,
                VertexID n_nz, TileIndex *bar_offset = nullptr,
                TileIndex *row_idx = nullptr, TileIndex *col_idx = nullptr,
                VertexLabel *data = nullptr, Mask *mask = nullptr);

  void FreeDevice();

  void MemcpyAsyncHost2Device(const Tile &tile, const cudaStream_t &stream);

  void MemcpyAsyncDevice2Host(const Tile &tile, const cudaStream_t &stream);

  void Show(bool is_transposed = false) const;

  void SetBit(VertexID x, VertexID y) { mask_->SetBit(x, y); }
  bool GetBit(VertexID x, VertexID y) { return mask_->GetBit(x, y); }

  void SetMaskPtr(Mask *ptr) { mask_ = ptr; }
  Mask *GetMaskPtr() const { return mask_; }
  Bitmap *GetMaskPtr2() const { return mask_ptr_; }

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
  // @Description: take as input the two masks of two Matrix and return the mask
  // that corresponds to the output matrix.
  // @Param: mask_a, the first mask, mask_b, the second mask.
  Mask *GetOutputMask(Mask &mask_a, Mask &mask_b);

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

  Bitmap *mask_ptr_ = nullptr;
};

class TiledMatrix {
private:
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;
  using TiledMatrixMetadata =
      sics::matrixgraph::core::data_structures::TiledMatrixMetadata;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using TileIndex = sics::matrixgraph::core::common::TileIndex;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;

public:
  void Init(const ImmutableCSR &immutable_csr, size_t tile_size);

  void InitTranspose(const ImmutableCSR &immutable_csr, size_t tile_size);

  void Init(VertexID n_rows, VertexID n_cols, VertexID n_nz_tile,
            Tile **data_ptr, Mask *mask, VertexID *tile_ptr,
            VertexID *tile_row_idx, VertexID *tile_col_idx,
            VertexID *tile_n_nz);

  void Write(const std::string &root_path);

public:
  TiledMatrix() = default;

  TiledMatrix(const ImmutableCSR &immutable_csr,
              size_t tile_size = kDefaultTileSize,
              TiledMatrixType type = kOrigin) {
    switch (type) {
    case kOrigin:
      Init(immutable_csr, tile_size);
      break;
    case kTransposed:
      InitTranspose(immutable_csr, tile_size);
      break;
    default:
      return;
    }
  }

  static VertexID XYToTileId(VertexID x, VertexID y, VertexID range) {
    return x * range + y;
  }

  static std::pair<VertexID, VertexID> TileIdToXY(VertexID tile_id, VertexID range,
                                           VertexID *x, VertexID *y) {
    *x = tile_id / range;
    *y = tile_id % range;
  }

  void ShowAbs() const;

  void Show() const;

  void Read(const std::string &root_path);

  Tile *GetTilebyIdx(VertexID idx) const { return data_ptr_[idx]; }

  void *get_data_ptr() const { return data_ptr_; }

  Mask *get_mask() const { return mask_; }

  VertexID get_tile_ptr_by_id(VertexID idx) const { return tile_ptr_[idx]; }

  TiledMatrixMetadata get_metadata() const { return metadata_; };

  VertexID *get_tile_ptr_ptr() const { return tile_ptr_; }

  VertexID *get_tile_col_idx_ptr() const { return tile_col_idx_; }

  VertexID *get_tile_col_row_ptr() const { return tile_col_idx_; }

  VertexID *get_tile_n_nz_ptr() const { return tile_n_nz_; }

private:
  TiledMatrixMetadata metadata_;

  Tile **data_ptr_;

  Mask *mask_;

  VertexID *tile_ptr_;
  VertexID *tile_row_idx_;
  VertexID *tile_col_idx_;
  VertexID *tile_n_nz_;
};

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // SICS_MATRIXGRAPH_CORE_DATA_STRUCTURES_TILED_MATRIX_H_
