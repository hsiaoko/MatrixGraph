#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_BIT_TILE_MATRIX_CUH_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_BIT_TILE_MATRIX_CUH_

#include <stdint.h>

#include "core/common/types.h"
#include "core/data_structures/bit_tile.cuh"
#include "core/data_structures/metadata.h"
#include "core/util/bitmap.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

class BitTiledMatrix {
  using TiledMatrixMetadata =
      sics::matrixgraph::core::data_structures::TiledMatrixMetadata;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using Bitmap = sics::matrixgraph::core::util::Bitmap;

public:
  BitTiledMatrix() = default;

  ~BitTiledMatrix();

  void Init(const TiledMatrixMetadata &metadata, Bitmap *nz_tile_bm = nullptr);

  void Init(VertexID tile_size, VertexID n_strips, Bitmap *bm);

  void SetNzBitmapPtr(Bitmap *nz_bitmap_ptr) { nz_tile_bm_ = nz_bitmap_ptr; }

  bool IsNzTile(size_t x, size_t y) const {
    return nz_tile_bm_->GetBit(x * metadata_.n_strips + y);
  }

  void InitBitTileVec();

  void Print() const;

  TiledMatrixMetadata GetMetadata() const { return metadata_; }

  VertexID *GetTileOffsetRowPtr() const { return tile_offset_row_; }

  VertexID *GetTileRowIdxPtr() const { return tile_row_idx_; }

  VertexID *GetTileColIdxPtr() const { return tile_col_idx_; }

  Bitmap *GetNzTileBitmapPtr() const { return nz_tile_bm_; }

  uint64_t *GetDataPtrByIdx(uint32_t idx) const;

  uint64_t *GetDataPtr() const { return data_; }

  uint64_t GetDataBufferSize() const;

private:
  TiledMatrixMetadata metadata_;

  Bitmap *nz_tile_bm_ = nullptr;

  uint64_t *data_ = nullptr;

  VertexID *tile_offset_row_ = nullptr;
  VertexID *tile_row_idx_ = nullptr;
  VertexID *tile_col_idx_ = nullptr;
};

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_DATA_STRUCTURES_BIT_TILE_MATRIX_CUH_