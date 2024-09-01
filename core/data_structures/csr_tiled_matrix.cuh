#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_CSR_TILED_MATRIX_CUH_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_CSR_TILED_MATRIX_CUH_

#include <memory>
#include <stdint.h>

#include "core/common/types.h"
#include "core/data_structures/metadata.h"
#include "core/util/bitmap.h"
#include "core/util/bitmap_no_ownership.h"
#include "core/util/gpu_bitmap.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

class CSRTiledMatrix {
  using TiledMatrixMetadata =
      sics::matrixgraph::core::data_structures::TiledMatrixMetadata;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using Bitmap = sics::matrixgraph::core::util::Bitmap;
  using GPUBitmap = sics::matrixgraph::core::util::GPUBitmap;
  using BitmapNoOwnerShip = sics::matrixgraph::core::util::BitmapNoOwnerShip;

public:
  CSRTiledMatrix() = default;

  ~CSRTiledMatrix();

  void Init(const TiledMatrixMetadata &metadata,
            GPUBitmap *nz_tile_bm = nullptr);

  void Init(VertexID tile_size, VertexID n_strips, Bitmap *bm);

  void SetNzBitmapPtr(GPUBitmap *nz_bitmap_ptr) { nz_tile_bm_ = nz_bitmap_ptr; }

  void SetData(uint8_t *buf) { data_ = buf; }

  void InitData(uint64_t size);

  void SetCSRMetadata(const SubGraphMetadata &sub_graph_metadata, VertexID i) {
    metadata_vec_[i] = sub_graph_metadata;
  }

  bool IsNzTile(size_t x, size_t y) const {
    return nz_tile_bm_->GetBit(x * metadata_.n_strips + y);
  }

  void InitBitTileVec();

  void Print() const;

  void SetBit(VertexID row, VertexID col);

  TiledMatrixMetadata GetMetadata() const { return metadata_; }

  SubGraphMetadata GetCSRMetadataByIdx(VertexID i) const {
    return metadata_vec_[i];
  }

  VertexID *GetTileOffsetRowPtr() const { return tile_offset_row_; }

  VertexID *GetTileRowIdxPtr() const { return tile_row_idx_; }

  VertexID *GetTileColIdxPtr() const { return tile_col_idx_; }

  GPUBitmap *GetNzTileBitmapPtr() const { return nz_tile_bm_; }

  uint64_t *GetDataPtrByIdx(uint32_t idx) const;

  uint64_t *GetCSROffsetPtr() const { return csr_offset_; }

  uint8_t *GetCSRBasePtrByIdx(uint32_t idx) const;

  uint8_t *GetDataPtr() const { return data_; }

  uint64_t GetDataBufferSize() const;

  void InitMetadataVec(VertexID n_csr) { metadata_vec_.resize(n_csr); }

private:
  TiledMatrixMetadata metadata_;

  std::vector<SubGraphMetadata> metadata_vec_;

  GPUBitmap *nz_tile_bm_ = nullptr;

  uint8_t *data_ = nullptr;
  uint64_t *csr_offset_ = nullptr;

  VertexID *tile_offset_row_ = nullptr;
  VertexID *tile_row_idx_ = nullptr;
  VertexID *tile_col_idx_ = nullptr;
};

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_DATA_STRUCTURES_CSR_TILED_MATRIX_CUH_