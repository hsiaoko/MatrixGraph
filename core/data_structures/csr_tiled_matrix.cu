#include "core/data_structures/csr_tiled_matrix.cuh"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <execution>
#include <mutex>
#include <numeric>
#include <thread>

#include "core/util/bitmap.h"
#include "core/util/bitmap_no_ownership.h"
#include "core/util/cuda_check.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

void CSRTiledMatrix::Print() const {
  if (metadata_.n_nz_tile == 0)
    return;
  std::cout << "[CSRTiledMatrix Print]" << std::endl;
  std::cout << " * tile_offset_row: ";

  for (VertexID _ = 0; _ < metadata_.n_strips + 1; _++) {
    std::cout << " " << tile_offset_row_[_];
  }
  std::cout << std::endl;

  for (VertexID _ = 0; _ < metadata_.n_nz_tile; _++) {
    std::cout << "offset: " << _ << ", x: " << tile_row_idx_[_]
              << " y: " << tile_col_idx_[_] << std::endl;
  }

  for (VertexID _ = 0; _ < metadata_.n_strips; _++) {
    for (VertexID __ = 0; __ < metadata_.n_strips; __++) {
      std::cout << GetNzTileBitmapPtr()->GetBit(_ * metadata_.n_strips + __)
                << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void CSRTiledMatrix::Init(const TiledMatrixMetadata &metadata,
                          GPUBitmap *nz_tile_bm) {

  metadata_.n_strips = metadata.n_strips;
  metadata_.n_nz_tile = metadata.n_nz_tile;
  metadata_.tile_size = metadata.tile_size;

  if (tile_row_idx_ != nullptr)
    delete[] tile_row_idx_;
  if (tile_col_idx_ != nullptr)
    delete[] tile_col_idx_;
  if (tile_offset_row_ != nullptr)
    delete[] tile_offset_row_;
  if (nz_tile_bm_ != nullptr)
    delete nz_tile_bm_;
  if (csr_offset_ != nullptr)
    delete[] csr_offset_;

  tile_row_idx_ = new VertexID[metadata_.n_nz_tile]();
  tile_col_idx_ = new VertexID[metadata_.n_nz_tile]();
  tile_offset_row_ = new VertexID[metadata_.n_strips + 1]();
  csr_offset_ = new uint64_t[metadata_.n_nz_tile + 1]();

  metadata_vec_.resize(metadata_.n_nz_tile);
  nz_tile_bm_ = nz_tile_bm;
}

void CSRTiledMatrix::Init(VertexID tile_size, VertexID n_strips, Bitmap *bm) {}

uint8_t *CSRTiledMatrix::GetCSRBasePtrByIdx(uint32_t idx) const {
  assert(idx < metadata_.n_nz_tile);
  return data_ + csr_offset_[idx];
}

void CSRTiledMatrix::InitData(uint64_t size) {
  CUDA_CHECK(cudaHostAlloc((void **)&data_, size, cudaHostAllocDefault));
}

void CSRTiledMatrix::SetBit(VertexID row, VertexID col) {
  nz_tile_bm_->SetBit(row * metadata_.n_strips + col);
}

uint64_t CSRTiledMatrix::GetDataBufferSize() const {
  return csr_offset_[metadata_.n_nz_tile];
}

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics