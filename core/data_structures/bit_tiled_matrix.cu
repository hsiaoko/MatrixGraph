#include "core/data_structures/bit_tiled_matrix.cuh"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <numeric>
#include <thread>
#include <execution>

#include "core/util/bitmap.h"
#include "core/util/bitmap_no_ownership.h"
#include "core/util/cuda_check.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

using VertexID = sics::matrixgraph::core::common::VertexID;
using BitmapNoOwnerShip = sics::matrixgraph::core::util::BitmapNoOwnerShip;

#define WORD_OFFSET(i) (i >> 6)
#define BIT_OFFSET(i) (i & 0x3f)

BitTiledMatrix::~BitTiledMatrix() {
  delete[] tile_row_idx_;
  delete[] tile_col_idx_;
  delete[] tile_offset_row_;
  delete[] data_;
}

void BitTiledMatrix::Init(const TiledMatrixMetadata &metadata,
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

  tile_row_idx_ = new VertexID[metadata_.n_nz_tile]();
  tile_col_idx_ = new VertexID[metadata_.n_nz_tile]();
  tile_offset_row_ = new VertexID[metadata_.n_strips + 1]();

  auto tile_buffer_size =
      sizeof(uint64_t) *
      std::max(1u, WORD_OFFSET(metadata_.tile_size * metadata_.tile_size));

  CUDA_CHECK(cudaHostAlloc((void **)&data_,
                           tile_buffer_size * metadata_.n_nz_tile,
                           cudaHostAllocDefault));

  if (nz_tile_bm == nullptr) {
    uint64_t *bm_data;
    size_t bm_size = pow(metadata_.n_strips, 2);
    CUDA_CHECK(cudaHostAlloc((void **)&bm_data,
                             sizeof(uint64_t) * (WORD_OFFSET(bm_size) + 1),
                             cudaHostAllocDefault));
    nz_tile_bm_ = new GPUBitmap(bm_size, bm_data);
  } else {
    nz_tile_bm_ = nz_tile_bm;
  }
}

void BitTiledMatrix::Print() const {
  if (metadata_.n_nz_tile == 0)
    return;
  std::cout << "[BitTiledMatrix Print]" << std::endl;
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

  for (VertexID _ = 0; _ < metadata_.n_nz_tile; _++) {
    std::cout << " * " << _ << " " << std::endl;
    auto *data = GetDataPtrByIdx(_);
    for (VertexID i = 0; i < metadata_.tile_size; i++) {
      for (VertexID j = 0; j < metadata_.tile_size; j++) {
        std::cout << (bool)(data[WORD_OFFSET(i * metadata_.tile_size + j)] &
                            (1ul << BIT_OFFSET(i * metadata_.tile_size + j)))
                  << " ";
      }
      std::cout << std::endl;
    }
  }
  std::cout << "[Print] Done!" << std::endl;
}

uint64_t *BitTiledMatrix::GetDataPtrByIdx(uint32_t idx) const {
  auto tile_unit =
      std::max(1u, (WORD_OFFSET(metadata_.tile_size * metadata_.tile_size)));
  return data_ + tile_unit * idx;
}

uint64_t BitTiledMatrix::GetDataBufferSize() const {
  return sizeof(uint64_t) *
         std::max(1u, WORD_OFFSET(metadata_.tile_size * metadata_.tile_size)) *
         metadata_.n_nz_tile;
}

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics