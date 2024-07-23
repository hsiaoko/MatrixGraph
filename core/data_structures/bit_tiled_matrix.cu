#include "core/data_structures/bit_tiled_matrix.cuh"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <numeric>
#include <thread>

#ifdef TBB_FOUND
#include <execution>
#endif

#include "core/util/bitmap.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

using VertexID = sics::matrixgraph::core::common::VertexID;

BitTiledMatrix::~BitTiledMatrix() {
  delete[] tile_row_idx_;
  delete[] tile_col_idx_;
  delete[] tile_offset_row_;
  for (auto &bit_tile : bit_tile_vec_) {
    delete bit_tile;
  }
}

void BitTiledMatrix::Init(const TiledMatrixMetadata &metadata) {
  assert(metadata.n_nz_tile != 0);

  metadata_.n_strips = metadata.n_strips;
  metadata_.n_nz_tile = metadata.n_nz_tile;
  metadata_.tile_size = metadata.tile_size;

  bit_tile_vec_.reserve(metadata.n_nz_tile);

  for(VertexID _ = 0; _ < metadata.n_nz_tile; _++){
    bit_tile_vec_.push_back(new BitTile());
  }

  if (tile_row_idx_ != nullptr)
    delete[] tile_row_idx_;
  if (tile_col_idx_ != nullptr)
    delete[] tile_col_idx_;
  if (tile_offset_row_ != nullptr)
    delete[] tile_offset_row_;
  if (nz_tile_bm_ != nullptr)
    delete nz_tile_bm_;

  tile_row_idx_ = new VertexID[metadata.n_nz_tile]();
  tile_col_idx_ = new VertexID[metadata.n_nz_tile]();
  tile_offset_row_ = new VertexID[metadata.n_strips + 1]();
  nz_tile_bm_ = new Bitmap(pow(metadata.n_strips, 2));
}

void BitTiledMatrix::Init(size_t n_nz_tile, size_t tile_size, size_t n_strips,
                          Bitmap *nz_tile_bm) {

  metadata_.n_strips = n_strips;
  metadata_.n_nz_tile = n_nz_tile;
  metadata_.tile_size = tile_size;

  bit_tile_vec_.resize(n_nz_tile);

  for (size_t _ = 0; _ < n_nz_tile; _++)
    bit_tile_vec_[_] = new BitTile(tile_size);

  tile_row_idx_ = new VertexID[n_nz_tile]();
  tile_col_idx_ = new VertexID[n_nz_tile]();
  tile_offset_row_ = new VertexID[n_strips + 1]();
  nz_tile_bm_ = nz_tile_bm;
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

  std::cout << " * offset to idx: " << std::endl;
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

  for (VertexID _ = 0; _ < metadata_.n_nz_tile; _++) {
    GetTileByIdx(_)->Print();
  }
}

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics