#include "core/data_structures/bit_tile.cuh"
#include <iostream>

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

BitTile::BitTile(uint32_t tile_size) {
  Init(tile_size);
}

BitTile::~BitTile() {
  delete bm_;
  delete bm_t_;
}

void BitTile::Init(uint32_t tile_size) {
  if (bm_ != nullptr)
    delete bm_;
  if (bm_t_ != nullptr)
    delete bm_t_;

  bm_ = new util::Bitmap(tile_size * tile_size);
  bm_t_ = new util::Bitmap(tile_size * tile_size);
}

void BitTile::Print() const {
  std::cout << "BitTile Print " << std::endl;
  for (uint32_t _ = 0; _ < tile_size(); _++) {
    for (uint32_t __ = 0; __ < tile_size(); __++) {
      std::cout << bm_->GetBit(_ * tile_size() + __) << " ";
    }
    std::cout << std::endl;
  }

  std::cout << " convert: " << std::endl;
  for (uint32_t _ = 0; _ < tile_size(); _++) {
    for (uint32_t __ = 0; __ < tile_size(); __++) {
      std::cout << bm_t_->GetBit(_ * tile_size() + __) << " ";
    }
    std::cout << std::endl;
  }
}

uint32_t BitTile::tile_size() const { return std::sqrt(bm_->size()); }

void BitTile::SetBit(uint32_t x, uint32_t y) {
  bm_->SetBit(x * tile_size() + y);
  bm_t_->SetBit(y * tile_size() + x);
}

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics