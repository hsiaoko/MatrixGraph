#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_BIT_MATRIX_CUH_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_BIT_MATRIX_CUH_

#include "core/common/types.h"
#include "core/util/bitmap.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

class BitTile {
  using VertexID = sics::matrixgraph::core::common::VertexID;

public:
  BitTile() = default;

  BitTile(uint32_t tile_size);

  ~BitTile();

  void Init(uint32_t tile_size);

  void SetBit(VertexID x, VertexID y);

  bool GetBit(VertexID x, VertexID y) const;

  void Transpose();

  void Print() const;

  util::Bitmap *GetBM() const { return bm_; }

  util::Bitmap *GetBMT() const { return bm_t_; }

  uint32_t tile_size() const;

private:
  uint32_t tile_id = 0;

  util::Bitmap *bm_ = nullptr;

  util::Bitmap *bm_t_ = nullptr;
};

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_DATA_STRUCTURES_BIT_MATRIX_CUH_