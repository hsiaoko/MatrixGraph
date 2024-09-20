#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_GRID_TILED_MATRIX_CUH_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_GRID_TILED_MATRIX_CUH_

#include <vector>

#include "core/common/types.h"
#include "core/data_structures/bit_tiled_matrix.cuh"
#include "core/data_structures/metadata.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

class GridBitTiledMatrix {
private:
  using GridGraphMetadata =
      sics::matrixgraph::core::data_structures::GridGraphMetadata;
  using GraphID = sics::matrixgraph::core::common::GraphID;
  using VertexID = sics::matrixgraph::core::common::VertexID;

public:
  GridBitTiledMatrix(const GridGraphMetadata &metadata);

  void SetMetadata(const GridGraphMetadata &metadata) { metadata_ = metadata; }

  GridGraphMetadata get_metadata() const { return metadata_; }

  BitTiledMatrix *GetBitTileMatrixPtrByIdx(GraphID idx) const {
    return tiled_matrix_vec_[idx];
  }

  void Print() const;

private:
  GridGraphMetadata metadata_;

  std::vector<BitTiledMatrix *> tiled_matrix_vec_;
};

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_DATA_STRUCTURES_GRID_TILED_MATRIX_CUH_