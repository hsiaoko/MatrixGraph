#include "core/data_structures/grid_tiled_matrix.cuh"

#include <cmath>

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

using GraphID = sics::matrixgraph::core::common::GraphID;
using VertexID = sics::matrixgraph::core::common::VertexID;

GridTiledMatrix::GridTiledMatrix(const GridGraphMetadata &metadata) {
  metadata_ = metadata;
  tiled_matrix_vec_.reserve(pow(metadata_.n_chunks, 2));
  for (VertexID _ = 0; _ < pow(metadata_.n_chunks, 2); _++) {
    tiled_matrix_vec_.push_back(new BitTiledMatrix());
  }
}

void GridTiledMatrix::Print() const {
  for (GraphID x = 0; x < metadata_.n_chunks; x++) {
    for (GraphID y = 0; y < metadata_.n_chunks; y++) {
      if (tiled_matrix_vec_[x * metadata_.n_chunks + y]
              ->GetMetadata()
              .n_nz_tile == 0) {
        continue;
      }
      std::cout << "Block (" << x << "," << y << ")" << std::endl;
      tiled_matrix_vec_[x * metadata_.n_chunks + y]->Print();
    }
  }
}

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics