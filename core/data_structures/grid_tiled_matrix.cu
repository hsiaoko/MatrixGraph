#include "core/data_structures/grid_tiled_matrix.cuh"

#include <algorithm>
#include <cmath>

#ifdef TBB_FOUND
#include <execution>
#endif

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

using GraphID = sics::matrixgraph::core::common::GraphID;
using VertexID = sics::matrixgraph::core::common::VertexID;

GridTiledMatrix::GridTiledMatrix(const GridGraphMetadata &metadata) {
  metadata_ = metadata;
  tiled_matrix_vec_.resize(pow(metadata_.n_chunks, 2));

  std::generate(std::execution::par, tiled_matrix_vec_.begin(),
                tiled_matrix_vec_.end(), []() { return new BitTiledMatrix(); });
}

void GridTiledMatrix::Print() const {
  std::cout << "[GridTiledMatrix Print] " << metadata_.n_chunks << "x"
            << metadata_.n_chunks << std::endl;
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