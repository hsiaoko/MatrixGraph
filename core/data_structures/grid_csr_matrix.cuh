#ifndef GRAPH_COMPUTING_MATRIXGRAPH_CORE_DATA_STRUCTURES_GRID_CSR_CUH_
#define GRAPH_COMPUTING_MATRIXGRAPH_CORE_DATA_STRUCTURES_GRID_CSR_CUH_

#include <vector>

#include "core/common/types.h"
#include "core/data_structures/csr_tiled_matrix.cuh"
#include "core/data_structures/metadata.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

class GridCSRMatrix {
private:
  using GridGraphMetadata =
      sics::matrixgraph::core::data_structures::GridGraphMetadata;
  using GraphID = sics::matrixgraph::core::common::GraphID;
  using VertexID = sics::matrixgraph::core::common::VertexID;

public:
  GridCSRMatrix(const GridGraphMetadata &metadata);

  void SetMetadata(const GridGraphMetadata &metadata) { metadata_ = metadata; }

  GridGraphMetadata get_metadata() const { return metadata_; }

  CSRTiledMatrix *GetBitTileMatrixPtrByIdx(GraphID idx) const {
    return tiled_matrix_vec_[idx];
  }

  void Print() const;

private:
  GridGraphMetadata metadata_;

  std::vector<CSRTiledMatrix *> tiled_matrix_vec_;
};

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // GRAPH_COMPUTING_MATRIXGRAPH_CORE_DATA_STRUCTURES_GRID_CSR_CUH_