#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_GRID_CSR_TILED_MATRIX_CUH_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_GRID_CSR_TILED_MATRIX_CUH_

#include <vector>

#include "core/common/types.h"
#include "core/data_structures/csr_tiled_matrix.cuh"
#include "core/data_structures/metadata.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

class GridCSRTiledMatrix {
private:
  using GridGraphMetadata =
      sics::matrixgraph::core::data_structures::GridGraphMetadata;
  using GraphID = sics::matrixgraph::core::common::GraphID;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;

public:
  GridCSRTiledMatrix(const GridGraphMetadata &metadata);

  void SetMetadata(const GridGraphMetadata &metadata) { metadata_ = metadata; }

  GridGraphMetadata get_metadata() const { return metadata_; }

  CSRTiledMatrix *GetTiledMatrixPtrByIdx(GraphID idx) const {
    return tiled_matrix_vec_[idx];
  }

  void Print(VertexID max_n_blocks = 2) const;

  void SetVLabelBasePointer(VertexLabel *buf) {
    vertex_label_base_pointer_ = buf;
  }

  VertexLabel *GetVLabelBasePointer() const {
    return vertex_label_base_pointer_;
  }

private:
  GridGraphMetadata metadata_;

  std::vector<CSRTiledMatrix *> tiled_matrix_vec_;

  VertexLabel *vertex_label_base_pointer_ = nullptr;
};

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_DATA_STRUCTURES_GRID_CSR_TILED_MATRIX_CUH_