#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_METADATA_H_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_METADATA_H_

#include "core/common/types.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

struct TiledMatrixMetadata {
private:
  using VertexID = sics::matrixgraph::core::common::VertexID;

public:
  VertexID n_cols;
  VertexID n_rows;
  VertexID n_nz_tile;
};

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // MATRIXGRAPH_CORE_DATA_STRUCTURES_METADATA_H_
