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

struct SubGraphMetadata {
  using GraphID = sics::matrixgraph::core::common::GraphID;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;

  // Subgraph Metadata
  GraphID gid = 0;
  VertexID num_vertices;
  EdgeIndex num_incoming_edges;
  EdgeIndex num_outgoing_edges;
  VertexID max_vid;
  VertexID min_vid;
};

struct GraphMetadata {
  using GraphID = sics::matrixgraph::core::common::GraphID;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;

  VertexID num_vertices;
  EdgeIndex num_edges;
  VertexID max_vid;
  VertexID min_vid;
  VertexID count_border_vertices;
  GraphID num_subgraphs;

  std::vector<SubGraphMetadata> subgraphs;
};

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // MATRIXGRAPH_CORE_DATA_STRUCTURES_METADATA_H_
