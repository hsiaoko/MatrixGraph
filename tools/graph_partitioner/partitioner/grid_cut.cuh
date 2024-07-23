#ifndef MATRIXGRAPH_TOOLS_GRAPH_PARTITIONER_PARTITIONER_GRID_CUT_H_
#define MATRIXGRAPH_TOOLS_GRAPH_PARTITIONER_PARTITIONER_GRID_CUT_H_

#include "core/common/types.h"
#include "tools/graph_partitioner/partitioner/partitioner_base.h"

#include "core/common/types.h"

namespace sics {
namespace matrixgraph {
namespace tools {
namespace partitioner {

class GridCutPartitioner : public PartitionerBase {
private:
  using StoreStrategy = sics::matrixgraph::tools::common::StoreStrategy;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
  using GraphID = sics::matrixgraph::core::common::GraphID;

public:
  GridCutPartitioner(const std::string &input_path,
                     const std::string &output_path,
                     StoreStrategy store_strategy, GraphID n_partitions)
      : PartitionerBase(input_path, output_path, store_strategy),
        n_partitions_(n_partitions) {}

  void RunPartitioner() override;

private:
  const GraphID n_partitions_;

  VertexID GetBucketID(VertexID vid, VertexID n_bucket,
                       size_t n_vertices) const;
};

} // namespace partitioner
} // namespace tools
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_TOOLS_GRAPH_PARTITIONER_PARTITIONER_GRID_CUT_H_