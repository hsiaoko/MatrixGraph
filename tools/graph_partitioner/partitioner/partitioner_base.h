#ifndef SICS_GRAPH_SYSTEMS_TOOLS_PARTITIONER_H_
#define SICS_GRAPH_SYSTEMS_TOOLS_PARTITIONER_H_

#include "core/common/types.h"

#include <string>

namespace sics {
namespace matrixgraph {
namespace tools {
namespace partitioner {

// This class defines the Partitioner interface, which is used to partition a
// graph into several subgraphs.
class PartitionerBase {
private:
  using StoreStrategy = sics::matrixgraph::core::common::StoreStrategy;

public:
  PartitionerBase(const std::string &input_path, const std::string &output_path,
                  StoreStrategy store_strategy)
      : input_path_(input_path), output_path_(output_path),
        store_strategy_(store_strategy) {}

  // This function will submit the partitioning task.
  virtual void RunPartitioner() = 0;

protected:
  const std::string input_path_;
  const std::string output_path_;
  const StoreStrategy store_strategy_;
};

} // namespace partitioner
} // namespace tools
} // namespace matrixgraph
} // namespace sics

#endif // SICS_GRAPH_SYSTEMS_TOOLS_PARTITIONER_H_