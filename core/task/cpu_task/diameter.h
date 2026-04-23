#ifndef MATRIXGRAPH_CORE_TASK_CPU_TASK_DIAMETER_H_
#define MATRIXGRAPH_CORE_TASK_CPU_TASK_DIAMETER_H_

#include <string>

#include "core/common/types.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/task/gpu_task/task_base.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

// Exact undirected diameter on CSR: treat each directed edge as undirected
// (traverse both out- and in-adjacency). O(n * (n + m)) time, parallelized
// over sources with std::execution::par.
class Diameter : public TaskBase {
 public:
  explicit Diameter(std::string data_graph_path)
      : data_graph_path_(std::move(data_graph_path)) {}

  void Run();

 private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;

  void LoadData();
  void ComputeUndirectedDiameter(const ImmutableCSR& g);

  ImmutableCSR g_;
  std::string data_graph_path_;
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_CPU_TASK_DIAMETER_H_
