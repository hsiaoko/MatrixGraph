#ifndef MATRIXGRAPH_CORE_TASK_CPU_TASK_DIAMETER_H_
#define MATRIXGRAPH_CORE_TASK_CPU_TASK_DIAMETER_H_

#include <cstdint>
#include <string>

#include "core/common/types.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/task/gpu_task/task_base.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

// Undirected diameter on CSR: treat each directed edge as undirected
// (traverse both out- and in-adjacency).
// Default: approximate — BFS from `sample_sources_` random vertices (max
// eccentricity among them; never exceeds true diameter). `sample_sources_ == 0`
// means exact: BFS from every vertex (O(n*(n+m))).
class Diameter : public TaskBase {
 public:
  // cpu_parallelism: max TBB workers for the parallel BFS-source loop; 0 = unlimited.
  Diameter(std::string data_graph_path, size_t sample_sources = 50,
           uint64_t random_seed = 42, size_t cpu_parallelism = 0)
      : data_graph_path_(std::move(data_graph_path)),
        sample_sources_(sample_sources),
        random_seed_(random_seed),
        cpu_parallelism_(cpu_parallelism) {}

  void Run();

 private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;

  void LoadData();
  void ComputeUndirectedDiameter(const ImmutableCSR& g);

  ImmutableCSR g_;
  std::string data_graph_path_;
  size_t sample_sources_;
  uint64_t random_seed_;
  size_t cpu_parallelism_;
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_CPU_TASK_DIAMETER_H_
