#ifndef MATRIXGRAPH_CORE_TASK_PAGERANK_CUH_
#define MATRIXGRAPH_CORE_TASK_PAGERANK_CUH_

#include <string>

#include "core/common/types.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/task/task_base.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

class PageRank : public TaskBase {
 private:
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;

 public:
  PageRank(const std::string& data_graph_path, const std::string& output_path,
           float damping_factor = 0.85f, float epsilon = 1e-6f,
           int max_iterations = 10)
      : data_graph_path_(data_graph_path),
        output_path_(output_path),
        damping_factor_(damping_factor),
        epsilon_(epsilon),
        max_iterations_(max_iterations) {}

  __host__ void Run();

 private:
  __host__ void LoadData();
  __host__ void ComputePageRank(const ImmutableCSR& g);

  std::string data_graph_path_;
  std::string output_path_;
  float damping_factor_;
  float epsilon_;
  int max_iterations_;
  ImmutableCSR g_;
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_PAGERANK_CUH_
