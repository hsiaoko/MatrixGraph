#ifndef MATRIXGRAPH_CORE_TASK_CPU_TASK_SKEW_H_
#define MATRIXGRAPH_CORE_TASK_CPU_TASK_SKEW_H_

#include <cstdint>
#include <string>

#include "core/common/types.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/task/gpu_task/task_base.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

// skew(G) ≈ d_hat(G) / d_bar, with the same undirected BFS setup as Diameter:
// d_hat = max eccentricity over sampled (or all) sources; edges treated as
// undirected via out- + in-adjacency.
// d_bar = mean total degree per vertex: (|E_out| + |E_in|) / n in the CSR.
class Skew : public TaskBase {
 public:
  Skew(std::string data_graph_path, size_t sample_sources = 50,
       uint64_t random_seed = 42)
      : data_graph_path_(std::move(data_graph_path)),
        sample_sources_(sample_sources),
        random_seed_(random_seed) {}

  void Run();

 private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;

  void LoadData();
  void ComputeSkew(const ImmutableCSR& g);

  ImmutableCSR g_;
  std::string data_graph_path_;
  size_t sample_sources_;
  uint64_t random_seed_;
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_CPU_TASK_SKEW_H_
