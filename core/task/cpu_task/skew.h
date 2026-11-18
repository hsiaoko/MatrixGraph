#ifndef MATRIXGRAPH_CORE_TASK_CPU_TASK_SKEW_H_
#define MATRIXGRAPH_CORE_TASK_CPU_TASK_SKEW_H_

#include "core/common/types.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/task/gpu_task/task_base.cuh"

#include <cstdint>
#include <string>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

class Skew : public TaskBase {
 public:
  Skew(std::string data_graph_path, size_t sample_sources = 50,
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
  void ComputeSkew(const ImmutableCSR& g);

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

#endif  // MATRIXGRAPH_CORE_TASK_CPU_TASK_SKEW_H_
