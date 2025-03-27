#ifndef MATRIXGRAPH_CORE_TASK_CPU_SUBISO_CUH_
#define MATRIXGRAPH_CORE_TASK_CPU_SUBISO_CUH_

#include <string>
#include <thread>
#include <vector>

#include "core/common/types.h"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/task/cpu_task/cpu_task_base.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

class CPUSubIso : public CPUTaskBase {
 private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using GraphID = sics::matrixgraph::core::common::GraphID;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;
  using Edges = sics::matrixgraph::core::data_structures::Edges;

 public:
  CPUSubIso(const std::string& pattern_path, const std::string& data_graph_path,
            const std::string& output_path, int num_threads)
      : pattern_path_(pattern_path),
        data_graph_path_(data_graph_path),
        output_path_(output_path),
        num_threads_(num_threads) {}

  void Run();

 private:
  void LoadData();
  void InitLabel(VertexLabel* label_p, VertexLabel* label_g);
  void InitLabel();
  void AllocMappingBuf();
  void Matching(const ImmutableCSR& p, const ImmutableCSR& g);
  void WOJMatching(const ImmutableCSR& p, const ImmutableCSR& g);

  // Thread pool for parallel processing
  void ParallelMatching(const ImmutableCSR& p, const ImmutableCSR& g,
                        size_t thread_id, size_t total_threads);

  ImmutableCSR p_;
  ImmutableCSR g_;

  const std::string pattern_path_;
  const std::string data_graph_path_;

  const std::string output_path_;
  const int num_threads_;

  VertexLabel* label_p_ = nullptr;
  VertexLabel* label_g_ = nullptr;
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_CPU_SUBISO_CUH_
