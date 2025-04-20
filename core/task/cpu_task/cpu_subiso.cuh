#ifndef MATRIXGRAPH_CORE_TASK_CPU_SUBISO_CUH_
#define MATRIXGRAPH_CORE_TASK_CPU_SUBISO_CUH_

#include <string>
#include <thread>
#include <vector>

#include "core/common/types.h"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/matrix.cuh"
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
  using Matrix = sics::matrixgraph::core::data_structures::Matrix;
  using Edges = sics::matrixgraph::core::data_structures::Edges;

 public:
  CPUSubIso(const std::string& pattern_path, const std::string& data_graph_path,
            const std::string& output_path, int num_threads,
            const std::string& matrix_path1 = "",
            const std::string& matrix_path2 = "")
      : pattern_path_(pattern_path),
        data_graph_path_(data_graph_path),
        output_path_(output_path),
        num_threads_(num_threads),
        matrix_path1_(matrix_path1),
        matrix_path2_(matrix_path2) {}

  void Run();

 private:
  void LoadData();

  void InitLabel(VertexLabel* label_p, VertexLabel* label_g);

  void InitLabel();

  void AllocMappingBuf();

  void RecursiveMatching(const ImmutableCSR& p, const ImmutableCSR& g,
                         const std::vector<Matrix>& m_vec);

  void WOJMatching(const ImmutableCSR& p, const ImmutableCSR& g);

  ImmutableCSR p_;
  ImmutableCSR g_;
  std::vector<Matrix> m_vec_;

  const std::string pattern_path_;
  const std::string data_graph_path_;
  const std::string matrix_path1_;
  const std::string matrix_path2_;

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
