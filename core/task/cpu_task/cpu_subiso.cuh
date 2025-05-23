#ifndef MATRIXGRAPH_CORE_TASK_CPU_SUBISO_CUH_
#define MATRIXGRAPH_CORE_TASK_CPU_SUBISO_CUH_

#include <string>
#include <thread>
#include <vector>

#include "core/common/types.h"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/matrix.cuh"
#include "core/data_structures/unified_buffer.cuh"
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
  using UnifiedOwnedBufferFloat =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<float>;

 public:
  CPUSubIso(const std::string& pattern_path, const std::string& data_graph_path,
            const std::string& output_path, int num_threads,
            const std::string& matrix_path1 = "",
            const std::string& matrix_path2 = "",
            const std::string& matrix_path3 = "",
            const std::string& matrix_path4 = "",
            const std::string& matrix_path5 = "",
            const std::string& matrix_path6 = "")
      : pattern_path_(pattern_path),
        data_graph_path_(data_graph_path),
        output_path_(output_path),
        num_threads_(num_threads),
        matrix_path1_(matrix_path1),
        matrix_path2_(matrix_path2),
        matrix_path3_(matrix_path3),
        matrix_path4_(matrix_path4),
        matrix_path5_(matrix_path5),
        matrix_path6_(matrix_path6) {}

  void Run();

 private:
  void LoadData();

  void InitLabel(VertexLabel* label_p, VertexLabel* label_g);

  void InitLabel();

  void AllocMappingBuf();

  void RecursiveMatching(
      const ImmutableCSR& p, const ImmutableCSR& g,
      const std::vector<Matrix>& m_vec,
      const std::vector<UnifiedOwnedBufferFloat*>& m_unified_buffer_vec);

  void WOJMatching(
      const ImmutableCSR& p, const ImmutableCSR& g,
      const std::vector<Matrix>& m_vec,
      const std::vector<UnifiedOwnedBufferFloat*>& m_unified_buffer_vec);

  ImmutableCSR p_;
  ImmutableCSR g_;
  std::vector<Matrix> m_vec_;
  std::vector<UnifiedOwnedBufferFloat*> m_unified_buffer_vec_;

  const std::string pattern_path_;
  const std::string data_graph_path_;
  const std::string matrix_path1_;
  const std::string matrix_path2_;
  const std::string matrix_path3_;
  const std::string matrix_path4_;
  const std::string matrix_path5_;
  const std::string matrix_path6_;

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
