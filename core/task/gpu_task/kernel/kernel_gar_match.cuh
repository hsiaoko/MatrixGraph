#ifndef MATRIXGRAPH_CORE_TASK_KERNEL_KERNEL_GAR_MATCH_CUH_
#define MATRIXGRAPH_CORE_TASK_KERNEL_KERNEL_GAR_MATCH_CUH_

#include "core/data_structures/gar_graph_arrays.h"
#include "core/data_structures/gar_match_arrays.h"
#include "core/data_structures/gar_pattern_arrays.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

class GARMatchKernelWrapper {
 public:
  using GARGraphArrays =
      sics::matrixgraph::core::data_structures::GARGraphArrays;
  using GARPatternArrays =
      sics::matrixgraph::core::data_structures::GARPatternArrays;
  using GARMatchArrays =
      sics::matrixgraph::core::data_structures::GARMatchArrays;

  GARMatchKernelWrapper(const GARMatchKernelWrapper& obj) = delete;
  void operator=(const GARMatchKernelWrapper&) = delete;

  static GARMatchKernelWrapper* GetInstance();

  // Placeholder: returns 0 and writes empty output.
  static int GARMatch(const GARGraphArrays& g, const GARPatternArrays& p,
                      GARMatchArrays* out);

 private:
  GARMatchKernelWrapper() = default;
  inline static GARMatchKernelWrapper* ptr_ = nullptr;
};

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_KERNEL_KERNEL_GAR_MATCH_CUH_
