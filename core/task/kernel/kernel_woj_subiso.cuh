#ifndef MATRIXGRAPH_CORE_TASK_KERNEL_KERNEL_WOJ_SUBISO_CUH_
#define MATRIXGRAPH_CORE_TASK_KERNEL_KERNEL_WOJ_SUBISO_CUH_

#include <vector>

#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/unified_buffer.cuh"
#include "core/task/kernel/data_structures/exec_plan.cuh"
#include "core/task/kernel/data_structures/woj_exec_plan.cuh"
#include "core/task/kernel/data_structures/woj_matches.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

class WOJSubIsoKernelWrapper {
private:
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using UnifiedOwnedBufferVertexID =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexID>;
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;
  using Edges = sics::matrixgraph::core::data_structures::Edges;

public:
  // deleting copy constructor
  WOJSubIsoKernelWrapper(const WOJSubIsoKernelWrapper &obj) = delete;

  void operator=(const WOJSubIsoKernelWrapper &) = delete;

  // @Description: GetInstance() is a method that returns an instance
  // when it is invoked. It returns the same instance if it is invoked more
  // than once as an instance of Singleton class is already created.
  static WOJSubIsoKernelWrapper *GetInstance();

  static std::vector<WOJMatches *> Filter(const WOJExecutionPlan &exec_plan,
                                          const ImmutableCSR &p,
                                          const ImmutableCSR &g,
                                          const Edges &e);

  static void Join(const WOJExecutionPlan &exec_plan,
                   const std::vector<WOJMatches *> &input_woj_matches,
                   WOJMatches *output_woj_matches);

private:
  WOJSubIsoKernelWrapper() = default;

  inline static WOJSubIsoKernelWrapper *ptr_ = nullptr;
};

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // INC_51_11_GRAPH_COMPUTING_MATRIXGRAPH_CORE_TASK_KERNEL_KERNEL_SUBISO_CUH_