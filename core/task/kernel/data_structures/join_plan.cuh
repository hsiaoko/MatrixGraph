#ifndef MATRIXGRAPH_CORE_TASK_KERNEL_DATA_STRUCTURES_JOIN_PLAN_CUH_
#define MATRIXGRAPH_CORE_TASK_KERNEL_DATA_STRUCTURES_JOIN_PLAN_CUH_

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

class JoinPlan {
public:
  __host__ get_join_key_idx() const { return join_key_idx_; }

private:
  std::vector<VertexID> join_order_;
  std::vector<VertexID> join_key_idx_;
};

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_TASK_KERNEL_DATA_STRUCTURES_JOIN_PLAN_CUH_