#ifndef MATRIXGRAPH_CORE_TASK_GPU_TASK_TASK_BASE_CUH_
#define MATRIXGRAPH_CORE_TASK_GPU_TASK_TASK_BASE_CUH_

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

// Base class for GPU tasks. Kept minimal: derived tasks own their CUDA
// streams and scratch buffers, so there is no shared state to manage here.
class TaskBase {
 public:
  ~TaskBase() = default;
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_GPU_TASK_TASK_BASE_CUH_
