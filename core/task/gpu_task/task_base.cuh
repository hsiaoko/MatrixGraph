#ifndef MATRIXGRAPH_CORE_COMPONENTS_TASK_TASK_BASE_CUH_
#define MATRIXGRAPH_CORE_COMPONENTS_TASK_TASK_BASE_CUH_

// #include <cuda_runtime.h>

#include <mutex>
#include <unordered_map>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

class TaskBase {
 public:
  ~TaskBase() = default;

  // Get CUDA stream for a task
  // @param task_id: Identifier for the task
  // cudaStream_t GetStream(size_t task_id);

  // // Remove CUDA stream for a task
  // // @param task_id: Identifier for the task
  // void ReleaseStream(size_t task_id);

  // // Check if a task is finished
  // // @param task_id: Identifier for the task
  // bool IsTaskFinished(size_t task_id);

  // protected:
  //  std::mutex streams_mtx_;
  //  std::unordered_map<size_t, cudaStream_t*> streams_by_task_id_;
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_COMPONENTS_TASK_TASK_BASE_CUH_
