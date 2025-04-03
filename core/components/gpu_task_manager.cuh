#ifndef MATRIXGRAPH_CORE_COMPONENTS_GPU_TASK_MANAGER_CUH_

#include "core/common/types.h"
#include "core/components/scheduler/CHBL_scheduler.h"
#include "core/components/scheduler/even_split_scheduler.h"
#include "core/components/scheduler/round_robin_scheduler.h"
#include "core/task/cpu_task/cpu_task_base.h"
#include "core/task/gpu_task/task_base.cuh"
#include <climits>
#include <condition_variable>
#include <cuda_runtime.h>
#include <mutex>
#include <string>
#include <unistd.h>

namespace sics {
namespace matrixgraph {
namespace core {
namespace components {

// Class for managing GPU tasks and resources
class GPUTaskManager {
 private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using TaskType = sics::matrixgraph::core::common::TaskType;
  using TaskBase = sics::matrixgraph::core::task::TaskBase;

 public:
  // Constructor
  GPUTaskManager() = default;

  // Destructor
  ~GPUTaskManager();

  // Function to submit a task to GPU
  // @param task_id: Identifier for the task
  // @param task_type: Type of the GPU task
  // @param kernel_wrap: Function pointer to the kernel wrap function
  // @param host_input: Pointer to the input buffer collection
  // @param host_output: Pointer to the output buffer collection
  void SubmitTask(size_t task_id, TaskType task_type, TaskBase* task);

  // Check if a task is finished
  // @param task_id: Identifier for the task
  bool IsTaskFinished(size_t task_id);

 private:
  // Get CUDA stream for a task
  // @param task_id: Identifier for the task
  cudaStream_t* GetStream(size_t task_id);

  // Remove CUDA stream for a task
  // @param task_id: Identifier for the task
  void ReleaseStream(size_t task_id);

  std::mutex streams_mtx_;

  std::unordered_map<size_t, cudaStream_t*> streams_by_task_id_;
};

}  // namespace components
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
#endif  // MATRIXGRAPH_CORE_COMPONENTS_GPU_TASK_MANAGER_CUH_