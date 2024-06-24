#ifndef MATRIXGRAPH_CORE_COMPONENTS_GPU_TASK_MANAGER_CUH_

#include <cuda_runtime.h>
#include <string>
#include <unistd.h>

#include <climits>
#include <condition_variable>
#include <mutex>

#include "core/common/types.h"
#include "core/components/scheduler/CHBL_scheduler.h"
#include "core/components/scheduler/even_split_scheduler.h"
#include "core/components/scheduler/round_robin_scheduler.h"
#include "core/data_structures/tiled_matrix.cuh"
#include "core/task/task_base.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace components {

// Class for managing GPU tasks and resources
class GPUTaskManager {
private:
  using TiledMatrix = sics::matrixgraph::core::data_structures::TiledMatrix;
  using Tile = sics::matrixgraph::core::data_structures::Tile;
  using Mask = sics::matrixgraph::core::data_structures::Mask;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using TaskBase = sics::matrixgraph::core::task::TaskBase;
  using GPUTaskType = sics::matrixgraph::core::task::GPUTaskType;

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
  void SubmitTask(size_t task_id, GPUTaskType task_type, TaskBase *task);

  // Check if a task is finished
  // @param task_id: Identifier for the task
  bool IsTaskFinished(size_t task_id);

private:
  // Get CUDA stream for a task
  // @param task_id: Identifier for the task
  cudaStream_t *GetStream(size_t task_id);

  // Remove CUDA stream for a task
  // @param task_id: Identifier for the task
  void ReleaseStream(size_t task_id);

  std::mutex streams_mtx_;

  std::unordered_map<size_t, cudaStream_t *> streams_by_task_id_;
};

} // namespace components
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // MATRIXGRAPH_CORE_COMPONENTS_GPU_TASK_MANAGER_CUH_