#ifndef MATRIXGRAPH_CORE_TASK_GPU_TASK_TASK_BASE_CUH_
#define MATRIXGRAPH_CORE_TASK_GPU_TASK_TASK_BASE_CUH_

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

// Base class for GPU tasks. Owns the CUDA streams used for per-stream
// parallelism; derived tasks own their own device/pinned scratch buffers.
class TaskBase {
 public:
  virtual ~TaskBase() = default;

 protected:
  // (Re)create n_streams_ CUDA streams. Safe to call after changing
  // n_streams_; destroys and recreates existing streams when the count differs.
  __host__ void EnsureStreams();

  // Destroy all streams owned by this task.
  __host__ void DestroyStreams();

  uint32_t n_streams_ = 1;
  std::vector<cudaStream_t> streams_;
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_GPU_TASK_TASK_BASE_CUH_
