#ifndef MATRIXGRAPH_CORE_TASK_GPU_TASK_TASK_BASE_CUH_
#define MATRIXGRAPH_CORE_TASK_GPU_TASK_TASK_BASE_CUH_

// TaskBase: the GPU task contract. Owns or borrows CUDA streams; service
// tasks keep typed APIs and leave Run empty rather than faking a job.
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

// Base class for GPU tasks. Streams have one owner per run: an engine
// attaches its pool stream before calling Run, and a task used directly
// (tests, tools) falls back to a stream it creates itself and destroys with
// the task. Derived tasks own their device/pinned scratch buffers.
class TaskBase {
 public:
  virtual ~TaskBase() { DestroyStreams(); }

  // Job entry point. Every job task overrides it; service-style tasks that
  // expose typed query APIs keep the empty default rather than faking a Run.
  virtual void Run() {}

  // Engine-side handoff: the stream this task should launch on. Only valid
  // until Run returns; the engine keeps ownership and destruction.
  __host__ void AttachPrimaryStream(cudaStream_t stream) {
    borrowed_stream_ = stream;
  }

 protected:
  // (Re)create n_streams_ CUDA streams. Safe to call after changing
  // n_streams_; destroys and recreates existing streams when the count differs.
  __host__ void EnsureStreams();

  // Destroy all streams owned by this task.
  __host__ void DestroyStreams();

  // Stream for tasks that run one launch sequence at a time (the common
  // case); multi-stream tasks index streams_ directly.
  __host__ cudaStream_t PrimaryStream() {
    if (borrowed_stream_ != nullptr) return borrowed_stream_;
    EnsureStreams();
    return streams_.empty() ? nullptr : streams_[0];
  }

  uint32_t n_streams_ = 1;
  std::vector<cudaStream_t> streams_;
  cudaStream_t borrowed_stream_ = nullptr;
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_GPU_TASK_TASK_BASE_CUH_
