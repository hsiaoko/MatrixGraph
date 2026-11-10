#include "core/task/gpu_task/task_base.cuh"

#include "core/util/cuda_check.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

void TaskBase::EnsureStreams() {
  if (streams_.size() == n_streams_) return;
  DestroyStreams();
  streams_.resize(n_streams_);
  for (uint32_t i = 0; i < n_streams_; ++i) {
    CUDA_CHECK(cudaStreamCreate(&streams_[i]));
  }
}

void TaskBase::DestroyStreams() {
  for (cudaStream_t stream : streams_) {
    if (stream) cudaStreamDestroy(stream);
  }
  streams_.clear();
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
