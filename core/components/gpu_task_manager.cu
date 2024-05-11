#include "core/components/gpu_task_manager.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace components {

bool GPUTaskManager::IsTaskFinished(size_t task_id) {
  auto iter = streams_by_task_id_.find(task_id);
  if (iter == streams_by_task_id_.end()) {
    return false;
  } else {
    cudaError_t err = cudaStreamQuery(*iter->second);
    if (err == cudaSuccess)
      return true;
    else if (err == cudaErrorNotReady)
      return false;
  }
}

cudaStream_t *GPUTaskManager::GetStream(size_t task_id) {
  std::lock_guard<std::mutex> lock(streams_mtx_);
  auto iter = streams_by_task_id_.find(task_id);
  if (iter == streams_by_task_id_.end()) {
    // If stream doesn't exist create a new CUDA stream
    cudaStream_t *p_stream = new cudaStream_t;
    cudaStreamCreate(p_stream);
    streams_by_task_id_.insert(std::make_pair(task_id, p_stream));
    return p_stream;
  } else {
    // Return the existing stream
    return iter->second;
  }
}

void GPUTaskManager::ReleaseStream(size_t task_id) {
  std::lock_guard<std::mutex> lock(streams_mtx_);
  // Find the stream for the task
  auto iter = streams_by_task_id_.find(task_id);
  if (iter != streams_by_task_id_.end()) {
    // If stream exist destroy the stream
    cudaStreamDestroy(*(iter->second));
    // Erase the stream from the map
    streams_by_task_id_.erase(iter);
  }
}
} // namespace components
} // namespace core
} // namespace matrixgraph
} // namespace sics
