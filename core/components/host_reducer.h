#ifndef HYPERBLOCKER_CORE_COMPONENTS_HOST_REDUCER_CUH_
#define HYPERBLOCKER_CORE_COMPONENTS_HOST_REDUCER_CUH_

#include <fstream>
#include <iostream>
#include <mutex>

#include <cuda_runtime.h>
#include <unistd.h>

namespace sics {
namespace matrixgraph {
namespace core {
namespace components {

class HostReducer {
public:
  HostReducer(scheduler::Scheduler *scheduler,
              std::unordered_map<int, cudaStream_t *> *p_streams,
              std::mutex *p_streams_mtx,
              std::unique_lock<std::mutex> *p_hr_start_lck,
              std::condition_variable *p_hr_start_cv, bool *p_hr_terminable)
      : p_streams_(p_streams), p_streams_mtx_(p_streams_mtx),
        p_hr_start_lck_(p_hr_start_lck), p_hr_start_cv_(p_hr_start_cv),
        scheduler_(scheduler), p_hr_terminable_(p_hr_terminable) {}

  ~HostReducer() = default;

  void Run() {
    p_hr_start_cv_->wait(*p_hr_start_lck_,
                         [&] { return p_streams_->size() > 0; });
  }

private:
  scheduler::Scheduler *scheduler_;

  std::mutex *p_streams_mtx_;
  std::unique_lock<std::mutex> *p_hr_start_lck_;
  std::condition_variable *p_hr_start_cv_;

  std::unordered_map<int, cudaStream_t *> *p_streams_;

  bool *p_hr_terminable_;
};

} // namespace components
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // HYPERBLOCKER_CORE_COMPONENTS_HOST_REDUCER_CUH_
