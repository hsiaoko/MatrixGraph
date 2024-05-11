#ifndef HYPERBLOCKER_CORE_HYPER_BLOCKER_CUH_
#define HYPERBLOCKER_CORE_HYPER_BLOCKER_CUH_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <condition_variable>
#include <ctime>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/components/gpu_task_manager.cuh"
#include "core/components/scheduler/CHBL_scheduler.h"
#include "core/components/scheduler/even_split_scheduler.h"
#include "core/components/scheduler/round_robin_scheduler.h"
#include "core/components/scheduler/scheduler.h"
#include "core/task/matrix_multiplier.cuh"
#include "core/task/task_base.cuh"
#include "core/util/cuda_check.cuh"

namespace sics {
namespace matrixgraph {
namespace core {

class MatrixGraph {
private:
  using SchedulerType =
      sics::matrixgraph::core::components::scheduler::SchedulerType;
  using GPUTaskType = sics::matrixgraph::core::task::GPUTaskType;
  using TaskBase = sics::matrixgraph::core::task::TaskBase;

public:
  MatrixGraph() = delete;

  MatrixGraph(SchedulerType scheduler_type = components::scheduler::kCHBL);

  ~MatrixGraph() = default;

  void Run(task::GPUTaskType task_type, task::TaskBase *task_ptr);

  void PrintDeviceInfo() const;

private:
  int n_device_ = 0;

  std::unique_ptr<std::mutex> p_streams_mtx_;

  std::mutex p_hr_start_mtx_;
  std::unique_lock<std::mutex> p_hr_start_lck_;
  std::condition_variable p_hr_start_cv_;

  std::unordered_map<int, cudaStream_t *> streams_;

  components::scheduler::Scheduler *scheduler_;

  bool p_hr_terminable_ = false;
};

} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // HYPERBLOCKER_CORE_HYPER_BLOCKER_CUH_