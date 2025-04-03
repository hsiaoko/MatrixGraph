#ifndef HYPERBLOCKER_CORE_HYPER_BLOCKER_CUH_
#define HYPERBLOCKER_CORE_HYPER_BLOCKER_CUH_

//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>

#include <condition_variable>
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
#include "core/task/cpu_task/cpu_task_base.h"
#include "core/task/gpu_task/task_base.cuh"
//#include "core/util/cuda_check.cuh"

namespace sics {
namespace matrixgraph {
namespace core {

class MatrixGraph {
 private:
  using SchedulerType =
      sics::matrixgraph::core::components::scheduler::SchedulerType;
  using TaskType = sics::matrixgraph::core::common::TaskType;
  using TaskBase = sics::matrixgraph::core::task::TaskBase;
  using CPUTaskBase = sics::matrixgraph::core::task::CPUTaskBase;

 public:
  MatrixGraph() = delete;

  MatrixGraph(SchedulerType scheduler_type = components::scheduler::kCHBL);

  ~MatrixGraph() = default;

  void Run(TaskType task_type, TaskBase* task_ptr);

  void Run(TaskType task_type, CPUTaskBase* task_ptr);

  void PrintDeviceInfo() const;

 private:
  int n_device_ = 0;

  std::unique_ptr<std::mutex> p_streams_mtx_;

  std::mutex p_hr_start_mtx_;
  std::unique_lock<std::mutex> p_hr_start_lck_;
  std::condition_variable p_hr_start_cv_;

  std::unordered_map<int, cudaStream_t*> streams_;

  components::scheduler::Scheduler* scheduler_;

  bool p_hr_terminable_ = false;
};

}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // HYPERBLOCKER_CORE_HYPER_BLOCKER_CUH_