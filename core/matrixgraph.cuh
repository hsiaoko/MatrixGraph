#ifndef MATRIXGRAPH_CORE_MATRIXGRAPH_CUH_
#define MATRIXGRAPH_CORE_MATRIXGRAPH_CUH_

#include "core/common/types.h"
#include "core/components/scheduler/scheduler.h"
#include "core/task/cpu_task/cpu_task_base.h"
#include "core/task/gpu_task/task_base.cuh"
#include "core/util/cuda_check.cuh"

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
};

}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_MATRIXGRAPH_CUH_