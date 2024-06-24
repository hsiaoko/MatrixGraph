#include <cuda_runtime.h>

#include <device_launch_parameters.h>
#include <gflags/gflags.h>

#include <fstream>
#include <iostream>
#include <list>
#include <utility>

#include <cutlass/gemm/device/gemm.h>

#include "core/common/types.h"
#include "core/common/yaml_config.h"
#include "core/components/scheduler/scheduler.h"
#include "core/matrixgraph.cuh"
#include "core/task/matrix_multiplier.cuh"
#include "core/task/task_base.cuh"

DEFINE_string(i, "", "data dir path.");
DEFINE_string(o, "", "output path.");
DEFINE_int32(gemm_count, 1, "GEMM count");
DEFINE_string(scheduler, "CHBL", "scheduler type.");

using sics::matrixgraph::core::components::scheduler::SchedulerType;
using sics::matrixgraph::core::task::MatrixMultiplier;

SchedulerType Scheduler2Enum(const std::string &s) {
  if (s == "EvenSplit")
    return sics::matrixgraph::core::components::scheduler::kEvenSplit;
  else if (s == "CHBL")
    return sics::matrixgraph::core::components::scheduler::kCHBL;
  else if (s == "RoundRobin")
    return sics::matrixgraph::core::components::scheduler::kRoundRobin;
  return sics::matrixgraph::core::components::scheduler::kCHBL;
};

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  auto scheduler_type = Scheduler2Enum(FLAGS_scheduler);
  sics::matrixgraph::core::MatrixGraph system(scheduler_type);

  auto matrix_multiplier =
      new MatrixMultiplier(FLAGS_i, FLAGS_o, FLAGS_gemm_count);

  // State which application is going to be running.
  system.PrintDeviceInfo();
  system.Run(sics::matrixgraph::core::task::kGEMM, matrix_multiplier);

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
