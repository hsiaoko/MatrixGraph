/**
 * @section DESCRIPTION
 *
 * Run HyperBlocker
 */

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
#include "core/gpu/global_func.cuh"
#include "core/matrixgraph.cuh"

DEFINE_string(data, "", "data dir path.");

DEFINE_string(o, "", "output path.");
DEFINE_string(sep, ",", "separator to split a line of csv file.");
DEFINE_string(scheduler, "CHBL", "scheduler type.");
DEFINE_bool(read_header, false, "whether to read header of csv.");
DEFINE_uint64(n_partitions, 1, "number of partitions.");

using sics::matrixgraph::core::components::scheduler::kCHBL;
using sics::matrixgraph::core::components::scheduler::kEvenSplit;
using sics::matrixgraph::core::components::scheduler::kRoundRobin;
using sics::matrixgraph::core::components::scheduler::SchedulerType;

SchedulerType Scheduler2Enum(const std::string &s) {
  if (s == "EvenSplit")
    return kEvenSplit;
  else if (s == "CHBL")
    return kCHBL;
  else if (s == "RoundRobin")
    return kRoundRobin;
  return kCHBL;
};

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  auto scheduler_type = Scheduler2Enum(FLAGS_scheduler);
  sics::matrixgraph::core::MatrixGraph system(FLAGS_data, scheduler_type);

  system.Run();

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
