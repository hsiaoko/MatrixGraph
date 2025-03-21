//#include <cuda_runtime.h>

//#include <device_launch_parameters.h>
#include <gflags/gflags.h>

#include <fstream>
#include <iostream>
#include <list>
#include <utility>

#include "core/common/types.h"
#include "core/common/yaml_config.h"
#include "core/components/scheduler/scheduler.h"
#include "core/matrixgraph.cuh"
#include "core/task/subiso.cuh"
#include "core/task/task_base.cuh"

DEFINE_string(p, "", "input data dir path for pattern.");
DEFINE_string(g, "", "input data dir path for data graph");
DEFINE_string(e, "", "input data dir path for edgelist of data graph");
DEFINE_string(o, "", "output path.");
DEFINE_string(scheduler, "CHBL", "scheduler type.");

using sics::matrixgraph::core::components::scheduler::SchedulerType;
using sics::matrixgraph::core::task::SubIso;

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

  auto *task = new SubIso(FLAGS_p, FLAGS_g, FLAGS_e, FLAGS_o);

  // State which application is going to be running.
  system.Run(sics::matrixgraph::core::task::kSubIso, task);

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
