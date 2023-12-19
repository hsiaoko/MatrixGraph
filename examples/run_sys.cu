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

DEFINE_string(data_l, "", "input dir 2.");
DEFINE_string(data_r, "", "input dir 1.");

DEFINE_string(rule_dir, "", "root path of rules.");
DEFINE_string(o, "", "output path.");
DEFINE_string(sep, ",", "separator to split a line of csv file.");
DEFINE_string(scheduler, "CHBL", "scheduler type.");
DEFINE_bool(read_header, false, "whether to read header of csv.");
DEFINE_uint64(n_partitions, 1, "number of partitions.");
DEFINE_uint64(prefix_hash_predicate_index, INT_MAX, "number of partitions.");

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
  sics::matrixgraph::core::HyperBlocker hb(
      FLAGS_rule_dir, FLAGS_data_l, FLAGS_data_r, FLAGS_o, FLAGS_n_partitions,
      FLAGS_prefix_hash_predicate_index, FLAGS_sep, scheduler_type);

  hb.Run();

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
