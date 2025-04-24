#include "core/task/cpu_task/cpu_subiso.cuh"

#include <gflags/gflags.h>

#include <fstream>
#include <iostream>
#include <list>
#include <utility>

#include "core/common/types.h"
#include "core/common/yaml_config.h"
#include "core/components/scheduler/scheduler.h"
#include "core/matrixgraph.cuh"
#include "core/task/cpu_task/cpu_task_base.h"

// Input/Output flags
DEFINE_string(p, "", "Path to the pattern graph file (required)");
DEFINE_string(g, "", "Path to the data graph file (required)");
DEFINE_string(m1, "", "Path to the matrix of pattern graph embedding");
DEFINE_string(m2, "", "Path to the matrix of data graph embedding");
DEFINE_string(o, "", "Path for output results (required)");
DEFINE_int32(t, 72, "Number of CPU threads to use (default: 1)");

// System configuration
DEFINE_string(
    scheduler, "CHBL",
    "Scheduler type (options: CHBL, EvenSplit, RoundRobin, default: CHBL)");

using sics::matrixgraph::core::components::scheduler::SchedulerType;
using sics::matrixgraph::core::task::CPUSubIso;

SchedulerType Scheduler2Enum(const std::string& s) {
  if (s == "EvenSplit")
    return sics::matrixgraph::core::components::scheduler::kEvenSplit;
  else if (s == "CHBL")
    return sics::matrixgraph::core::components::scheduler::kCHBL;
  else if (s == "RoundRobin")
    return sics::matrixgraph::core::components::scheduler::kRoundRobin;
  return sics::matrixgraph::core::components::scheduler::kCHBL;
}

bool ValidateParameters() {
  bool is_valid = true;

  // Check required parameters
  if (FLAGS_p.empty()) {
    std::cerr << "Error: Pattern graph path (-p) is required" << std::endl;
    is_valid = false;
  }
  if (FLAGS_g.empty()) {
    std::cerr << "Error: Data graph path (-g) is required" << std::endl;
    is_valid = false;
  }
  if (FLAGS_t < 1) {
    std::cerr << "Error: Number of threads (-t) must be at least 1"
              << std::endl;
    is_valid = false;
  }

  return is_valid;
}

void PrintConfig() {
  std::cout << "\n=== CPU SubIso Configuration ===" << std::endl;
  std::cout << "Pattern Graph: " << FLAGS_p << std::endl;
  std::cout << "Data Graph: " << FLAGS_g << std::endl;
  std::cout << "Matrix 1: " << FLAGS_m1 << std::endl;
  std::cout << "Matrix 2: " << FLAGS_m2 << std::endl;
  std::cout << "Output Path: " << FLAGS_o << std::endl;
  std::cout << "Num Threads: " << FLAGS_t << std::endl;
  std::cout << "Scheduler: " << FLAGS_scheduler << std::endl;
  std::cout << "==============================\n" << std::endl;
}

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "CPU Subgraph Isomorphism computation using MatrixGraph\n"
      "Usage: " +
      std::string(argv[0]) +
      " -p <pattern_path> -g <graph_path> -e <edge_list> -o <output_path> "
      "[-t <num_threads>] [options]");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!ValidateParameters()) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "apps/cpu_subiso.cu");
    return EXIT_FAILURE;
  }

  PrintConfig();

  try {
    auto scheduler_type = Scheduler2Enum(FLAGS_scheduler);
    sics::matrixgraph::core::MatrixGraph system(scheduler_type);

    auto* task =
        new CPUSubIso(FLAGS_p, FLAGS_g, FLAGS_o, FLAGS_t, FLAGS_m1, FLAGS_m2);
    system.Run(sics::matrixgraph::core::common::kCPUSubIso, task);
    delete task;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
