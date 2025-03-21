#include <gflags/gflags.h>

#include <fstream>
#include <iostream>
#include <list>
#include <string>
#include <utility>

#include "core/common/types.h"
#include "core/common/yaml_config.h"
#include "core/components/scheduler/scheduler.h"
#include "core/matrixgraph.cuh"
#include "core/task/pagerank.cuh"
#include "core/task/task_base.cuh"

// Input/Output flags
DEFINE_string(g, "", "Path to the input graph file (required)");
DEFINE_string(o, "", "Path to save PageRank values (required)");

// Algorithm parameters
DEFINE_double(damping, 0.85,
              "Damping factor for PageRank (range: 0.0 to 1.0, default: 0.85)");
DEFINE_double(epsilon, 1e-6,
              "Convergence threshold (range: > 0.0, default: 1e-6)");
DEFINE_int32(max_iter, 10,
             "Maximum number of iterations (range: > 0, default: 10)");

// System configuration
DEFINE_string(
    scheduler, "CHBL",
    "Scheduler type (options: CHBL, EvenSplit, RoundRobin, default: CHBL)");

using sics::matrixgraph::core::components::scheduler::SchedulerType;
using sics::matrixgraph::core::task::PageRank;

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
  if (FLAGS_g.empty()) {
    std::cerr << "Error: Input graph path (-g) is required" << std::endl;
    is_valid = false;
  }

  // Validate parameter ranges
  if (FLAGS_damping <= 0.0 || FLAGS_damping >= 1.0) {
    std::cerr << "Error: Damping factor must be between 0.0 and 1.0"
              << std::endl;
    is_valid = false;
  }
  if (FLAGS_epsilon <= 0.0) {
    std::cerr << "Error: Epsilon must be greater than 0.0" << std::endl;
    is_valid = false;
  }
  if (FLAGS_max_iter <= 0) {
    std::cerr << "Error: Maximum iterations must be greater than 0"
              << std::endl;
    is_valid = false;
  }

  return is_valid;
}

void PrintConfig() {
  std::cout << "\n=== PageRank Configuration ===" << std::endl;
  std::cout << "Input Graph: " << FLAGS_g << std::endl;
  std::cout << "Damping Factor: " << FLAGS_damping << std::endl;
  std::cout << "Convergence Threshold: " << FLAGS_epsilon << std::endl;
  std::cout << "Max Iterations: " << FLAGS_max_iter << std::endl;
  std::cout << "Scheduler: " << FLAGS_scheduler << std::endl;
  std::cout << "==========================\n" << std::endl;
}

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "PageRank computation using MatrixGraph\n"
      "Usage: " +
      std::string(argv[0]) + " -g <graph_path> -o <output_path> [options]");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!ValidateParameters()) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "apps/pagerank.cu");
    return EXIT_FAILURE;
  }

  PrintConfig();

  try {
    auto scheduler_type = Scheduler2Enum(FLAGS_scheduler);
    sics::matrixgraph::core::MatrixGraph system(scheduler_type);

    auto* task = new PageRank(FLAGS_g, FLAGS_o, FLAGS_damping, FLAGS_epsilon,
                              FLAGS_max_iter);
    system.Run(sics::matrixgraph::core::task::kPageRank, task);
    delete task;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
