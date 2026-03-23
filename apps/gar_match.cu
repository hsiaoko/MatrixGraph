#include <gflags/gflags.h>

#include <iostream>

#include "core/components/scheduler/scheduler.h"
#include "core/matrixgraph.cuh"
#include "core/task/gpu_task/gar_match.cuh"

// Input/Output flags
DEFINE_string(config, "", "Path to GARMatch config (required)");
DEFINE_string(o, "", "Path for output results (required)");

DEFINE_string(
    scheduler, "CHBL",
    "Scheduler type (options: CHBL, EvenSplit, RoundRobin, default: CHBL)");

using sics::matrixgraph::core::components::scheduler::SchedulerType;
using sics::matrixgraph::core::task::GARMatch;

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
  if (FLAGS_config.empty()) {
    std::cerr << "Error: GARMatch config path (-config) is required"
              << std::endl;
    is_valid = false;
  }
  if (FLAGS_o.empty()) {
    std::cerr << "Error: Output path (-o) is required" << std::endl;
    is_valid = false;
  }
  return is_valid;
}

void PrintConfig() {
  std::cout << "\n=== GARMatch App Configuration ===" << std::endl;
  std::cout << "Config Path: " << FLAGS_config << std::endl;
  std::cout << "Output Path: " << FLAGS_o << std::endl;
  std::cout << "Scheduler: " << FLAGS_scheduler << std::endl;
  std::cout << "==================================\n" << std::endl;
}

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "GARMatch app using MatrixGraph task interface\n"
      "Usage: " +
      std::string(argv[0]) +
      " -config <config_path> -o <output_path> [options]");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!ValidateParameters()) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "apps/gar_match.cu");
    return EXIT_FAILURE;
  }

  PrintConfig();

  try {
    auto scheduler_type = Scheduler2Enum(FLAGS_scheduler);
    sics::matrixgraph::core::MatrixGraph system(scheduler_type);

    auto* task = new GARMatch(FLAGS_config, FLAGS_o);
    system.Run(sics::matrixgraph::core::common::kGARMatch, task);
    delete task;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
