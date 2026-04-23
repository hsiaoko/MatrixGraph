#include <gflags/gflags.h>

#include <iostream>
#include <string>
#include <utility>

#include "core/common/types.h"
#include "core/components/scheduler/scheduler.h"
#include "core/matrixgraph.cuh"
#include "core/task/cpu_task/diameter.h"
#include "core/task/gpu_task/task_base.cuh"

DEFINE_string(g, "", "Path to the input graph file (required)");
DEFINE_string(
    scheduler, "CHBL",
    "Scheduler type (options: CHBL, EvenSplit, RoundRobin, default: CHBL)");

using sics::matrixgraph::core::components::scheduler::SchedulerType;
using sics::matrixgraph::core::task::Diameter;

SchedulerType Scheduler2Enum(const std::string& s) {
  if (s == "EvenSplit")
    return sics::matrixgraph::core::components::scheduler::kEvenSplit;
  if (s == "CHBL")
    return sics::matrixgraph::core::components::scheduler::kCHBL;
  if (s == "RoundRobin")
    return sics::matrixgraph::core::components::scheduler::kRoundRobin;
  return sics::matrixgraph::core::components::scheduler::kCHBL;
}

bool ValidateParameters() {
  if (FLAGS_g.empty()) {
    std::cerr << "Error: Input graph path (-g) is required" << std::endl;
    return false;
  }
  return true;
}

void PrintConfig() {
  std::cout << "\n=== Graph Diameter ===" << std::endl;
  std::cout << "Input Graph: " << FLAGS_g << std::endl;
  std::cout << "Scheduler: " << FLAGS_scheduler << std::endl;
  std::cout << "Metric: undirected diameter (out + in adjacency as undirected)"
            << std::endl;
  std::cout << "Note: exact O(n*(n+m)); large graphs can take a long time."
            << std::endl;
  std::cout << "=======================\n" << std::endl;
}

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "Undirected graph diameter (exact) using MatrixGraph\n"
      "Usage: " +
      std::string(argv[0]) + " -g <graph_path> [options]");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!ValidateParameters()) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "apps/diameter.cpp");
    return EXIT_FAILURE;
  }

  PrintConfig();

  try {
    auto scheduler_type = Scheduler2Enum(FLAGS_scheduler);
    sics::matrixgraph::core::MatrixGraph system(scheduler_type);

    auto* task = new Diameter(FLAGS_g);
    system.Run(sics::matrixgraph::core::common::kDiameter, task);
    delete task;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
