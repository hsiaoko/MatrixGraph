#include <gflags/gflags.h>

#include <iostream>
#include <string>
#include <utility>

#include "core/common/types.h"
#include "core/components/scheduler/scheduler.h"
#include "core/matrixgraph.cuh"
#include "core/task/cpu_task/skew.h"
#include "core/task/gpu_task/task_base.cuh"

DEFINE_string(g, "", "Path to the input graph file (required)");
DEFINE_int32(
    skew_samples, 50,
    "Approximate d_hat: number of random BFS source vertices. "
    "Use 0 for exact d_hat (BFS from every vertex; can be very slow).");
DEFINE_uint64(skew_seed, 42, "RNG seed for sampling sources when skew_samples > 0.");
DEFINE_uint32(
    cpu_parallel, 0,
    "Cap TBB parallelism for the BFS-source loop (matches batch cpu_cores); "
    "0 = default / unlimited.");
DEFINE_string(
    scheduler, "CHBL",
    "Scheduler type (options: CHBL, EvenSplit, RoundRobin, default: CHBL)");

using sics::matrixgraph::core::components::scheduler::SchedulerType;
using sics::matrixgraph::core::task::Skew;

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
  if (FLAGS_skew_samples < 0) {
    std::cerr << "Error: skew_samples must be >= 0 (0 = exact d_hat)"
              << std::endl;
    return false;
  }
  return true;
}

void PrintConfig() {
  std::cout << "\n=== Graph Skew ===" << std::endl;
  std::cout << "Input Graph: " << FLAGS_g << std::endl;
  std::cout << "skew(G) ≈ d_hat(G) / d_bar" << std::endl;
  std::cout << "  d_hat: max BFS eccentricity (undirected: out+in adjacency), "
               "same sampling as Diameter"
            << std::endl;
  std::cout << "  d_bar: mean total degree (|E_out|+|E_in|)/n" << std::endl;
  std::cout << "Scheduler: " << FLAGS_scheduler << std::endl;
  if (FLAGS_skew_samples == 0) {
    std::cout << "Mode: exact d_hat (all vertices as BFS sources)" << std::endl;
  } else {
    std::cout << "Mode: approximate d_hat, skew_samples=" << FLAGS_skew_samples
              << ", skew_seed=" << FLAGS_skew_seed << std::endl;
  }
  if (FLAGS_cpu_parallel > 0) {
    std::cout << "CPU parallelism cap (TBB, BFS sources): " << FLAGS_cpu_parallel
              << std::endl;
  } else {
    std::cout << "CPU parallelism cap (TBB): default (unlimited)" << std::endl;
  }
  std::cout << "=======================\n" << std::endl;
}

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "Graph skew ≈ d_hat / d_bar (default: approximate d_hat)\n"
      "Usage: " +
      std::string(argv[0]) +
      " -g <graph_path> [-skew_samples N] [-skew_seed S] [options]");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!ValidateParameters()) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "apps/skew.cpp");
    return EXIT_FAILURE;
  }

  PrintConfig();

  try {
    auto scheduler_type = Scheduler2Enum(FLAGS_scheduler);
    sics::matrixgraph::core::MatrixGraph system(scheduler_type);

    auto* task = new Skew(FLAGS_g, static_cast<size_t>(FLAGS_skew_samples),
                          FLAGS_skew_seed,
                          static_cast<size_t>(FLAGS_cpu_parallel));
    system.Run(sics::matrixgraph::core::common::kSkew, task);
    delete task;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
