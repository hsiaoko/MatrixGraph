#include "core/task/cpu_task/subiso_cpu.cuh"

#include <gflags/gflags.h>

#include <fstream>
#include <iostream>
#include <list>
#include <utility>

#include "core/common/consts.h"
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
DEFINE_string(o, "",
              "Path for output results; if empty, only the match count is "
              "reported (default: empty = count-only)");
DEFINE_string(reject_output, "",
              "Path to write rejected (u,v) candidates, empty disables it");
DEFINE_int32(t, 72, "Number of CPU threads to use (default: 1)");
DEFINE_int32(filter_hop, 1,
             "Number of hops for min-wise filter neighbor collection (default: 1)");
DEFINE_int32(filter_k, 3,
             "Number of minimum hash values (k) for k-min-wise filter (default: 3)");
DEFINE_bool(disable_min_wise_filter, false,
            "Disable min-wise IP filter");
DEFINE_bool(disable_label_degree_filter, false,
            "Disable label-degree filter");
DEFINE_bool(disable_nlc_filter, false,
            "Disable neighbor-label-counter filter");
DEFINE_bool(disable_matching_order, false,
            "Disable cost-model-based matching order (use default local-id DFS)");

// System configuration
DEFINE_string(
    scheduler, "CHBL",
    "Scheduler type (options: CHBL, EvenSplit, RoundRobin, default: CHBL)");

using sics::matrixgraph::core::components::scheduler::SchedulerType;
using sics::matrixgraph::core::task::SubIsoCPU;

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
  if (FLAGS_filter_hop < 1) {
    std::cerr << "Error: filter_hop (--filter_hop) must be at least 1"
              << std::endl;
    is_valid = false;
  }
  if (FLAGS_filter_k < 1 ||
      static_cast<uint32_t>(FLAGS_filter_k) >
          sics::matrixgraph::core::common::kDefaultHeapCapacity) {
    std::cerr << "Error: filter_k (--filter_k) must be in [1, "
              << sics::matrixgraph::core::common::kDefaultHeapCapacity << "]"
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
  std::cout << "Output Path: "
            << (FLAGS_o.empty() ? "(none, count-only)" : FLAGS_o) << std::endl;
  std::cout << "Reject Output Path: " << FLAGS_reject_output << std::endl;
  std::cout << "Num Threads: " << FLAGS_t << std::endl;
  std::cout << "Filter Hop: " << FLAGS_filter_hop << std::endl;
  std::cout << "Filter K: " << FLAGS_filter_k << std::endl;
  std::cout << "Min-Wise Filter: "
            << (FLAGS_disable_min_wise_filter ? "disabled" : "enabled")
            << std::endl;
  std::cout << "Label-Degree Filter: "
            << (FLAGS_disable_label_degree_filter ? "disabled" : "enabled")
            << std::endl;
  std::cout << "NLC Filter: "
            << (FLAGS_disable_nlc_filter ? "disabled" : "enabled") << std::endl;
  std::cout << "Matching Order: "
            << (FLAGS_disable_matching_order ? "default" : "cost-model")
            << std::endl;
  std::cout << "Scheduler: " << FLAGS_scheduler << std::endl;
  std::cout << "==============================\n" << std::endl;
}

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "CPU Subgraph Isomorphism computation using MatrixGraph\n"
      "Usage: " +
      std::string(argv[0]) +
      " -p <pattern_path> -g <graph_path> [-o <output_path>] "
      "[-t <num_threads>] [options]");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!ValidateParameters()) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "apps/subiso_cpu.cpp");
    return EXIT_FAILURE;
  }

  PrintConfig();

  try {
    auto scheduler_type = Scheduler2Enum(FLAGS_scheduler);
    sics::matrixgraph::core::MatrixGraph system(scheduler_type);

    auto* task = new SubIsoCPU(
        FLAGS_p, FLAGS_g, FLAGS_o, FLAGS_t, FLAGS_m1, FLAGS_m2, "", "", "",
        "", FLAGS_reject_output, FLAGS_filter_hop, FLAGS_filter_k,
        !FLAGS_disable_min_wise_filter, !FLAGS_disable_label_degree_filter,
        !FLAGS_disable_nlc_filter, !FLAGS_disable_matching_order);
    system.Run(sics::matrixgraph::core::common::kSubIsoCPU, task);
    delete task;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
