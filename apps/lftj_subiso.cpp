#include <gflags/gflags.h>

#include <iostream>
#include <string>

#include "core/common/types.h"
#include "core/components/scheduler/scheduler.h"
#include "core/matrixgraph.cuh"
#include "core/task/cpu_task/lftj_subiso.cuh"

DEFINE_string(p, "", "Path to the pattern graph directory (required)");
DEFINE_string(g, "", "Path to the data graph directory (required)");
DEFINE_string(o, "", "Path for optional output (default: none)");
DEFINE_int32(t, 1, "Number of CPU threads (currently reserved, default: 1)");
DEFINE_uint64(limit, std::numeric_limits<uint64_t>::max(),
              "Maximum number of embeddings to count (default: unlimited)");
DEFINE_bool(canonical, false,
            "Enforce increasing vertex IDs with matching depth to avoid "
            "automorphic duplicates (correct for cliques, default: false)");
DEFINE_bool(disable_min_wise_filter, false,
            "Disable k-min-wise label-hash pre-filter (default: false)");
DEFINE_int32(filter_hop, 1,
             "Number of hops for min-wise filter neighbor collection "
             "(default: 1)");
DEFINE_int32(filter_k, 3,
             "Number of minimum hash values (k) for k-min-wise filter "
             "(default: 3)");
DEFINE_bool(disable_matching_order, false,
            "Disable greedy matching order and use natural order 0,1,2,... "
            "(default: false)");

static sics::matrixgraph::core::components::scheduler::SchedulerType
Scheduler2Enum(const std::string& s) {
  using sics::matrixgraph::core::components::scheduler::SchedulerType;
  if (s == "EvenSplit")
    return sics::matrixgraph::core::components::scheduler::kEvenSplit;
  else if (s == "CHBL")
    return sics::matrixgraph::core::components::scheduler::kCHBL;
  else if (s == "RoundRobin")
    return sics::matrixgraph::core::components::scheduler::kRoundRobin;
  return sics::matrixgraph::core::components::scheduler::kCHBL;
}

static bool ValidateParameters() {
  bool ok = true;
  if (FLAGS_p.empty()) {
    std::cerr << "Error: -p (pattern graph path) is required." << std::endl;
    ok = false;
  }
  if (FLAGS_g.empty()) {
    std::cerr << "Error: -g (data graph path) is required." << std::endl;
    ok = false;
  }
  return ok;
}

int main(int argc, char** argv) {
  gflags::SetUsageMessage(
      "LFTJ-style subgraph isomorphism counter (count-only, no materialization)"
      "\nUsage: " +
      std::string(argv[0]) +
      " -p <pattern_dir> -g <graph_dir> [-o <output>] [-limit <n>] "
      "[-canonical] [-disable_min_wise_filter] [-filter_hop <hop>] "
      "[-filter_k <k>] [-disable_matching_order]");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!ValidateParameters()) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "apps/lftj_subiso.cpp");
    return EXIT_FAILURE;
  }

  std::cout << "=== LFTJ SubIso Configuration ===" << std::endl;
  std::cout << "Pattern Graph: " << FLAGS_p << std::endl;
  std::cout << "Data Graph: " << FLAGS_g << std::endl;
  std::cout << "Output Path: " << (FLAGS_o.empty() ? "(none)" : FLAGS_o)
            << std::endl;
  std::cout << "Thread Count: " << FLAGS_t << std::endl;
  std::cout << "Count Limit: "
            << (FLAGS_limit == std::numeric_limits<uint64_t>::max()
                    ? "unlimited"
                    : std::to_string(FLAGS_limit))
            << std::endl;
  std::cout << "Canonical: " << (FLAGS_canonical ? "true" : "false")
            << std::endl;
  std::cout << "Min-Wise Filter: "
            << (FLAGS_disable_min_wise_filter ? "disabled" : "enabled")
            << std::endl;
  std::cout << "Filter Hop: " << FLAGS_filter_hop << std::endl;
  std::cout << "Filter K: " << FLAGS_filter_k << std::endl;
  std::cout << "Matching Order: "
            << (FLAGS_disable_matching_order ? "natural" : "greedy")
            << std::endl;
  std::cout << "=================================\n" << std::endl;

  try {
    auto scheduler_type = Scheduler2Enum("CHBL");
    sics::matrixgraph::core::MatrixGraph system(scheduler_type);

    auto* task = new sics::matrixgraph::core::task::LFTJSubIso(
        FLAGS_p, FLAGS_g, FLAGS_o, FLAGS_t, FLAGS_limit, FLAGS_canonical,
        !FLAGS_disable_min_wise_filter, FLAGS_filter_hop, FLAGS_filter_k,
        FLAGS_disable_matching_order);
    system.Run(sics::matrixgraph::core::common::kLFTJSubIso, task);
    delete task;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
