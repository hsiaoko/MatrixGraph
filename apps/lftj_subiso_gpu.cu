#include <gflags/gflags.h>

#include <iostream>
#include <string>

#include "core/common/types.h"
#include "core/components/scheduler/scheduler.h"
#include "core/matrixgraph.cuh"
#include "core/task/gpu_task/lftj_subiso_gpu.cuh"

DEFINE_string(p, "", "Path to the pattern graph directory (required)");
DEFINE_string(g, "", "Path to the data graph directory (required)");
DEFINE_string(o, "", "Path for optional output (default: none)");
DEFINE_int32(t, 0,
             "Number of GPU threads; 0 means use default (default: 0)");
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
DEFINE_bool(disable_ldf_filter, true,
            "Disable label-degree filter (directed out/in degree check). "
            "Default is disabled because LFTJ matches undirected edges; "
            "only enable (set to false) when the CSR is symmetric/directed. "
            "(default: true)");
DEFINE_bool(disable_nlc_filter, false,
            "Disable neighborhood-label-count filter (default: false)");
DEFINE_bool(disable_lpf_filter, false,
            "Disable label-pair-frequency filter (default: false)");
DEFINE_bool(disable_lcf_filter, false,
            "Disable global label-count filter (default: false)");
DEFINE_bool(disable_bloom_filter, false,
            "Disable Bloom neighbor-label-set filter (default: false)");
DEFINE_bool(disable_min_wise_bloom_filter, false,
            "Disable k-min-wise Bloom filter (default: false)");

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
      "GPU LFTJ-style subgraph isomorphism counter (count-only).\n"
      "Usage: " +
      std::string(argv[0]) +
      " -p <pattern_dir> -g <graph_dir> [-t <threads>] [-o <output>] "
      "[-canonical] [-disable_min_wise_filter] [-filter_hop <hop>] "
      "[-filter_k <k>] [-disable_matching_order] [-disable_ldf_filter] "
      "[-disable_nlc_filter] [-disable_lpf_filter] [-disable_lcf_filter] "
      "[-disable_bloom_filter] [-disable_min_wise_bloom_filter]");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!ValidateParameters()) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "apps/lftj_subiso_gpu.cu");
    return EXIT_FAILURE;
  }

  std::cout << "=== LFTJ SubIso GPU Configuration ===" << std::endl;
  std::cout << "Pattern Graph: " << FLAGS_p << std::endl;
  std::cout << "Data Graph: " << FLAGS_g << std::endl;
  std::cout << "Output Path: " << (FLAGS_o.empty() ? "(none)" : FLAGS_o)
            << std::endl;
  std::cout << "GPU Threads: " << (FLAGS_t == 0 ? "default" : std::to_string(FLAGS_t))
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
  std::cout << "LDF Filter: "
            << (FLAGS_disable_ldf_filter ? "disabled" : "enabled") << std::endl;
  std::cout << "NLC Filter: "
            << (FLAGS_disable_nlc_filter ? "disabled" : "enabled") << std::endl;
  std::cout << "LPF Filter: "
            << (FLAGS_disable_lpf_filter ? "disabled" : "enabled") << std::endl;
  std::cout << "LCF Filter: "
            << (FLAGS_disable_lcf_filter ? "disabled" : "enabled") << std::endl;
  std::cout << "Bloom Filter: "
            << (FLAGS_disable_bloom_filter ? "disabled" : "enabled")
            << std::endl;
  std::cout << "Min-Wise Bloom Filter: "
            << (FLAGS_disable_min_wise_bloom_filter ? "disabled" : "enabled")
            << std::endl;
  std::cout << "=====================================\n" << std::endl;

  try {
    auto scheduler_type = Scheduler2Enum("CHBL");
    sics::matrixgraph::core::MatrixGraph system(scheduler_type);

    auto* task = new sics::matrixgraph::core::task::LFTJSubIsoGpu(
        FLAGS_p, FLAGS_g, FLAGS_o, FLAGS_t, FLAGS_canonical,
        !FLAGS_disable_min_wise_filter, FLAGS_filter_hop, FLAGS_filter_k,
        FLAGS_disable_matching_order, !FLAGS_disable_ldf_filter,
        !FLAGS_disable_nlc_filter, !FLAGS_disable_lpf_filter,
        !FLAGS_disable_lcf_filter, !FLAGS_disable_bloom_filter,
        !FLAGS_disable_min_wise_bloom_filter);
    system.Run(sics::matrixgraph::core::common::kLFTJSubIsoGpu, task);
    delete task;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
