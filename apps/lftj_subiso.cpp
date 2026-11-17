#include "core/task/cpu_task/lftj_subiso.cuh"

#include "apps/backend_flag.h"
#include "core/common/types.h"
#include "core/engine/engine.h"

#include <gflags/gflags.h>
#include <iostream>
#include <string>
#include <thread>

DEFINE_string(p, "", "Path to the pattern graph directory (required)");
DEFINE_string(g, "", "Path to the data graph directory (required)");
DEFINE_string(o, "", "Path for optional output (default: none)");
DEFINE_string(reject_output, "",
              "Path to write filtered (u,v) pairs: data vertices with the "
              "same label as pattern vertex u that did not enter candidates[u] "
              "(default: none)");
DEFINE_int32(t, 0,
             "Number of CPU threads; 0 means use all hardware cores (default: "
             "0 = all cores)");
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
DEFINE_string(backend, "cpu",
              "Execution backend (options: gpu, cpu, cpu_serial; "
              "default: cpu)");

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
      " -p <pattern_dir> -g <graph_dir> [-o <output>] [-reject_output <path>] "
      "[-t <threads>] [-limit <n>] [-canonical] [-disable_min_wise_filter] "
      "[-filter_hop <hop>] [-filter_k <k>] [-disable_matching_order] "
      "[-disable_ldf_filter] [-disable_nlc_filter] [-disable_lpf_filter] "
      "[-disable_lcf_filter] [-disable_bloom_filter] "
      "[-disable_min_wise_bloom_filter]");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!ValidateParameters()) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "apps/lftj_subiso.cpp");
    return EXIT_FAILURE;
  }

  int num_threads = FLAGS_t;
  if (num_threads <= 0) {
    num_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (num_threads <= 0) num_threads = 1;
  }

  std::cout << "=== LFTJ SubIso Configuration ===" << std::endl;
  std::cout << "Pattern Graph: " << FLAGS_p << std::endl;
  std::cout << "Data Graph: " << FLAGS_g << std::endl;
  std::cout << "Output Path: " << (FLAGS_o.empty() ? "(none)" : FLAGS_o)
            << std::endl;
  std::cout << "Reject Output Path: "
            << (FLAGS_reject_output.empty() ? "(none)" : FLAGS_reject_output)
            << std::endl;
  std::cout << "Thread Count: " << num_threads
            << (FLAGS_t == 0 ? " (auto = all cores)" : " (user-specified)")
            << std::endl;
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
  std::cout << "=================================\n" << std::endl;

  try {
    const auto backend = mg_app::ParseBackend(FLAGS_backend);
    auto engine = sics::matrixgraph::core::engine::MakeEngine(backend);

    auto* task = new sics::matrixgraph::core::task::LFTJSubIso(
        FLAGS_p, FLAGS_g, FLAGS_o, num_threads, FLAGS_limit, FLAGS_canonical,
        !FLAGS_disable_min_wise_filter, FLAGS_filter_hop, FLAGS_filter_k,
        FLAGS_disable_matching_order, !FLAGS_disable_ldf_filter,
        !FLAGS_disable_nlc_filter, !FLAGS_disable_lpf_filter,
        !FLAGS_disable_lcf_filter, !FLAGS_disable_bloom_filter,
        !FLAGS_disable_min_wise_bloom_filter, FLAGS_reject_output);
    engine->Run(*task);
    delete task;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
