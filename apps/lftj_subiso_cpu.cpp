#include <gflags/gflags.h>

#include <iostream>
#include <string>
#include <thread>

#include "core/common/types.h"
#include "core/task/cpu_task/lftj_subiso.cuh"

DEFINE_string(p, "", "Path to the pattern graph directory (required)");
DEFINE_string(g, "", "Path to the data graph directory (required)");
DEFINE_string(o, "", "Path for optional output (default: none)");
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
      "CPU-only LFTJ-style subgraph isomorphism counter.\n"
      "Directly runs the CPU task without GPU dispatch.\n"
      "Usage: " +
      std::string(argv[0]) +
      " -p <pattern_dir> -g <graph_dir> [-t <threads>] [-o <output>] "
      "[-limit <n>] [-canonical] [-disable_min_wise_filter] "
      "[-filter_hop <hop>] [-filter_k <k>]");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!ValidateParameters()) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "apps/lftj_subiso_cpu.cpp");
    return EXIT_FAILURE;
  }

  int num_threads = FLAGS_t;
  if (num_threads <= 0) {
    num_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (num_threads <= 0) num_threads = 1;
  }

  std::cout << "=== LFTJ SubIso CPU-only Configuration ===" << std::endl;
  std::cout << "Pattern Graph: " << FLAGS_p << std::endl;
  std::cout << "Data Graph: " << FLAGS_g << std::endl;
  std::cout << "Output Path: " << (FLAGS_o.empty() ? "(none)" : FLAGS_o)
            << std::endl;
  std::cout << "Threads: " << num_threads
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
  std::cout << "==========================================\n" << std::endl;

  try {
    sics::matrixgraph::core::task::LFTJSubIso task(
        FLAGS_p, FLAGS_g, FLAGS_o, num_threads, FLAGS_limit, FLAGS_canonical,
        !FLAGS_disable_min_wise_filter, FLAGS_filter_hop, FLAGS_filter_k);
    task.Run();
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
