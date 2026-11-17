#include <gflags/gflags.h>

#include <fstream>
#include <iostream>
#include <list>
#include <string>
#include <utility>

#include "core/common/types.h"
#include "core/common/yaml_config.h"
#include "apps/backend_flag.h"
#include "core/engine/engine.h"
#include "core/task/gpu_task/task_base.cuh"
#include "core/task/gpu_task/wcc.cuh"

// Input/Output flags
DEFINE_string(g, "", "Path to the input graph file (required)");

// System configuration
DEFINE_string(backend, "gpu",
              "Execution backend (options: gpu, cpu, cpu_serial; "
              "default: gpu)");

using sics::matrixgraph::core::task::WCC;

bool ValidateParameters() {
  bool is_valid = true;

  // Check required parameters
  if (FLAGS_g.empty()) {
    std::cerr << "Error: Input graph path (-g) is required" << std::endl;
    is_valid = false;
  }

  return is_valid;
}

void PrintConfig() {
  std::cout << "\n=== WCC Configuration ===" << std::endl;
  std::cout << "Input Graph: " << FLAGS_g << std::endl;
    std::cout << "=======================\n" << std::endl;
}

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "Weakly Connected Components computation using MatrixGraph\n"
      "Usage: " +
      std::string(argv[0]) + " -g <graph_path> [options]");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!ValidateParameters()) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "apps/wcc.cu");
    return EXIT_FAILURE;
  }

  PrintConfig();

  try {
    const auto backend = mg_app::ParseBackend(FLAGS_backend);
    auto engine = sics::matrixgraph::core::engine::MakeEngine(backend);

    auto* task = new WCC(FLAGS_g);
    engine->Run(*task);
    delete task;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
