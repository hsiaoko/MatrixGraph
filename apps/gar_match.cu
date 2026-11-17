#include <gflags/gflags.h>

#include <cuda_runtime.h>
#include <iostream>

#include "apps/backend_flag.h"
#include "core/engine/engine.h"
#include "core/task/gpu_task/gar_match.cuh"

// Input/Output flags
DEFINE_string(config, "", "Path to GARMatch config (required)");
DEFINE_string(o, "", "Path for output results (required)");

DEFINE_string(backend, "gpu",
              "Execution backend (options: gpu, cpu, cpu_serial; "
              "default: gpu)");
DEFINE_int32(device, 2, "CUDA device id to use (default: 0)");

using sics::matrixgraph::core::task::GARMatch;

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
  if (FLAGS_device < 0) {
    std::cerr << "Error: device id must be >= 0" << std::endl;
    is_valid = false;
  }
  return is_valid;
}

void PrintConfig() {
  std::cout << "\n=== GARMatch App Configuration ===" << std::endl;
  std::cout << "Config Path: " << FLAGS_config << std::endl;
  std::cout << "Output Path: " << FLAGS_o << std::endl;
    std::cout << "Device: " << FLAGS_device << std::endl;
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
    int device_count = 0;
    cudaError_t dev_err = cudaGetDeviceCount(&device_count);
    if (dev_err != cudaSuccess) {
      std::cerr << "Error: cudaGetDeviceCount failed: "
                << cudaGetErrorString(dev_err) << std::endl;
      return EXIT_FAILURE;
    }
    if (FLAGS_device >= device_count) {
      std::cerr << "Error: device id " << FLAGS_device
                << " is out of range, available devices: " << device_count
                << std::endl;
      return EXIT_FAILURE;
    }
    dev_err = cudaSetDevice(FLAGS_device);
    if (dev_err != cudaSuccess) {
      std::cerr << "Error: cudaSetDevice(" << FLAGS_device
                << ") failed: " << cudaGetErrorString(dev_err) << std::endl;
      return EXIT_FAILURE;
    }

    const auto backend = mg_app::ParseBackend(FLAGS_backend);
    auto engine = sics::matrixgraph::core::engine::MakeEngine(backend);

    auto* task = new GARMatch(FLAGS_config, FLAGS_o);
    engine->Run(*task);
    delete task;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
