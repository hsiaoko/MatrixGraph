#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <gflags/gflags.h>

#include <fstream>
#include <iostream>
#include <list>
#include <utility>

#include "core/common/types.h"
#include "core/common/yaml_config.h"
#include "apps/backend_flag.h"
#include "core/engine/engine.h"
#include "core/task/gpu_task/gemm.cuh"
#include "core/task/gpu_task/task_base.cuh"

DEFINE_string(i, "", "input data dir path for graph.");
DEFINE_string(it, "", "input data dir path for transposed graph.");
DEFINE_string(o, "", "output path.");
DEFINE_int32(count, 1, "count");
DEFINE_string(backend, "gpu",
              "Execution backend (options: gpu, cpu, cpu_serial; "
              "default: gpu)");

using sics::matrixgraph::core::task::GEMM;


int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const auto backend = mg_app::ParseBackend(FLAGS_backend);
  auto engine = sics::matrixgraph::core::engine::MakeEngine(backend);

  auto* gemm = new GEMM(FLAGS_i, FLAGS_it, FLAGS_o, FLAGS_count);

  // State which application is going to be running.
  engine->Run(*gemm);

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
