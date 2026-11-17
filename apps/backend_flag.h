#ifndef MATRIXGRAPH_APPS_BACKEND_FLAG_H_
#define MATRIXGRAPH_APPS_BACKEND_FLAG_H_

#include "core/engine/engine.h"
#include "core/util/error.h"

#include <string>

namespace mg_app {

// Single parser for the -backend flag every executable shares, so the mapping
// lives in one place instead of fourteen copies of Scheduler2Enum.
inline sics::matrixgraph::core::engine::Backend ParseBackend(
    const std::string& name) {
  using sics::matrixgraph::core::engine::Backend;
  if (name == "gpu") return Backend::kGpu;
  if (name == "cpu") return Backend::kCpuParallel;
  if (name == "cpu_serial" || name == "serial") return Backend::kCpuSerial;
  throw sics::matrixgraph::core::util::Error(
      sics::matrixgraph::core::util::ErrorCode::kInvalidArgument,
      "unknown -backend value '" + name + "' (options: gpu, cpu, cpu_serial)");
}

}  // namespace mg_app

#endif  // MATRIXGRAPH_APPS_BACKEND_FLAG_H_