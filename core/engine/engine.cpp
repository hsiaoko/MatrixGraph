#include "core/engine/engine.h"

#include "core/engine/engine_internal.h"
#include "core/util/error.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace engine {

std::unique_ptr<Engine> MakeEngine(Backend backend,
                                   const EngineOptions& options) {
  switch (backend) {
    case Backend::kCpuSerial: {
      EngineOptions serial = options;
      serial.max_threads = 1;
      return detail::MakeCpuEngine(serial);
    }
    case Backend::kCpuParallel:
      return detail::MakeCpuEngine(options);
    case Backend::kGpu:
      return detail::MakeGpuEngine(options);
  }
  throw util::Error(util::ErrorCode::kInvalidArgument,
                    "MakeEngine: unknown backend");
}

}  // namespace engine
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics