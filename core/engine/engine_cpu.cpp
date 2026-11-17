#include "core/engine/engine.h"
#include "core/engine/engine_internal.h"
#include "core/task/cpu_task/cpu_task_base.h"
#include "core/util/error.h"

#include <algorithm>
#include <oneapi/tbb/global_control.h>
#include <thread>

namespace sics {
namespace matrixgraph {
namespace core {
namespace engine {
namespace detail {
namespace {

class CpuEngine : public Engine {
 public:
  explicit CpuEngine(const EngineOptions& options)
      : threads_(options.max_threads > 0
                     ? options.max_threads
                     : std::max(1u, std::thread::hardware_concurrency())) {
    if (IsParallel()) {
      // global_control is process-wide and nests; an engine scopes its cap
      // for its lifetime. Per-engine arenas are the documented refinement.
      control_ = std::make_unique<oneapi::tbb::global_control>(
          oneapi::tbb::global_control::max_allowed_parallelism, threads_);
    }
  }

  Backend backend() const override {
    return IsParallel() ? Backend::kCpuParallel : Backend::kCpuSerial;
  }

  void Run(task::TaskBase&) override {
    throw util::Error(util::ErrorCode::kUnsupported,
                      "CPU engine cannot run a GPU task");
  }

  void Run(task::CPUTaskBase& task) override { task.Run(); }

 private:
  bool IsParallel() const { return threads_ > 1; }

  uint32_t threads_;
  std::unique_ptr<oneapi::tbb::global_control> control_;
};

}  // namespace

std::unique_ptr<Engine> MakeCpuEngine(const EngineOptions& options) {
  return std::make_unique<CpuEngine>(options);
}

}  // namespace detail
}  // namespace engine
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics