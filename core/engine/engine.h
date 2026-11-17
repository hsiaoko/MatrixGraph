#ifndef MATRIXGRAPH_CORE_ENGINE_ENGINE_H_
#define MATRIXGRAPH_CORE_ENGINE_ENGINE_H_

#include <cstdint>
#include <memory>
#include <vector>

namespace sics {
namespace matrixgraph {
namespace core {

namespace task {
class CPUTaskBase;
class TaskBase;
}  // namespace task

namespace engine {

/**
 * The three execution intents the library supports. Backend selection is an
 * explicit caller choice (MakeEngine), never a compiler macro: a .cu suffix
 * used to decide whether host code ran in parallel, which surprised every
 * reader of the code.
 */
enum class Backend {
  kCpuSerial,
  kCpuParallel,
  kGpu,
};

struct EngineOptions {
  // GPU logical device indices; empty means the MATRIXGRAPH_CUDA_* env
  // contract decides (see docs/MATRIXGRAPH_ENV.md).
  std::vector<int> devices;
  // Streams per device for overlap; 0 keeps the env/default (2, minimum 1).
  uint32_t num_streams = 0;
  // CPU worker cap; 0 means hardware concurrency. kCpuSerial forces 1.
  uint32_t max_threads = 0;
};

/**
 * Owns one backend's execution resources and runs tasks inside that context:
 * CPU engines install the thread cap, GPU engines select the device. Tasks
 * borrow the context; the engine outlives every task it runs.
 *
 * A task handed to the wrong engine is a programming error and throws
 * util::Error(kUnsupported) rather than silently doing nothing.
 */
class Engine {
 public:
  virtual ~Engine() = default;

  virtual Backend backend() const = 0;

  virtual void Run(task::TaskBase& task) = 0;

  virtual void Run(task::CPUTaskBase& task) = 0;
};

std::unique_ptr<Engine> MakeEngine(Backend backend,
                                   const EngineOptions& options = {});

}  // namespace engine
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_ENGINE_ENGINE_H_