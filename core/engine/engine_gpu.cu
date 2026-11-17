#include <algorithm>

#include "core/engine/engine.h"
#include "core/engine/engine_internal.h"
#include "core/task/cpu_task/cpu_task_base.h"
#include "core/task/gpu_task/task_base.cuh"
#include "core/util/cuda_check.cuh"
#include "core/util/cuda_device.cuh"
#include "core/util/error.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace engine {
namespace detail {
namespace {

class GpuEngine : public Engine {
 public:
  explicit GpuEngine(const EngineOptions& options) {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
      throw util::Error(util::ErrorCode::kCuda, "GpuEngine: no CUDA devices");
    }
    devices_ = options.devices.empty()
                   ? util::MatrixGraphCudaDeviceList()
                   : options.devices;
    if (devices_.empty()) devices_.push_back(util::MatrixGraphCudaDevice());
    for (int device : devices_) {
      if (device < 0 || device >= device_count) {
        throw util::Error(util::ErrorCode::kInvalidArgument,
                          "GpuEngine: device index " + std::to_string(device) +
                              " out of range");
      }
    }
    num_streams_ = options.num_streams > 0
                       ? options.num_streams
                       : util::MatrixGraphCudaStreamsPerGpu();
  }

  Backend backend() const override { return Backend::kGpu; }

  void Run(task::TaskBase& task) override {
    // Tasks drive their own per-GPU stream matrices for now; the engine
    // guarantees the primary device context and grows the stream pool when
    // tasks start borrowing it (M4).
    CUDA_CHECK(cudaSetDevice(devices_.front()));
    task.Run();
  }

  void Run(task::CPUTaskBase&) override {
    throw util::Error(util::ErrorCode::kUnsupported,
                      "GPU engine cannot run a CPU task");
  }

 private:
  std::vector<int> devices_;
  uint32_t num_streams_ = 2;
};

}  // namespace

std::unique_ptr<Engine> MakeGpuEngine(const EngineOptions& options) {
  return std::make_unique<GpuEngine>(options);
}

}  // namespace detail
}  // namespace engine
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics