#include <chrono>
#include <iostream>

#include "core/matrixgraph.cuh"
#include "core/util/error.h"

namespace sics {
namespace matrixgraph {
namespace core {

MatrixGraph::MatrixGraph(SchedulerType /*scheduler_type*/) {
  // The scheduler parameter predates the engine layer; apps still pass
  // -scheduler and M4 migrates them to -backend. Each facade owns one engine
  // per kind so Run enters the right execution context.
  gpu_engine_ = engine::MakeEngine(engine::Backend::kGpu);
  cpu_engine_ = engine::MakeEngine(engine::Backend::kCpuParallel);
}

void MatrixGraph::Run(TaskBase* task_ptr) {
  if (task_ptr == nullptr) {
    throw util::Error(util::ErrorCode::kInvalidArgument,
                      "MatrixGraph::Run: null task");
  }
  auto start_time = std::chrono::system_clock::now();
  gpu_engine_->Run(*task_ptr);
  auto end_time = std::chrono::system_clock::now();
  std::cout << "MatrixGraph GPU Run elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   end_time - start_time)
                       .count() /
                   1e6
            << " s" << std::endl;
}

void MatrixGraph::Run(CPUTaskBase* task_ptr) {
  if (task_ptr == nullptr) {
    throw util::Error(util::ErrorCode::kInvalidArgument,
                      "MatrixGraph::Run: null task");
  }
  auto start_time = std::chrono::system_clock::now();
  cpu_engine_->Run(*task_ptr);
  auto end_time = std::chrono::system_clock::now();
  std::cout << "MatrixGraph CPU Run elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   end_time - start_time)
                       .count() /
                   1e6
            << " s" << std::endl;
}

// Deprecated shims: apps still pass TaskType and -scheduler. The type was
// only ever a manual substitute for virtual dispatch; M4 migrates the apps
// to Run(task) plus -backend and removes these overloads and TaskType.
void MatrixGraph::Run(TaskType /*task_type*/, TaskBase* task_ptr) {
  Run(task_ptr);
}

void MatrixGraph::Run(TaskType /*task_type*/, CPUTaskBase* task_ptr) {
  Run(task_ptr);
}

void MatrixGraph::PrintDeviceInfo() const {
  int dev = 0;
  cudaDeviceProp dev_prop;
  CUDA_CHECK(cudaGetDeviceCount(&dev));
  std::cout << "Device Info for " << dev << " devices." << std::endl;
  size_t size;
  cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
  for (int i = 0; i < dev; i++) {
    cudaGetDeviceProperties(&dev_prop, i);
    std::cout << "Device " << i << ": " << dev_prop.name << std::endl;
    std::cout << "multiProcessorCount: " << dev_prop.multiProcessorCount
              << std::endl;
    std::cout << "sharedMemPerBlock: " << dev_prop.sharedMemPerBlock / 1024.0
              << " KB" << std::endl;
    std::cout << "maxThreadsPerBlock：" << dev_prop.maxThreadsPerBlock
              << std::endl;
    std::cout << "maxThreadsPerMultiProcessor："
              << dev_prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "defaultCUDAHeapSize：" << size << std::endl;
    std::cout << std::endl;
  }
}

}  // namespace core
}  // namespace matrixgraph
}  // namespace sics