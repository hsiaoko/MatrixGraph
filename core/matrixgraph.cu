#include "core/matrixgraph.cuh"

#include <ctime>

#include "core/task/gemm.cuh"

namespace sics {
namespace matrixgraph {
namespace core {

MatrixGraph::MatrixGraph(SchedulerType scheduler_type) {

  PrintDeviceInfo();

  CUDA_CHECK(cudaGetDeviceCount(&n_device_));

  switch (scheduler_type) {
  case components::scheduler::kCHBL:
    scheduler_ = new components::scheduler::CHBLScheduler(n_device_);
    break;
  default:
    break;
  }
}

void MatrixGraph::Run(GPUTaskType task_type, TaskBase *task_ptr) {

  auto start_time = std::chrono::system_clock::now();

  auto prepare_end_time = std::chrono::system_clock::now();
  switch (task_type) {
  case task::kGEMM: {
    auto task = reinterpret_cast<task::GEMM *>(task_ptr);
    task->Run();
    break;
  }
  case task::kMatrixAnalysis:
    break;
  case task::kPPRQuery: {
    std::cout << "[PPR Query]" << std::endl;
    auto task = reinterpret_cast<task::PPRQuery *>(task_ptr);
    task->Run();
    break;
  }
  default:
    break;
  }

  auto end_time = std::chrono::system_clock::now();

  std::cout << "MatrixGraph.Run() elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                     start_time)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;
}

void MatrixGraph::PrintDeviceInfo() const {
  int dev = 0;
  cudaDeviceProp devProp;
  CUDA_CHECK(cudaGetDeviceCount(&dev));
  std::cout << "Device Info for " << dev << " devices." << std::endl;
  for (int i = 0; i < dev; i++) {
    cudaGetDeviceProperties(&devProp, i);
    std::cout << "Device " << i << ": " << devProp.name << std::endl;
    std::cout << "multiProcessorCount: " << devProp.multiProcessorCount
              << std::endl;
    std::cout << "sharedMemPerBlock: " << devProp.sharedMemPerBlock / 1024.0
              << " KB" << std::endl;
    std::cout << "maxThreadsPerBlock：" << devProp.maxThreadsPerBlock
              << std::endl;
    std::cout << "maxThreadsPerMultiProcessor："
              << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << std::endl;
  }
}

} // namespace core
} // namespace matrixgraph
} // namespace sics