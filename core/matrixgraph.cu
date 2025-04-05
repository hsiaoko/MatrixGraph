#include "core/matrixgraph.cuh"
#include "core/task/cpu_task/cpu_subiso.cuh"
#include "core/task/gpu_task/bfs.cuh"
#include "core/task/gpu_task/gemm.cuh"
#include "core/task/gpu_task/gemv.cuh"
#include "core/task/gpu_task/pagerank.cuh"
#include "core/task/gpu_task/ppr_query.cuh"
#include "core/task/gpu_task/subiso.cuh"
#include "core/task/gpu_task/wcc.cuh"
#include <chrono>
#include <ctime>

namespace sics {
namespace matrixgraph {
namespace core {

MatrixGraph::MatrixGraph(SchedulerType scheduler_type) {
  CUDA_CHECK(cudaGetDeviceCount(&n_device_));

  switch (scheduler_type) {
    case components::scheduler::kCHBL:
      scheduler_ = new components::scheduler::CHBLScheduler(n_device_);
      break;
    default:
      break;
  }
}

void MatrixGraph::Run(TaskType task_type, TaskBase* task_ptr) {
  PrintDeviceInfo();
  auto start_time = std::chrono::system_clock::now();

  auto prepare_end_time = std::chrono::system_clock::now();
  switch (task_type) {
    case common::kGEMM: {
      auto task = reinterpret_cast<task::GEMM*>(task_ptr);
      task->Run();
      break;
    }
    case common::kSubIso: {
      std::cout << "SubIso Query" << std::endl;
      auto task = reinterpret_cast<task::SubIso*>(task_ptr);
      task->Run();
      break;
    }
    case common::kPPRQuery: {
      std::cout << "[PPR Query]" << std::endl;
      auto task = reinterpret_cast<task::PPRQuery*>(task_ptr);
      task->Run();
      break;
    }
    case common::kWCC: {
      std::cout << "[WCC Query]" << std::endl;
      auto task = reinterpret_cast<task::WCC*>(task_ptr);
      task->Run();
      break;
    }
    case common::kPageRank: {
      std::cout << "[PageRank Query]" << std::endl;
      auto task = reinterpret_cast<task::PageRank*>(task_ptr);
      task->Run();
      break;
    }
    case common::kBFS: {
      std::cout << "[BFS Traverse]" << std::endl;
      auto task = reinterpret_cast<task::BFS*>(task_ptr);
      task->Run();
      break;
    }
    case common::kGEMV: {
      std::cout << "[GEMV Traverse]" << std::endl;
      auto task = reinterpret_cast<task::GEMV*>(task_ptr);
      task->Run();
      break;
    }
    default:
      break;
  }

  auto end_time = std::chrono::system_clock::now();

  std::cout << "MatrixGraph GPU Run elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                     start_time)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;
}

void MatrixGraph::Run(TaskType task_type, CPUTaskBase* task_ptr) {
  PrintDeviceInfo();
  auto start_time = std::chrono::system_clock::now();

  auto prepare_end_time = std::chrono::system_clock::now();
  switch (task_type) {
    case common::kCPUSubIso: {
      std::cout << "[SubIso Query]" << std::endl;
      auto task = reinterpret_cast<task::CPUSubIso*>(task_ptr);
      task->Run();
      break;
    }
    default:
      break;
  }

  auto end_time = std::chrono::system_clock::now();

  std::cout << "MatrixGraph CPU Run() elapsed: "
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
  size_t size;
  cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
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
    std::cout << "defaultCUDAHeapSize：" << size << std::endl;
    std::cout << std::endl;
  }
}

}  // namespace core
}  // namespace matrixgraph
}  // namespace sics