#ifndef HYPERBLOCKER_CORE_HYPER_BLOCKER_CUH_
#define HYPERBLOCKER_CORE_HYPER_BLOCKER_CUH_

#include <condition_variable>
#include <ctime>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <experimental/filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/components/data_mngr/data_mngr_base.h"
#include "core/components/data_mngr/tiled_matrix_mngr.h"
#include "core/components/execution_plan_generator.h"
#include "core/components/host_producer.h"
#include "core/components/host_reducer.h"
#include "core/components/scheduler/CHBL_scheduler.h"
#include "core/components/scheduler/even_split_scheduler.h"
#include "core/components/scheduler/round_robin_scheduler.h"
#include "core/components/scheduler/scheduler.h"

namespace sics {
namespace matrixgraph {
namespace core {

class MatrixGraph {
private:
  using DataMngr = sics::matrixgraph::core::components::DataMngrBase;
  using ExecutionPlanGenerator =
      sics::matrixgraph::core::components::ExecutionPlanGenerator;
  using HostProducer = sics::matrixgraph::core::components::HostProducer;
  using HostReducer = sics::matrixgraph::core::components::HostReducer;
  using TiledMatrixMngr = sics::matrixgraph::core::components::TiledMatrixMngr;

public:
  MatrixGraph() = delete;

  MatrixGraph(const std::string &data_path, const std::string &pattern_path,
              sics::matrixgraph::core::components::scheduler::SchedulerType
                  scheduler_type = sics::matrixgraph::core::components::
                      scheduler::SchedulerType::kCHBL)
      : data_path_(data_path), pattern_path_(pattern_path) {

    Init();

    auto start_time = std::chrono::system_clock::now();

    p_streams_mtx_ = std::make_unique<std::mutex>();

    p_hr_start_mtx_ = std::make_unique<std::mutex>();
    p_hr_start_lck_ =
        std::make_unique<std::unique_lock<std::mutex>>(*p_hr_start_mtx_.get());
    p_hr_start_cv_ = std::make_unique<std::condition_variable>();
    streams_ = std::make_unique<std::unordered_map<int, cudaStream_t *>>();

    switch (scheduler_type) {
    case sics::matrixgraph::core::components::scheduler::SchedulerType::kCHBL:
      scheduler_ =
          std::make_unique<components::scheduler::CHBLScheduler>(n_device_);
      break;
    }

    data_mngr_ = std::make_unique<TiledMatrixMngr>(data_path_);
    execution_plan_generator_ =
        std::make_unique<ExecutionPlanGenerator>(pattern_path_);

    execution_plan_generator_->GenerateExecutionPlan();

    p_hr_terminable_ = std::make_unique<bool>(false);

    auto end_time = std::chrono::system_clock::now();
    std::cout << "MatrixGraph.Initialize() elapsed: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     end_time - start_time)
                         .count() /
                     (double)CLOCKS_PER_SEC
              << std::endl;
  }

  ~MatrixGraph() = default;

  void Run() {

    auto start_time = std::chrono::system_clock::now();

    // ShowDeviceProperties();
    HostProducer hp(data_mngr_.get(), scheduler_.get(), streams_.get(),
                    p_streams_mtx_.get(), p_hr_start_lck_.get(),
                    p_hr_start_cv_.get(), p_hr_terminable_.get());
    HostReducer hr(scheduler_.get(), streams_.get(), p_streams_mtx_.get(),
                   p_hr_start_lck_.get(), p_hr_start_cv_.get(),
                   p_hr_terminable_.get());

    std::thread hp_thread(&HostProducer::Run, &hp);
    std::thread hr_thread(&HostReducer::Run, &hr);
    auto prepare_end_time = std::chrono::system_clock::now();

    hp_thread.join();
    hr_thread.join();

    auto end_time = std::chrono::system_clock::now();

    std::cout << "MatrixGraph.Run() elapsed: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     end_time - start_time)
                         .count() /
                     (double)CLOCKS_PER_SEC
              << std::endl;
  }

  void ShowDeviceProperties() {
    cudaError_t cudaStatus;
    std::cout << "Device properties" << std::endl;
    int dev = 0;
    cudaDeviceProp devProp;
    cudaStatus = cudaGetDeviceCount(&dev);
    printf("error %d\n", cudaStatus);
    // if (cudaStatus) return;
    for (int i = 0; i < dev; i++) {
      cudaGetDeviceProperties(&devProp, i);
      std::cout << "Device " << dev << ": " << devProp.name << std::endl;
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
    n_device_ = dev;
  }

  void Init() {
    cudaError_t cudaStatus;
    int dev = 0;
    cudaDeviceProp devProp;
    cudaStatus = cudaGetDeviceCount(&dev);
    n_device_ = dev;
  }

private:
  int n_device_ = 0;

  const std::string data_path_;
  const std::string pattern_path_;

  std::unique_ptr<std::mutex> p_streams_mtx_;

  std::unique_ptr<std::mutex> p_hr_start_mtx_;
  std::unique_ptr<std::unique_lock<std::mutex>> p_hr_start_lck_;
  std::unique_ptr<std::condition_variable> p_hr_start_cv_;

  std::unique_ptr<DataMngr> data_mngr_;
  std::unique_ptr<ExecutionPlanGenerator> execution_plan_generator_;

  std::unique_ptr<std::unordered_map<int, cudaStream_t *>> streams_;

  std::unique_ptr<components::scheduler::Scheduler> scheduler_;

  std::unique_ptr<bool> p_hr_terminable_;
};

} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // HYPERBLOCKER_CORE_HYPER_BLOCKER_CUH_
