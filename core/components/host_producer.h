#ifndef HYPERBLOCKER_CORE_COMPONENTS_HOST_PRODUCER_H_
#define HYPERBLOCKER_CORE_COMPONENTS_HOST_PRODUCER_H_

#include <cuda_runtime.h>
#include <unistd.h>

#include <climits>
#include <condition_variable>
#include <mutex>

#include "core/common/types.h"
#include "core/components/execution_plan_generator.h"
#include "core/components/scheduler/CHBL_scheduler.h"
#include "core/components/scheduler/even_split_scheduler.h"
#include "core/components/scheduler/round_robin_scheduler.h"
#include "core/data_structures/match.h"
#include "core/data_structures/table.h"
#include "core/gpu/global_func.cuh"
#include "core/gpu/host_func.cuh"
#include "core/gpu/kernel_data_structures/kernel_bitmap.cuh"
#include "core/gpu/kernel_data_structures/kernel_table.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace components {

class HostProducer {

  // using ColumnMajor = cutlass::layout::ColumnMajor;
  // using RowMajor = cutlass::layout::RowMajor;
  // using CutlassGemm = cutlass::gemm::device::Gemm<float, RowMajor, float,
  //                                                 RowMajor, float, RowMajor>;

public:
  HostProducer(int n_partitions, DataMngr *data_mngr,
               ExecutionPlanGenerator *epg, scheduler::Scheduler *scheduler,
               std::unordered_map<int, cudaStream_t *> *p_streams,
               std::mutex *p_streams_mtx,
               // Match *p_match,
               std::unique_lock<std::mutex> *p_hr_start_lck,
               std::condition_variable *p_hr_start_cv, bool *p_hr_terminable,
               int prefix_hash_predicate_index = INT_MAX)
      : n_partitions_(n_partitions), data_mngr_(data_mngr), epg_(epg),
        p_streams_(p_streams),
        p_streams_mtx_(p_streams_mtx), // p_match_(p_match),
        prefix_hash_predicate_index_(prefix_hash_predicate_index),
        p_hr_start_lck_(p_hr_start_lck), p_hr_start_cv_(p_hr_start_cv),
        scheduler_(scheduler), p_hr_terminable_(p_hr_terminable) {

    cudaGetDeviceCount(&n_device_);
  }

  void Run() {
    std::cout << "Host Producer running on " << n_device_ << " devices."
              << std::endl;

    for (size_t i = 0; i < 1; i++) {
      auto bin_id = 0;
      cudaSetDevice(bin_id);
      cudaStream_t *p_stream = new cudaStream_t;
      cudaStreamCreate(p_stream);

      auto start_time = std::chrono::system_clock::now();
      // AsyncSubmit(*p_stream);
      Submit();

      auto end_time = std::chrono::system_clock::now();

      std::cout << "Task elapsed: "
                << std::chrono::duration_cast<std::chrono::microseconds>(
                       end_time - start_time)
                           .count() /
                       (double)CLOCKS_PER_SEC
                << std::endl;

      std::lock_guard<std::mutex> lock(*p_streams_mtx_);
      p_streams_->insert(std::make_pair(i, p_stream));
      p_hr_start_cv_->notify_all();
    }

    *p_hr_terminable_ = true;
  }

  __host__ void Submit() {
    int M = 10000;
    int N = 10000;
    int K = 10000;

    float alpha = 1.0f;
    float beta = 1.0f;

    std::cout << "Submit." << std::endl;

    auto start_time = std::chrono::system_clock::now();
    cudaError_t result =
        sics::matrixgraph::core::gpu::cuBLASGemm_host(M, N, K, alpha, beta);
    auto end_time = std::chrono::system_clock::now();
    std::cout << "Submit() elapsed: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     end_time - start_time)
                         .count() /
                     (double)CLOCKS_PER_SEC
              << std::endl;
  }

private:
  int n_device_ = 0;
  int n_partitions_ = 1;

  int prefix_hash_predicate_index_ = INT_MAX;

  std::unique_lock<std::mutex> *p_hr_start_lck_;
  std::condition_variable *p_hr_start_cv_;

  scheduler::Scheduler *scheduler_;
  DataMngr *data_mngr_;
  ExecutionPlanGenerator *epg_;

  std::unordered_map<int, cudaStream_t *> *p_streams_;
  std::mutex *p_streams_mtx_;

  // Match *p_match_;

  bool *p_hr_terminable_;
};

} // namespace components
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // HYPERBLOCKER_CORE_COMPONENTS_HOST_PRODUCER_CUH_
