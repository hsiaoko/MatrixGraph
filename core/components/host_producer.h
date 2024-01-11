#ifndef HYPERBLOCKER_CORE_COMPONENTS_HOST_PRODUCER_H_
#define HYPERBLOCKER_CORE_COMPONENTS_HOST_PRODUCER_H_

#include <cuda_runtime.h>
#include <unistd.h>

#include <climits>
#include <condition_variable>
#include <mutex>

#include "core/common/types.h"
#include "core/components/scheduler/CHBL_scheduler.h"
#include "core/components/scheduler/even_split_scheduler.h"
#include "core/components/scheduler/round_robin_scheduler.h"
#include "core/gpu/global_func.cuh"
#include "core/gpu/host_func.cuh"
#include "core/gpu/kernel_data_structures/kernel_bitmap.cuh"
#include "core/gpu/kernel_data_structures/kernel_table.cuh"
#include "core/util/set_operations.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace components {

class HostProducer {
private:
  using DataMngr = sics::matrixgraph::core::components::DataMngrBase;
  using TiledMatrixMngr = sics::matrixgraph::core::components::TiledMatrixMngr;
  using VertexID = sics::matrixgraph::core::common::VertexID;

public:
  HostProducer(DataMngr *data_mngr, scheduler::Scheduler *scheduler,
               std::unordered_map<int, cudaStream_t *> *p_streams,
               std::mutex *p_streams_mtx,
               std::unique_lock<std::mutex> *p_hr_start_lck,
               std::condition_variable *p_hr_start_cv, bool *p_hr_terminable)
      : data_mngr_(data_mngr), p_streams_(p_streams),
        p_streams_mtx_(p_streams_mtx), p_hr_start_lck_(p_hr_start_lck),
        p_hr_start_cv_(p_hr_start_cv), scheduler_(scheduler),
        p_hr_terminable_(p_hr_terminable) {

    cudaGetDeviceCount(&n_device_);
  }

  //  void Run() {
  //    std::cout << "Host Producer running on " << n_device_ << " devices."
  //              << std::endl;
  //
  //    for (size_t i = 0; i < 1; i++) {
  //      auto bin_id = 0;
  //      cudaSetDevice(bin_id);
  //      cudaStream_t *p_stream = new cudaStream_t;
  //      cudaStreamCreate(p_stream);
  //
  //      auto start_time = std::chrono::system_clock::now();
  //      // AsyncSubmit(*p_stream);
  //      // Submit();
  //      SubmitTask();
  //
  //      auto end_time = std::chrono::system_clock::now();
  //
  //      std::cout << "Task elapsed: "
  //                << std::chrono::duration_cast<std::chrono::microseconds>(
  //                       end_time - start_time)
  //                           .count() /
  //                       (double)CLOCKS_PER_SEC
  //                << std::endl;
  //
  //      std::lock_guard<std::mutex> lock(*p_streams_mtx_);
  //      p_streams_->insert(std::make_pair(i, p_stream));
  //      p_hr_start_cv_->notify_all();
  //    }
  //
  //    *p_hr_terminable_ = true;
  //  }

  __host__ void Run() {
    std::cout << "Host Producer running on " << n_device_ << " devices."
              << std::endl;
    TiledMatrixMngr *tiled_matrix_mngr =
        reinterpret_cast<TiledMatrixMngr *>(data_mngr_);
    auto tiled_matrix = tiled_matrix_mngr->GetTiledMatrixPtr();
    auto tiled_transposed_matrix =
        tiled_matrix_mngr->GetTransposedTiledMatrixPtr();

    tiled_matrix->Show();
    tiled_transposed_matrix->Show();

    assert(tiled_matrix->get_metadata().n_cols ==
           tiled_transposed_matrix->get_metadata().n_rows);

    auto max_val = tiled_matrix->get_metadata().n_cols >
                           tiled_transposed_matrix->get_metadata().n_rows
                       ? tiled_matrix->get_metadata().n_cols
                       : tiled_transposed_matrix->get_metadata().n_rows;

    for (VertexID i = 0; i < tiled_matrix->get_metadata().n_rows; i++) {

      // Create a new stream for each task.
      auto bin_id = i % n_device_;
      cudaSetDevice(bin_id);
      cudaStream_t *p_stream = new cudaStream_t;
      cudaStreamCreate(p_stream);

      auto tile_ptr = tiled_matrix->get_tile_ptr_by_id(i);
      auto tile_ptr_t = tiled_transposed_matrix->get_tile_ptr_by_id(i);

      auto tile_scope = tiled_matrix->get_tile_ptr_by_id(i + 1) - tile_ptr;
      auto tile_scope_t =
          tiled_transposed_matrix->get_tile_ptr_by_id(i + 1) - tile_ptr_t;

      auto &&intersection = sics::matrixgraph::core::util::set::GetIntersection(
          tiled_matrix->get_tile_col_idx_ptr() + tile_ptr, tile_scope,
          tiled_transposed_matrix->get_tile_col_idx_ptr() + tile_ptr_t,
          tile_scope_t, max_val);

      std::cout << "Intersection size: " << intersection.size() << std::endl;
      for (size_t j = 0; j < intersection.size(); j++) {
        std::cout << "Intersection: " << (intersection[j]).first << ", "
                  << (intersection[j]).second << std::endl;
        auto tile =
            tiled_matrix->GetTilebyIdx(tile_ptr + (intersection[j]).first);
        auto tile_t = tiled_transposed_matrix->GetTilebyIdx(
            tile_ptr_t + (intersection[j]).second);

        tile->Show();
        tile_t->Show();

        sics::matrixgraph::core::gpu::TiledMatrixGemm_host(*tile, *tile_t,
                                                           *p_stream);
        std::lock_guard<std::mutex> lock(*p_streams_mtx_);
        p_streams_->insert(std::make_pair(i, p_stream));
        p_hr_start_cv_->notify_all();
      }
    }
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

  std::unique_lock<std::mutex> *p_hr_start_lck_;
  std::condition_variable *p_hr_start_cv_;

  scheduler::Scheduler *scheduler_;
  DataMngr *data_mngr_;

  std::unordered_map<int, cudaStream_t *> *p_streams_;
  std::mutex *p_streams_mtx_;

  bool *p_hr_terminable_;
};

} // namespace components
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // HYPERBLOCKER_CORE_COMPONENTS_HOST_PRODUCER_CUH_
