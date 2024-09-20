#include "core/task/gemm.cuh"

#include <ctime>
#include <cuda_runtime.h>
#include <execution>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/host_buffer.cuh"
#include "core/data_structures/metadata.h"
#include "core/data_structures/unified_buffer.cuh"
#include "core/io/grid_csr_tiled_matrix_io.cuh"
#include "core/task/kernel/matrix_operations.cuh"
#include "core/util/atomic.h"
#include "core/util/bitmap.h"
#include "core/util/bitmap_no_ownership.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using sics::matrixgraph::core::data_structures::GridCSRTiledMatrix;
using sics::matrixgraph::core::io::GridCSRTiledMatrixIO;
using GirdTiledMatrix =
    sics::matrixgraph::core::data_structures::GridCSRTiledMatrix;
using DeviceOwnedBufferUint64 =
    sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<uint64_t>;
using DeviceOwnedBufferUint32 =
    sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<uint32_t>;
using UnifiedOwnedBufferUint32 =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint32_t>;
using UnifiedOwnedBufferUint64 =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint64_t>;
using BufferUint64 = sics::matrixgraph::core::data_structures::Buffer<uint64_t>;
using BufferUint32 = sics::matrixgraph::core::data_structures::Buffer<uint32_t>;
using DeviceOwnedBufferUint64 =
    sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<uint64_t>;
using MatrixOperationsKernelWrapper =
    sics::matrixgraph::core::task::kernel::MatrixOperationsKernelWrapper;
using Bitmap = sics::matrixgraph::core::util::Bitmap;
using GPUBitmap = sics::matrixgraph::core::util::GPUBitmap;
using BitmapNoOwnerShip = sics::matrixgraph::core::util::BitmapNoOwnerShip;
using sics::matrixgraph::core::util::atomic::WriteAdd;
using TiledMatrixMetadata =
    sics::matrixgraph::core::data_structures::TiledMatrixMetadata;

// CUDA kernel to add elements of two arrays
__host__ void GEMM::LoadData() {
  std::cout << "[GEMM] LoadData()" << std::endl;
  GridCSRTiledMatrixIO grid_csr_tiled_matrix_io;

  grid_csr_tiled_matrix_io.Read(input_path_, &A_);
  grid_csr_tiled_matrix_io.Read(input_path_transposed_, &B_);
  A_->Print();
  B_->Print();
  C_ = new GridCSRTiledMatrix(A_->get_metadata());
}

__host__ void GEMM::InitResultMatrix() {}

__host__ void GEMM::FillTilesUnifiedMemory() {}

__host__ void GEMM::FillTiles() {}

__host__ void GEMM::Count(const GridCSRTiledMatrix &G) {}

__host__ void GEMM::Run() {

  auto start_time_0 = std::chrono::system_clock::now();
  InitResultMatrix();

  auto start_time_1 = std::chrono::system_clock::now();

  std::cout << "[GEMM] Run Step1 InitResultMatrix() elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_1 - start_time_0)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;

  FillTilesUnifiedMemory();
  auto start_time_2 = std::chrono::system_clock::now();
  std::cout << "[GEMM] Run Step2 FillTiles() elapsed:"
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_2 - start_time_1)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;

  Count(*C_);

  auto start_time_3 = std::chrono::system_clock::now();
  std::cout << "[GEMM] Run Step3 Count() elapsed:"
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_3 - start_time_2)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;
}

} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics