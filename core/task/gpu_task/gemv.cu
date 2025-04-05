#include "core/common/consts.h"
#include "core/common/host_algorithms.cuh"
#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/host_buffer.cuh"
#include "core/data_structures/metadata.h"
#include "core/data_structures/unified_buffer.cuh"
#include "core/task/gpu_task/gemv.cuh"
#include "core/util/atomic.h"
#include "core/util/bitmap_no_ownership.h"
#include "core/util/bitmap_ownership.h"
#include <ctime>
#include <cuda_runtime.h>
#include <execution>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using DeviceOwnedBufferUint64 =
    sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<uint64_t>;
using DeviceOwnedBufferUint32 =
    sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<uint32_t>;
using DeviceOwnedBufferUint8 =
    sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<uint8_t>;
using UnifiedOwnedBufferUint32 =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint32_t>;
using UnifiedOwnedBufferUint64 =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint64_t>;
using UnifiedOwnedBufferUint8 =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint8_t>;
using BufferUint64 = sics::matrixgraph::core::data_structures::Buffer<uint64_t>;
using BufferUint8 = sics::matrixgraph::core::data_structures::Buffer<uint8_t>;
using BufferUint32 = sics::matrixgraph::core::data_structures::Buffer<uint32_t>;
using DeviceOwnedBufferUint64 =
    sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<uint64_t>;
using BitmapOwnership = sics::matrixgraph::core::util::BitmapOwnership;
using BitmapNoOwnerShip = sics::matrixgraph::core::util::BitmapNoOwnerShip;
using sics::matrixgraph::core::util::atomic::WriteAdd;
using Edges = sics::matrixgraph::core::data_structures::Edges;
using sics::matrixgraph::core::util::atomic::WriteAdd;
using sics::matrixgraph::core::util::atomic::WriteMax;
using sics::matrixgraph::core::util::atomic::WriteMin;
using GraphMetadata = sics::matrixgraph::core::data_structures::GraphMetadata;
using Edge = sics::matrixgraph::core::data_structures::Edge;
using EdgelistMetadata =
    sics::matrixgraph::core::data_structures::EdgelistMetadata;
using GridGraphMetadata =
    sics::matrixgraph::core::data_structures::GridGraphMetadata;
using sics::matrixgraph::core::common::kDefalutNumEdgesPerBlock;
using sics::matrixgraph::core::common::kDefalutNumEdgesPerTile;
using sics::matrixgraph::core::common::kMaxNumEdges;
using sics::matrixgraph::core::common::kMaxNumEdgesPerBlock;

// CUDA kernel to add elements of two arrays
__host__ void GEMV::LoadData() {
  std::cout << "[GEMV] LoadData()" << std::endl;
}

__host__ void GEMV::Run() {
  auto start_time_0 = std::chrono::system_clock::now();
  LoadData();
  auto start_time_1 = std::chrono::system_clock::now();

  auto start_time_2 = std::chrono::system_clock::now();

  std::cout << "[GEMV] LoadData() elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_1 - start_time_0)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;

  std::cout << "[GEMV] Run() elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_2 - start_time_1)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics