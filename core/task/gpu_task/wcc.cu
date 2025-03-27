#include <cuda_runtime.h>

#include <algorithm>
#include <ctime>
#include <execution>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "core/common/consts.h"
#include "core/common/host_algorithms.cuh"
#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/host_buffer.cuh"
#include "core/data_structures/metadata.h"
#include "core/data_structures/unified_buffer.cuh"
#include "core/io/grid_csr_tiled_matrix_io.cuh"
#include "core/task/gpu_task/kernel/data_structures/exec_plan.cuh"
#include "core/task/gpu_task/kernel/data_structures/matches.cuh"
#include "core/task/gpu_task/kernel/data_structures/woj_matches.cuh"
#include "core/task/gpu_task/kernel/kernel_wcc.cuh"
#include "core/task/gpu_task/wcc.cuh"
#include "core/util/atomic.h"
#include "core/util/bitmap.h"
#include "core/util/bitmap_no_ownership.h"
#include "core/util/format_converter.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using VertexID = sics::matrixgraph::core::common::VertexID;
using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
using sics::matrixgraph::core::data_structures::GridCSRTiledMatrix;
using sics::matrixgraph::core::io::GridCSRTiledMatrixIO;
using GirdTiledMatrix =
    sics::matrixgraph::core::data_structures::GridCSRTiledMatrix;
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
using UnifiedOwnedBufferEdgeIndex =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<EdgeIndex>;
using UnifiedOwnedBufferVertexID =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexID>;
using UnifiedOwnedBufferVertexLabel =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexLabel>;
using UnifiedOwnedBufferUint8 =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint8_t>;
using BufferUint64 = sics::matrixgraph::core::data_structures::Buffer<uint64_t>;
using BufferUint8 = sics::matrixgraph::core::data_structures::Buffer<uint8_t>;
using BufferUint32 = sics::matrixgraph::core::data_structures::Buffer<uint32_t>;
using BufferEdgeIndex =
    sics::matrixgraph::core::data_structures::Buffer<EdgeIndex>;
using BufferVertexID =
    sics::matrixgraph::core::data_structures::Buffer<VertexID>;
using BufferVertexLabel =
    sics::matrixgraph::core::data_structures::Buffer<VertexLabel>;
using DeviceOwnedBufferUint64 =
    sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<uint64_t>;
using WCCKernelWrapper =
    sics::matrixgraph::core::task::kernel::WCCKernelWrapper;
using Bitmap = sics::matrixgraph::core::util::Bitmap;
using GPUBitmap = sics::matrixgraph::core::util::GPUBitmap;
using BitmapNoOwnerShip = sics::matrixgraph::core::util::BitmapNoOwnerShip;
using sics::matrixgraph::core::util::atomic::WriteAdd;
using TiledMatrixMetadata =
    sics::matrixgraph::core::data_structures::TiledMatrixMetadata;
using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;
using Edges = sics::matrixgraph::core::data_structures::Edges;
using sics::matrixgraph::core::util::atomic::WriteAdd;
using sics::matrixgraph::core::util::atomic::WriteMax;
using sics::matrixgraph::core::util::atomic::WriteMin;
using sics::matrixgraph::core::util::format_converter::Edgelist2CSRTiledMatrix;
using GraphMetadata = sics::matrixgraph::core::data_structures::GraphMetadata;
using Edge = sics::matrixgraph::core::data_structures::Edge;
using EdgelistMetadata =
    sics::matrixgraph::core::data_structures::EdgelistMetadata;
using GridGraphMetadata =
    sics::matrixgraph::core::data_structures::GridGraphMetadata;
using sics::matrixgraph::core::common::kDefalutNumEdgesPerBlock;
using sics::matrixgraph::core::common::kDefalutNumEdgesPerTile;
using sics::matrixgraph::core::common::kMaxNumCandidates;
using sics::matrixgraph::core::common::kMaxNumEdges;
using sics::matrixgraph::core::common::kMaxNumEdgesPerBlock;
using sics::matrixgraph::core::common::kMaxVertexID;
using Matches = sics::matrixgraph::core::task::kernel::Matches;
using WOJMatches = sics::matrixgraph::core::task::kernel::WOJMatches;
using sics::matrixgraph::core::common::kMaxNumWeft;

__host__ void WCC::LoadData() {
  std::cout << "[WCC] LoadData()" << std::endl;

  g_.Read(data_graph_path_);
  // g_.PrintGraph();
}

__host__ void WCC::HashMin(const ImmutableCSR& g) {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;

  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  cudaSetDevice(1);

  // Init data_graph.
  BufferUint8 data_g;
  BufferVertexLabel v_label_g;
  BufferVertexID data_edgelist_g;

  UnifiedOwnedBufferUint8 unified_data_g;
  UnifiedOwnedBufferVertexLabel unified_v_label_g;
  UnifiedOwnedBufferVertexID unified_edgelist_g;

  data_g.data = g.GetGraphBuffer();
  data_g.size = sizeof(VertexID) * g.get_num_vertices() +
                sizeof(VertexID) * g.get_num_vertices() +
                sizeof(VertexID) * g.get_num_vertices() +
                sizeof(EdgeIndex) * (g.get_num_vertices() + 1) +
                sizeof(EdgeIndex) * (g.get_num_vertices() + 1) +
                sizeof(VertexID) * g.get_num_incoming_edges() +
                sizeof(VertexID) * g.get_num_outgoing_edges() +
                sizeof(VertexID) * (g.get_max_vid() + 1);

  unified_data_g.Init(data_g);

  v_label_g.data = g.GetVLabelBasePointer();
  v_label_g.size = sizeof(VertexLabel) * g.get_max_vid();

  unified_v_label_g.Init(v_label_g);

  // Start HashMin ...
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  WCCKernelWrapper::WCC(stream, g.get_num_vertices(),
                        g.get_num_outgoing_edges(), unified_data_g,
                        unified_v_label_g);

  cudaStreamDestroy(stream);
}

__host__ void WCC::Run() {
  auto start_time_0 = std::chrono::system_clock::now();
  LoadData();
  auto start_time_1 = std::chrono::system_clock::now();

  HashMin(g_);

  auto start_time_2 = std::chrono::system_clock::now();

  std::cout << "[WCC] LoadData() elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_1 - start_time_0)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;

  std::cout << "[WCC] HashMin() elapsed: "
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
