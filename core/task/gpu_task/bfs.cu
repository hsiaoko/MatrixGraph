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
#include "core/task/gpu_task/bfs.cuh"
#include "core/task/gpu_task/kernel/data_structures/exec_plan.cuh"
#include "core/task/gpu_task/kernel/kernel_bfs.cuh"
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
using GridCSRTiledMatrix =
    sics::matrixgraph::core::data_structures::GridCSRTiledMatrix;
using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;
using Edges = sics::matrixgraph::core::data_structures::Edges;
using UnifiedOwnedBufferUint8 =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint8_t>;
using UnifiedOwnedBufferVertexLabel =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexLabel>;
using BFSKernelWrapper =
    sics::matrixgraph::core::task::kernel::BFSKernelWrapper;
using BufferVertexLabel =
    sics::matrixgraph::core::data_structures::Buffer<VertexLabel>;
using BufferUint8 = sics::matrixgraph::core::data_structures::Buffer<uint8_t>;

__host__ void BFS::LoadData() {
  std::cout << "[BFS] LoadData()" << std::endl;
  g_.Read(data_graph_path_);
  // g_.PrintGraph();
}

__host__ void BFS::ExecuteBFS(const ImmutableCSR& g, VertexID src) {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;

  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  cudaSetDevice(1);

  // Initialize buffers
  UnifiedOwnedBufferUint8 unified_data_g;
  UnifiedOwnedBufferVertexLabel unified_v_label_g;

  // Set up data buffer
  BufferUint8 data_g;
  data_g.data = g.GetGraphBuffer();
  data_g.size =
      sizeof(VertexID) * g.get_num_vertices() *
          3 +  // globalid, in_degree, out_degree
      sizeof(EdgeIndex) * (g.get_num_vertices() + 1) *
          2 +  // in_offset, out_offset
      sizeof(VertexID) * (g.get_num_incoming_edges() +
                          g.get_num_outgoing_edges() + g.get_max_vid() + 1);
  unified_data_g.Init(data_g);

  // Set up vertex label buffer
  BufferVertexLabel v_label_g;
  v_label_g.data = g.GetVLabelBasePointer();
  v_label_g.size = sizeof(VertexLabel) * g.get_max_vid();
  unified_v_label_g.Init(v_label_g);

  // Execute BFS
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  BFSKernelWrapper::BFS(stream, g.get_num_vertices(),
                        g.get_num_outgoing_edges(), src, unified_data_g,
                        unified_v_label_g);

  cudaStreamDestroy(stream);
}

__host__ void BFS::Run() {
  auto start_time_0 = std::chrono::system_clock::now();
  LoadData();
  auto start_time_1 = std::chrono::system_clock::now();

  ExecuteBFS(g_, src_);

  auto start_time_2 = std::chrono::system_clock::now();

  std::cout << "[BFS] LoadData() elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_1 - start_time_0)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;

  std::cout << "[BFS] ExecuteBFS() elapsed: "
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
