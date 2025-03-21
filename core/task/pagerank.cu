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
#include "core/task/kernel/kernel_pagerank.cuh"
#include "core/task/pagerank.cuh"
#include "core/util/atomic.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using VertexID = sics::matrixgraph::core::common::VertexID;
using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
using DeviceOwnedBufferUint8 =
    sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<uint8_t>;
using UnifiedOwnedBufferUint8 =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint8_t>;
using UnifiedOwnedBufferFloat =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<float>;
using BufferUint8 = sics::matrixgraph::core::data_structures::Buffer<uint8_t>;
using PageRankKernelWrapper =
    sics::matrixgraph::core::task::kernel::PageRankKernelWrapper;
using BufferVertexLabel =
    sics::matrixgraph::core::data_structures::Buffer<VertexLabel>;
using UnifiedOwnedBufferVertexLabel =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexLabel>;
using BufferVertexID =
    sics::matrixgraph::core::data_structures::Buffer<VertexID>;
using UnifiedOwnedBufferVertexID =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexID>;
using UnifiedOwnedBufferFloat =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<float>;
using BufferFloat = sics::matrixgraph::core::data_structures::Buffer<float>;

__host__ void PageRank::LoadData() {
  std::cout << "[PageRank] LoadData()" << std::endl;

  g_.Read(data_graph_path_);
  g_.PrintGraph();
}

__host__ void PageRank::ComputePageRank(const ImmutableCSR& g) {
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

  BufferFloat page_ranks;
  page_ranks.data = new float[g.get_max_vid()]();
  page_ranks.size = sizeof(float) * g.get_max_vid();

  UnifiedOwnedBufferFloat unified_page_ranks;
  unified_page_ranks.Init(page_ranks);

  // Start PageRank ...
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  PageRankKernelWrapper::PageRank(
      stream, g.get_num_vertices(), g.get_num_outgoing_edges(), unified_data_g,
      unified_page_ranks, damping_factor_, epsilon_, max_iterations_);

  cudaStreamDestroy(stream);
}

__host__ void PageRank::Run() {
  auto start_time_0 = std::chrono::system_clock::now();
  LoadData();
  auto start_time_1 = std::chrono::system_clock::now();

  ComputePageRank(g_);

  auto start_time_2 = std::chrono::system_clock::now();

  std::cout << "[PageRank] LoadData() elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_1 - start_time_0)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;

  std::cout << "[PageRank] ComputePageRank() elapsed: "
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
