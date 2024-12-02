#include "core/task/subiso.cuh"

#include <algorithm>
#include <ctime>
#include <cuda_runtime.h>
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
#include "core/data_structures/matches.cuh"
#include "core/data_structures/metadata.h"
#include "core/data_structures/unified_buffer.cuh"
#include "core/io/grid_csr_tiled_matrix_io.cuh"
#include "core/task/kernel/kernel_subiso.cuh"
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
using SubIsoKernelWrapper =
    sics::matrixgraph::core::task::kernel::SubIsoKernelWrapper;
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
using ExecutionPlan = sics::matrixgraph::core::task::SubIso::ExecutionPlan;
using Matches = sics::matrixgraph::core::data_structures::Matches;

// CUDA kernel to add elements of two arrays
__host__ void SubIso::LoadData() {
  std::cout << "[SubIso] LoadData()" << std::endl;

  p_.Read(pattern_path_);
  p_.PrintGraph(100);

  g_.Read(data_graph_path_);
  g_.PrintGraph(999);
}

__host__ void SubIso::InitLabel() {
  VertexLabel *p_vlabel = p_.GetVLabelBasePointer();
  p_vlabel[0] = 0;
  p_vlabel[1] = 1;
  p_vlabel[2] = 2;
  p_vlabel[3] = 3;
  p_vlabel[4] = 4;
  p_vlabel[5] = 3;

  VertexLabel *g_vlabel = g_.GetVLabelBasePointer();

  g_vlabel[0] = 0;
  g_vlabel[1] = 1;
  g_vlabel[2] = 2;
  g_vlabel[3] = 3;
  g_vlabel[4] = 3;
  g_vlabel[5] = 4;
  g_vlabel[6] = 4;
  g_vlabel[7] = 1;
  g_vlabel[8] = 0;
}

static void DFSTraverse(VertexID vid, Bitmap &visited, const ImmutableCSR &g,
                        std::vector<VertexID> &output, VertexID depth,
                        VertexID &max_depth) {

  auto u = g.GetVertexByLocalID(vid);

  auto globalid = g.GetGlobalIDByLocalID(vid);
  output.emplace_back(globalid);

  visited.SetBit(vid);

  max_depth = std::max(depth, max_depth);

  for (VertexID _ = 0; _ < u.outdegree; _++) {
    if (!visited.GetBit(u.outgoing_edges[_])) {
      DFSTraverse(u.outgoing_edges[_], visited, g, output, depth + 1,
                  max_depth);
    }
  }
}

__host__ void SubIso::GenerateDFSExecutionPlan(const ImmutableCSR &p,
                                               const ImmutableCSR &g,
                                               ExecutionPlan *exec_plan) {

  Bitmap visited(p.get_max_vid());
  auto global_vid = p.GetGlobalIDByLocalID(0);
  std::vector<VertexID> output;
  output.reserve(p.get_max_vid());

  VertexID depth = 0;
  DFSTraverse(0, visited, p, output, 0, depth);

  exec_plan->sequential_exec_path.Init(sizeof(VertexID) *
                                       (p.get_num_vertices()));
  cudaMemcpy(exec_plan->sequential_exec_path.GetPtr(), output.data(),
             sizeof(VertexID) * p.get_num_vertices(), cudaMemcpyHostToHost);
  exec_plan->n_vertices = p.get_num_vertices();
  exec_plan->depth = depth;
}

__host__ void SubIso::Matching(const ImmutableCSR &p, const ImmutableCSR &g) {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;

  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  // Init pattern.
  BufferUint8 data_p;
  BufferVertexLabel v_label_p;
  UnifiedOwnedBufferUint8 unified_data_p;
  UnifiedOwnedBufferVertexLabel unified_v_label_p;

  data_p.data = p.GetGraphBuffer();
  data_p.size = sizeof(VertexID) * (p.get_num_vertices() * 3) +
                sizeof(VertexID) * p.get_num_incoming_edges() +
                sizeof(VertexID) * p.get_num_outgoing_edges() +
                sizeof(EdgeIndex) * p.get_num_vertices() * 2 +
                sizeof(VertexID) * (p.get_max_vid() + 1);

  unified_data_p.Init(data_p);

  v_label_p.data = p.GetVLabelBasePointer();
  v_label_p.size = sizeof(VertexLabel) * p.get_num_vertices();

  unified_v_label_p.Init(v_label_p);

  // Init data_graph.
  BufferUint8 data_g;
  BufferVertexLabel v_label_g;

  UnifiedOwnedBufferUint8 unified_data_g;
  UnifiedOwnedBufferVertexLabel unified_v_label_g;

  data_g.data = g.GetGraphBuffer();
  data_g.size = sizeof(VertexID) * (g.get_num_vertices() * 3) +
                sizeof(VertexID) * g.get_num_incoming_edges() +
                sizeof(VertexID) * g.get_num_outgoing_edges() +
                sizeof(EdgeIndex) * g.get_num_vertices() * 2 +
                sizeof(VertexID) * (g.get_max_vid() + 1);
  unified_data_g.Init(data_g);

  v_label_g.data = g.GetVLabelBasePointer();
  v_label_g.size = sizeof(VertexLabel) * g.get_num_vertices();

  unified_v_label_g.Init(v_label_g);

  // Init output.
  std::vector<UnifiedOwnedBufferVertexID> unified_edgelist_m;
  unified_edgelist_m.resize(p.get_num_vertices() - 1);
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, step, &mtx, &unified_edgelist_m, &p](auto w) {
        for (VertexID _ = w; _ < p.get_num_vertices() - 1; _ += step) {
          unified_edgelist_m[_].Init(sizeof(VertexID) * kMaxNumCandidates * 2);
        }
      });

  sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexID *>
      unified_m_ptr;
  unified_m_ptr.Init(sizeof(VertexID *) * (p.get_num_vertices() - 1) * 2);
  for (auto _ = 0; _ < unified_edgelist_m.size(); _++) {
    unified_m_ptr.GetPtr()[_] = unified_edgelist_m[_].GetPtr();
  }

  UnifiedOwnedBufferEdgeIndex unified_m_offset;
  unified_m_offset.Init(sizeof(EdgeIndex) * p.get_num_vertices());

  Matches matches(p.get_num_vertices(), g.get_num_vertices());

  UnifiedOwnedBufferVertexID unified_m0_data;
  UnifiedOwnedBufferVertexID unified_m0_offset;

  unified_m0_data.Init(sizeof(VertexID) * g.get_num_vertices() / 65536);

  // Generate Execution Plan
  ExecutionPlan exec_plan;
  GenerateDFSExecutionPlan(p, g, &exec_plan);

  // Start Matching ...
  cudaDeviceSynchronize();
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  SubIsoKernelWrapper::SubIso(
      stream, exec_plan.depth, exec_plan.sequential_exec_path,
      p.get_num_vertices(), p.get_num_outgoing_edges(), unified_data_p,
      unified_v_label_p, g.get_num_vertices(), g.get_num_outgoing_edges(),
      unified_data_g, unified_v_label_g, matches.weft_count_,
      matches.weft_offset_, matches.weft_size_,
      matches.v_candidate_offset_for_each_weft_, matches.matches_data_);

  cudaDeviceSynchronize();

  matches.Print();
  // Print Output.
  for (auto _ = 0; _ < p.get_num_vertices(); _++) {
    std::cout << " level " << _ << " get " << unified_m_offset.GetPtr()[_]
              << " matches: " << std::endl;
    for (auto __ = 0; __ < unified_m_offset.GetPtr()[_]; __++) {
      std::cout << unified_m_ptr.GetPtr()[_][__] << " ";
    }
    std::cout << std::endl;
  }
}

__host__ void SubIso::Run() {

  auto start_time_0 = std::chrono::system_clock::now();
  LoadData();
  auto start_time_1 = std::chrono::system_clock::now();
  InitLabel();

  auto start_time_2 = std::chrono::system_clock::now();

  Matching(p_, g_);

  std::cout << "[SubIso] LoadData() elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_1 - start_time_0)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;
}

} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics