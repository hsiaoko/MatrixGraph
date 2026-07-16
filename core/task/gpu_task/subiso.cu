#include <cuda_runtime.h>

#include <algorithm>
#include <ctime>
#include "core/util/execution_policy.h"
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include <unordered_map>

#include "core/common/host_algorithms.cuh"
#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/exec_plan.cuh"
#include "core/data_structures/host_buffer.cuh"
#include "core/data_structures/matches.cuh"
#include "core/data_structures/metadata.h"
#include "core/data_structures/unified_buffer.cuh"
#include "core/data_structures/woj_matches.cuh"
#include "core/io/grid_csr_tiled_matrix_io.cuh"
#include "core/task/gpu_task/kernel/kernel_subiso.cuh"
#include "core/task/gpu_task/kernel/kernel_woj_subiso.cuh"
#include "core/task/gpu_task/subiso.cuh"
#include "core/util/atomic.h"
#include "core/util/bitmap_no_ownership.h"
#include "core/util/cuda_check.cuh"
#include "core/util/cuda_device.cuh"
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
using WOJSubIsoKernelWrapper =
    sics::matrixgraph::core::task::kernel::WOJSubIsoKernelWrapper;
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
using Matches = sics::matrixgraph::core::data_structures::Matches;
using WOJMatches = sics::matrixgraph::core::data_structures::WOJMatches;

__host__ void SubIso::LoadData() {
  std::cout << "[SubIso] LoadData() ..." << std::endl;

  p_.Read(pattern_path_);

  g_.Read(data_graph_path_);

  //p_.PrintGraph(10);
  //g_.PrintGraph(1);
}

__host__ void SubIso::Matching(const ImmutableCSR& p, const ImmutableCSR& g) {
  std::cout << "Matching ..." << std::endl;
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;

  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  CUDA_CHECK(cudaSetDevice(
      sics::matrixgraph::core::util::MatrixGraphCudaDevice()));

  // Init pattern.
  BufferUint8 data_p;
  BufferVertexLabel v_label_p;
  UnifiedOwnedBufferUint8 unified_data_p;
  UnifiedOwnedBufferVertexLabel unified_v_label_p;

  data_p.data = p.GetGraphBuffer();
  data_p.size = sizeof(VertexID) * p.get_num_vertices() +
                sizeof(VertexID) * p.get_num_vertices() +
                sizeof(VertexID) * p.get_num_vertices() +
                sizeof(EdgeIndex) * (p.get_num_vertices() + 1) +
                sizeof(EdgeIndex) * (p.get_num_vertices() + 1) +
                sizeof(VertexID) * p.get_num_incoming_edges() +
                sizeof(VertexID) * p.get_num_outgoing_edges() +
                sizeof(VertexID) * (p.get_max_vid() + 1);

  unified_data_p.Init(data_p);

  v_label_p.data = p.GetVLabelBasePointer();
  v_label_p.size = sizeof(VertexLabel) * p.get_num_vertices();

  unified_v_label_p.Init(v_label_p);

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
  v_label_g.size = sizeof(VertexLabel) * g.get_num_vertices();

  unified_v_label_g.Init(v_label_g);

  UnifiedOwnedBufferEdgeIndex unified_m_offset;
  unified_m_offset.Init(sizeof(EdgeIndex) * p.get_num_vertices());

  Matches matches(p.get_num_vertices(), g.get_num_vertices());

  // Generate Execution Plan
  ExecutionPlan exec_plan;
  exec_plan.GenerateDFSExecutionPlan(p, g);

  // Start Matching ...
  cudaDeviceSynchronize();
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  SubIsoKernelWrapper::SubIso(
      stream, exec_plan.get_depth(), *exec_plan.get_sequential_exec_path_ptr(),
      *exec_plan.get_inverted_index_of_sequential_exec_path_ptr(),
      *exec_plan.get_sequential_exec_path_in_edges_ptr(), p.get_num_vertices(),
      p.get_num_outgoing_edges(), unified_data_p, unified_v_label_p,
      g.get_num_vertices(), g.get_num_outgoing_edges(), unified_data_g,
      unified_edgelist_g, unified_v_label_g, matches.weft_count_,
      matches.weft_offset_, matches.weft_size_,
      matches.v_candidate_offset_for_each_weft_, matches.matches_data_);

  cudaDeviceSynchronize();

  //matches.Print(3);
}

__host__ void SubIso::WOJMatching(
    const ImmutableCSR& p, const ImmutableCSR& g, bool enable_min_wise_filter,
    bool enable_label_degree_filter, bool enable_nlc_filter,
    bool enable_lpf_filter, bool enable_lcf_filter, bool enable_bloom_filter,
    bool enable_min_wise_bloom_filter) {
  // Generate Execution Plan
  WOJExecutionPlan exec_plan;
  exec_plan.GenerateWOJExecutionPlan(p, g);
  const std::vector<int> cuda_devices =
      sics::matrixgraph::core::util::MatrixGraphCudaDeviceList();
  exec_plan.SetCudaDeviceIds(cuda_devices);

  std::cout << "[SubIso] WOJMatching: execution plan |V_p|="
            << exec_plan.get_n_vertices_p() << " |E_p|="
            << exec_plan.get_n_edges_p() << " |V_g|="
            << exec_plan.get_n_vertices_g() << " |E_g|="
            << exec_plan.get_n_edges_g() << std::endl;
  std::cout << "[SubIso] WOJMatching: plan uses " << exec_plan.get_n_devices()
            << " logical GPU(s), device id(s): ";
  for (size_t i = 0; i < cuda_devices.size(); ++i) {
    if (i) {
      std::cout << ", ";
    }
    std::cout << cuda_devices[i];
  }
  std::cout << std::endl;
  std::cout << "[SubIso] WOJMatching: step 1 WOJSubIsoKernelWrapper::Filter, "
               "step 2 Join ..."
            << std::endl;

  auto start_time_0 = std::chrono::system_clock::now();
  auto woj_matches = WOJSubIsoKernelWrapper::Filter(
      exec_plan, p, g, enable_min_wise_filter, enable_label_degree_filter,
      enable_nlc_filter, enable_lpf_filter, enable_lcf_filter,
      enable_bloom_filter, enable_min_wise_bloom_filter);
  auto start_time_1 = std::chrono::system_clock::now();

  auto output_woj_matches_vec =
      WOJSubIsoKernelWrapper::Join(exec_plan, woj_matches);
  std::cout << "Join Down" << std::endl;

  for (auto _ = 0; _ < output_woj_matches_vec.size(); _++) {
    output_woj_matches_vec[_]->Print();
  }

  auto start_time_2 = std::chrono::system_clock::now();

  std::cout << "[WOJMatching] Filter() elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_1 - start_time_0)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;
  std::cout << "[WOJMatching] Join() elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_2 - start_time_1)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;
}

__host__ void SubIso::Run() {
  auto start_time_0 = std::chrono::system_clock::now();
  LoadData();
  auto start_time_1 = std::chrono::system_clock::now();

  WOJMatching(p_, g_, enable_min_wise_filter_, enable_label_degree_filter_,
              enable_nlc_filter_, enable_lpf_filter_, enable_lcf_filter_,
              enable_bloom_filter_, enable_min_wise_bloom_filter_);
  // Matching(p_, g_);

  auto start_time_2 = std::chrono::system_clock::now();
  auto start_time_3 = std::chrono::system_clock::now();

  std::cout << "[SubIso] LoadData() elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_1 - start_time_0)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;

  std::cout << "[SubIso] Matching() elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_2 - start_time_1)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;
}

// ---------------------------------------------------------------------------
// C API helper: build ImmutableCSR from flat buffers and run WOJMatching.
// ---------------------------------------------------------------------------
__host__ int SubIso::Run(
    uint32_t p_num_vertices, uint32_t p_num_in_edges, uint32_t p_num_out_edges,
    uint32_t p_max_vid, uint32_t p_min_vid, const uint8_t* p_csr_data,
    uint64_t p_csr_data_size, const uint32_t* p_labels,
    uint32_t g_num_vertices, uint32_t g_num_in_edges, uint32_t g_num_out_edges,
    uint32_t g_max_vid, uint32_t g_min_vid, const uint8_t* g_csr_data,
    uint64_t g_csr_data_size, const uint32_t* g_labels,
    int max_result_tables, int max_result_rows, int max_result_cols,
    uint32_t* out_table_cols, uint32_t* out_table_rows,
    uint32_t* out_headers_flat, uint32_t* out_data_flat, int* out_num_tables) {
  if (!p_csr_data || !p_labels || !g_csr_data || !g_labels || !out_num_tables) {
    return 1;
  }

  // Helper: build an ImmutableCSR from flat arrays.
  auto build_csr = [](uint32_t num_vertices, uint32_t num_in_edges,
                      uint32_t num_out_edges, uint32_t max_vid,
                      uint32_t min_vid, const uint8_t* csr_data,
                      uint64_t csr_data_size, const uint32_t* labels) {
    ImmutableCSR* csr = new ImmutableCSR();
    csr->SetNumVertices(num_vertices);
    csr->SetNumIncomingEdges(num_in_edges);
    csr->SetNumOutgoingEdges(num_out_edges);
    csr->SetMaxVid(max_vid);
    csr->SetMinVid(min_vid);

    uint8_t* buf = new uint8_t[csr_data_size];
    std::memcpy(buf, csr_data, csr_data_size);
    csr->SetGraphBuffer(buf);
    csr->ParseBasePtr(buf);

    VertexLabel* lbl = new VertexLabel[num_vertices];
    std::memcpy(lbl, labels, sizeof(VertexLabel) * num_vertices);
    csr->SetVertexLabelBuffer(lbl);

    return csr;
  };

  ImmutableCSR* p = build_csr(p_num_vertices, p_num_in_edges, p_num_out_edges,
                              p_max_vid, p_min_vid, p_csr_data,
                              p_csr_data_size, p_labels);
  ImmutableCSR* g = build_csr(g_num_vertices, g_num_in_edges, g_num_out_edges,
                              g_max_vid, g_min_vid, g_csr_data,
                              g_csr_data_size, g_labels);

  // Run WOJ matching.
  std::vector<WOJMatches*> results;
  try {
    WOJExecutionPlan exec_plan;
    exec_plan.GenerateWOJExecutionPlan(*p, *g);
    const std::vector<int> cuda_devices =
        sics::matrixgraph::core::util::MatrixGraphCudaDeviceList();
    exec_plan.SetCudaDeviceIds(cuda_devices);

    results = WOJSubIsoKernelWrapper::Filter(exec_plan, *p, *g, true, true, true,
                                             true, true, false, false);
    results = WOJSubIsoKernelWrapper::Join(exec_plan, results);
  } catch (...) {
    delete[] p->GetVLabelBasePointer();
    delete[] g->GetVLabelBasePointer();
    delete p;
    delete g;
    return 1;
  }

  int n_tables = static_cast<int>(results.size());
  if (n_tables > max_result_tables) n_tables = max_result_tables;
  *out_num_tables = n_tables;

  for (int t = 0; t < n_tables; ++t) {
    WOJMatches* m = results[t];
    uint32_t cols = m->get_x_offset();
    uint32_t rows = m->get_y_offset();
    out_table_cols[t] = cols;
    out_table_rows[t] = rows;

    // Copy headers (column IDs).
    uint32_t header_copy = cols > static_cast<uint32_t>(max_result_cols)
                               ? static_cast<uint32_t>(max_result_cols)
                               : cols;
    if (header_copy > 0 && out_headers_flat) {
      uint32_t* dst_header =
          out_headers_flat + t * max_result_cols;
      std::memcpy(dst_header, m->get_header_ptr(),
                  sizeof(uint32_t) * header_copy);
    }

    // Copy data (row-major, each row has 'cols' elements).
    uint32_t row_copy = rows > static_cast<uint32_t>(max_result_rows)
                            ? static_cast<uint32_t>(max_result_rows)
                            : rows;
    if (row_copy > 0 && cols > 0 && out_data_flat) {
      uint32_t* dst_data = out_data_flat + t * max_result_rows * max_result_cols;
      const uint32_t* src_data = m->get_data_ptr();
      for (uint32_t r = 0; r < row_copy; ++r) {
        uint32_t col_copy = cols > static_cast<uint32_t>(max_result_cols)
                                ? static_cast<uint32_t>(max_result_cols)
                                : cols;
        std::memcpy(dst_data + r * max_result_cols,
                    src_data + r * m->get_x(),
                    sizeof(uint32_t) * col_copy);
      }
    }
  }

  // Cleanup WOJMatches (allocated inside Filter/Join).
  for (auto* m : results) {
    if (m) m->Free();
    delete m;
  }

  // Cleanup CSR labels (graph buffers are owned by unique_ptr in ImmutableCSR).
  delete[] p->GetVLabelBasePointer();
  delete[] g->GetVLabelBasePointer();
  delete p;
  delete g;

  return 0;
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
