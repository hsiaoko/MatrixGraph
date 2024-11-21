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
#include "core/data_structures/metadata.h"
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

// CUDA kernel to add elements of two arrays
__host__ void SubIso::LoadData() {
  std::cout << "[SubIso] LoadData()" << std::endl;
  GridCSRTiledMatrixIO grid_csr_tiled_matrix_io;

  p_.Read(pattern_path_);
  grid_csr_tiled_matrix_io.Read(data_graph_path_, &g_);

  p_.PrintGraph(100);
  g_->Print(999);
}

__host__ void SubIso::InitLabel() {
  auto *p_vlabel = p_.GetVLabelBasePointer();
  std::cout << std::endl;
  p_vlabel[0] = 0;
  p_vlabel[1] = 1;
  p_vlabel[2] = 2;
  p_vlabel[3] = 3;
  p_vlabel[4] = 4;
  p_vlabel[5] = 3;

  auto *g_vlabel = g_->GetVLabelBasePointer();

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

__host__ void SubIso::AllocMappingBuf() {}

__host__ void SubIso::Matching(const ImmutableCSR &p,
                               const GridCSRTiledMatrix &g) {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;

  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  auto block = g.GetTiledMatrixPtrByIdx(0);
  VertexID tile_size = block->GetMetadata().tile_size;
  VertexID n_strips = block->GetMetadata().n_strips;

  auto tile_buffer_size =
      sizeof(uint64_t) * std::max(1u, WORD_OFFSET(tile_size * tile_size));

  // Init Streams.
  VertexID M = g.get_metadata().n_chunks;
  VertexID N = g.get_metadata().n_chunks;

  std::vector<cudaStream_t> p_streams_vec;
  p_streams_vec.resize(M * N);
  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [this, M, N, step, &p_streams_vec, &mtx](auto w) {
                  for (VertexID i = w; i < N; i += step) {
                    for (VertexID j = 0; j < M; j++) {
                      cudaSetDevice(common::hash_function(i * N + j) % 4);
                      cudaStreamCreate(&p_streams_vec[i * N + j]);
                    }
                  }
                });

  // Init pattern.
  BufferUint8 data_p;
  BufferVertexLabel v_label_p;

  UnifiedOwnedBufferUint8 unified_data_p;
  UnifiedOwnedBufferVertexLabel unified_v_label_p;

  data_p.data = p.GetGraphBuffer();
  data_p.size = sizeof(VertexID) * (p.get_num_vertices() * 3) +
                sizeof(VertexID) * p.get_num_incoming_edges() +
                sizeof(VertexID) * p.get_num_outgoing_edges() +
                sizeof(EdgeIndex) * (p.get_num_vertices() * 2 + 2);

  unified_data_p.Init(data_p);

  v_label_p.data = p.GetVLabelBasePointer();
  v_label_p.size = sizeof(VertexLabel) * p.get_num_vertices();

  unified_v_label_p.Init(v_label_p);

  // Init Data Graph.
  std::vector<BufferVertexID> tile_offset_row_g;
  std::vector<BufferVertexID> tile_row_idx_g;
  std::vector<BufferVertexID> tile_col_idx_g;
  std::vector<BufferUint64> csr_offset_g;
  std::vector<BufferUint8> data_g;

  tile_offset_row_g.resize(N * M);
  tile_row_idx_g.resize(N * M);
  tile_col_idx_g.resize(N * M);
  csr_offset_g.resize(N * M);
  data_g.resize(N * M);

  std::vector<UnifiedOwnedBufferVertexID> unified_csr_n_vertices_g;
  std::vector<UnifiedOwnedBufferVertexID> unified_csr_n_edges_g;
  std::vector<UnifiedOwnedBufferVertexID> unified_tile_offset_row_g;
  std::vector<UnifiedOwnedBufferVertexID> unified_tile_row_idx_g;
  std::vector<UnifiedOwnedBufferVertexID> unified_tile_col_idx_g;
  std::vector<UnifiedOwnedBufferUint64> unified_csr_offset_g;
  std::vector<UnifiedOwnedBufferUint8> unified_data_g;

  unified_csr_n_vertices_g.resize(N * M);
  unified_csr_n_edges_g.resize(N * M);
  unified_tile_offset_row_g.resize(N * M);
  unified_tile_row_idx_g.resize(N * M);
  unified_tile_col_idx_g.resize(N * M);
  unified_csr_offset_g.resize(N * M);
  unified_data_g.resize(N * M);

  std::for_each(
      // std::execution::par,
      worker.begin(), worker.end(),
      [this, M, N, step, &g, &p_streams_vec, tile_size, tile_buffer_size, &mtx,
       &tile_offset_row_g, &tile_row_idx_g, &tile_col_idx_g, &csr_offset_g,
       &data_g, &unified_tile_offset_row_g, &unified_tile_row_idx_g,
       &unified_tile_col_idx_g, &unified_csr_offset_g, &unified_data_g,
       &unified_csr_n_vertices_g, &unified_csr_n_edges_g](auto w) {
        for (VertexID i = w; i < M; i += step) {
          for (VertexID k = 0; k < N; k++) {
            auto block = g.GetTiledMatrixPtrByIdx(i * N + k);
            if (block->GetMetadata().n_nz_tile == 0)
              continue;

            tile_offset_row_g[i * N + k].data = block->GetTileOffsetRowPtr();
            tile_offset_row_g[i * N + k].size =
                sizeof(VertexID) * (block->GetMetadata().n_strips + 1);

            tile_row_idx_g[i * N + k].data = block->GetTileRowIdxPtr();
            tile_row_idx_g[i * N + k].size =
                block->GetMetadata().n_nz_tile * sizeof(VertexID);

            tile_col_idx_g[i * N + k].data = block->GetTileColIdxPtr();
            tile_col_idx_g[i * N + k].size =
                block->GetMetadata().n_nz_tile * sizeof(VertexID);
            csr_offset_g[i * N + k].data = block->GetCSROffsetPtr();
            csr_offset_g[i * N + k].size =
                block->GetMetadata().n_nz_tile * sizeof(uint64_t);

            data_g[i * N + k].data = block->GetDataPtr();
            data_g[i * N + k].size = block->GetDataBufferSize();

            {
              std::lock_guard<std::mutex> lock(mtx);
              unified_tile_offset_row_g[i * N + k].Init(
                  tile_offset_row_g[i * N + k]);
              unified_tile_row_idx_g[i * N + k].Init(tile_row_idx_g[i * N + k]);
              unified_tile_col_idx_g[i * N + k].Init(tile_col_idx_g[i * N + k]);
              unified_csr_offset_g[i * N + k].Init(csr_offset_g[i * N + k]);
              unified_data_g[i * N + k].Init(data_g[i * N + k]);

              auto subgraph_metadata = block->GetCSRMetadata();
              unified_csr_n_vertices_g[i * N + k].Init(
                  sizeof(uint32_t) * block->GetMetadata().n_nz_tile);
              unified_csr_n_edges_g[i * N + k].Init(
                  sizeof(uint64_t) * block->GetMetadata().n_nz_tile);
              for (VertexID gid = 0; gid < block->GetMetadata().n_nz_tile;
                   gid++) {
                unified_csr_n_vertices_g[i * N + k].SetElement(
                    subgraph_metadata[gid].num_vertices, gid);
                unified_csr_n_edges_g[i * N + k].SetElement(
                    subgraph_metadata[gid].num_outgoing_edges, gid);
              }
            }
          }
        }
      });

  BufferVertexLabel v_label_g;
  UnifiedOwnedBufferVertexLabel unified_v_label_g;

  v_label_g.data = g.GetVLabelBasePointer();
  v_label_g.size = sizeof(VertexLabel) * g.get_metadata().n_vertices;
  unified_v_label_g.Init(v_label_g);

  // Init output.
  std::vector<UnifiedOwnedBufferVertexID> unified_edgelist_m;
  unified_edgelist_m.resize(p.get_num_outgoing_edges());
  std::vector<UnifiedOwnedBufferEdgeIndex> unified_output_offset_m;
  unified_output_offset_m.resize(p.get_num_outgoing_edges());

  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, step, &p_streams_vec, &mtx, &unified_edgelist_m,
       &unified_output_offset_m, &p](auto w) {
        for (VertexID _ = w; _ < p.get_num_outgoing_edges(); _ += step) {
          unified_edgelist_m[_].Init(sizeof(VertexID) * kMaxNumCandidates * 2);
          unified_output_offset_m[_].Init(sizeof(EdgeIndex));
        }
      });

  cudaDeviceSynchronize();

  // Start Matching ...
  std::for_each(
      // std::execution::par,
      worker.begin(), worker.end(),
      [this, M, N, step, &p, &g, &p_streams_vec, tile_size, tile_buffer_size,
       n_strips, &mtx, &unified_data_p, &unified_v_label_p,
       &unified_csr_n_vertices_g, &unified_csr_n_edges_g,
       &unified_tile_offset_row_g, &unified_tile_row_idx_g,
       &unified_tile_col_idx_g, &unified_csr_offset_g, &unified_data_g,
       &unified_edgelist_m, &unified_output_offset_m](auto w) {
        for (VertexID i = w; i < M; i += step) {
          for (VertexID j = 0; j < N; j++) {
            auto block = g.GetTiledMatrixPtrByIdx(i * N + j);
            if (block == nullptr || block->GetMetadata().n_nz_tile == 0)
              continue;

            cudaSetDevice(common::hash_function(i * N + j) % 4);

            {
              std::lock_guard<std::mutex> lock(mtx);
              cudaStream_t &p_stream = p_streams_vec[i * N + j];
              SubIsoKernelWrapper::SubIso(
                  p_stream, tile_size, n_strips, block->GetMetadata().n_nz_tile,
                  unified_data_p, unified_v_label_p,
                  unified_csr_n_vertices_g[i * N + j],
                  unified_csr_n_edges_g[i * N + j],
                  unified_tile_offset_row_g[i * j + j],
                  unified_tile_row_idx_g[i * N + j],
                  unified_tile_col_idx_g[i * N + j],
                  unified_csr_offset_g[i * N + j], unified_data_g[i * N + j],
                  unified_edgelist_m, unified_output_offset_m);
            }
          }
        }
      });

  cudaDeviceSynchronize();
}

__host__ void SubIso::Run() {

  auto start_time_0 = std::chrono::system_clock::now();
  LoadData();
  auto start_time_1 = std::chrono::system_clock::now();
  InitLabel();

  auto start_time_2 = std::chrono::system_clock::now();

  Matching(p_, *g_);

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