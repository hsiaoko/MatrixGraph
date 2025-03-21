#include <cuda_runtime.h>

#include <chrono>
#include <iostream>

#include "core/common/consts.h"
#include "core/common/types.h"
#include "core/task/kernel/data_structures/kernel_bitmap.cuh"
#include "core/task/kernel/data_structures/kernel_bitmap_no_ownership.cuh"
#include "core/task/kernel/data_structures/mini_kernel_bitmap.cuh"
#include "core/task/kernel/kernel_bfs.cuh"
#include "core/util/bitmap_no_ownership.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
using VertexID = sics::matrixgraph::core::common::VertexID;
using sics::matrixgraph::core::common::kBlockDim;
using sics::matrixgraph::core::common::kGridDim;
using sics::matrixgraph::core::common::kMaxVertexID;
using sics::matrixgraph::core::task::kernel::KernelBitmapNoOwnership;
using BitmapNoOwnerShip = sics::matrixgraph::core::util::BitmapNoOwnerShip;

struct ParametersBFS {
  VertexID n_vertices_g;
  EdgeIndex n_edges_g;
  uint8_t* data_g;
  VertexLabel* v_level_g;
  VertexID* in_active_vertices;
  VertexID* out_active_vertices;
  VertexID* in_active_vertices_offset;
  VertexID* out_active_vertices_offset;
  uint64_t* in_visited_bitmap_data;
  uint64_t* out_visited_bitmap_data;
  uint64_t* visited_bitmap_data;
  VertexLabel current_level;
};

BFSKernelWrapper* BFSKernelWrapper::GetInstance() {
  if (ptr_ == nullptr) {
    ptr_ = new BFSKernelWrapper();
  }
  return ptr_;
}

static __global__ void InitKernel(ParametersBFS params,
                                  VertexID source_vertex) {
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int step = blockDim.x * gridDim.x;

  for (VertexID v_idx = tid; v_idx < params.n_vertices_g; v_idx += step) {
    params.v_level_g[v_idx] = kMaxVertexID;  // Initialize to infinity
    if (v_idx == source_vertex) {
      params.v_level_g[v_idx] = 0;  // Source vertex has level 0
      params.in_active_vertices[0] = v_idx;
      *(params.in_active_vertices_offset) = 1;
    }
  }
}

static __global__ void BFSKernel(ParametersBFS params) {
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int step = blockDim.x * gridDim.x;

  VertexID* const globalid_g = reinterpret_cast<VertexID*>(params.data_g);
  VertexID* const in_degree_g = globalid_g + params.n_vertices_g;
  VertexID* const out_degree_g = in_degree_g + params.n_vertices_g;
  EdgeIndex* const in_offset_g =
      reinterpret_cast<EdgeIndex*>(out_degree_g + params.n_vertices_g);
  EdgeIndex* const out_offset_g = in_offset_g + params.n_vertices_g + 1;
  EdgeIndex* const in_edges_g = out_offset_g + params.n_vertices_g + 1;
  VertexID* const out_edges_g =
      reinterpret_cast<VertexID*>(in_edges_g + params.n_edges_g);

  KernelBitmapNoOwnership in_visited(params.n_vertices_g,
                                     params.in_visited_bitmap_data);
  KernelBitmapNoOwnership out_visited(params.n_vertices_g,
                                      params.out_visited_bitmap_data);
  KernelBitmapNoOwnership visited(params.n_vertices_g,
                                  params.visited_bitmap_data);

  for (VertexID offset = tid; offset < *(params.in_active_vertices_offset);
       offset += step) {
    VertexID v_idx = params.in_active_vertices[offset];
    EdgeIndex v_offset_base = out_offset_g[v_idx];
    VertexLabel v_level = params.v_level_g[v_idx];

    for (VertexID nbr_v_idx = 0; nbr_v_idx < out_degree_g[v_idx]; nbr_v_idx++) {
      VertexID nbr_v = out_edges_g[v_offset_base + nbr_v_idx];
      VertexLabel label_nbr_v = params.v_level_g[nbr_v];

      if (label_nbr_v > v_level + 1) {
        atomicMin(params.v_level_g + nbr_v, v_level + 1);
        out_visited.SetBit(nbr_v);
        auto new_offset =
            atomicAdd(params.out_active_vertices_offset, VertexID(1));
        params.out_active_vertices[new_offset] = nbr_v;
      }
    }
  }
}

void BFSKernelWrapper::BFS(
    const cudaStream_t& stream, VertexID n_vertices_g, EdgeIndex n_edges_g,
    VertexID source_vertex,
    const data_structures::UnifiedOwnedBuffer<uint8_t>& data_g,
    const data_structures::UnifiedOwnedBuffer<VertexLabel>& v_level_g) {
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8388608 * 128);

  uint64_t* visited_bitmap_data;
  CUDA_CHECK(cudaMallocManaged(
      &visited_bitmap_data,
      sizeof(uint64_t) * (KERNEL_WORD_OFFSET(n_vertices_g) + 1)));
  uint64_t* in_visited_bitmap_data;
  CUDA_CHECK(cudaMallocManaged(
      &in_visited_bitmap_data,
      sizeof(uint64_t) * (KERNEL_WORD_OFFSET(n_vertices_g) + 1)));
  uint64_t* out_visited_bitmap_data;
  CUDA_CHECK(cudaMallocManaged(
      &out_visited_bitmap_data,
      sizeof(uint64_t) * (KERNEL_WORD_OFFSET(n_vertices_g) + 1)));

  VertexID* in_active_vertices;
  VertexID* out_active_vertices;
  VertexID* in_active_vertices_offset;
  VertexID* out_active_vertices_offset;
  CUDA_CHECK(
      cudaMallocManaged(&in_active_vertices, sizeof(VertexID) * n_vertices_g));
  CUDA_CHECK(
      cudaMallocManaged(&out_active_vertices, sizeof(VertexID) * n_vertices_g));
  CUDA_CHECK(cudaMallocManaged(&in_active_vertices_offset, sizeof(VertexID)));
  CUDA_CHECK(cudaMallocManaged(&out_active_vertices_offset, sizeof(VertexID)));
  *in_active_vertices_offset = 0;  // Will be set to 1 in InitKernel
  *out_active_vertices_offset = 0;

  BitmapNoOwnerShip out_visited(n_vertices_g, out_visited_bitmap_data);
  BitmapNoOwnerShip in_visited(n_vertices_g, in_visited_bitmap_data);
  BitmapNoOwnerShip visited(n_vertices_g, visited_bitmap_data);

  ParametersBFS params{.n_vertices_g = n_vertices_g,
                       .n_edges_g = n_edges_g,
                       .data_g = data_g.GetPtr(),
                       .v_level_g = v_level_g.GetPtr(),
                       .in_active_vertices = in_active_vertices,
                       .out_active_vertices = out_active_vertices,
                       .in_active_vertices_offset = in_active_vertices_offset,
                       .out_active_vertices_offset = out_active_vertices_offset,
                       .in_visited_bitmap_data = in_visited.data(),
                       .out_visited_bitmap_data = out_visited.data(),
                       .visited_bitmap_data = visited.data(),
                       .current_level = 0};

  dim3 dimBlock(kBlockDim);
  dim3 dimGrid(kGridDim);

  InitKernel<<<dimGrid, dimBlock, 0, stream>>>(params, source_vertex);
  cudaStreamSynchronize(stream);

  auto time1 = std::chrono::system_clock::now();

  while (*(params.in_active_vertices_offset) > 0) {
    std::cout << "Level " << params.current_level
              << " Active vertices: " << *(params.in_active_vertices_offset)
              << std::endl;

    BFSKernel<<<dimGrid, dimBlock, 0, stream>>>(params);
    cudaStreamSynchronize(stream);

    std::swap(in_visited, out_visited);
    std::swap(in_active_vertices, out_active_vertices);
    std::swap(in_active_vertices_offset, out_active_vertices_offset);
    params.in_active_vertices = in_active_vertices;
    params.out_active_vertices = out_active_vertices;
    params.in_active_vertices_offset = in_active_vertices_offset;
    params.out_active_vertices_offset = out_active_vertices_offset;
    params.in_visited_bitmap_data = in_visited.data();
    params.out_visited_bitmap_data = out_visited.data();

    *(params.out_active_vertices_offset) = 0;
    out_visited.Clear();
    visited.Clear();
    params.current_level++;
  }

  auto time2 = std::chrono::system_clock::now();

  std::cout << "[BFS]:"
            << std::chrono::duration_cast<std::chrono::microseconds>(time2 -
                                                                     time1)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << " seconds, " << params.current_level << " levels" << std::endl;

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    CUDA_CHECK(err);
  }

  cudaFree(visited_bitmap_data);
  cudaFree(in_visited_bitmap_data);
  cudaFree(out_visited_bitmap_data);
  cudaFree(in_active_vertices);
  cudaFree(out_active_vertices);
  cudaFree(in_active_vertices_offset);
  cudaFree(out_active_vertices_offset);
}

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
