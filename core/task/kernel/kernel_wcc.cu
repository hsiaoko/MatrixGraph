#include <cuda_runtime.h>

#include <chrono>
#include <iostream>

#include "core/common/consts.h"
#include "core/common/types.h"
#include "core/task/kernel/data_structures/kernel_bitmap.cuh"
#include "core/task/kernel/data_structures/kernel_bitmap_no_ownership.cuh"
#include "core/task/kernel/data_structures/mini_kernel_bitmap.cuh"
#include "core/task/kernel/kernel_wcc.cuh"
#include "core/util/bitmap_no_ownership.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
using sics::matrixgraph::core::common::kMaxNumCandidatesPerThread;
using VertexID = sics::matrixgraph::core::common::VertexID;
using VertexID = sics::matrixgraph::core::common::VertexID;
using sics::matrixgraph::core::common::kBlockDim;
using sics::matrixgraph::core::common::kGridDim;
using sics::matrixgraph::core::common::kMaxNumWeft;
using sics::matrixgraph::core::common::kMaxVertexID;
using sics::matrixgraph::core::task::kernel::HostKernelBitmap;
using sics::matrixgraph::core::task::kernel::HostMiniKernelBitmap;
using sics::matrixgraph::core::task::kernel::KernelBitmap;
using sics::matrixgraph::core::task::kernel::KernelBitmapNoOwnership;
using sics::matrixgraph::core::task::kernel::MiniKernelBitmap;
using sics::matrixgraph::core::util::BitmapNoOwnerShip;

struct ParametersWCC {
  VertexID n_vertices_g;
  EdgeIndex n_edges_g;
  uint8_t* data_g;
  VertexLabel* v_label_g;
  VertexID* in_active_vertices;
  VertexID* out_active_vertices;
  VertexID* in_active_vertices_offset;
  VertexID* out_active_vertices_offset;
  uint64_t* in_visited_bitmap_data;
  uint64_t* out_visited_bitmap_data;
};

static __global__ void InitKernel(ParametersWCC params) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;
  for (VertexID v_idx = tid; v_idx < params.n_vertices_g; v_idx += step) {
    params.v_label_g[v_idx] = v_idx;
    params.in_active_vertices[v_idx] = v_idx;
  }
}

static __global__ void HashMinKernel(ParametersWCC params) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  VertexID* globalid_g = (VertexID*)(params.data_g);
  VertexID* in_degree_g = globalid_g + params.n_vertices_g;
  VertexID* out_degree_g = in_degree_g + params.n_vertices_g;
  EdgeIndex* in_offset_g = (EdgeIndex*)(out_degree_g + params.n_vertices_g);
  EdgeIndex* out_offset_g = (EdgeIndex*)(in_offset_g + params.n_vertices_g + 1);
  EdgeIndex* in_edges_g = (EdgeIndex*)(out_offset_g + params.n_vertices_g + 1);
  VertexID* out_edges_g = in_edges_g + params.n_edges_g;
  VertexID* edges_globalid_by_localid_g = out_edges_g + params.n_edges_g;

  KernelBitmapNoOwnership in_visited(params.n_vertices_g,
                                     params.in_visited_bitmap_data);
  KernelBitmapNoOwnership out_visited(params.n_vertices_g,
                                      params.out_visited_bitmap_data);

  for (VertexID v_idx = tid; v_idx < params.n_vertices_g; v_idx += step) {
    if (!in_visited.GetBit(v_idx)) continue;
    EdgeIndex v_offset_base = out_offset_g[v_idx];

    VertexLabel v_label = params.v_label_g[v_idx];
    for (VertexID nbr_v_idx = 0; nbr_v_idx < out_degree_g[v_idx]; nbr_v_idx++) {
      VertexID nbr_v = out_edges_g[v_offset_base + nbr_v_idx];

      // if (nbr_v >= 65608366) {
      //   printf("%d/ 65608366\n", nbr_v);
      // }

      VertexLabel label_nbr_v = *(params.v_label_g + nbr_v);

      if (label_nbr_v > v_label) {
        atomicMin(params.v_label_g + nbr_v, v_label);
        out_visited.SetBit(nbr_v);
      }
    }
  }
}

static __global__ void HashMinKernelActiveVertices(ParametersWCC params) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;
  VertexID* globalid_g = (VertexID*)(params.data_g);
  VertexID* in_degree_g = globalid_g + params.n_vertices_g;
  VertexID* out_degree_g = in_degree_g + params.n_vertices_g;
  EdgeIndex* in_offset_g = (EdgeIndex*)(out_degree_g + params.n_vertices_g);
  EdgeIndex* out_offset_g = (EdgeIndex*)(in_offset_g + params.n_vertices_g + 1);
  EdgeIndex* in_edges_g = (EdgeIndex*)(out_offset_g + params.n_vertices_g + 1);
  VertexID* out_edges_g = in_edges_g + params.n_edges_g;
  VertexID* edges_globalid_by_localid_g = out_edges_g + params.n_edges_g;

  KernelBitmapNoOwnership in_visited(params.n_vertices_g,
                                     params.in_visited_bitmap_data);
  KernelBitmapNoOwnership out_visited(params.n_vertices_g,
                                      params.out_visited_bitmap_data);
  for (VertexID offset = tid; offset < *(params.in_active_vertices_offset);
       offset += step) {
    auto v_idx = params.in_active_vertices[offset];
    EdgeIndex v_offset_base = in_offset_g[v_idx];

    VertexLabel v_label = params.v_label_g[v_idx];
    VertexLabel min_label = v_label;

    for (VertexID nbr_v_idx = 0; nbr_v_idx < in_degree_g[v_idx]; nbr_v_idx++) {
      VertexID nbr_v = in_edges_g[v_offset_base + nbr_v_idx];

      VertexLabel label_nbr_v = *(params.v_label_g + nbr_v);

      if (label_nbr_v < v_label) {
        min_label = label_nbr_v;
      }
    }
    if (min_label < v_label) {
      atomicMin(params.v_label_g + v_idx, min_label);
      out_visited.SetBit(v_idx);
      auto offset = atomicAdd(params.out_active_vertices_offset, VertexID(1));
      // params.out_active_vertices[offset] = v_idx;
      //  printf("%d->%d ", label_nbr_v, v_label);
      //  printf("active: %d\n", params.out_active_vertices[offset]);
    }
  }
}

void WCCKernelWrapper::WCC(
    const cudaStream_t& stream, VertexID n_vertices_g, EdgeIndex n_edges_g,
    const data_structures::UnifiedOwnedBuffer<uint8_t>& data_g,
    const data_structures::UnifiedOwnedBuffer<VertexLabel>& v_label_g) {
  dim3 dimBlock(kBlockDim);
  dim3 dimGrid(kGridDim);
  // dim3 dimBlock(1);
  // dim3 dimGrid(1);

  // The default heap size is 8M.
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8388608 * 128);

  // Initialize.
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
  *in_active_vertices_offset = n_vertices_g;
  *out_active_vertices_offset = 0;

  BitmapNoOwnerShip out_visited(n_vertices_g, out_visited_bitmap_data);
  BitmapNoOwnerShip in_visited(n_vertices_g, in_visited_bitmap_data);

  ParametersWCC params{.n_vertices_g = n_vertices_g,
                       .n_edges_g = n_edges_g,
                       .data_g = data_g.GetPtr(),
                       .v_label_g = v_label_g.GetPtr(),
                       .in_active_vertices = in_active_vertices,
                       .out_active_vertices = out_active_vertices,
                       .in_active_vertices_offset = in_active_vertices_offset,
                       .out_active_vertices_offset = out_active_vertices_offset,
                       .in_visited_bitmap_data = in_visited.data(),
                       .out_visited_bitmap_data = out_visited.data()};

  InitKernel<<<dimGrid, dimBlock, 0, stream>>>(params);

  cudaStreamSynchronize(stream);

  in_visited.Fill();
  size_t round = 0;
  auto time1 = std::chrono::system_clock::now();
  while (!in_visited.IsEmpty()) {
    ParametersWCC params{
        .n_vertices_g = n_vertices_g,
        .n_edges_g = n_edges_g,
        .data_g = data_g.GetPtr(),
        .v_label_g = v_label_g.GetPtr(),
        .in_active_vertices = in_active_vertices,
        .out_active_vertices = out_active_vertices,
        .in_active_vertices_offset = in_active_vertices_offset,
        .out_active_vertices_offset = out_active_vertices_offset,
        .in_visited_bitmap_data = in_visited.data(),
        .out_visited_bitmap_data = out_visited.data()};

    // std::cout << "Round " << round++
    //           << " Active vertices: " << in_visited.Count() << std::endl;
    std::cout << "Round " << round++
              << " Active vertices: " << *(in_active_vertices_offset)
              << std::endl;
    // HashMinKernel<<<dimGrid, dimBlock, 0, stream>>>(params);
    HashMinKernelActiveVertices<<<dimGrid, dimBlock, 0, stream>>>(params);
    cudaStreamSynchronize(stream);

    std::swap(in_visited, out_visited);
    std::swap(in_active_vertices, out_active_vertices);
    std::swap(in_active_vertices_offset, out_active_vertices_offset);
    cudaMemset(out_active_vertices, 0,
               sizeof(VertexID) * *(out_active_vertices_offset));
    *(out_active_vertices_offset) = 0;
    out_visited.Clear();
  }
  auto time2 = std::chrono::system_clock::now();

  for (int i = 0; i < 5; i++) {
    std::cout << v_label_g.GetPtr()[i] << " ";
  }
  std::cout << "[WCC]:"
            << std::chrono::duration_cast<std::chrono::microseconds>(time2 -
                                                                     time1)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << "\n\t HashMin:"
            << std::chrono::duration_cast<std::chrono::microseconds>(time2 -
                                                                     time1)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    CUDA_CHECK(err);
  }
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
