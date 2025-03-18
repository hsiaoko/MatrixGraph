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
  uint64_t* in_visited_bitmap_data;
  uint64_t* out_visited_bitmap_data;
};

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

  out_visited.SetBit(1);
  // for (VertexID vid = tid; vid < params.n_vertices_g; vid += step) {
  //   printf("%d ", vid);
  // }
}

void WCCKernelWrapper::WCC(
    const cudaStream_t& stream, VertexID n_vertices_g, EdgeIndex n_edges_g,
    const data_structures::UnifiedOwnedBuffer<uint8_t>& data_g,
    const data_structures::UnifiedOwnedBuffer<VertexLabel>& v_label_g) {
  dim3 dimBlock(kBlockDim);
  dim3 dimGrid(kGridDim);

  // The default heap size is 8M.
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8388608 * 128);
  auto time1 = std::chrono::system_clock::now();

  // Initialize.
  uint64_t* in_visited_bitmap_data;
  CUDA_CHECK(cudaMallocManaged(&in_visited_bitmap_data,
                               KERNEL_WORD_OFFSET(n_vertices_g)));
  uint64_t* out_visited_bitmap_data;
  CUDA_CHECK(cudaMallocManaged(&out_visited_bitmap_data,
                               KERNEL_WORD_OFFSET(n_vertices_g)));

  BitmapNoOwnerShip out_visited(n_vertices_g, out_visited_bitmap_data);
  std::cout << "Count: " << out_visited.Count() << std::endl;
  ParametersWCC params{
      .n_vertices_g = n_vertices_g,
      .n_edges_g = n_edges_g,
      .data_g = data_g.GetPtr(),
      .v_label_g = v_label_g.GetPtr(),
      .in_visited_bitmap_data = in_visited_bitmap_data,
      .out_visited_bitmap_data = out_visited_bitmap_data,
  };

  HashMinKernel<<<dimGrid, dimBlock, 0, stream>>>(params);
  cudaStreamSynchronize(stream);
  auto time2 = std::chrono::system_clock::now();

  std::cout << "Count: " << out_visited.Count() << std::endl;

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
}

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
