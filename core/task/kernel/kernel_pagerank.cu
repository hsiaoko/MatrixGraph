#include <cuda_runtime.h>

#include <chrono>
#include <iostream>

#include "core/common/consts.h"
#include "core/common/types.h"
#include "core/task/kernel/kernel_pagerank.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using VertexID = sics::matrixgraph::core::common::VertexID;
using sics::matrixgraph::core::common::kBlockDim;
using sics::matrixgraph::core::common::kGridDim;

struct ParametersPageRank {
  VertexID n_vertices_g;
  EdgeIndex n_edges_g;
  uint8_t* data_g;
  float* curr_page_ranks;
  float* next_page_ranks;
  float damping_factor;
  float epsilon;
};

static __global__ void InitKernel(ParametersPageRank params) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  float init_rank = 1.0f / params.n_vertices_g;
  for (VertexID v_idx = tid; v_idx < params.n_vertices_g; v_idx += step) {
    params.curr_page_ranks[v_idx] = init_rank;
    params.next_page_ranks[v_idx] = 0.0f;
  }
}

static __global__ void PageRankKernel(ParametersPageRank params) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  VertexID* globalid_g = (VertexID*)(params.data_g);
  VertexID* in_degree_g = globalid_g + params.n_vertices_g;
  VertexID* out_degree_g = in_degree_g + params.n_vertices_g;
  EdgeIndex* in_offset_g = (EdgeIndex*)(out_degree_g + params.n_vertices_g);
  EdgeIndex* out_offset_g = (EdgeIndex*)(in_offset_g + params.n_vertices_g + 1);
  EdgeIndex* in_edges_g = (EdgeIndex*)(out_offset_g + params.n_vertices_g + 1);
  VertexID* out_edges_g = in_edges_g + params.n_edges_g;

  // Process each vertex
  for (VertexID v_idx = tid; v_idx < params.n_vertices_g; v_idx += step) {
    float sum = 0.0f;
    EdgeIndex v_offset_base = in_offset_g[v_idx];

    // Sum up contributions from incoming edges
    for (VertexID nbr_idx = 0; nbr_idx < in_degree_g[v_idx]; nbr_idx++) {
      VertexID nbr_v = in_edges_g[v_offset_base + nbr_idx];
      float nbr_rank = params.curr_page_ranks[nbr_v];
      float out_degree = static_cast<float>(out_degree_g[nbr_v]);
      if (out_degree > 0) {
        sum += nbr_rank / out_degree;
      }
    }

    // Calculate new PageRank value
    params.next_page_ranks[v_idx] =
        (1.0f - params.damping_factor) / params.n_vertices_g +
        params.damping_factor * sum;
  }
}

static __global__ void SwapAndCheckConvergenceKernel(ParametersPageRank params,
                                                     float* max_diff) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  __shared__ float block_max_diff;
  if (threadIdx.x == 0) {
    block_max_diff = 0.0f;
  }
  __syncthreads();

  for (VertexID v_idx = tid; v_idx < params.n_vertices_g; v_idx += step) {
    float diff =
        fabsf(params.next_page_ranks[v_idx] - params.curr_page_ranks[v_idx]);
    atomicMax((int*)&block_max_diff, __float_as_int(diff));
    params.curr_page_ranks[v_idx] = params.next_page_ranks[v_idx];
    params.next_page_ranks[v_idx] = 0.0f;
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    atomicMax((int*)max_diff, __float_as_int(block_max_diff));
  }
}

PageRankKernelWrapper* PageRankKernelWrapper::GetInstance() {
  if (ptr_ == nullptr) {
    ptr_ = new PageRankKernelWrapper();
  }
  return ptr_;
}

void PageRankKernelWrapper::PageRank(
    const cudaStream_t& stream, VertexID n_vertices_g, EdgeIndex n_edges_g,
    const data_structures::UnifiedOwnedBuffer<uint8_t>& data_g,
    data_structures::UnifiedOwnedBuffer<float>& page_ranks,
    float damping_factor, float epsilon, int max_iterations) {
  dim3 dimBlock(kBlockDim);
  dim3 dimGrid(kGridDim);

  // Allocate device memory for next_page_ranks and max_diff
  float* d_next_page_ranks;
  float* d_max_diff;
  cudaMalloc(&d_next_page_ranks, sizeof(float) * n_vertices_g);
  cudaMalloc(&d_max_diff, sizeof(float));

  ParametersPageRank params{
      n_vertices_g,      n_edges_g,      data_g.GetPtr(), page_ranks.GetPtr(),
      d_next_page_ranks, damping_factor, epsilon};

  // Initialize PageRank values
  InitKernel<<<dimGrid, dimBlock, 0, stream>>>(params);
  cudaStreamSynchronize(stream);

  // Main PageRank iteration loop
  int iteration = 0;
  float h_max_diff = epsilon + 1.0f;  // Ensure we enter the loop

  while (iteration < max_iterations && h_max_diff > epsilon) {
    // Reset max_diff for this iteration
    cudaMemset(d_max_diff, 0, sizeof(float));

    // Compute new PageRank values
    PageRankKernel<<<dimGrid, dimBlock, 0, stream>>>(params);

    // Swap buffers and check for convergence
    SwapAndCheckConvergenceKernel<<<dimGrid, dimBlock, 0, stream>>>(params,
                                                                    d_max_diff);

    // Copy max_diff back to host
    cudaMemcpyAsync(&h_max_diff, d_max_diff, sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    iteration++;
  }

  std::cout << "[PageRank] Converged after " << iteration
            << " iterations with diff: " << h_max_diff << std::endl;

  // Clean up
  cudaFree(d_next_page_ranks);
  cudaFree(d_max_diff);
}

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
