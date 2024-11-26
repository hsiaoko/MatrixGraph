#include <cuda_runtime.h>
#include <iostream>

#include "core/common/consts.h"
#include "core/common/types.h"
#include "core/task/kernel/data_structures/kernel_bitmap.cuh"
#include "core/task/kernel/kernel_subiso.cuh"

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
using sics::matrixgraph::core::task::kernel::KernelBitmap;

struct LocalMatches {
  VertexID *data;
  VertexID *offset;
};

struct ParametersSubIso {
  VertexID depth_p;
  VertexID *exec_path;
  VertexID n_vertices_p;
  EdgeIndex n_edges_p;
  uint8_t *data_p;
  VertexLabel *v_label_p;
  VertexID n_vertices_g;
  EdgeIndex n_edges_g;
  uint8_t *data_g;
  VertexLabel *v_label_g;
  VertexID **m_ptr;
  EdgeIndex *m_offset;
};
// candidates in M should be all index instead of globalid.

static __device__ void get_first_matches_kernel(const ParametersSubIso &params,
                                                VertexID u, VertexID *m0,
                                                VertexID *m0_offset) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  VertexID *globalid_g = (VertexID *)(params.data_g);
  VertexID *in_degree_g = globalid_g + params.n_vertices_g;
  VertexID *out_degree_g = in_degree_g + params.n_vertices_g;
  EdgeIndex *in_offset_g = (EdgeIndex *)(out_degree_g + params.n_vertices_g);
  EdgeIndex *out_offset_g =
      (EdgeIndex *)(in_offset_g + params.n_vertices_g + 1);
  EdgeIndex *in_edges_g = (EdgeIndex *)(out_offset_g + params.n_vertices_g + 1);
  VertexID *out_edges_g = in_edges_g + params.n_edges_g;
  VertexID *edges_globalid_by_localid_g = out_edges_g + params.n_edges_g;

  auto v_label_p = params.v_label_p[u];
  for (VertexID v_idx = tid; v_idx < params.n_vertices_g; v_idx += step) {
    auto global_id = globalid_g[v_idx];
    auto v_label_g = params.v_label_g[global_id];
    if (v_label_g == v_label_p) {
      auto local_offset = atomicAdd((unsigned *)params.m_offset, 1);
      m0[local_offset] = v_idx;
    }
  }
}

static __device__ bool label_filter_kernel() { return true; }

static __device__ void dfs_kernel(const ParametersSubIso &params,
                                  uint32_t level, VertexID v_idx,
                                  KernelBitmap &global_visited,
                                  KernelBitmap local_visited, bool &match,
                                  LocalMatches &local_matches) {

  if (level > params.depth_p)
    return;

  VertexID exec_plan_idx = local_visited.Count();
  VertexID u = params.exec_path[exec_plan_idx];
  VertexLabel u_label = params.v_label_p[u];

  VertexID global_exec_plan_idx = global_visited.Count();

  VertexID *globalid_g = (VertexID *)(params.data_g);
  VertexID *in_degree_g = globalid_g + params.n_vertices_g;
  VertexID *out_degree_g = in_degree_g + params.n_vertices_g;
  EdgeIndex *in_offset_g = (EdgeIndex *)(out_degree_g + params.n_vertices_g);
  EdgeIndex *out_offset_g =
      (EdgeIndex *)(in_offset_g + params.n_vertices_g + 1);
  EdgeIndex *in_edges_g = (EdgeIndex *)(out_offset_g + params.n_vertices_g + 1);
  VertexID *out_edges_g = in_edges_g + params.n_edges_g;
  VertexID *edges_globalid_by_localid_g = out_edges_g + params.n_edges_g;

  // Plast filter function gere.
  // printf("dfs to %d, u %d, u label %d, global_exe_idx %d\n",
  // globalid_g[v_idx],
  //       u, u_label, global_exec_plan_idx);
  // printf("evel %d\n",level);
  VertexLabel v_label = params.v_label_g[globalid_g[v_idx]];

  if (v_label != u_label)
    return;

  global_visited.SetBit(u);
  local_matches.offset[exec_plan_idx]++;

  if (global_exec_plan_idx == params.n_vertices_p - 1) {
    printf("#match end at %d\n", globalid_g[v_idx]);
    match = true;
    return;
  }

  EdgeIndex offset_base = out_offset_g[v_idx];
  for (VertexID nbr_idx = 0; nbr_idx < out_degree_g[v_idx]; nbr_idx++) {
    VertexID candidate_v_idx = *(out_edges_g + offset_base + nbr_idx);
    VertexID candidate_global_v_id = globalid_g[candidate_v_idx];
    VertexLabel candidate_v_label = params.v_label_g[candidate_global_v_id];
    printf(" \t pre %d candidate %d, level %d, v_idx: %d, %d/%d\n",
           globalid_g[v_idx], candidate_global_v_id, level, v_idx, nbr_idx,
           out_degree_g[v_idx]);
    dfs_kernel(params, level + 1, candidate_v_idx, global_visited,
               global_visited, match, local_matches);
  }
}

static __device__ void extend_kernel(const ParametersSubIso &params,
                                     uint32_t level) {

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  VertexID u_idx = 0;

  KernelBitmap visited(params.n_vertices_p);

  VertexID *globalid_g = (VertexID *)(params.data_g);
  VertexID *in_degree_g = globalid_g + params.n_vertices_g;
  VertexID *out_degree_g = in_degree_g + params.n_vertices_g;
  EdgeIndex *in_offset_g = (EdgeIndex *)(out_degree_g + params.n_vertices_g);
  EdgeIndex *out_offset_g =
      (EdgeIndex *)(in_offset_g + params.n_vertices_g + 1);
  EdgeIndex *in_edges_g = (EdgeIndex *)(out_offset_g + params.n_vertices_g + 1);
  VertexID *out_edges_g = in_edges_g + params.n_edges_g;
  VertexID *edges_globalid_by_localid_g = out_edges_g + params.n_edges_g;

  LocalMatches local_matches;
  local_matches.data = (VertexID *)malloc(sizeof(VertexID) * params.n_edges_p *
                                          2 * kMaxNumCandidatesPerThread);

  local_matches.offset =
      (VertexID *)malloc(sizeof(VertexID) * params.n_vertices_p);

  printf("EXTEND: %d\n", local_matches.offset);
  for (VertexID candidate_idx = tid; candidate_idx < params.m_offset[level];
       candidate_idx += step) {
    visited.Clear();
    memset(local_matches.data, 0,
           params.n_edges_p * 2 * kMaxNumCandidatesPerThread *
               sizeof(VertexID));
    memset(local_matches.offset, 0, params.n_vertices_p * sizeof(VertexID));

    VertexID v_idx = params.m_ptr[level][candidate_idx];
    VertexID global_vid = globalid_g[v_idx];

    bool match = false;
    dfs_kernel(params, level, v_idx, visited, visited, match, local_matches);

    printf("#done!\n");
    // Write result to the global buffer.
    if (match) {
      for (auto _ = 0; _ < params.n_vertices_p; _++) {
        printf("eid _ %d, offset %d\n", _, local_matches.offset[_]);
      }
    }
  }
  free(local_matches.data);
  free(local_matches.offset);
}

static __global__ void subiso_kernel(ParametersSubIso params) {

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  // Phrase pattern
  VertexID *globalid_p = (VertexID *)params.data_p;
  VertexID *in_degree_p = globalid_p + params.n_vertices_p;
  VertexID *out_degree_p = in_degree_p + params.n_vertices_p;
  EdgeIndex *in_offset_p = (EdgeIndex *)(out_degree_p + params.n_vertices_p);
  EdgeIndex *out_offset_p =
      (EdgeIndex *)(in_offset_p + params.n_vertices_p + 1);
  EdgeIndex *in_edges_p = (EdgeIndex *)(out_offset_p + params.n_vertices_p + 1);
  VertexID *out_edges_p = in_edges_p + params.n_edges_p;
  VertexID *edges_globalid_by_localid_p = out_edges_p + params.n_edges_p;

  VertexID level = 0;
  VertexID u = globalid_p[level];
  printf("##########tid: %d, u %d, %d\n", tid, u, params.n_vertices_p);
  VertexID *m0 = params.m_ptr[level];
  EdgeIndex *m0_offset = params.m_offset + level;

  get_first_matches_kernel(params, u, m0, m0_offset);

  extend_kernel(params, level);
}

void SubIsoKernelWrapper::SubIso(
    const cudaStream_t &stream, VertexID depth_p,
    const data_structures::UnifiedOwnedBuffer<VertexID> &exec_path,
    VertexID n_vertices_p, EdgeIndex n_edges_p,
    const data_structures::UnifiedOwnedBuffer<uint8_t> &data_p,
    const data_structures::UnifiedOwnedBuffer<VertexLabel> &v_label_p,
    VertexID n_vertices_g, EdgeIndex n_edges_g,
    const data_structures::UnifiedOwnedBuffer<uint8_t> &data_g,
    const data_structures::UnifiedOwnedBuffer<VertexLabel> &v_label_g,
    const data_structures::UnifiedOwnedBuffer<VertexID *> &m_ptr,
    const data_structures::UnifiedOwnedBuffer<EdgeIndex> &m_offset) {

  dim3 dimBlock(32);
  dim3 dimGrid(32);

  ParametersSubIso params{.depth_p = depth_p,
                          .exec_path = exec_path.GetPtr(),
                          .n_vertices_p = n_vertices_p,
                          .n_edges_p = n_edges_p,
                          .data_p = data_p.GetPtr(),
                          .v_label_p = v_label_p.GetPtr(),
                          .n_vertices_g = n_vertices_g,
                          .n_edges_g = n_edges_g,
                          .data_g = data_g.GetPtr(),
                          .v_label_g = v_label_g.GetPtr(),
                          .m_ptr = m_ptr.GetPtr(),
                          .m_offset = m_offset.GetPtr()};

  subiso_kernel<<<dimGrid, dimBlock, 0, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    CUDA_CHECK(err);
  }
}

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics