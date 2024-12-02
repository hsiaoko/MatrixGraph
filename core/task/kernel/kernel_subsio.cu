#include <cuda_runtime.h>
#include <iostream>

#include "core/common/consts.h"
#include "core/common/types.h"
#include "core/task/kernel/data_structures/kernel_bitmap.cuh"
#include "core/task/kernel/data_structures/mini_kernel_bitmap.cuh"
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
using sics::matrixgraph::core::common::kMaxVertexID;
using sics::matrixgraph::core::task::kernel::HostMiniKernelBitmap;
using sics::matrixgraph::core::task::kernel::KernelBitmap;
using sics::matrixgraph::core::task::kernel::MiniKernelBitmap;

struct LocalMatches {
  VertexID *data = nullptr;
  VertexID *size = nullptr;
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
  VertexID *weft_count;
  EdgeIndex *weft_offset;
  VertexID *weft_size;
  VertexID *v_candidate_offset_for_each_weft;
  VertexID *matches_data;
};
// candidates in M should be all index instead of globalid.

static __device__ void dfs_kernel(const ParametersSubIso &params,
                                  uint32_t level, VertexID pre_v_idx,
                                  VertexID v_idx,
                                  MiniKernelBitmap &global_visited,
                                  MiniKernelBitmap local_visited, bool &match,
                                  LocalMatches &local_matches) {

  if (level > params.depth_p) {
    return;
  }
  unsigned exec_plan_idx = local_visited.Count();
  unsigned global_exec_plan_idx = global_visited.Count();

  if (global_exec_plan_idx == params.n_vertices_p) {
    return;
  }

  VertexID u = params.exec_path[exec_plan_idx];
  VertexLabel u_label = params.v_label_p[exec_plan_idx];

  VertexID global_u = params.exec_path[global_exec_plan_idx];
  VertexLabel global_u_label = params.v_label_p[global_exec_plan_idx];

  VertexID *globalid_g = (VertexID *)(params.data_g);
  VertexID *in_degree_g = globalid_g + params.n_vertices_g;
  VertexID *out_degree_g = in_degree_g + params.n_vertices_g;
  EdgeIndex *in_offset_g = (EdgeIndex *)(out_degree_g + params.n_vertices_g);
  EdgeIndex *out_offset_g =
      (EdgeIndex *)(in_offset_g + params.n_vertices_g + 1);
  EdgeIndex *in_edges_g = (EdgeIndex *)(out_offset_g + params.n_vertices_g + 1);
  VertexID *out_edges_g = in_edges_g + params.n_edges_g;
  VertexID *edges_globalid_by_localid_g = out_edges_g + params.n_edges_g;

  // Paste filter function gere.
  printf("dfs to %d, u %d, u label %d, \n", globalid_g[v_idx], u, u_label);
  VertexLabel v_label = params.v_label_g[globalid_g[v_idx]];
  if (v_label == u_label && v_label == global_u_label) {
    global_visited.SetBit(global_u);

    VertexID offset = local_matches.size[exec_plan_idx];

    if (pre_v_idx == kMaxVertexID) {
      local_matches
          .data[kMaxNumCandidatesPerThread * 2 * exec_plan_idx + 2 * offset] =
          pre_v_idx;
    } else {
      local_matches
          .data[kMaxNumCandidatesPerThread * 2 * exec_plan_idx + 2 * offset] =
          globalid_g[pre_v_idx];
    }
    local_matches
        .data[kMaxNumCandidatesPerThread * 2 * exec_plan_idx + 2 * offset + 1] =
        globalid_g[v_idx];

    local_matches.size[exec_plan_idx]++;
    local_visited.SetBit(u);
  } else if (v_label == u_label) {
    auto offset = local_matches.size[exec_plan_idx];

    local_matches
        .data[kMaxNumCandidatesPerThread * 2 * exec_plan_idx + 2 * offset] =
        globalid_g[pre_v_idx];
    local_matches
        .data[kMaxNumCandidatesPerThread * 2 * exec_plan_idx + 2 * offset + 1] =
        globalid_g[v_idx];
    local_matches.size[exec_plan_idx]++;
    local_visited.SetBit(u);

  } else if (v_label == global_u_label) {
    auto offset = local_matches.size[global_exec_plan_idx];

    local_matches.data[kMaxNumCandidatesPerThread * 2 * global_exec_plan_idx +
                       2 * offset] = globalid_g[pre_v_idx];
    local_matches.data[kMaxNumCandidatesPerThread * 2 * global_exec_plan_idx +
                       2 * offset + 1] = globalid_g[v_idx];

    local_matches.size[global_exec_plan_idx]++;
    global_visited.SetBit(global_u);
  } else {
    return;
  }

  printf("set %d\n", u);

  EdgeIndex offset_base = out_offset_g[v_idx];
  for (VertexID nbr_idx = 0; nbr_idx < out_degree_g[v_idx]; nbr_idx++) {
    VertexID candidate_v_idx = *(out_edges_g + offset_base + nbr_idx);
    VertexID candidate_global_v_id = globalid_g[candidate_v_idx];
    VertexLabel candidate_v_label = params.v_label_g[candidate_global_v_id];
    printf(" \t  pre %d candidate %d, level %d, v_idx: %d, %d/%d\n",
           globalid_g[v_idx], candidate_global_v_id, level, v_idx, nbr_idx,
           out_degree_g[v_idx]);
    dfs_kernel(params, level + 1, v_idx, candidate_v_idx, global_visited,
               local_visited, match, local_matches);
    printf(" back! after check %d\n", candidate_global_v_id);
  }
}

static __host__ void host_dfs_kernel(const ParametersSubIso &params,
                                     uint32_t level, VertexID pre_v_idx,
                                     VertexID v_idx,
                                     HostMiniKernelBitmap &global_visited,
                                     HostMiniKernelBitmap local_visited,
                                     bool &match, LocalMatches &local_matches) {

  unsigned int tid = 0;
  unsigned int step = 1;
  if (level > params.depth_p) {
    return;
  }
  VertexID exec_plan_idx = local_visited.Count();
  VertexID global_exec_plan_idx = global_visited.Count();

  if (global_exec_plan_idx == params.n_vertices_p) {
    return;
  }

  VertexID u = params.exec_path[exec_plan_idx];
  VertexLabel u_label = params.v_label_p[u];

  VertexID global_u = params.exec_path[global_exec_plan_idx];
  VertexLabel global_u_label = params.v_label_p[global_u];

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
  printf("dfs to %d, u %d, u label %d, \n", globalid_g[v_idx], u, u_label);
  VertexLabel v_label = params.v_label_g[globalid_g[v_idx]];
  if (v_label == u_label && v_label == global_u_label) {
    global_visited.SetBit(global_u);
    VertexID offset = local_matches.size[exec_plan_idx];

    if (pre_v_idx == kMaxVertexID) {
      local_matches
          .data[kMaxNumCandidatesPerThread * 2 * exec_plan_idx + 2 * offset] =
          pre_v_idx;
    } else {
      local_matches
          .data[kMaxNumCandidatesPerThread * 2 * exec_plan_idx + 2 * offset] =
          globalid_g[pre_v_idx];
    }
    local_matches
        .data[kMaxNumCandidatesPerThread * 2 * exec_plan_idx + 2 * offset + 1] =
        globalid_g[v_idx];

    local_matches.size[exec_plan_idx]++;
    local_visited.SetBit(u);
  } else if (v_label == u_label) {
    auto offset = local_matches.size[exec_plan_idx];
    local_matches
        .data[kMaxNumCandidatesPerThread * 2 * exec_plan_idx + 2 * offset] =
        globalid_g[pre_v_idx];
    local_matches
        .data[kMaxNumCandidatesPerThread * 2 * exec_plan_idx + 2 * offset + 1] =
        globalid_g[v_idx];

    local_matches.size[exec_plan_idx]++;
    local_visited.SetBit(u);
  } else if (v_label == global_u_label) {
    auto offset = local_matches.size[global_exec_plan_idx];
    local_matches.data[kMaxNumCandidatesPerThread * 2 * global_exec_plan_idx +
                       2 * offset] = globalid_g[pre_v_idx];
    local_matches.data[kMaxNumCandidatesPerThread * 2 * global_exec_plan_idx +
                       2 * offset + 1] = globalid_g[v_idx];

    local_matches.size[global_exec_plan_idx]++;
    global_visited.SetBit(global_u);
  } else {
    return;
  }
  printf("set %d\n", u);

  EdgeIndex offset_base = out_offset_g[v_idx];
  if (level == params.depth_p) {
    return;
  } else {
    for (VertexID nbr_idx = tid; nbr_idx < out_degree_g[v_idx];
         nbr_idx += step) {
      VertexID candidate_v_idx = *(out_edges_g + offset_base + nbr_idx);
      VertexID candidate_global_v_id = globalid_g[candidate_v_idx];
      VertexLabel candidate_v_label = params.v_label_g[candidate_global_v_id];
      printf(" \t pre %d candidate %d, level %d, v_idx: %d, %d/%d\n",
             globalid_g[v_idx], candidate_global_v_id, level, v_idx, nbr_idx,
             out_degree_g[v_idx]);
      host_dfs_kernel(params, level + 1, v_idx, candidate_v_idx, global_visited,
                      local_visited, match, local_matches);
    }
  }
}

static __device__ void extend_kernel(const ParametersSubIso &params) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  MiniKernelBitmap visited(params.n_vertices_p);

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
  local_matches.data = (VertexID *)malloc(
      sizeof(VertexID) * params.n_vertices_p * 2 * kMaxNumCandidatesPerThread);

  local_matches.size =
      (VertexID *)malloc(sizeof(VertexID) * params.n_vertices_p);

  for (VertexID candidate_idx = tid; candidate_idx < params.n_vertices_g;
       candidate_idx += step) {

    visited.Clear();
    memset(local_matches.data, 0,
           params.n_vertices_p * 2 * kMaxNumCandidatesPerThread *
               sizeof(VertexID));
    memset(local_matches.size, 0, params.n_vertices_p * sizeof(VertexID));

    VertexID v_idx = candidate_idx;
    VertexID global_vid = globalid_g[v_idx];

    printf("tid: %d, extend: %d\n", tid, global_vid);
    bool match = false;
    dfs_kernel(params, 0, kMaxVertexID, v_idx, visited, visited, match,
               local_matches);

    printf("#done!\n");

    // Write result to the global buffer. Obtain space for each. The last
    // pattern vertex dosen't get a match.
    if (local_matches.size[params.n_vertices_p - 1] != 0) {
      VertexID total_n_candidate = 0;

      VertexID *local_offset =
          (VertexID *)malloc(sizeof(VertexID) * (params.n_vertices_p + 1));
      memset(local_offset, 0, sizeof(VertexID) * (params.n_vertices_p + 1));

      for (VertexID _ = 0; _ < params.n_vertices_p; _++) {
        total_n_candidate += local_matches.size[_];
        local_offset[_ + 1] = local_offset[_] + local_matches.size[_];
      }

      // atomic operations
      VertexID weft_id = atomicAdd(params.weft_count, (VertexID)1);

      params.weft_size[weft_id] = total_n_candidate;
      memcpy(params.v_candidate_offset_for_each_weft + weft_id, local_offset,
             sizeof(VertexID) * (params.n_vertices_p + 1));

      __syncthreads();

      if (tid == 0) {
        VertexID weft_count = *(params.weft_count);
        for (VertexID _ = 0; _ < weft_count; _++) {
          params.weft_offset[_ + 1] =
              params.weft_offset[_] + params.weft_size[_];
        }
      }

      __syncthreads();

      // Fill weft forest.
      VertexID *base_ptr = params.matches_data + params.weft_offset[weft_id] *
                                                     2 * params.n_vertices_p;

      for (VertexID _ = 0; _ < params.n_vertices_p; _++) {
        memcpy(base_ptr + local_offset[_] * 2,
               local_matches.data + kMaxNumCandidatesPerThread * 2 * _,
               local_matches.size[_] * sizeof(VertexID) * 2);
      }
    }
  }
  free(local_matches.data);
  free(local_matches.size);
}

static __host__ void host_extend_kernel(const ParametersSubIso &params) {
  unsigned int tid = 0;
  unsigned int step = 1;

  HostMiniKernelBitmap visited(params.n_vertices_p);

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
  local_matches.data = (VertexID *)malloc(
      sizeof(VertexID) * params.n_vertices_p * 2 * kMaxNumCandidatesPerThread);

  local_matches.size =
      (VertexID *)malloc(sizeof(VertexID) * params.n_vertices_p);

  for (VertexID candidate_idx = tid; candidate_idx < params.n_vertices_g;
       candidate_idx += step) {
    visited.Clear();
    memset(local_matches.data, 0,
           params.n_vertices_p * 2 * kMaxNumCandidatesPerThread *
               sizeof(VertexID));
    memset(local_matches.size, 0, params.n_vertices_p * sizeof(VertexID));

    // VertexID v_idx = m0.data[candidate_idx];
    // VertexID global_vid = globalid_g[v_idx];
    VertexID v_idx = candidate_idx;
    VertexID global_vid = globalid_g[v_idx];

    printf("extend: %d\n", global_vid);

    bool match = false;
    host_dfs_kernel(params, 0, kMaxVertexID, v_idx, visited, visited, match,
                    local_matches);

    printf("#done!\n");

    // Write result to the global buffer. Obtain space for each. The last
    // pattern vertex dosen't get a match.
    if (local_matches.size[params.n_vertices_p - 1] != 0) {
      printf("global %d\n", (*params.weft_count));
      VertexID total_n_candidate = 0;

      VertexID *local_offset =
          (VertexID *)malloc(sizeof(VertexID) * (params.n_vertices_p + 1));
      memset(local_offset, 0, sizeof(VertexID) * (params.n_vertices_p + 1));

      for (VertexID _ = 0; _ < params.n_vertices_p; _++) {
        total_n_candidate += local_matches.size[_];
        local_offset[_ + 1] = local_offset[_] + local_matches.size[_];
      }

      // atomic operations
      VertexID weft_id = (*params.weft_count)++;

      params.weft_size[weft_id] = total_n_candidate;
      memcpy(params.v_candidate_offset_for_each_weft + weft_id, local_offset,
             sizeof(VertexID) * (params.n_vertices_p + 1));

      //__syncthreads();

      if (tid == 0) {
        VertexID weft_count = *(params.weft_count);
        for (VertexID _ = 0; _ < weft_count; _++) {
          params.weft_offset[_ + 1] =
              params.weft_offset[_] + params.weft_size[_];
        }
      }

      //__syncthreads();

      // Fill weft forest.
      VertexID *base_ptr = params.matches_data + params.weft_offset[weft_id] *
                                                     2 * params.n_vertices_p;

      for (VertexID _ = 0; _ < params.n_vertices_p; _++) {
        memcpy(base_ptr + local_offset[_] * 2,
               local_matches.data + kMaxNumCandidatesPerThread * 2 * _,
               local_matches.size[_] * sizeof(VertexID) * 2);
        printf("local_match:%d, local_offset%d\n", local_matches.size[_],
               local_offset[_]);
        for (VertexID j = 0; j < local_matches.size[_]; j++) {
          printf("j:%d, data: %d->%d\n", j,
                 *(local_matches.data + kMaxNumCandidatesPerThread * 2 * _ +
                   2 * j),
                 *(local_matches.data + kMaxNumCandidatesPerThread * 2 * _ +
                   2 * j + 1));
          printf("\t data: %d->%d\n", *(base_ptr + 2 * local_offset[_] + 2 * j),
                 *(base_ptr + 2 * local_offset[_] + 2 * j + 1)

          );
        }
      }
    }
  }
  free(local_matches.data);
  free(local_matches.size);
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

  extend_kernel(params);
}

static __host__ void host_subiso_kernel(ParametersSubIso params) {

  unsigned int tid = 0;
  unsigned int step = 1;

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

  host_extend_kernel(params);
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
    const data_structures::UnifiedOwnedBuffer<VertexID> &weft_count,
    const data_structures::UnifiedOwnedBuffer<EdgeIndex> &weft_offset,
    const data_structures::UnifiedOwnedBuffer<VertexID> &weft_size,
    const data_structures::UnifiedOwnedBuffer<VertexID>
        &v_candidate_offset_for_each_weft,
    const data_structures::UnifiedOwnedBuffer<VertexID> &matches_data) {

  dim3 dimBlock(1);
  dim3 dimGrid(1);

  LocalMatches m0;
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
                          .weft_count = weft_count.GetPtr(),
                          .weft_offset = weft_offset.GetPtr(),
                          .weft_size = weft_size.GetPtr(),
                          .v_candidate_offset_for_each_weft =
                              v_candidate_offset_for_each_weft.GetPtr(),
                          .matches_data = matches_data.GetPtr()};

  subiso_kernel<<<dimGrid, dimBlock, 0, stream>>>(params);
  // host_subiso_kernel(params);

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