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
  VertexID *inverted_index_of_exec_path;
  VertexID *exec_path_in_edges;
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
// candidates in M should be an index (localid) instead of globalid.

static __host__ __device__ bool label_filter(const ParametersSubIso &params,
                                             VertexID exec_plan_idx,
                                             VertexID v_idx) {

  VertexID *globalid_g = (VertexID *)(params.data_g);
  VertexLabel v_label = params.v_label_g[globalid_g[v_idx]];

  VertexID u = params.exec_path[exec_plan_idx];
  VertexLabel u_label = params.v_label_p[u];

  return u_label == v_label;
}

static __host__ __device__ bool filter(const ParametersSubIso &params,
                                       VertexID exec_plan_idx, VertexID v_idx) {
  return label_filter(params, exec_plan_idx, v_idx);
}

static __device__ bool check(const ParametersSubIso &params,
                             LocalMatches &local_matches) {

  VertexID *globalid_p = (VertexID *)(params.data_p);
  VertexID *in_degree_p = globalid_p + params.n_vertices_p;
  VertexID *out_degree_p = in_degree_p + params.n_vertices_p;
  EdgeIndex *in_offset_p = (EdgeIndex *)(out_degree_p + params.n_vertices_p);
  EdgeIndex *out_offset_p =
      (EdgeIndex *)(in_offset_p + params.n_vertices_p + 1);
  EdgeIndex *in_edges_p = (EdgeIndex *)(out_offset_p + params.n_vertices_p + 1);
  VertexID *out_edges_p = in_edges_p + params.n_edges_p;
  VertexID *edges_globalid_by_localid_p = out_edges_p + params.n_edges_p;

  VertexID *globalid_g = (VertexID *)(params.data_g);
  VertexID *in_degree_g = globalid_g + params.n_vertices_g;
  VertexID *out_degree_g = in_degree_g + params.n_vertices_g;
  EdgeIndex *in_offset_g = (EdgeIndex *)(out_degree_g + params.n_vertices_g);
  EdgeIndex *out_offset_g =
      (EdgeIndex *)(in_offset_g + params.n_vertices_g + 1);
  EdgeIndex *in_edges_g = (EdgeIndex *)(out_offset_g + params.n_vertices_g + 1);
  VertexID *out_edges_g = in_edges_g + params.n_edges_g;
  VertexID *edges_globalid_by_localid_g = out_edges_g + params.n_edges_g;

  KernelBitmap visited(params.n_vertices_g);

  for (VertexID u_idx = 0; u_idx < params.n_vertices_p; u_idx++) {
    VertexID n_candidates = local_matches.size[u_idx];
    EdgeIndex in_offset_base_u = in_offset_p[u_idx];

    // Check for each candidate of U
    for (VertexID _ = 0; _ < n_candidates; _++) {
      VertexID v_idx =
          local_matches
              .data[u_idx * kMaxNumCandidatesPerThread * 2 + 2 * _ + 1];

      // Check for each neighbor of candidate
      for (VertexID nbr_u_idx = 0; nbr_u_idx < in_degree_p[u_idx];
           nbr_u_idx++) {
        VertexID nbr_u = in_edges_p[in_offset_base_u + nbr_u_idx];

        VertexID n_candidates_nbr_u = local_matches.size[nbr_u];

        visited.Clear();

        // Check neighbor of u' candidate
        for (VertexID candidate_idx_nbr_u = 0;
             candidate_idx_nbr_u < n_candidates_nbr_u; candidate_idx_nbr_u++) {
          VertexID candidate_nbr_u =
              local_matches.data[nbr_u * kMaxNumCandidatesPerThread * 2 +
                                 2 * candidate_idx_nbr_u + 1];
          visited.SetBit(candidate_nbr_u);
        }

        EdgeIndex in_offset_base_v = in_offset_g[v_idx];
        VertexID match_count = 0;

        // Check neighbor of v
        for (VertexID nbr_v_idx = 0; nbr_v_idx < in_degree_g[v_idx];
             nbr_v_idx++) {
          VertexID nbr_v = in_edges_g[in_offset_base_v + nbr_v_idx];
          if (visited.GetBit(nbr_v)) {
            match_count++;
            break;
          }
        }

        // there is a mismatch between v'->v and u'->u. we have to remove it.
        if (match_count == 0) {
          local_matches
              .data[u_idx * kMaxNumCandidatesPerThread * 2 + 2 * _ + 1] =
              kMaxVertexID;
          local_matches.data[u_idx * kMaxNumCandidatesPerThread * 2 + 2 * _] =
              kMaxVertexID;
        }
      }
    }
  }
}

static __device__ void dfs_kernel(const ParametersSubIso &params,
                                  uint32_t level, VertexID pre_v_idx,
                                  VertexID v_idx,
                                  MiniKernelBitmap &global_visited,
                                  MiniKernelBitmap local_visited, bool &match,
                                  LocalMatches &local_matches) {

  if (level > params.depth_p)
    return;

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
  VertexLabel v_label = params.v_label_g[globalid_g[v_idx]];

  VertexID global_pre_u = params.exec_path_in_edges[2 * global_exec_plan_idx];
  VertexID local_pre_u = params.exec_path_in_edges[2 * exec_plan_idx];

  bool local_match_tag = false;
  bool global_match_tag = false;

  // if true, we need check both global u and local u. it false we only need to
  // check local u.
  if (local_pre_u == global_pre_u) {
    local_match_tag = filter(params, exec_plan_idx, v_idx);
    global_match_tag = filter(params, global_exec_plan_idx, v_idx);
  } else {
    local_match_tag = filter(params, exec_plan_idx, v_idx);
  }

  // printf("\t global pre_u: %d, local pre_u: %d\n",
  //        params.exec_path_in_edges[2 * global_exec_plan_idx],
  //        params.exec_path_in_edges[2 * exec_plan_idx]);
  // printf("dfs to %d, u %d, u label %d, \n", globalid_g[v_idx], u, u_label);
  //  VertexLabel v_label = params.v_label_g[globalid_g[v_idx]];

  if (local_match_tag && global_match_tag) {
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
    printf("\t 1 set %d->%d\n", pre_v_idx, globalid_g[v_idx]);
    local_matches.size[exec_plan_idx]++;
    local_visited.SetBit(u);
    global_visited.SetBit(u);
  } else if (local_match_tag) {
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
    printf("\t 2 set %d->%d\n", pre_v_idx, globalid_g[v_idx]);
  } else if (global_match_tag) {

    VertexID offset = local_matches.size[global_exec_plan_idx];
    if (pre_v_idx == kMaxVertexID) {
      local_matches.data[kMaxNumCandidatesPerThread * 2 * global_exec_plan_idx +
                         2 * offset] = pre_v_idx;
    } else {
      local_matches.data[kMaxNumCandidatesPerThread * 2 * global_exec_plan_idx +
                         2 * offset] = globalid_g[pre_v_idx];
    }
    local_matches.data[kMaxNumCandidatesPerThread * 2 * global_exec_plan_idx +
                       2 * offset + 1] = globalid_g[v_idx];
    local_matches.size[global_exec_plan_idx]++;
    local_visited.SetBit(u);
    printf("\t 3 set %d->%d\n", pre_v_idx, globalid_g[v_idx]);
  } else {
    printf("back\n");
    return;
  }

  printf("set %d\n", u);
  EdgeIndex offset_base = out_offset_g[v_idx];
  for (VertexID nbr_idx = 0; nbr_idx < out_degree_g[v_idx]; nbr_idx++) {
    VertexID candidate_v_idx = *(out_edges_g + offset_base + nbr_idx);
    VertexID candidate_global_v_id = globalid_g[candidate_v_idx];
    VertexLabel candidate_v_label = params.v_label_g[candidate_global_v_id];
    printf(" \t  pre %d candidate %d, level %d, v_idx: %d, %d/%d\n",
           globalid_g[v_idx], candidate_global_v_id, level, candidate_v_idx,
           nbr_idx, out_degree_g[v_idx]);
    dfs_kernel(params, level + 1, v_idx, candidate_v_idx, global_visited,
               local_visited, match, local_matches);
  }
}

static __host__ void host_dfs_kernel(const ParametersSubIso &params,
                                     uint32_t level, VertexID pre_v_idx,
                                     VertexID v_idx,
                                     HostMiniKernelBitmap &global_visited,
                                     HostMiniKernelBitmap local_visited,
                                     bool &match, LocalMatches &local_matches) {

  if (level > params.depth_p)
    return;

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
  VertexLabel v_label = params.v_label_g[globalid_g[v_idx]];

  VertexID global_pre_u = params.exec_path_in_edges[2 * global_exec_plan_idx];
  VertexID local_pre_u = params.exec_path_in_edges[2 * exec_plan_idx];

  bool local_match_tag = false;
  bool global_match_tag = false;

  // if true, we need check both global u and local u. it false we only need to
  // check local u.
  if (local_pre_u == global_pre_u) {
    local_match_tag = filter(params, exec_plan_idx, v_idx);
    global_match_tag = filter(params, global_exec_plan_idx, v_idx);
  } else {
    local_match_tag = filter(params, exec_plan_idx, v_idx);
  }

  // printf("\t global pre_u: %d, local pre_u: %d\n",
  //        params.exec_path_in_edges[2 * global_exec_plan_idx],
  //        params.exec_path_in_edges[2 * exec_plan_idx]);
  // printf("dfs to %d, u %d, u label %d, \n", globalid_g[v_idx], u, u_label);
  //  VertexLabel v_label = params.v_label_g[globalid_g[v_idx]];

  if (local_match_tag && global_match_tag) {
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
    printf("\t 1 set %d->%d\n", pre_v_idx, globalid_g[v_idx]);
    local_matches.size[exec_plan_idx]++;
    local_visited.SetBit(u);
    global_visited.SetBit(u);
  } else if (local_match_tag) {
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
    printf("\t 2 set %d->%d\n", pre_v_idx, globalid_g[v_idx]);
  } else if (global_match_tag) {

    VertexID offset = local_matches.size[global_exec_plan_idx];
    if (pre_v_idx == kMaxVertexID) {
      local_matches.data[kMaxNumCandidatesPerThread * 2 * global_exec_plan_idx +
                         2 * offset] = pre_v_idx;
    } else {
      local_matches.data[kMaxNumCandidatesPerThread * 2 * global_exec_plan_idx +
                         2 * offset] = globalid_g[pre_v_idx];
    }
    local_matches.data[kMaxNumCandidatesPerThread * 2 * global_exec_plan_idx +
                       2 * offset + 1] = globalid_g[v_idx];
    local_matches.size[global_exec_plan_idx]++;
    local_visited.SetBit(u);
    printf("\t 3 set %d->%d\n", pre_v_idx, globalid_g[v_idx]);
  } else {
    return;
  }

  printf("set %d\n", u);
  EdgeIndex offset_base = out_offset_g[v_idx];
  for (VertexID nbr_idx = 0; nbr_idx < out_degree_g[v_idx]; nbr_idx++) {
    VertexID candidate_v_idx = *(out_edges_g + offset_base + nbr_idx);
    VertexID candidate_global_v_id = globalid_g[candidate_v_idx];
    VertexLabel candidate_v_label = params.v_label_g[candidate_global_v_id];
    // printf(" \t  pre %d candidate %d, level %d, v_idx: %d, %d/%d\n",
    //        globalid_g[v_idx], candidate_global_v_id, level, candidate_v_idx,
    //        nbr_idx, out_degree_g[v_idx]);
    host_dfs_kernel(params, level + 1, v_idx, candidate_v_idx, global_visited,
                    local_visited, match, local_matches);
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

    bool match = false;
    dfs_kernel(params, 0, kMaxVertexID, v_idx, visited, visited, match,
               local_matches);

    // Write result to the global buffer. Obtain space for each. The last
    // pattern vertex dosen't get a match.
    if (local_matches.size[params.n_vertices_p - 1] != 0) {
      // check(params, local_matches);
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
      memcpy(params.v_candidate_offset_for_each_weft +
                 weft_id * (params.n_vertices_p + 1),
             local_offset, sizeof(VertexID) * (params.n_vertices_p + 1));

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
      free(local_offset);
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

    VertexID v_idx = candidate_idx;
    VertexID global_vid = globalid_g[v_idx];

    bool match = false;
    host_dfs_kernel(params, 0, kMaxVertexID, v_idx, visited, visited, match,
                    local_matches);

    // Write result to the global buffer. Obtain space for each. The last
    // pattern vertex dosen't get a match.
    if (local_matches.size[params.n_vertices_p - 1] != 0) {
      // check(params, local_matches);

      VertexID total_n_candidate = 0;

      VertexID local_offset[params.n_vertices_p + 1];
      memset(local_offset, 0, sizeof(VertexID) * (params.n_vertices_p + 1));

      for (VertexID _ = 0; _ < params.n_vertices_p; _++) {
        total_n_candidate += local_matches.size[_];
        local_offset[_ + 1] = local_offset[_] + local_matches.size[_];
      }

      // atomic operations
      VertexID weft_id = *(params.weft_count);
      *(params.weft_count) = *(params.weft_count) + 1;

      params.weft_size[weft_id] = total_n_candidate;
      memcpy(params.v_candidate_offset_for_each_weft +
                 weft_id * (params.n_vertices_p + 1),
             local_offset, sizeof(VertexID) * (params.n_vertices_p + 1));

      // __synchronize();
      if (tid == 0) {
        VertexID weft_count = *(params.weft_count);
        for (VertexID _ = 0; _ < weft_count; _++) {
          params.weft_offset[_ + 1] =
              params.weft_offset[_] + params.weft_size[_];
        }
      }
      // __synchronize();

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

static __global__ void subiso_kernel(ParametersSubIso params) {
  extend_kernel(params);
  printf("#done\n");
}

static __host__ void host_subiso_kernel(ParametersSubIso params) {
  host_extend_kernel(params);
}

void SubIsoKernelWrapper::SubIso(
    const cudaStream_t &stream, VertexID depth_p,
    const data_structures::UnifiedOwnedBuffer<VertexID> &exec_path,
    const data_structures::UnifiedOwnedBuffer<VertexID>
        &inverted_index_of_exec_path,
    const data_structures::UnifiedOwnedBuffer<VertexID> &exec_path_in_edges,
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
                          .inverted_index_of_exec_path =
                              inverted_index_of_exec_path.GetPtr(),
                          .exec_path_in_edges = exec_path_in_edges.GetPtr(),
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
  //host_subiso_kernel(params);

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