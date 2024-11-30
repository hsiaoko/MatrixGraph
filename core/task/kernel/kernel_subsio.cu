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
  VertexID *matches_count;
  EdgeIndex *weft_offset;
  VertexID *weft_size;
  VertexID *v_candidate_offset_for_each_weft;
  VertexID *matches_data;
};
// candidates in M should be all index instead of globalid.

static __device__ void get_first_matches_kernel(const ParametersSubIso &params,
                                                VertexID u, LocalMatches &m0) {}

static __host__ void host_get_first_matches_kernel(ParametersSubIso &params,
                                                   VertexID u,
                                                   LocalMatches &m0) {
  unsigned int tid = 0;
  unsigned int step = 1;

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
      auto local_offset = *m0.size;
      *m0.size = *(m0.size) + 1;
      m0.data[local_offset] = v_idx;
    }
  }
}

static __device__ void dfs_kernel(const ParametersSubIso &params,
                                  uint32_t level, VertexID v_idx,
                                  MiniKernelBitmap &global_visited,
                                  MiniKernelBitmap local_visited, bool &match,
                                  LocalMatches &local_matches) {}

static __host__ void host_dfs_kernel(const ParametersSubIso &params,
                                     uint32_t level, VertexID pre_v_idx,
                                     VertexID v_idx,
                                     HostMiniKernelBitmap &global_visited,
                                     HostMiniKernelBitmap local_visited,
                                     bool &match, LocalMatches &local_matches) {

  unsigned int tid = 0;
  unsigned int step = 1;
  if (level > params.depth_p) {
    printf(" ### out %d/%d\n", level, params.depth_p);
    return;
  }
  VertexID exec_plan_idx = local_visited.Count();
  VertexID global_exec_plan_idx = global_visited.Count();

  if (global_exec_plan_idx == params.n_vertices_p) {
    std::cout << "MATCH" << std::endl;
    return;
  }

  VertexID u = params.exec_path[exec_plan_idx];
  VertexLabel u_label = params.v_label_p[u];

  VertexID global_u = params.exec_path[global_exec_plan_idx];
  VertexLabel global_u_label = params.v_label_p[global_u];

  printf("local count %ld, global count %ld\n", local_visited.Count(),
         global_visited.Count());

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
    printf("push %d to %d, global_offset%d\n", pre_v_idx, offset,
           kMaxNumCandidatesPerThread * 2 * exec_plan_idx + 2 * offset);
    printf("push %d to %d, global_offset%d\n", globalid_g[v_idx], offset + 1,
           kMaxNumCandidatesPerThread * 2 * exec_plan_idx + 2 * offset + 1);
    printf("offset %d, pos %d\n ", offset,
           params.n_vertices_p * offset + exec_plan_idx);

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
    printf("both match%d\n", u);
  } else if (v_label == u_label) {
    printf("\tlocal_match, %d\n", u);
    auto offset = local_matches.size[exec_plan_idx];

    printf("push %d to %d, global_offset%d\n", globalid_g[pre_v_idx], offset,
           kMaxNumCandidatesPerThread * 2 * exec_plan_idx + 2 * offset);
    printf("push %d to %d, global_offset%d\n", globalid_g[v_idx], offset + 1,
           kMaxNumCandidatesPerThread * 2 * exec_plan_idx + 2 * offset + 1);

    local_matches
        .data[kMaxNumCandidatesPerThread * 2 * exec_plan_idx + 2 * offset] =
        globalid_g[pre_v_idx];
    local_matches
        .data[kMaxNumCandidatesPerThread * 2 * exec_plan_idx + 2 * offset + 1] =
        globalid_g[v_idx];

    local_matches.size[exec_plan_idx]++;
    local_visited.SetBit(u);
  } else if (v_label == global_u_label) {
    printf("\tglobal_match, %d,  global_exec_idx: %d\n", global_u,
           global_exec_plan_idx);
    auto offset = local_matches.size[global_exec_plan_idx];

    printf("push %d to %d, global_offset%d\n", globalid_g[pre_v_idx], offset,
           kMaxNumCandidatesPerThread * 2 * global_exec_plan_idx + 2 * offset);
    printf("push %d to %d, global_offset%d\n", globalid_g[v_idx], offset + 1,
           kMaxNumCandidatesPerThread * 2 * global_exec_plan_idx + 2 * offset +
               1);

    local_matches.data[kMaxNumCandidatesPerThread * 2 * global_exec_plan_idx +
                       2 * offset] = globalid_g[pre_v_idx];
    local_matches.data[kMaxNumCandidatesPerThread * 2 * global_exec_plan_idx +
                       2 * offset + 1] = globalid_g[v_idx];

    local_matches.size[global_exec_plan_idx]++;
    global_visited.SetBit(global_u);
  } else {
    printf("dis match u: %d or global_u", u, global_u);
    return;
  }

  printf("set %d\n", u);

  EdgeIndex offset_base = out_offset_g[v_idx];
  printf("level %d\n", level);
  if (level == params.depth_p) {
    printf("return\n");
    return;
  } else {
    printf("push\n");
    for (VertexID nbr_idx = tid; nbr_idx < out_degree_g[v_idx];
         nbr_idx += step) {
      VertexID candidate_v_idx = *(out_edges_g + offset_base + nbr_idx);
      VertexID candidate_global_v_id = globalid_g[candidate_v_idx];
      VertexLabel candidate_v_label = params.v_label_g[candidate_global_v_id];
      printf(" global count %ld, local count %ld\n", global_visited.Count(),
             local_visited.Count());
      printf(" \t pre %d candidate %d, level %d, v_idx: %d, %d/%d\n",
             globalid_g[v_idx], candidate_global_v_id, level, v_idx, nbr_idx,
             out_degree_g[v_idx]);
      host_dfs_kernel(params, level + 1, v_idx, candidate_v_idx, global_visited,
                      local_visited, match, local_matches);
      printf("back!\n");
    }
  }
}

static __device__ void extend_kernel(const ParametersSubIso &params,
                                     uint32_t level, LocalMatches &m0) {}

static __host__ void host_extend_kernel(const ParametersSubIso &params,
                                        uint32_t level, LocalMatches &m0) {
  unsigned int tid = 0;
  unsigned int step = 1;
  VertexID u_idx = 0;

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

  for (VertexID candidate_idx = tid; candidate_idx < *m0.size;
       candidate_idx += step) {
    visited.Clear();
    memset(local_matches.data, 0,
           params.n_vertices_p * 2 * kMaxNumCandidatesPerThread *
               sizeof(VertexID));
    memset(local_matches.size, 0, params.n_vertices_p * sizeof(VertexID));

    VertexID v_idx = m0.data[candidate_idx];
    VertexID global_vid = globalid_g[v_idx];

    printf("extend: %d\n", global_vid);

    bool match = false;
    host_dfs_kernel(params, level, kMaxVertexID, v_idx, visited, visited, match,
                    local_matches);

    printf("#done!\n");

    // Write result to the global buffer. Obtain space for each. The last
    // pattern vertex dosen't get a match.
    if (local_matches.size[params.n_vertices_p - 1] != 0) {
      printf("global %d\n", (*params.matches_count));
      VertexID total_n_candidate = 0;

      VertexID *local_offset =
          (VertexID *)malloc(sizeof(VertexID) * params.n_vertices_p);

      for (VertexID _ = 0; _ < params.n_vertices_p; _++) {
        total_n_candidate += local_matches.size[_];
        local_offset[_ + 1] = local_offset[_] + local_matches.size[_];
      }

      // atomic operations
      VertexID weft_id = (*params.matches_count)++;

      params.weft_size[weft_id] = total_n_candidate;

      //__syncthreads();

      if (tid == 0) {
        VertexID matches_count = *(params.matches_count);
        for (VertexID _ = 0; _ < matches_count; _++) {
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
        params.v_candidate_offset_for_each_weft[weft_id * params.n_vertices_p +
                                                _] = local_matches.size[_];
      }

      // Show Result.
      auto weft_offset = params.weft_offset[weft_id];
      printf("weft_offset: %d\n", weft_offset);

      for (auto col = 0; col < params.n_vertices_p; col++) {

        printf("u_ %d, n_candidate %d\n", col,
               params.v_candidate_offset_for_each_weft[weft_id *
                                                           params.n_vertices_p +
                                                       col]);
        for (auto candidate_idx = 0;
             candidate_idx < params.v_candidate_offset_for_each_weft
                                 [weft_id * params.n_vertices_p + col];
             candidate_idx++) {
          printf("%d ->%d\n", *(base_ptr + 2 * candidate_idx),
                 *(base_ptr + 2 * candidate_idx + 1));
        }
      }

      // for (VertexID weft_id = 0; weft_id < *params.matches_count; weft_id++)
      // {
      //   printf("weft_id %d\n ", weft_id);

      //   for (VertexID _ = 1; _ < params.n_vertices_p; _++) {
      //     printf("u%d, n_candidate:%d\n", _,
      //            params.v_candidate_offset_for_each_weft
      //                [weft_id * params.n_vertices_p + _]);
      //     auto base_ptr = params.matches_data + weft_offset +
      //                     _ * 2 * kMaxNumCandidatesPerThread;
      //     for (auto candidate_idx = 0;
      //          candidate_idx < params.v_candidate_offset_for_each_weft
      //                              [weft_id * params.n_vertices_p + _];
      //          candidate_idx++) {
      //       printf("%d->%d\n ", base_ptr[2 * v_idx], base_ptr[2 * v_idx +
      //       1]);
      //     }
      //     printf("\n");
      //   }
      // }
    }

    // Fill matches
    if (local_matches.size[params.n_vertices_p - 1] != 0) {
    }
  }
  free(local_matches.data);
  free(local_matches.size);
}

static __global__ void subiso_kernel(ParametersSubIso params) {}

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
  // VertexID *m0 = params.m_ptr[level];
  // EdgeIndex *m0_offset = params.m_offset + level;
  LocalMatches m0;
  m0.data = (VertexID *)malloc(sizeof(VertexID) * params.n_vertices_g / step);
  m0.size = (VertexID *)malloc(sizeof(VertexID));

  host_get_first_matches_kernel(params, u, m0);

  for (int i = 0; i < *m0.size; i++) {
    printf("%d\n", m0.data[i]);
  }

  host_extend_kernel(params, level, m0);

  free(m0.data);
  free(m0.size);
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
    const data_structures::UnifiedOwnedBuffer<VertexID> &matches_count,
    const data_structures::UnifiedOwnedBuffer<EdgeIndex> &weft_offset,
    const data_structures::UnifiedOwnedBuffer<VertexID> &weft_size,
    const data_structures::UnifiedOwnedBuffer<VertexID>
        &v_candidate_offset_for_each_weft,
    const data_structures::UnifiedOwnedBuffer<VertexID> &matches_data) {

  dim3 dimBlock(1);
  dim3 dimGrid(1);

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
                          .matches_count = matches_count.GetPtr(),
                          .weft_offset = weft_offset.GetPtr(),
                          .weft_size = weft_size.GetPtr(),
                          .v_candidate_offset_for_each_weft =
                              v_candidate_offset_for_each_weft.GetPtr(),
                          .matches_data = matches_data.GetPtr()};

  for (int i = 0; i < 10; i++) {
    std::cout << params.exec_path[i] << " ";
  }
  // subiso_kernel<<<dimGrid, dimBlock, 0, stream>>>(params);

  host_subiso_kernel(params);
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