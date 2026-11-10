#include "core/task/gpu_task/kernel/kernel_gar_deduce.cuh"

#include <cuda_runtime.h>

#include "core/common/consts.h"
#include "core/util/cuda_check.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

using sics::matrixgraph::core::common::kBlockDim;
using sics::matrixgraph::core::common::kGridDim;

namespace {

// Candidate test for one graph edge against one pattern edge: edge label
// plus endpoint vertex labels (-1 = wildcard). No degree/neighborhood
// pruning — homomorphism needs none.
__device__ __forceinline__ bool GARDeduceCandidateMatch(
    const GARDeduceFilterParams& params, int pe, int ge) {
  if (params.p.edge_label[pe] >= 0 &&
      params.g_e_label[ge] != params.p.edge_label[pe]) {
    return false;
  }
  if (params.g_v_label != nullptr) {
    const uint32_t gs = params.g_e_src[ge];
    const uint32_t gd = params.g_e_dst[ge];
    const int32_t sl = params.p.node_label[params.p.edge_src[pe]];
    const int32_t dl = params.p.node_label[params.p.edge_dst[pe]];
    if (sl >= 0 && params.g_v_label[gs] != sl) return false;
    if (dl >= 0 && params.g_v_label[gd] != dl) return false;
  }
  return true;
}

__global__ void GARDeduceCountCandidatesKernel(GARDeduceFilterParams params) {
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int step = blockDim.x * gridDim.x;
  for (int ge = static_cast<int>(tid); ge < params.g_n_edges;
       ge += static_cast<int>(step)) {
    for (int pe = 0; pe < params.p.n_edges; ++pe) {
      if (GARDeduceCandidateMatch(params, pe, ge)) {
        atomicAdd(&params.cand_counts[pe], 1);
      }
    }
  }
}

__global__ void GARDeduceFillCandidatesKernel(GARDeduceFilterParams params,
                                              int* cursors) {
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int step = blockDim.x * gridDim.x;
  for (int ge = static_cast<int>(tid); ge < params.g_n_edges;
       ge += static_cast<int>(step)) {
    for (int pe = 0; pe < params.p.n_edges; ++pe) {
      if (GARDeduceCandidateMatch(params, pe, ge)) {
        const int slot = atomicAdd(&cursors[pe], 1);
        params.cand_src[slot] = params.g_e_src[ge];
        params.cand_dst[slot] = params.g_e_dst[ge];
      }
    }
  }
}

__global__ void GARDeducePackCandidatesKernel(const uint32_t* cand_src,
                                              const uint32_t* cand_dst,
                                              int n_cand, uint64_t* fwd_keys,
                                              uint64_t* bwd_keys) {
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int step = blockDim.x * gridDim.x;
  for (int c = static_cast<int>(tid); c < n_cand; c += static_cast<int>(step)) {
    const uint64_t gs = cand_src[c];
    const uint64_t gd = cand_dst[c];
    fwd_keys[c] = (gs << 32) | gd;
    bwd_keys[c] = (gd << 32) | gs;
  }
}

// lower_bound over sorted uint64 keys.
__device__ __forceinline__ int GARDeduceLowerBound(const uint64_t* keys,
                                                   int n, uint64_t key) {
  int lo = 0, hi = n;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (keys[mid] < key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

__device__ __forceinline__ void GARDeduceEmitRow(const int32_t* row,
                                                 int n_nodes, int32_t* next,
                                                 int* next_count, int cap) {
  const int slot = atomicAdd(next_count, 1);
  if (slot >= cap) return;
  int32_t* out_row = next + static_cast<size_t>(slot) * n_nodes;
  for (int i = 0; i < n_nodes; ++i) out_row[i] = row[i];
}

__global__ void GARDeduceInitEmbeddingsKernel(const uint64_t* fwd_keys,
                                              int n_cand, int p_u, int p_v,
                                              int n_nodes, int32_t* out_emb,
                                              int* out_count, int cap) {
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int step = blockDim.x * gridDim.x;
  for (int c = static_cast<int>(tid); c < n_cand; c += static_cast<int>(step)) {
    const uint32_t gs = static_cast<uint32_t>(fwd_keys[c] >> 32);
    const uint32_t gd = static_cast<uint32_t>(fwd_keys[c]);
    if (p_u == p_v && gs != gd) continue;
    // Homomorphism: p_u != p_v with gs == gd is allowed — the CPU
    // PatternMatcher also maps distinct pattern vertices to one data vertex.
    const int slot = atomicAdd(out_count, 1);
    if (slot >= cap) continue;
    int32_t* row = out_emb + static_cast<size_t>(slot) * n_nodes;
    for (int i = 0; i < n_nodes; ++i) row[i] = -1;
    row[p_u] = static_cast<int32_t>(gs);
    row[p_v] = static_cast<int32_t>(gd);
  }
}

// Per-row adjacency-driven expansion (mirrors the CPU matcher's binary
// search over sorted pair arrays):
//   both endpoints bound  -> existence check, emit at most one row;
//   src bound only        -> scan the fwd segment with src == au;
//   dst bound only        -> scan the bwd segment with dst == av;
//   neither bound         -> full scan (BFS join order prevents this).
__global__ void GARDeduceExpandEmbeddingsKernel(
    const int32_t* curr, int n_curr, int n_nodes, int p_u, int p_v,
    const uint64_t* fwd_keys, const uint64_t* bwd_keys, int n_cand,
    int32_t* next, int* next_count, int cap) {
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int step = blockDim.x * gridDim.x;
  for (int t = static_cast<int>(tid); t < n_curr; t += static_cast<int>(step)) {
    const int32_t* row = curr + static_cast<size_t>(t) * n_nodes;
    const int32_t au = row[p_u];
    const int32_t av = row[p_v];
    if (p_u == p_v) {
      if (au < 0) continue;  // join order guarantees the endpoint is bound
      const uint64_t key =
          (static_cast<uint64_t>(au) << 32) | static_cast<uint32_t>(au);
      const int lo = GARDeduceLowerBound(fwd_keys, n_cand, key);
      if (lo < n_cand && fwd_keys[lo] == key) {
        GARDeduceEmitRow(row, n_nodes, next, next_count, cap);
      }
      continue;
    }
    if (au >= 0 && av >= 0) {
      const uint64_t key =
          (static_cast<uint64_t>(au) << 32) | static_cast<uint32_t>(av);
      const int lo = GARDeduceLowerBound(fwd_keys, n_cand, key);
      if (lo < n_cand && fwd_keys[lo] == key) {
        GARDeduceEmitRow(row, n_nodes, next, next_count, cap);
      }
    } else if (au >= 0) {
      const uint64_t prefix = static_cast<uint64_t>(au) << 32;
      const int lo = GARDeduceLowerBound(fwd_keys, n_cand, prefix);
      for (int j = lo; j < n_cand && (fwd_keys[j] >> 32) ==
                                     static_cast<uint64_t>(au);
           ++j) {
        const int slot = atomicAdd(next_count, 1);
        if (slot >= cap) return;
        int32_t* out_row = next + static_cast<size_t>(slot) * n_nodes;
        for (int i = 0; i < n_nodes; ++i) out_row[i] = row[i];
        out_row[p_v] = static_cast<int32_t>(
            static_cast<uint32_t>(fwd_keys[j]));
      }
    } else if (av >= 0) {
      const uint64_t prefix = static_cast<uint64_t>(av) << 32;
      const int lo = GARDeduceLowerBound(bwd_keys, n_cand, prefix);
      for (int j = lo; j < n_cand && (bwd_keys[j] >> 32) ==
                                     static_cast<uint64_t>(av);
           ++j) {
        const int slot = atomicAdd(next_count, 1);
        if (slot >= cap) return;
        int32_t* out_row = next + static_cast<size_t>(slot) * n_nodes;
        for (int i = 0; i < n_nodes; ++i) out_row[i] = row[i];
        out_row[p_u] = static_cast<int32_t>(
            static_cast<uint32_t>(bwd_keys[j]));
      }
    } else {
      for (int j = 0; j < n_cand; ++j) {
        const int slot = atomicAdd(next_count, 1);
        if (slot >= cap) return;
        int32_t* out_row = next + static_cast<size_t>(slot) * n_nodes;
        for (int i = 0; i < n_nodes; ++i) out_row[i] = row[i];
        out_row[p_u] = static_cast<int32_t>(
            static_cast<uint32_t>(fwd_keys[j] >> 32));
        out_row[p_v] = static_cast<int32_t>(
            static_cast<uint32_t>(fwd_keys[j]));
      }
    }
  }
}

__device__ __forceinline__ bool GARDeduceEvalCmp(int op, double lhs,
                                                 double rhs) {
  switch (op) {
    case 0: return lhs == rhs;
    case 1: return lhs != rhs;
    case 2: return lhs < rhs;
    case 3: return lhs <= rhs;
    case 4: return lhs > rhs;
    case 5: return lhs >= rhs;
  }
  return false;
}

// Edge existence over graph edge arrays sorted by (src, dst): binary search
// for the (src, dst) run, then a linear scan of the label run.
__device__ __forceinline__ bool GARDeduceHasEdge(const uint32_t* e_src,
                                                 const uint32_t* e_dst,
                                                 const int32_t* e_label,
                                                 int n_edges, uint32_t s,
                                                 uint32_t d, int32_t label) {
  int lo = 0, hi = n_edges;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (e_src[mid] < s || (e_src[mid] == s && e_dst[mid] < d)) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  for (int i = lo; i < n_edges && e_src[i] == s && e_dst[i] == d; ++i) {
    if (e_label[i] == label) return true;
  }
  return false;
}

__global__ void GARDeduceEvalEmitKernel(GARDeduceEvalParams params) {
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int step = blockDim.x * gridDim.x;
  for (int t = static_cast<int>(tid); t < params.n_emb;
       t += static_cast<int>(step)) {
    int32_t row[kGARDeduceMaxNodes];
    for (int j = 0; j < params.n_nodes; ++j) {
      row[j] = params.emb[static_cast<size_t>(t) * params.n_nodes + j];
    }
    bool ok = true;
    for (int pi = 0; pi < params.n_preds && ok; ++pi) {
      const GARPredicate& pr = params.preds[pi];
      switch (pr.kind) {
        case 0: {  // const: node.col <op> cval
          const uint32_t r = static_cast<uint32_t>(row[pr.node]);
          const uint8_t* vm = params.attr_valid[pr.col];
          if (vm != nullptr && !vm[r]) {
            ok = false;
            break;
          }
          ok = GARDeduceEvalCmp(pr.op, params.attr_vals[pr.col][r], pr.cval);
          break;
        }
        case 1: {  // var: node.col <op> rhs_node.rhs_col
          const uint32_t ri = static_cast<uint32_t>(row[pr.node]);
          const uint32_t rj = static_cast<uint32_t>(row[pr.rhs_node]);
          const uint8_t* vmi = params.attr_valid[pr.col];
          const uint8_t* vmj = params.attr_valid[pr.rhs_col];
          if ((vmi != nullptr && !vmi[ri]) ||
              (vmj != nullptr && !vmj[rj])) {
            ok = false;
            break;
          }
          ok = GARDeduceEvalCmp(pr.op, params.attr_vals[pr.col][ri],
                                params.attr_vals[pr.rhs_col][rj]);
          break;
        }
        case 2:  // edge: edge_label(node, rhs_node) exists
          ok = GARDeduceHasEdge(params.g_e_src, params.g_e_dst,
                                params.g_e_label, params.g_n_edges,
                                static_cast<uint32_t>(row[pr.node]),
                                static_cast<uint32_t>(row[pr.rhs_node]),
                                pr.edge_label);
          break;
        case 3: {  // exists: node.col has a value
          const uint32_t r = static_cast<uint32_t>(row[pr.node]);
          const uint8_t* vm = params.attr_valid[pr.col];
          ok = (vm == nullptr) || vm[r];
          break;
        }
        default:
          ok = false;
          break;
      }
    }
    if (params.x_ok != nullptr) params.x_ok[t] = ok ? 1 : 0;
    if (!ok) continue;
    atomicAdd(params.x_count, 1ULL);
    if (params.has_head) {
      const int slot = atomicAdd(params.head_count, 1);
      if (slot < params.head_cap) {
        params.head_src[slot] = static_cast<uint32_t>(row[params.head_u]);
        params.head_dst[slot] = static_cast<uint32_t>(row[params.head_v]);
      }
    }
  }
}

}  // namespace

void LaunchGARDeduceCountCandidates(const GARDeduceFilterParams& params) {
  GARDeduceCountCandidatesKernel<<<kGridDim, kBlockDim>>>(params);
  CUDA_CHECK(cudaGetLastError());
}

void LaunchGARDeduceFillCandidates(const GARDeduceFilterParams& params,
                                   int* cursors) {
  GARDeduceFillCandidatesKernel<<<kGridDim, kBlockDim>>>(params, cursors);
  CUDA_CHECK(cudaGetLastError());
}

void LaunchGARDeducePackCandidates(const uint32_t* cand_src,
                                   const uint32_t* cand_dst, int n_cand,
                                   uint64_t* fwd_keys, uint64_t* bwd_keys) {
  GARDeducePackCandidatesKernel<<<kGridDim, kBlockDim>>>(
      cand_src, cand_dst, n_cand, fwd_keys, bwd_keys);
  CUDA_CHECK(cudaGetLastError());
}

void LaunchGARDeduceInitEmbeddings(const uint64_t* fwd_keys, int n_cand,
                                   int p_u, int p_v, int n_nodes,
                                   int32_t* out_emb, int* out_count,
                                   int cap) {
  GARDeduceInitEmbeddingsKernel<<<kGridDim, kBlockDim>>>(
      fwd_keys, n_cand, p_u, p_v, n_nodes, out_emb, out_count, cap);
  CUDA_CHECK(cudaGetLastError());
}

void LaunchGARDeduceExpandEmbeddings(const int32_t* curr, int n_curr,
                                     int n_nodes, int p_u, int p_v,
                                     const uint64_t* fwd_keys,
                                     const uint64_t* bwd_keys, int n_cand,
                                     int32_t* next, int* next_count,
                                     int cap) {
  GARDeduceExpandEmbeddingsKernel<<<kGridDim, kBlockDim>>>(
      curr, n_curr, n_nodes, p_u, p_v, fwd_keys, bwd_keys, n_cand, next,
      next_count, cap);
  CUDA_CHECK(cudaGetLastError());
}

void LaunchGARDeduceEvalEmit(const GARDeduceEvalParams& params) {
  GARDeduceEvalEmitKernel<<<kGridDim, kBlockDim>>>(params);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
