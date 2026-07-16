#include "core/task/gpu_task/kernel/kernel_lftj_subiso.cuh"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "core/common/consts.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/unified_buffer.cuh"
#include "core/util/cuda_check.cuh"
#include "core/util/cuda_device.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

using VertexID = sics::matrixgraph::core::common::VertexID;
using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;

namespace {

// Maximum per-depth local candidate buffer for a single DFS thread.
constexpr uint32_t kMaxLocalCandidates = 4096;
// Maximum pattern vertices supported by the per-thread stack.
constexpr uint32_t kMaxPatternVertices = 32;
// Maximum backward neighbors per depth.
constexpr uint32_t kMaxBackwardNeighbors = 8;

struct ParametersLFTJ {
  // Pattern graph.
  VertexID n_vertices_p;
  EdgeIndex n_edges_p;
  VertexID* out_degree_p;
  VertexID* in_degree_p;
  EdgeIndex* out_offset_p;
  EdgeIndex* in_offset_p;
  VertexID* out_edges_p;
  VertexID* in_edges_p;
  VertexLabel* v_label_p;

  // Data graph.
  VertexID n_vertices_g;
  EdgeIndex n_edges_g;
  VertexID* out_degree_g;
  VertexID* in_degree_g;
  EdgeIndex* out_offset_g;
  EdgeIndex* in_offset_g;
  VertexID* out_edges_g;
  VertexID* in_edges_g;
  VertexLabel* v_label_g;

  // Unified undirected data adjacency.
  EdgeIndex* data_offsets;
  VertexID* data_neighbors;

  // Candidate sets: flat array + per-vertex offsets.
  VertexID* cand_data;
  EdgeIndex* cand_offsets;

  // Matching order and backward neighbors.
  VertexID* order;
  VertexID* bn_offsets;     // size pn+1
  VertexID* bn_data;        // flattened bn_list
  VertexID pn;

  bool canonical;
  uint64_t* match_count;

  // Per-thread workspace: each thread gets kMaxPatternVertices * kMaxLocalCandidates.
  VertexID* local_cand_workspace;
  uint32_t* visited_workspace;  // each thread gets n_vertices_g bits
};

__device__ __forceinline__ VertexID* LocalCandBuf(VertexID* workspace,
                                                   uint32_t tid, uint32_t depth) {
  return workspace + tid * kMaxPatternVertices * kMaxLocalCandidates +
         depth * kMaxLocalCandidates;
}

// Binary search for value v in sorted array [arr, arr+n).
__device__ __forceinline__ bool DeviceBinarySearch(const VertexID* arr,
                                                    uint32_t n, VertexID v) {
  int lo = 0, hi = static_cast<int>(n) - 1;
  while (lo <= hi) {
    int mid = (lo + hi) >> 1;
    VertexID midv = arr[mid];
    if (midv == v) return true;
    if (midv < v)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return false;
}

__device__ __forceinline__ void DeviceIntersectSorted(
    const VertexID* a, uint32_t na, const VertexID* b, uint32_t nb,
    VertexID* out, uint32_t& out_size) {
  out_size = 0;
  if (na == 0 || nb == 0) return;
  if (na > nb) {
    DeviceIntersectSorted(b, nb, a, na, out, out_size);
    return;
  }
  for (uint32_t i = 0; i < na; ++i) {
    VertexID v = a[i];
    if (DeviceBinarySearch(b, nb, v)) {
      out[out_size++] = v;
    }
  }
}

__device__ __forceinline__ bool VisitedGet(const uint32_t* visited,
                                            VertexID v) {
  return visited[v >> 5] & (1u << (v & 31u));
}

__device__ __forceinline__ void VisitedSet(uint32_t* visited, VertexID v) {
  atomicOr((int*)(visited + (v >> 5)), 1u << (v & 31u));
}

__device__ __forceinline__ void VisitedClear(uint32_t* visited, VertexID v) {
  atomicAnd((int*)(visited + (v >> 5)), ~(1u << (v & 31u)));
}

static __global__ void LFTJEnumerateKernel(ParametersLFTJ params) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  VertexID pn = params.pn;
  VertexID dn = params.n_vertices_g;
  if (pn == 0) return;

  uint32_t visited_words = (dn + 31) >> 5;
  uint32_t* visited = params.visited_workspace + tid * visited_words;
  for (uint32_t i = 0; i < visited_words; ++i) visited[i] = 0;

  VertexID embedding[kMaxPatternVertices];
  uint32_t idx[kMaxPatternVertices];
  for (uint32_t i = 0; i < pn; ++i) idx[i] = 0;

  VertexID* local_cand[kMaxPatternVertices];
  uint32_t local_cand_size[kMaxPatternVertices];

  // Root vertex = order[0].
  VertexID root_u = params.order[0];
  EdgeIndex root_cand_begin = params.cand_offsets[root_u];
  EdgeIndex root_cand_end = params.cand_offsets[root_u + 1];
  EdgeIndex root_total = root_cand_end - root_cand_begin;

  // Each thread processes root candidates [tid, tid+step, ...).
  VertexID* root_buf = LocalCandBuf(params.local_cand_workspace, tid, 0);
  uint32_t root_size = 0;
  for (EdgeIndex i = tid; i < root_total && root_size < kMaxLocalCandidates;
       i += step) {
    root_buf[root_size++] = params.cand_data[root_cand_begin + i];
  }
  local_cand[0] = root_buf;
  local_cand_size[0] = root_size;

  int32_t depth = 0;
  uint64_t local_count = 0;

  while (true) {
    if (depth < 0) break;

    if (idx[depth] >= local_cand_size[depth]) {
      if (depth > 0) {
        VisitedClear(visited, embedding[depth - 1]);
      }
      --depth;
      continue;
    }

    VertexID v = local_cand[depth][idx[depth]++];
    if (VisitedGet(visited, v)) continue;
    if (params.canonical && depth > 0 && v <= embedding[depth - 1]) continue;

    embedding[depth] = v;
    VisitedSet(visited, v);

    if (depth == pn - 1) {
      ++local_count;
      VisitedClear(visited, v);
      continue;
    }

    // Compute local candidates for depth+1.
    VertexID next_depth = depth + 1;
    VertexID next_u = params.order[next_depth];
    VertexID* base = params.cand_data + params.cand_offsets[next_u];
    uint32_t base_size =
        params.cand_offsets[next_u + 1] - params.cand_offsets[next_u];

    if (base_size == 0) {
      VisitedClear(visited, v);
      continue;
    }

    VertexID bn[kMaxBackwardNeighbors];
    const VertexID* bn_nbrs[kMaxBackwardNeighbors];
    uint32_t bn_deg[kMaxBackwardNeighbors];
    uint32_t bn_count = params.bn_offsets[next_depth + 1] -
                        params.bn_offsets[next_depth];
    if (bn_count > kMaxBackwardNeighbors) bn_count = kMaxBackwardNeighbors;

    for (uint32_t i = 0; i < bn_count; ++i) {
      bn[i] = params.bn_data[params.bn_offsets[next_depth] + i];
      VertexID mapped_v = embedding[bn[i]];
      bn_nbrs[i] = params.data_neighbors + params.data_offsets[mapped_v];
      bn_deg[i] =
          params.data_offsets[mapped_v + 1] - params.data_offsets[mapped_v];
    }

    VertexID* out_buf = LocalCandBuf(params.local_cand_workspace, tid, next_depth);
    uint32_t out_size = 0;

    if (bn_count == 0) {
      out_size = base_size < kMaxLocalCandidates ? base_size
                                                 : kMaxLocalCandidates;
      for (uint32_t i = 0; i < out_size; ++i) out_buf[i] = base[i];
    } else {
      // Find smallest list to iterate over.
      const VertexID* iter_list = base;
      uint32_t iter_size = base_size;
      bool iter_is_base = true;
      for (uint32_t i = 0; i < bn_count; ++i) {
        if (bn_deg[i] < iter_size) {
          iter_list = bn_nbrs[i];
          iter_size = bn_deg[i];
          iter_is_base = false;
        }
      }

      for (uint32_t i = 0; i < iter_size && out_size < kMaxLocalCandidates;
           ++i) {
        VertexID cand = iter_list[i];
        if (iter_is_base) {
          bool ok = true;
          for (uint32_t j = 0; j < bn_count; ++j) {
            if (!DeviceBinarySearch(bn_nbrs[j], bn_deg[j], cand)) {
              ok = false;
              break;
            }
          }
          if (ok) out_buf[out_size++] = cand;
        } else {
          if (!DeviceBinarySearch(base, base_size, cand)) continue;
          bool ok = true;
          for (uint32_t j = 0; j < bn_count; ++j) {
            if (bn_nbrs[j] == iter_list) continue;
            if (!DeviceBinarySearch(bn_nbrs[j], bn_deg[j], cand)) {
              ok = false;
              break;
            }
          }
          if (ok) out_buf[out_size++] = cand;
        }
      }
    }

    if (out_size == 0) {
      VisitedClear(visited, v);
      continue;
    }

    local_cand[next_depth] = out_buf;
    local_cand_size[next_depth] = out_size;
    idx[next_depth] = 0;
    ++depth;
  }

  atomicAdd(reinterpret_cast<unsigned long long*>(params.match_count),
            local_count);
}

uint32_t LFTJLaunchTotalThreads(int requested) {
  if (const char* s = std::getenv("MG_LFTJ_TOTAL_THREADS")) {
    int v = std::atoi(s);
    if (v > 0) return static_cast<uint32_t>(v);
  }
  if (requested > 0) return static_cast<uint32_t>(requested);
  return 256;  // default reasonable GPU occupancy
}

uint32_t LFTJLaunchBlockDim() {
  if (const char* s = std::getenv("MG_LFTJ_BLOCK")) {
    int v = std::atoi(s);
    if (v > 0) return static_cast<uint32_t>(v);
  }
  return 128;
}

uint32_t LFTJLaunchGridDim(int requested) {
  if (const char* s = std::getenv("MG_LFTJ_GRID")) {
    int v = std::atoi(s);
    if (v > 0) return static_cast<uint32_t>(v);
  }
  uint32_t total = LFTJLaunchTotalThreads(requested);
  uint32_t block = LFTJLaunchBlockDim();
  return (total + block - 1) / block;
}

}  // namespace

uint64_t LFTJSubIsoKernelWrapper::Enumerate(
    const ImmutableCSR& pattern, const ImmutableCSR& data_graph,
    const std::vector<EdgeIndex>& data_offsets,
    const std::vector<VertexID>& data_neighbors,
    const std::vector<std::vector<VertexID>>& pattern_adj,
    const std::vector<std::vector<VertexID>>& candidates,
    const std::vector<VertexID>& order,
    const std::vector<std::vector<VertexID>>& bn_list,
    bool canonical, int num_threads) {
  using UnifiedOwnedBufferUint8 =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint8_t>;
  using UnifiedOwnedBufferUint32 =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint32_t>;
  using UnifiedOwnedBufferUint64 =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint64_t>;
  using UnifiedOwnedBufferVertexID =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexID>;
  using UnifiedOwnedBufferEdgeIndex =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<EdgeIndex>;
  using UnifiedOwnedBufferVertexLabel =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexLabel>;

  VertexID pn = pattern.get_num_vertices();
  VertexID dn = data_graph.get_num_vertices();

  // Pattern CSR buffers.
  auto pattern_buf_size =
      sizeof(VertexID) * pn +           // globalid
      sizeof(VertexID) * pn +           // in_degree
      sizeof(VertexID) * pn +           // out_degree
      sizeof(EdgeIndex) * (pn + 1) +    // in_offset
      sizeof(EdgeIndex) * (pn + 1) +    // out_offset
      sizeof(VertexID) * pattern.get_num_incoming_edges() +
      sizeof(VertexID) * pattern.get_num_outgoing_edges();
  UnifiedOwnedBufferUint8 d_pattern_csr;
  d_pattern_csr.Init(pattern_buf_size);
  std::memcpy(d_pattern_csr.GetPtr(), pattern.GetGraphBuffer(),
              pattern_buf_size);

  VertexID* d_globalid_p = reinterpret_cast<VertexID*>(d_pattern_csr.GetPtr());
  VertexID* d_in_degree_p = d_globalid_p + pn;
  VertexID* d_out_degree_p = d_in_degree_p + pn;
  EdgeIndex* d_in_offset_p = reinterpret_cast<EdgeIndex*>(d_out_degree_p + pn);
  EdgeIndex* d_out_offset_p = d_in_offset_p + pn + 1;
  VertexID* d_in_edges_p = reinterpret_cast<VertexID*>(d_out_offset_p + pn + 1);
  VertexID* d_out_edges_p = d_in_edges_p + pattern.get_num_incoming_edges();

  UnifiedOwnedBufferVertexLabel d_v_label_p;
  d_v_label_p.Init(sizeof(VertexLabel) * pn);
  std::memcpy(d_v_label_p.GetPtr(), pattern.GetVLabelBasePointer(),
              sizeof(VertexLabel) * pn);

  // Data CSR buffers.
  auto data_buf_size =
      sizeof(VertexID) * dn +           // globalid
      sizeof(VertexID) * dn +           // in_degree
      sizeof(VertexID) * dn +           // out_degree
      sizeof(EdgeIndex) * (dn + 1) +    // in_offset
      sizeof(EdgeIndex) * (dn + 1) +    // out_offset
      sizeof(VertexID) * data_graph.get_num_incoming_edges() +
      sizeof(VertexID) * data_graph.get_num_outgoing_edges();
  UnifiedOwnedBufferUint8 d_data_csr;
  d_data_csr.Init(data_buf_size);
  std::memcpy(d_data_csr.GetPtr(), data_graph.GetGraphBuffer(),
              data_buf_size);

  VertexID* d_globalid_g = reinterpret_cast<VertexID*>(d_data_csr.GetPtr());
  VertexID* d_in_degree_g = d_globalid_g + dn;
  VertexID* d_out_degree_g = d_in_degree_g + dn;
  EdgeIndex* d_in_offset_g = reinterpret_cast<EdgeIndex*>(d_out_degree_g + dn);
  EdgeIndex* d_out_offset_g = d_in_offset_g + dn + 1;
  VertexID* d_in_edges_g = reinterpret_cast<VertexID*>(d_out_offset_g + dn + 1);
  VertexID* d_out_edges_g =
      d_in_edges_g + data_graph.get_num_incoming_edges();

  UnifiedOwnedBufferVertexLabel d_v_label_g;
  d_v_label_g.Init(sizeof(VertexLabel) * dn);
  std::memcpy(d_v_label_g.GetPtr(), data_graph.GetVLabelBasePointer(),
              sizeof(VertexLabel) * dn);

  // Undirected data adjacency.
  UnifiedOwnedBufferEdgeIndex d_data_offsets;
  d_data_offsets.Init(sizeof(EdgeIndex) * data_offsets.size());
  std::memcpy(d_data_offsets.GetPtr(), data_offsets.data(),
              sizeof(EdgeIndex) * data_offsets.size());

  UnifiedOwnedBufferVertexID d_data_neighbors;
  d_data_neighbors.Init(sizeof(VertexID) * data_neighbors.size());
  std::memcpy(d_data_neighbors.GetPtr(), data_neighbors.data(),
              sizeof(VertexID) * data_neighbors.size());

  // Candidate sets.
  EdgeIndex total_cand = 0;
  std::vector<EdgeIndex> cand_offsets_host(pn + 1, 0);
  for (VertexID u = 0; u < pn; ++u) {
    cand_offsets_host[u + 1] =
        cand_offsets_host[u] + candidates[u].size();
  }
  total_cand = cand_offsets_host[pn];

  UnifiedOwnedBufferVertexID d_cand_data;
  d_cand_data.Init(sizeof(VertexID) * total_cand);
  for (VertexID u = 0; u < pn; ++u) {
    std::memcpy(d_cand_data.GetPtr() + cand_offsets_host[u],
                candidates[u].data(), sizeof(VertexID) * candidates[u].size());
  }

  UnifiedOwnedBufferEdgeIndex d_cand_offsets;
  d_cand_offsets.Init(sizeof(EdgeIndex) * cand_offsets_host.size());
  std::memcpy(d_cand_offsets.GetPtr(), cand_offsets_host.data(),
              sizeof(EdgeIndex) * cand_offsets_host.size());

  // Matching order.
  UnifiedOwnedBufferVertexID d_order;
  d_order.Init(sizeof(VertexID) * pn);
  std::memcpy(d_order.GetPtr(), order.data(), sizeof(VertexID) * pn);

  // Backward neighbors.
  std::vector<VertexID> bn_offsets_host(pn + 1, 0);
  std::vector<VertexID> bn_data_host;
  for (VertexID d = 0; d < pn; ++d) {
    bn_offsets_host[d + 1] =
        bn_offsets_host[d] + static_cast<VertexID>(bn_list[d].size());
    bn_data_host.insert(bn_data_host.end(), bn_list[d].begin(),
                        bn_list[d].end());
  }
  UnifiedOwnedBufferVertexID d_bn_offsets;
  d_bn_offsets.Init(sizeof(VertexID) * bn_offsets_host.size());
  std::memcpy(d_bn_offsets.GetPtr(), bn_offsets_host.data(),
              sizeof(VertexID) * bn_offsets_host.size());

  UnifiedOwnedBufferVertexID d_bn_data;
  d_bn_data.Init(sizeof(VertexID) * bn_data_host.size());
  std::memcpy(d_bn_data.GetPtr(), bn_data_host.data(),
              sizeof(VertexID) * bn_data_host.size());

  // Match counter.
  UnifiedOwnedBufferUint64 d_match_count;
  d_match_count.Init(sizeof(uint64_t));
  std::memset(d_match_count.GetPtr(), 0, sizeof(uint64_t));

  // Per-thread workspace: allocate for the physical launched thread count
  // (grid * block), even if the logical step is smaller.
  uint32_t total_threads = LFTJLaunchTotalThreads(num_threads);
  uint32_t grid = LFTJLaunchGridDim(num_threads);
  uint32_t block = LFTJLaunchBlockDim();
  uint32_t physical_threads = grid * block;

  UnifiedOwnedBufferVertexID d_local_cand_workspace;
  d_local_cand_workspace.Init(sizeof(VertexID) * physical_threads *
                              kMaxPatternVertices * kMaxLocalCandidates);

  uint32_t visited_words = (dn + 31) >> 5;
  UnifiedOwnedBufferUint32 d_visited_workspace;
  d_visited_workspace.Init(sizeof(uint32_t) * physical_threads * visited_words);

  ParametersLFTJ params{
      .n_vertices_p = pn,
      .n_edges_p = pattern.get_num_outgoing_edges(),
      .out_degree_p = d_out_degree_p,
      .in_degree_p = d_in_degree_p,
      .out_offset_p = d_out_offset_p,
      .in_offset_p = d_in_offset_p,
      .out_edges_p = d_out_edges_p,
      .in_edges_p = d_in_edges_p,
      .v_label_p = d_v_label_p.GetPtr(),
      .n_vertices_g = dn,
      .n_edges_g = data_graph.get_num_outgoing_edges(),
      .out_degree_g = d_out_degree_g,
      .in_degree_g = d_in_degree_g,
      .out_offset_g = d_out_offset_g,
      .in_offset_g = d_in_offset_g,
      .out_edges_g = d_out_edges_g,
      .in_edges_g = d_in_edges_g,
      .v_label_g = d_v_label_g.GetPtr(),
      .data_offsets = d_data_offsets.GetPtr(),
      .data_neighbors = d_data_neighbors.GetPtr(),
      .cand_data = d_cand_data.GetPtr(),
      .cand_offsets = d_cand_offsets.GetPtr(),
      .order = d_order.GetPtr(),
      .bn_offsets = d_bn_offsets.GetPtr(),
      .bn_data = d_bn_data.GetPtr(),
      .pn = pn,
      .canonical = canonical,
      .match_count = d_match_count.GetPtr(),
      .local_cand_workspace = d_local_cand_workspace.GetPtr(),
      .visited_workspace = d_visited_workspace.GetPtr(),
  };

  auto t0 = std::chrono::high_resolution_clock::now();
  std::cout << "[LFTJSubIsoGpu] launch grid=" << grid << " block=" << block
            << " threads=" << total_threads << std::endl;

  LFTJEnumerateKernel<<<grid, block>>>(params);
  CUDA_CHECK(cudaDeviceSynchronize());

  auto t1 = std::chrono::high_resolution_clock::now();
  double ms =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() /
      1000.0;
  std::cout << "[LFTJSubIsoGpu] device enumeration kernel: " << ms << " ms"
            << std::endl;

  uint64_t match_count = 0;
  std::memcpy(&match_count, d_match_count.GetPtr(), sizeof(uint64_t));
  return match_count;
}

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
