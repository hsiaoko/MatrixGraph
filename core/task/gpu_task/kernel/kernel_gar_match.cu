#include "core/task/gpu_task/kernel/kernel_gar_match.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#include "core/common/consts.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/heap.cuh"
#include "core/data_structures/host_buffer.cuh"
#include "core/data_structures/mini_kernel_bitmap.cuh"
#include "core/task/gpu_task/kernel/algorithms/hash.cuh"
#include "core/util/cuda_check.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

using GARGraphArrays = sics::matrixgraph::core::data_structures::GARGraphArrays;
using GARPatternArrays =
    sics::matrixgraph::core::data_structures::GARPatternArrays;
using GARMatchArrays = sics::matrixgraph::core::data_structures::GARMatchArrays;
using BufferUint32 = sics::matrixgraph::core::data_structures::Buffer<uint32_t>;
using BufferInt32 = sics::matrixgraph::core::data_structures::Buffer<int32_t>;
using BufferInt = sics::matrixgraph::core::data_structures::Buffer<int>;
using DeviceOwnedBufferUint32 =
    sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<uint32_t>;
using DeviceOwnedBufferInt32 =
    sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<int32_t>;
using DeviceOwnedBufferInt =
    sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<int>;
using MiniKernelBitmap = sics::matrixgraph::core::task::kernel::MiniKernelBitmap;
using MinHeap = sics::matrixgraph::core::task::kernel::MinHeap;
using sics::matrixgraph::core::common::kDefaultHeapCapacity;
using sics::matrixgraph::core::common::kMaxVertexID;
using sics::matrixgraph::core::common::kBlockDim;
using sics::matrixgraph::core::common::kGridDim;

GARMatchKernelWrapper* GARMatchKernelWrapper::GetInstance() {
  if (ptr_ == nullptr) {
    ptr_ = new GARMatchKernelWrapper();
  }
  return ptr_;
}

static __noinline__ __device__ bool LabelDegreeFilter(
    const GARPatternArrays& p, const GARGraphArrays& g, int u_idx, uint32_t v_idx) {
  if (u_idx < 0 || u_idx >= p.n_nodes) return false;
  if (v_idx >= static_cast<uint32_t>(g.n_vertices)) return false;
  if (p.node_label_idx == nullptr || g.v_label_idx == nullptr) return false;

  const int32_t u_label = p.node_label_idx[u_idx];
  const int32_t v_label = g.v_label_idx[v_idx];
  if (u_label != v_label) return false;

  int p_out = 0, p_in = 0, g_out = 0, g_in = 0;
  for (int e = 0; e < p.n_edges; ++e) {
    if (p.edge_src[e] == u_idx) ++p_out;
    if (p.edge_dst[e] == u_idx) ++p_in;
  }
  for (int e = 0; e < g.n_edges; ++e) {
    if (g.e_src[e] == v_idx) ++g_out;
    if (g.e_dst[e] == v_idx) ++g_in;
  }
  return g_out >= p_out && g_in >= p_in;
}

static __noinline__ __device__ bool NeighborLabelCounterFilter(
    const GARPatternArrays& p, const GARGraphArrays& g, int u_idx, uint32_t v_idx) {
  if (u_idx < 0 || u_idx >= p.n_nodes) return false;
  if (v_idx >= static_cast<uint32_t>(g.n_vertices)) return false;
  if (p.node_label_idx == nullptr || g.v_label_idx == nullptr) return false;

  const int32_t u_label = p.node_label_idx[u_idx];
  const int32_t v_label = g.v_label_idx[v_idx];
  if (u_label != v_label) return false;

  MiniKernelBitmap u_label_visited(32);
  MiniKernelBitmap v_label_visited(32);
  u_label_visited.Clear();
  v_label_visited.Clear();
  for (int e = 0; e < p.n_edges; ++e) {
    if (p.edge_src[e] == u_idx && p.edge_dst[e] >= 0 && p.edge_dst[e] < p.n_nodes) {
      const uint32_t l = static_cast<uint32_t>(p.node_label_idx[p.edge_dst[e]]) & 31U;
      u_label_visited.SetBit(l);
    }
    if (p.edge_dst[e] == u_idx && p.edge_src[e] >= 0 && p.edge_src[e] < p.n_nodes) {
      const uint32_t l = static_cast<uint32_t>(p.node_label_idx[p.edge_src[e]]) & 31U;
      u_label_visited.SetBit(l);
    }
  }
  for (int e = 0; e < g.n_edges; ++e) {
    if (g.e_src[e] == v_idx) {
      const uint32_t nbr = g.e_dst[e];
      if (nbr < static_cast<uint32_t>(g.n_vertices)) {
        const uint32_t l = static_cast<uint32_t>(g.v_label_idx[nbr]) & 31U;
        v_label_visited.SetBit(l);
      }
    }
    if (g.e_dst[e] == v_idx) {
      const uint32_t nbr = g.e_src[e];
      if (nbr < static_cast<uint32_t>(g.n_vertices)) {
        const uint32_t l = static_cast<uint32_t>(g.v_label_idx[nbr]) & 31U;
        v_label_visited.SetBit(l);
      }
    }
  }
  return v_label_visited.Count() >= u_label_visited.Count();
}

static __noinline__ __device__ bool KMinWiseIPFilter(
    const GARPatternArrays& p, const GARGraphArrays& g, int u_idx, uint32_t v_idx) {
  if (u_idx < 0 || u_idx >= p.n_nodes) return false;
  if (v_idx >= static_cast<uint32_t>(g.n_vertices)) return false;
  if (p.node_label_idx == nullptr || g.v_label_idx == nullptr) return false;

  const int32_t u_label = p.node_label_idx[u_idx];
  const int32_t v_label = g.v_label_idx[v_idx];
  if (u_label != v_label) return false;

  int p_out = 0, p_in = 0, g_out = 0, g_in = 0;
  for (int e = 0; e < p.n_edges; ++e) {
    if (p.edge_src[e] == u_idx) ++p_out;
    if (p.edge_dst[e] == u_idx) ++p_in;
  }
  for (int e = 0; e < g.n_edges; ++e) {
    if (g.e_src[e] == v_idx) ++g_out;
    if (g.e_dst[e] == v_idx) ++g_in;
  }

  MiniKernelBitmap u_label_visited(32);
  MiniKernelBitmap v_label_visited(32);
  MinHeap u_k_min_heap;
  MinHeap v_k_min_heap;
  uint32_t u_k_min_heap_data[kDefaultHeapCapacity];
  uint32_t v_k_min_heap_data[kDefaultHeapCapacity];

  for (int e = 0; e < p.n_edges; ++e) {
    if (p.edge_src[e] == u_idx && p.edge_dst[e] >= 0 && p.edge_dst[e] < p.n_nodes) {
      const uint32_t l = static_cast<uint32_t>(p.node_label_idx[p.edge_dst[e]]) & 31U;
      u_label_visited.SetBit(l);
      u_k_min_heap.Insert(HashTable(l));
    }
    if (p.edge_dst[e] == u_idx && p.edge_src[e] >= 0 && p.edge_src[e] < p.n_nodes) {
      const uint32_t l = static_cast<uint32_t>(p.node_label_idx[p.edge_src[e]]) & 31U;
      u_label_visited.SetBit(l);
      u_k_min_heap.Insert(HashTable(l));
    }
  }
  for (int e = 0; e < g.n_edges; ++e) {
    if (g.e_src[e] == v_idx) {
      const uint32_t nbr = g.e_dst[e];
      if (nbr < static_cast<uint32_t>(g.n_vertices)) {
        const uint32_t l = static_cast<uint32_t>(g.v_label_idx[nbr]) & 31U;
        v_label_visited.SetBit(l);
        v_k_min_heap.Insert(HashTable(l));
      }
    }
    if (g.e_dst[e] == v_idx) {
      const uint32_t nbr = g.e_src[e];
      if (nbr < static_cast<uint32_t>(g.n_vertices)) {
        const uint32_t l = static_cast<uint32_t>(g.v_label_idx[nbr]) & 31U;
        v_label_visited.SetBit(l);
        v_k_min_heap.Insert(HashTable(l));
      }
    }
  }

  u_k_min_heap.CopyData(u_k_min_heap_data);
  v_k_min_heap.CopyData(v_k_min_heap_data);
  for (uint32_t i = 0; i < v_k_min_heap.get_offset(); ++i) {
    const uint32_t v_ip = v_k_min_heap_data[i];
    for (uint32_t j = 0; j < u_k_min_heap.get_offset(); ++j) {
      if (u_k_min_heap_data[j] == v_ip) {
        v_k_min_heap_data[i] = kMaxVertexID;
        u_k_min_heap_data[j] = kMaxVertexID;
        break;
      }
    }
  }
  uint32_t min_v = kMaxVertexID;
  for (uint32_t i = 0; i < v_k_min_heap.get_offset(); ++i) {
    if (v_k_min_heap_data[i] < min_v) min_v = v_k_min_heap_data[i];
  }
  for (uint32_t i = 0; i < u_k_min_heap.get_offset(); ++i) {
    if (u_k_min_heap_data[i] < min_v) return false;
  }

  return v_label_visited.Count() >= u_label_visited.Count() && g_out >= p_out &&
         g_in >= p_in;
}

static __noinline__ __device__ bool GARVertexFilter(
    const GARPatternArrays& p, const GARGraphArrays& g, int u_idx, uint32_t v_idx) {
  return LabelDegreeFilter(p, g, u_idx, v_idx) &&
         NeighborLabelCounterFilter(p, g, u_idx, v_idx) &&
         KMinWiseIPFilter(p, g, u_idx, v_idx);
}

struct ParametersGARFilter {
  const uint32_t* g_v_id;
  const int32_t* g_v_label_idx;
  int n_vertices_g;

  const uint32_t* g_e_src;
  const uint32_t* g_e_dst;
  const int32_t* g_e_label_idx;
  int n_edges_g;

  int32_t p_src_label;
  int32_t p_dst_label;
  int32_t p_edge_label;
  const int32_t* p_node_label_idx;
  int p_n_nodes;
  const int32_t* p_edge_src;
  const int32_t* p_edge_dst;
  int p_n_edges;
  int32_t p_src_idx;
  int32_t p_dst_idx;

  uint32_t* cand_src;
  uint32_t* cand_dst;
  int* cand_count;
  int cand_capacity;
};

static __global__ void GARFilterEdgeCandidatesKernel(ParametersGARFilter params) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  for (int ge = static_cast<int>(tid); ge < params.n_edges_g;
       ge += static_cast<int>(step)) {
    if (params.g_e_label_idx &&
        params.g_e_label_idx[ge] != params.p_edge_label) {
      continue;
    }
    const uint32_t gs = params.g_e_src[ge];
    const uint32_t gd = params.g_e_dst[ge];
    if (gs >= static_cast<uint32_t>(params.n_vertices_g) ||
        gd >= static_cast<uint32_t>(params.n_vertices_g) ||
        params.g_v_label_idx == nullptr) {
      continue;
    }
    const int32_t src_label = params.g_v_label_idx[gs];
    const int32_t dst_label = params.g_v_label_idx[gd];
    if (src_label != params.p_src_label || dst_label != params.p_dst_label) {
      continue;
    }
    GARGraphArrays g_view{
        params.g_v_id,      params.g_v_label_idx, params.n_vertices_g,
        params.g_e_src,     params.g_e_dst,       nullptr,
        params.g_e_label_idx, params.n_edges_g};
    GARPatternArrays p_view{
        params.p_node_label_idx, params.p_n_nodes, params.p_edge_src,
        params.p_edge_dst,       nullptr,          params.p_n_edges};
    if (!GARVertexFilter(p_view, g_view, params.p_src_idx, gs)) continue;
    if (!GARVertexFilter(p_view, g_view, params.p_dst_idx, gd)) continue;
    int slot = atomicAdd(params.cand_count, 1);
    if (slot < params.cand_capacity) {
      params.cand_src[slot] = gs;
      params.cand_dst[slot] = gd;
    }
  }
}

static __device__ __forceinline__ bool GARContainsAssigned(
    const int32_t* row, int n_nodes, uint32_t v) {
  for (int i = 0; i < n_nodes; ++i) {
    if (row[i] >= 0 && static_cast<uint32_t>(row[i]) == v) return true;
  }
  return false;
}

static __global__ void GARInitEmbeddingsKernel(
    const uint32_t* cand_src, const uint32_t* cand_dst, int n_cand, int p_u,
    int p_v, int n_nodes, int32_t* out_embeddings, int* out_count,
    int out_capacity) {
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int step = blockDim.x * gridDim.x;
  for (int c = static_cast<int>(tid); c < n_cand; c += static_cast<int>(step)) {
    const uint32_t gs = cand_src[c];
    const uint32_t gd = cand_dst[c];
    if (p_u == p_v && gs != gd) continue;
    if (p_u != p_v && gs == gd) continue;
    int slot = atomicAdd(out_count, 1);
    if (slot >= out_capacity) continue;
    int32_t* row = out_embeddings + static_cast<size_t>(slot) * n_nodes;
    for (int i = 0; i < n_nodes; ++i) row[i] = -1;
    row[p_u] = static_cast<int32_t>(gs);
    row[p_v] = static_cast<int32_t>(gd);
  }
}

static __global__ void GARExpandEmbeddingsKernel(
    const int32_t* curr_embeddings, int curr_count, int n_nodes, int p_u,
    int p_v, const uint32_t* cand_src, const uint32_t* cand_dst, int n_cand,
    int32_t* next_embeddings, int* next_count, int next_capacity) {
  const unsigned long long tid =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long step =
      static_cast<unsigned long long>(blockDim.x) * gridDim.x;
  const unsigned long long total_pairs =
      static_cast<unsigned long long>(curr_count) *
      static_cast<unsigned long long>(n_cand);
  for (unsigned long long k = tid; k < total_pairs; k += step) {
    const int emb_idx = static_cast<int>(k / static_cast<unsigned long long>(n_cand));
    const int cand_idx = static_cast<int>(k % static_cast<unsigned long long>(n_cand));
    const int32_t* row =
        curr_embeddings + static_cast<size_t>(emb_idx) * n_nodes;
    const uint32_t gs = cand_src[cand_idx];
    const uint32_t gd = cand_dst[cand_idx];
    if (p_u == p_v && gs != gd) continue;
    if (p_u != p_v && gs == gd) continue;

    const int32_t au = row[p_u];
    const int32_t av = row[p_v];
    if (au >= 0 && static_cast<uint32_t>(au) != gs) continue;
    if (av >= 0 && static_cast<uint32_t>(av) != gd) continue;

    if (au < 0 && GARContainsAssigned(row, n_nodes, gs)) continue;
    if (av < 0 && GARContainsAssigned(row, n_nodes, gd)) continue;

    int slot = atomicAdd(next_count, 1);
    if (slot >= next_capacity) continue;
    int32_t* out_row = next_embeddings + static_cast<size_t>(slot) * n_nodes;
    for (int i = 0; i < n_nodes; ++i) out_row[i] = row[i];
    out_row[p_u] = static_cast<int32_t>(gs);
    out_row[p_v] = static_cast<int32_t>(gd);
  }
}

int GARMatchKernelWrapper::GARMatch(const GARGraphArrays& g,
                                    const GARPatternArrays& p,
                                    GARMatchArrays* out) {
  std::cout << "GARMatchKernelWrapper::GARMatch (placeholder, filters exposed)"
            << std::endl;
  g.Print(3);
  p.Print(3);

  // CUDA kernel stage 1: candidate filtering for each pattern edge.
  std::vector<std::vector<uint32_t>> cand_src_by_edge(
      static_cast<size_t>(std::max(0, p.n_edges)));
  std::vector<std::vector<uint32_t>> cand_dst_by_edge(
      static_cast<size_t>(std::max(0, p.n_edges)));
  if (g.n_edges > 0 && p.n_edges > 0 && g.e_src && g.e_dst && p.edge_src &&
      p.edge_dst && p.node_label_idx) {
    BufferUint32 h_g_v_id{const_cast<uint32_t*>(g.v_id),
                          sizeof(uint32_t) * static_cast<size_t>(g.n_vertices)};
    BufferInt32 h_g_v_label_idx{const_cast<int32_t*>(g.v_label_idx),
                                sizeof(int32_t) *
                                    static_cast<size_t>(g.n_vertices)};
    BufferUint32 h_g_e_src{const_cast<uint32_t*>(g.e_src),
                           sizeof(uint32_t) * static_cast<size_t>(g.n_edges)};
    BufferUint32 h_g_e_dst{const_cast<uint32_t*>(g.e_dst),
                           sizeof(uint32_t) * static_cast<size_t>(g.n_edges)};
    BufferInt32 h_g_e_label_idx{
        const_cast<int32_t*>(g.e_label_idx),
        sizeof(int32_t) * static_cast<size_t>(g.n_edges)};
    BufferInt32 h_p_node_label_idx{
        const_cast<int32_t*>(p.node_label_idx),
        sizeof(int32_t) * static_cast<size_t>(p.n_nodes)};
    BufferInt32 h_p_edge_src{const_cast<int32_t*>(p.edge_src),
                             sizeof(int32_t) * static_cast<size_t>(p.n_edges)};
    BufferInt32 h_p_edge_dst{const_cast<int32_t*>(p.edge_dst),
                             sizeof(int32_t) * static_cast<size_t>(p.n_edges)};

    DeviceOwnedBufferUint32 d_g_v_id;
    DeviceOwnedBufferInt32 d_g_v_label_idx;
    DeviceOwnedBufferUint32 d_g_e_src;
    DeviceOwnedBufferUint32 d_g_e_dst;
    DeviceOwnedBufferInt32 d_g_e_label_idx;
    DeviceOwnedBufferInt32 d_p_node_label_idx;
    DeviceOwnedBufferInt32 d_p_edge_src;
    DeviceOwnedBufferInt32 d_p_edge_dst;
    d_g_v_id.Init(h_g_v_id);
    d_g_v_label_idx.Init(h_g_v_label_idx);
    d_g_e_src.Init(h_g_e_src);
    d_g_e_dst.Init(h_g_e_dst);
    d_g_e_label_idx.Init(h_g_e_label_idx);
    d_p_node_label_idx.Init(h_p_node_label_idx);
    d_p_edge_src.Init(h_p_edge_src);
    d_p_edge_dst.Init(h_p_edge_dst);

    dim3 dimBlock(kBlockDim);
    dim3 dimGrid(kGridDim);
    for (int pe = 0; pe < p.n_edges; ++pe) {
      const int32_t pu = p.edge_src[pe];
      const int32_t pv = p.edge_dst[pe];
      if (pu < 0 || pu >= p.n_nodes || pv < 0 || pv >= p.n_nodes) continue;

      DeviceOwnedBufferUint32 d_cand_src(
          sizeof(uint32_t) * static_cast<size_t>(g.n_edges));
      DeviceOwnedBufferUint32 d_cand_dst(
          sizeof(uint32_t) * static_cast<size_t>(g.n_edges));
      DeviceOwnedBufferInt d_cand_count(sizeof(int));

      ParametersGARFilter params{
          .g_v_id = d_g_v_id.GetPtr(),
          .g_v_label_idx = d_g_v_label_idx.GetPtr(),
          .n_vertices_g = g.n_vertices,
          .g_e_src = d_g_e_src.GetPtr(),
          .g_e_dst = d_g_e_dst.GetPtr(),
          .g_e_label_idx = d_g_e_label_idx.GetPtr(),
          .n_edges_g = g.n_edges,
          .p_src_label = p.node_label_idx[pu],
          .p_dst_label = p.node_label_idx[pv],
          .p_edge_label = p.edge_label_idx ? p.edge_label_idx[pe] : 0,
          .p_node_label_idx = d_p_node_label_idx.GetPtr(),
          .p_n_nodes = p.n_nodes,
          .p_edge_src = d_p_edge_src.GetPtr(),
          .p_edge_dst = d_p_edge_dst.GetPtr(),
          .p_n_edges = p.n_edges,
          .p_src_idx = pu,
          .p_dst_idx = pv,
          .cand_src = d_cand_src.GetPtr(),
          .cand_dst = d_cand_dst.GetPtr(),
          .cand_count = d_cand_count.GetPtr(),
          .cand_capacity = g.n_edges,
      };
      GARFilterEdgeCandidatesKernel<<<dimGrid, dimBlock>>>(params);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      int h_count = 0;
      BufferInt h_count_buf{&h_count, sizeof(int)};
      d_cand_count.Device2Host(&h_count_buf);
      int keep = std::max(0, std::min(h_count, g.n_edges));
      if (keep > 0) {
        std::vector<uint32_t> tmp_src(static_cast<size_t>(g.n_edges), 0);
        std::vector<uint32_t> tmp_dst(static_cast<size_t>(g.n_edges), 0);
        BufferUint32 h_cand_src{tmp_src.data(),
                                sizeof(uint32_t) * static_cast<size_t>(g.n_edges)};
        BufferUint32 h_cand_dst{tmp_dst.data(),
                                sizeof(uint32_t) * static_cast<size_t>(g.n_edges)};
        d_cand_src.Device2Host(&h_cand_src);
        d_cand_dst.Device2Host(&h_cand_dst);
        cand_src_by_edge[static_cast<size_t>(pe)].assign(
            tmp_src.begin(), tmp_src.begin() + keep);
        cand_dst_by_edge[static_cast<size_t>(pe)].assign(
            tmp_dst.begin(), tmp_dst.begin() + keep);
      } else {
        cand_src_by_edge[static_cast<size_t>(pe)].clear();
        cand_dst_by_edge[static_cast<size_t>(pe)].clear();
      }
      std::cout << "[GARMatch][CUDA] pattern_edge=" << pe
                << " candidate_count=" << h_count << std::endl;
    }
  }



  if (out == nullptr) {
    return 1;
  }

  if (out->num_conditions) {
    *(out->num_conditions) = 1;
  }
  if (out->row_size) {
    *(out->row_size) = 0;
  }
  if (out->match_size) {
    *(out->match_size) = 0;
  }

  // Join phase (CUDA):
  // Initialize from first pattern-edge candidates, then iteratively expand
  // frontier by remaining pattern edges with endpoint consistency:
  // src==dst or dst==src across linked pattern vertices.
  if (p.n_nodes <= 0 || p.n_edges <= 0) {
    std::cout << "[GARMatch] empty pattern, skip join." << std::endl;
    std::cout << "GARMatchKernelWrapper::GARMatch END" << std::endl;
    return 0;
  }
  for (int pe = 0; pe < p.n_edges; ++pe) {
    if (cand_src_by_edge[static_cast<size_t>(pe)].empty()) {
      std::cout << "[GARMatch] pattern edge " << pe
                << " has no candidates, no match." << std::endl;
      std::cout << "GARMatchKernelWrapper::GARMatch END" << std::endl;
      return 0;
    }
  }

  auto local_to_instance = [&](uint32_t local_vid) -> uint32_t {
    if (g.v_id && local_vid < static_cast<uint32_t>(g.n_vertices)) {
      return g.v_id[local_vid];
    }
    return local_vid;
  };

  std::vector<std::vector<uint32_t>> embeddings;
  const int max_embeddings =
      std::max(1, out->match_capacity / std::max(1, p.n_nodes));
  const size_t emb_bytes = sizeof(int32_t) * static_cast<size_t>(max_embeddings) *
                           static_cast<size_t>(p.n_nodes);
  DeviceOwnedBufferInt32 d_frontier_a(emb_bytes);
  DeviceOwnedBufferInt32 d_frontier_b(emb_bytes);
  DeviceOwnedBufferInt d_curr_count(sizeof(int));
  DeviceOwnedBufferInt d_next_count(sizeof(int));
  dim3 dimBlock(kBlockDim);
  dim3 dimGrid(kGridDim);

  const int e0 = 0;
  const int32_t p0_u = p.edge_src[e0];
  const int32_t p0_v = p.edge_dst[e0];
  const auto& init_src = cand_src_by_edge[static_cast<size_t>(e0)];
  const auto& init_dst = cand_dst_by_edge[static_cast<size_t>(e0)];
  BufferUint32 h_init_src{const_cast<uint32_t*>(init_src.data()),
                          sizeof(uint32_t) * init_src.size()};
  BufferUint32 h_init_dst{const_cast<uint32_t*>(init_dst.data()),
                          sizeof(uint32_t) * init_dst.size()};
  DeviceOwnedBufferUint32 d_init_src;
  DeviceOwnedBufferUint32 d_init_dst;
  d_init_src.Init(h_init_src);
  d_init_dst.Init(h_init_dst);
  CUDA_CHECK(cudaMemset(d_curr_count.GetPtr(), 0, sizeof(int)));
  GARInitEmbeddingsKernel<<<dimGrid, dimBlock>>>(
      d_init_src.GetPtr(), d_init_dst.GetPtr(),
      static_cast<int>(std::min(init_src.size(), init_dst.size())), p0_u, p0_v,
      p.n_nodes, d_frontier_a.GetPtr(), d_curr_count.GetPtr(), max_embeddings);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int h_curr_count = 0;
  BufferInt h_curr_count_buf{&h_curr_count, sizeof(int)};
  d_curr_count.Device2Host(&h_curr_count_buf);
  h_curr_count = std::min(h_curr_count, max_embeddings);

  bool use_a_as_curr = true;
  for (int pe = 1; pe < p.n_edges && h_curr_count > 0; ++pe) {
    const int32_t pu = p.edge_src[pe];
    const int32_t pv = p.edge_dst[pe];
    if (pu < 0 || pu >= p.n_nodes || pv < 0 || pv >= p.n_nodes) {
      h_curr_count = 0;
      break;
    }
    const auto& c_src = cand_src_by_edge[static_cast<size_t>(pe)];
    const auto& c_dst = cand_dst_by_edge[static_cast<size_t>(pe)];
    const int n_cand = static_cast<int>(std::min(c_src.size(), c_dst.size()));
    if (n_cand <= 0) {
      h_curr_count = 0;
      break;
    }

    BufferUint32 h_c_src{const_cast<uint32_t*>(c_src.data()),
                         sizeof(uint32_t) * c_src.size()};
    BufferUint32 h_c_dst{const_cast<uint32_t*>(c_dst.data()),
                         sizeof(uint32_t) * c_dst.size()};
    DeviceOwnedBufferUint32 d_c_src;
    DeviceOwnedBufferUint32 d_c_dst;
    d_c_src.Init(h_c_src);
    d_c_dst.Init(h_c_dst);
    CUDA_CHECK(cudaMemset(d_next_count.GetPtr(), 0, sizeof(int)));

    int32_t* curr_ptr =
        use_a_as_curr ? d_frontier_a.GetPtr() : d_frontier_b.GetPtr();
    int32_t* next_ptr =
        use_a_as_curr ? d_frontier_b.GetPtr() : d_frontier_a.GetPtr();
    GARExpandEmbeddingsKernel<<<dimGrid, dimBlock>>>(
        curr_ptr, h_curr_count, p.n_nodes, pu, pv, d_c_src.GetPtr(),
        d_c_dst.GetPtr(), n_cand, next_ptr, d_next_count.GetPtr(),
        max_embeddings);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_next_count = 0;
    BufferInt h_next_count_buf{&h_next_count, sizeof(int)};
    d_next_count.Device2Host(&h_next_count_buf);
    h_curr_count = std::min(h_next_count, max_embeddings);
    use_a_as_curr = !use_a_as_curr;
  }

  if (h_curr_count > 0) {
    std::vector<int32_t> h_embeddings_raw(
        static_cast<size_t>(max_embeddings) * static_cast<size_t>(p.n_nodes), -1);
    BufferInt32 h_embeddings_buf{h_embeddings_raw.data(),
                                 sizeof(int32_t) * h_embeddings_raw.size()};
    DeviceOwnedBufferInt32& d_curr =
        use_a_as_curr ? d_frontier_a : d_frontier_b;
    d_curr.Device2Host(&h_embeddings_buf);
    embeddings.reserve(static_cast<size_t>(h_curr_count));
    for (int r = 0; r < h_curr_count; ++r) {
      bool complete = true;
      std::vector<uint32_t> emb(static_cast<size_t>(p.n_nodes), 0);
      for (int j = 0; j < p.n_nodes; ++j) {
        const int32_t v =
            h_embeddings_raw[static_cast<size_t>(r) * p.n_nodes + j];
        if (v < 0) {
          complete = false;
          break;
        }
        emb[static_cast<size_t>(j)] = static_cast<uint32_t>(v);
      }
      if (complete) embeddings.push_back(std::move(emb));
    }
  }

  if (embeddings.empty()) {
    std::cout << "[GARMatch] join found 0 complete matches." << std::endl;
    std::cout << "GARMatchKernelWrapper::GARMatch END" << std::endl;
    return 0;
  }

  // Flatten to GARMatchArrays.
  int row_size = 0;
  int match_size = 0;
  std::sort(embeddings.begin(), embeddings.end(),
            [](const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
              return a[0] < b[0];
            });
  size_t group_begin = 0;
  while (group_begin < embeddings.size()) {
    const uint32_t pivot_local = embeddings[group_begin][0];
    const uint32_t pivot_id = local_to_instance(pivot_local);
    size_t group_end = group_begin;
    while (group_end < embeddings.size() && embeddings[group_end][0] == pivot_local) {
      ++group_end;
    }
    for (int pos = 0; pos < p.n_nodes; ++pos) {
      std::vector<uint32_t> vals;
      vals.reserve(group_end - group_begin);
      for (size_t k = group_begin; k < group_end; ++k) {
        vals.push_back(local_to_instance(
            embeddings[k][static_cast<size_t>(pos)]));
      }
      std::sort(vals.begin(), vals.end());
      vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
      if (row_size >= out->row_capacity) break;
      const int offset = match_size;
      int write_n = static_cast<int>(vals.size());
      if (offset + write_n > out->match_capacity) {
        write_n = std::max(0, out->match_capacity - offset);
      }
      if (out->row_pivot_id) out->row_pivot_id[row_size] = pivot_id;
      if (out->row_cond_j) out->row_cond_j[row_size] = pos;
      if (out->row_pos) out->row_pos[row_size] = pos;
      if (out->row_offset) out->row_offset[row_size] = offset;
      if (out->row_count) out->row_count[row_size] = write_n;
      if (out->matched_v_ids) {
        for (int i = 0; i < write_n; ++i) {
          out->matched_v_ids[offset + i] = vals[static_cast<size_t>(i)];
        }
      }
      match_size += write_n;
      row_size += 1;
      if (row_size >= out->row_capacity || match_size >= out->match_capacity) break;
    }
    if (row_size >= out->row_capacity || match_size >= out->match_capacity) break;
    group_begin = group_end;
  }
  if (out->row_size) *(out->row_size) = row_size;
  if (out->match_size) *(out->match_size) = match_size;
  if (out->num_conditions) *(out->num_conditions) = p.n_nodes;
  std::cout << "[GARMatch] join complete, embeddings=" << embeddings.size()
            << ", rows=" << row_size << ", match_pool=" << match_size
            << std::endl;
  out->Print(5);
  std::cout << "GARMatchKernelWrapper::GARMatch END" << std::endl;
  return 0;
}

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
