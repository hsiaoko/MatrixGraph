#include "core/task/gpu_task/kernel/kernel_gar_match.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/host_buffer.cuh"
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

GARMatchKernelWrapper* GARMatchKernelWrapper::GetInstance() {
  if (ptr_ == nullptr) {
    ptr_ = new GARMatchKernelWrapper();
  }
  return ptr_;
}

static bool LabelDegreeFilter(const GARMatchArrays& m,
                              const GARPatternArrays& p,
                              const GARGraphArrays& g, int u_idx,
                              uint32_t v_id) {
  (void)m;
  if (u_idx < 0 || u_idx >= p.n_nodes) return false;
  int32_t u_label = p.node_label_idx[u_idx];
  int32_t v_label = -1;
  for (int i = 0; i < g.n_vertices; ++i) {
    if (g.v_id[i] == v_id) {
      v_label = g.v_label_idx[i];
      break;
    }
  }
  if (v_label < 0) return false;
  if (u_label != v_label) return false;
  int p_out = 0, p_in = 0, g_out = 0, g_in = 0;
  for (int e = 0; e < p.n_edges; ++e) {
    if (p.edge_src[e] == u_idx) ++p_out;
    if (p.edge_dst[e] == u_idx) ++p_in;
  }
  for (int e = 0; e < g.n_edges; ++e) {
    if (g.e_src[e] == v_id) ++g_out;
    if (g.e_dst[e] == v_id) ++g_in;
  }
  return g_out >= p_out && g_in >= p_in;
}

static bool NeighborLabelCounterFilter(const GARMatchArrays& m,
                                       const GARPatternArrays& p,
                                       const GARGraphArrays& g, int u_idx,
                                       uint32_t v_id) {
  (void)m;
  if (u_idx < 0 || u_idx >= p.n_nodes) return false;
  int32_t u_label = p.node_label_idx[u_idx];
  int32_t v_label = -1;
  for (int i = 0; i < g.n_vertices; ++i) {
    if (g.v_id[i] == v_id) {
      v_label = g.v_label_idx[i];
      break;
    }
  }
  if (v_label < 0 || u_label != v_label) return false;

  std::unordered_set<int32_t> p_labels;
  std::unordered_set<int32_t> g_labels;
  for (int e = 0; e < p.n_edges; ++e) {
    if (p.edge_src[e] == u_idx)
      p_labels.insert(p.node_label_idx[p.edge_dst[e]]);
    if (p.edge_dst[e] == u_idx)
      p_labels.insert(p.node_label_idx[p.edge_src[e]]);
  }
  for (int e = 0; e < g.n_edges; ++e) {
    if (g.e_src[e] == v_id) {
      int li = -1;
      for (int i = 0; i < g.n_vertices; ++i) {
        if (g.v_id[i] == g.e_dst[e]) {
          li = i;
          break;
        }
      }
      if (li >= 0) g_labels.insert(g.v_label_idx[li]);
    }
    if (g.e_dst[e] == v_id) {
      int li = -1;
      for (int i = 0; i < g.n_vertices; ++i) {
        if (g.v_id[i] == g.e_src[e]) {
          li = i;
          break;
        }
      }
      if (li >= 0) g_labels.insert(g.v_label_idx[li]);
    }
  }
  return g_labels.size() >= p_labels.size();
}

static bool KMinWiseIPFilter(const GARMatchArrays& m, const GARPatternArrays& p,
                             const GARGraphArrays& g, int u_idx,
                             uint32_t v_id) {
  (void)m;
  if (u_idx < 0 || u_idx >= p.n_nodes) return false;
  std::vector<uint32_t> pu;
  std::vector<uint32_t> gv;
  auto hasher = std::hash<int32_t>{};

  for (int e = 0; e < p.n_edges; ++e) {
    if (p.edge_dst[e] == u_idx) {
      int nbr = p.edge_src[e];
      pu.push_back(
          static_cast<uint32_t>((hasher(p.node_label_idx[nbr]) << 3) % 64));
    }
  }
  for (int e = 0; e < g.n_edges; ++e) {
    if (g.e_dst[e] == v_id) {
      int li = -1;
      for (int i = 0; i < g.n_vertices; ++i) {
        if (g.v_id[i] == g.e_src[e]) {
          li = i;
          break;
        }
      }
      if (li >= 0) {
        gv.push_back(
            static_cast<uint32_t>((hasher(g.v_label_idx[li]) << 3) % 64));
      }
    }
  }

  std::sort(pu.begin(), pu.end());
  std::sort(gv.begin(), gv.end());
  size_t i = 0, j = 0;
  std::vector<uint32_t> pu_only;
  std::vector<uint32_t> gv_only;
  while (i < pu.size() && j < gv.size()) {
    if (pu[i] == gv[j]) {
      ++i;
      ++j;
    } else if (pu[i] < gv[j]) {
      pu_only.push_back(pu[i++]);
    } else {
      gv_only.push_back(gv[j++]);
    }
  }
  while (i < pu.size()) pu_only.push_back(pu[i++]);
  while (j < gv.size()) gv_only.push_back(gv[j++]);
  if (pu_only.empty()) return true;
  if (gv_only.empty()) return false;
  uint32_t min_gv = gv_only.front();
  for (uint32_t x : pu_only) {
    if (x < min_gv) return false;
  }
  return true;
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
    int slot = atomicAdd(params.cand_count, 1);
    if (slot < params.cand_capacity) {
      params.cand_src[slot] = gs;
      params.cand_dst[slot] = gd;
    }
  }
}

int GARMatchKernelWrapper::GARMatch(const GARGraphArrays& g,
                                    const GARPatternArrays& p,
                                    GARMatchArrays* out) {
  std::cout << "GARMatchKernelWrapper::GARMatch (placeholder, filters exposed)"
            << std::endl;
  constexpr int kPrintTopN = 3;
  auto print_u32 = [](const char* name, const uint32_t* arr, int n,
                      int top_n = 3) {
    const int limit = std::min(n, std::max(0, top_n));
    std::cout << name << " (n=" << n << ", top=" << limit << "): [";
    for (int i = 0; i < limit; ++i) {
      if (i) std::cout << ", ";
      std::cout << arr[i];
    }
    if (limit < n) std::cout << ", ...";
    std::cout << "]" << std::endl;
  };
  auto print_i32 = [](const char* name, const int32_t* arr, int n,
                      int top_n = 3) {
    const int limit = std::min(n, std::max(0, top_n));
    std::cout << name << " (n=" << n << ", top=" << limit << "): [";
    for (int i = 0; i < limit; ++i) {
      if (i) std::cout << ", ";
      std::cout << arr[i];
    }
    if (limit < n) std::cout << ", ...";
    std::cout << "]" << std::endl;
  };

  std::cout << "[GARMatch] ---- Graph Arrays (g) ----" << std::endl;
  std::cout << "g.n_vertices: " << g.n_vertices << std::endl;
  std::cout << "g.n_edges: " << g.n_edges << std::endl;
  if (g.v_id && g.n_vertices > 0)
    print_u32("g.v_id", g.v_id, g.n_vertices, kPrintTopN);
  if (g.v_label_idx && g.n_vertices > 0)
    print_i32("g.v_label_idx", g.v_label_idx, g.n_vertices, kPrintTopN);
  if (g.e_src && g.n_edges > 0)
    print_u32("g.e_src", g.e_src, g.n_edges, kPrintTopN);
  if (g.e_dst && g.n_edges > 0)
    print_u32("g.e_dst", g.e_dst, g.n_edges, kPrintTopN);
  if (g.e_id && g.n_edges > 0)
    print_u32("g.e_id", g.e_id, g.n_edges, kPrintTopN);
  if (g.e_label_idx && g.n_edges > 0)
    print_i32("g.e_label_idx", g.e_label_idx, g.n_edges, kPrintTopN);

  std::cout << "[GARMatch] ---- Pattern Arrays (p) ----" << std::endl;
  std::cout << "p.n_nodes: " << p.n_nodes << std::endl;
  std::cout << "p.n_edges: " << p.n_edges << std::endl;
  if (p.node_label_idx && p.n_nodes > 0)
    print_i32("p.node_label_idx", p.node_label_idx, p.n_nodes, kPrintTopN);
  if (p.edge_src && p.n_edges > 0)
    print_i32("p.edge_src", p.edge_src, p.n_edges, kPrintTopN);
  if (p.edge_dst && p.n_edges > 0)
    print_i32("p.edge_dst", p.edge_dst, p.n_edges, kPrintTopN);
  if (p.edge_label_idx && p.n_edges > 0)
    print_i32("p.edge_label_idx", p.edge_label_idx, p.n_edges, kPrintTopN);

  // CUDA kernel stage 1: candidate filtering for each pattern edge.
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

    DeviceOwnedBufferUint32 d_g_v_id(h_g_v_id);
    DeviceOwnedBufferInt32 d_g_v_label_idx(h_g_v_label_idx);
    DeviceOwnedBufferUint32 d_g_e_src(h_g_e_src);
    DeviceOwnedBufferUint32 d_g_e_dst(h_g_e_dst);
    DeviceOwnedBufferInt32 d_g_e_label_idx(h_g_e_label_idx);

    const int threads = 256;
    const int blocks = std::max(1, std::min((g.n_edges + threads - 1) / threads, 1024));
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
          .cand_src = d_cand_src.GetPtr(),
          .cand_dst = d_cand_dst.GetPtr(),
          .cand_count = d_cand_count.GetPtr(),
          .cand_capacity = g.n_edges,
      };
      GARFilterEdgeCandidatesKernel<<<blocks, threads>>>(params);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      int h_count = 0;
      BufferInt h_count_buf{&h_count, sizeof(int)};
      d_cand_count.Device2Host(&h_count_buf);
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
  std::cout << "GARMatchKernelWrapper::GARMatch (placeholder, filters exposed) END"
            << std::endl;
  return 0;
}

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
