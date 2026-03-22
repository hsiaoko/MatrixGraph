#include "core/task/gpu_task/kernel/kernel_gar_match.cuh"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <unordered_set>
#include <vector>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

using GARGraphArrays = sics::matrixgraph::core::data_structures::GARGraphArrays;
using GARPatternArrays =
    sics::matrixgraph::core::data_structures::GARPatternArrays;
using GARMatchArrays = sics::matrixgraph::core::data_structures::GARMatchArrays;

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

int GARMatchKernelWrapper::GARMatch(const GARGraphArrays& g,
                                    const GARPatternArrays& p,
                                    GARMatchArrays* out) {
  std::cout << "GARMatchKernelWrapper::GARMatch (placeholder, filters exposed)"
            << std::endl;
  auto print_u32 = [](const char* name, const uint32_t* arr, int n) {
    std::cout << name << " (n=" << n << "): [";
    for (int i = 0; i < n; ++i) {
      if (i) std::cout << ", ";
      std::cout << arr[i];
    }
    std::cout << "]" << std::endl;
  };
  auto print_i32 = [](const char* name, const int32_t* arr, int n) {
    std::cout << name << " (n=" << n << "): [";
    for (int i = 0; i < n; ++i) {
      if (i) std::cout << ", ";
      std::cout << arr[i];
    }
    std::cout << "]" << std::endl;
  };

  std::cout << "[GARMatch] ---- Graph Arrays (g) ----" << std::endl;
  std::cout << "g.n_vertices: " << g.n_vertices << std::endl;
  std::cout << "g.n_edges: " << g.n_edges << std::endl;
  if (g.v_id && g.n_vertices > 0) print_u32("g.v_id", g.v_id, g.n_vertices);
  if (g.v_label_idx && g.n_vertices > 0)
    print_i32("g.v_label_idx", g.v_label_idx, g.n_vertices);
  if (g.e_src && g.n_edges > 0) print_u32("g.e_src", g.e_src, g.n_edges);
  if (g.e_dst && g.n_edges > 0) print_u32("g.e_dst", g.e_dst, g.n_edges);
  if (g.e_id && g.n_edges > 0) print_u32("g.e_id", g.e_id, g.n_edges);
  if (g.e_label_idx && g.n_edges > 0)
    print_i32("g.e_label_idx", g.e_label_idx, g.n_edges);

  std::cout << "[GARMatch] ---- Pattern Arrays (p) ----" << std::endl;
  std::cout << "p.n_nodes: " << p.n_nodes << std::endl;
  std::cout << "p.n_edges: " << p.n_edges << std::endl;
  if (p.node_label_idx && p.n_nodes > 0)
    print_i32("p.node_label_idx", p.node_label_idx, p.n_nodes);
  if (p.edge_src && p.n_edges > 0) print_i32("p.edge_src", p.edge_src, p.n_edges);
  if (p.edge_dst && p.n_edges > 0) print_i32("p.edge_dst", p.edge_dst, p.n_edges);
  if (p.edge_label_idx && p.n_edges > 0)
    print_i32("p.edge_label_idx", p.edge_label_idx, p.n_edges);

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
