#include "core/task/gpu_task/lftj_subiso_gpu.cuh"

#include <array>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <unordered_set>

#include "core/task/gpu_task/kernel/kernel_lftj_subiso.cuh"
#include "core/util/cuda_check.cuh"
#include "core/util/cuda_device.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using VertexID = sics::matrixgraph::core::common::VertexID;
using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using LFTJSubIsoKernelWrapper =
    sics::matrixgraph::core::task::kernel::LFTJSubIsoKernelWrapper;

void LFTJSubIsoGpu::LoadData() {
  pattern_.Read(pattern_path_);
  data_graph_.Read(data_graph_path_);
}

void LFTJSubIsoGpu::BuildUndirectedAdjacency() {
  VertexID pn = pattern_.get_num_vertices();
  pattern_adj_.assign(pn, {});
  for (VertexID u = 0; u < pn; ++u) {
    std::unordered_set<VertexID> nbrs;
    VertexID out_deg = pattern_.GetOutDegreeByLocalID(u);
    const VertexID* out_edges = pattern_.GetOutgoingEdgesByLocalID(u);
    for (VertexID i = 0; i < out_deg; ++i) nbrs.insert(out_edges[i]);
    VertexID in_deg = pattern_.GetInDegreeByLocalID(u);
    const VertexID* in_edges = pattern_.GetIncomingEdgesByLocalID(u);
    for (VertexID i = 0; i < in_deg; ++i) nbrs.insert(in_edges[i]);
    pattern_adj_[u].assign(nbrs.begin(), nbrs.end());
    std::sort(pattern_adj_[u].begin(), pattern_adj_[u].end());
  }

  VertexID dn = data_graph_.get_num_vertices();
  data_offsets_.assign(dn + 1, 0);

  std::vector<std::vector<VertexID>> tmp(dn);
  for (VertexID v = 0; v < dn; ++v) {
    VertexID out_deg = data_graph_.GetOutDegreeByLocalID(v);
    const VertexID* out_edges = data_graph_.GetOutgoingEdgesByLocalID(v);
    for (VertexID i = 0; i < out_deg; ++i) tmp[v].push_back(out_edges[i]);
    VertexID in_deg = data_graph_.GetInDegreeByLocalID(v);
    const VertexID* in_edges = data_graph_.GetIncomingEdgesByLocalID(v);
    for (VertexID i = 0; i < in_deg; ++i) tmp[v].push_back(in_edges[i]);
    std::sort(tmp[v].begin(), tmp[v].end());
    tmp[v].erase(std::unique(tmp[v].begin(), tmp[v].end()), tmp[v].end());
    data_offsets_[v + 1] = data_offsets_[v] + tmp[v].size();
  }

  data_neighbors_.resize(data_offsets_[dn]);
  for (VertexID v = 0; v < dn; ++v) {
    std::memcpy(data_neighbors_.data() + data_offsets_[v], tmp[v].data(),
                tmp[v].size() * sizeof(VertexID));
  }
}

void LFTJSubIsoGpu::BuildMinWiseFilterCaches() {
  p_min_wise_cache_.clear();
  g_min_wise_cache_.clear();
  p_bloom_signature_.clear();
  g_bloom_signature_.clear();
  if (!enable_min_wise_filter_ && !enable_nlc_filter_ &&
      !enable_min_wise_bloom_filter_)
    return;

  BuildMinWiseFilterCache(pattern_, p_min_wise_cache_, filter_hop_, filter_k_);
  BuildMinWiseFilterCache(data_graph_, g_min_wise_cache_, filter_hop_,
                          filter_k_);

  if (enable_bloom_filter_) {
    auto n_p = pattern_.get_num_vertices();
    p_bloom_signature_.assign(n_p, 0);
    const VertexLabel* plabels = pattern_.GetVLabelBasePointer();
    for (VertexID u = 0; u < n_p; ++u) {
      uint64_t sig = 0;
      for (VertexID nbr : pattern_adj_[u]) {
        VertexLabel lbl = plabels[nbr];
        if (lbl < 64) sig |= (1ULL << lbl);
      }
      p_bloom_signature_[u] = sig;
    }

    auto n_g = data_graph_.get_num_vertices();
    g_bloom_signature_.assign(n_g, 0);
    const VertexLabel* glabels = data_graph_.GetVLabelBasePointer();
    for (VertexID v = 0; v < n_g; ++v) {
      uint64_t sig = 0;
      VertexID deg = data_offsets_[v + 1] - data_offsets_[v];
      const VertexID* nbrs = data_neighbors_.data() + data_offsets_[v];
      for (VertexID i = 0; i < deg; ++i) {
        VertexLabel lbl = glabels[nbrs[i]];
        if (lbl < 64) sig |= (1ULL << lbl);
      }
      g_bloom_signature_[v] = sig;
    }
  }
}

void LFTJSubIsoGpu::BuildCandidateSets() {
  // Read filter-order override from environment.
  filter_order_ = 0;
  if (const char* e = std::getenv("MG_LFTJ_FILTER_ORDER")) {
    std::string s(e);
    if (s == "minwise_nlc_lpf") {
      filter_order_ = 1;
    } else if (s == "lpf_nlc_minwise") {
      filter_order_ = 2;
    } else if (s == "nlc_lpf_minwise") {
      filter_order_ = 3;
    }
  }

  VertexID pn = pattern_.get_num_vertices();
  VertexID dn = data_graph_.get_num_vertices();
  const VertexLabel* plabels = pattern_.GetVLabelBasePointer();
  const VertexLabel* dlabels = data_graph_.GetVLabelBasePointer();

  candidates_.assign(pn, {});
  for (VertexID u = 0; u < pn; ++u) {
    VertexLabel u_label = plabels[u];
    VertexID u_deg = pattern_adj_[u].size();
    VertexID u_out_deg = pattern_.GetOutDegreeByLocalID(u);
    VertexID u_in_deg = pattern_.GetInDegreeByLocalID(u);
    uint64_t u_bloom_mask = enable_bloom_filter_ ? p_bloom_signature_[u] : 0;

    auto do_ldf = [&](VertexID v) {
      return !(enable_ldf_filter_ &&
               (data_graph_.GetOutDegreeByLocalID(v) < u_out_deg ||
                data_graph_.GetInDegreeByLocalID(v) < u_in_deg));
    };
    auto do_nlc = [&](VertexID v) {
      return !(enable_nlc_filter_ &&
               g_min_wise_cache_[v].all_neighbor_label_count <
                   p_min_wise_cache_[u].all_neighbor_label_count);
    };
    auto do_lpf = [&](VertexID v) {
      if (!enable_lpf_filter_) return true;
      const int kLpfLabelCap = 32;
      int u_freq[kLpfLabelCap] = {0};
      int v_freq[kLpfLabelCap] = {0};
      const VertexLabel* plabels = pattern_.GetVLabelBasePointer();
      const VertexLabel* glabels = data_graph_.GetVLabelBasePointer();
      for (VertexID nbr : pattern_adj_[u]) {
        VertexLabel lbl = plabels[nbr];
        if (lbl < kLpfLabelCap) ++u_freq[lbl];
      }
      VertexID v_deg = data_offsets_[v + 1] - data_offsets_[v];
      const VertexID* v_nbrs = data_neighbors_.data() + data_offsets_[v];
      for (VertexID i = 0; i < v_deg; ++i) {
        VertexLabel lbl = glabels[v_nbrs[i]];
        if (lbl < kLpfLabelCap) ++v_freq[lbl];
      }
      for (int lbl = 0; lbl < kLpfLabelCap; ++lbl) {
        if (v_freq[lbl] < u_freq[lbl]) return false;
      }
      return true;
    };
    auto do_bloom = [&](VertexID v) {
      return !(enable_bloom_filter_ &&
               (g_bloom_signature_[v] & u_bloom_mask) != u_bloom_mask);
    };
    auto do_minwise_bloom = [&](VertexID v) {
      return !(enable_min_wise_bloom_filter_ &&
               !KMinBloomIPFilter(u, v, pattern_, data_graph_, p_min_wise_cache_,
                                  g_min_wise_cache_));
    };
    auto do_minwise = [&](VertexID v) {
      return !(enable_min_wise_filter_ &&
               !KMinWiseIPFilter(u, v, pattern_, data_graph_, p_min_wise_cache_,
                                 g_min_wise_cache_));
    };

    auto apply_default = [&](VertexID v) {
      if (!do_ldf(v)) return false;
      if (!do_nlc(v)) return false;
      if (!do_lpf(v)) return false;
      if (!do_bloom(v)) return false;
      if (!do_minwise_bloom(v)) return false;
      if (!do_minwise(v)) return false;
      return true;
    };

    auto apply_minwise_nlc_lpf = [&](VertexID v) {
      if (!do_minwise(v)) return false;
      if (!do_nlc(v)) return false;
      if (!do_lpf(v)) return false;
      if (!do_bloom(v)) return false;
      if (!do_minwise_bloom(v)) return false;
      return true;
    };

    auto apply_lpf_nlc_minwise = [&](VertexID v) {
      if (!do_lpf(v)) return false;
      if (!do_nlc(v)) return false;
      if (!do_minwise(v)) return false;
      if (!do_bloom(v)) return false;
      if (!do_minwise_bloom(v)) return false;
      return true;
    };

    auto apply_nlc_lpf_minwise = [&](VertexID v) {
      if (!do_nlc(v)) return false;
      if (!do_lpf(v)) return false;
      if (!do_minwise(v)) return false;
      if (!do_bloom(v)) return false;
      if (!do_minwise_bloom(v)) return false;
      return true;
    };

    for (VertexID v = 0; v < dn; ++v) {
      if (dlabels[v] != u_label) continue;
      VertexID v_deg = data_offsets_[v + 1] - data_offsets_[v];
      if (v_deg < u_deg) continue;
      if (enable_ldf_filter_ &&
          (data_graph_.GetOutDegreeByLocalID(v) < u_out_deg ||
           data_graph_.GetInDegreeByLocalID(v) < u_in_deg))
        continue;

      bool ok;
      if (filter_order_ == 1) {
        ok = apply_minwise_nlc_lpf(v);
      } else if (filter_order_ == 2) {
        ok = apply_lpf_nlc_minwise(v);
      } else if (filter_order_ == 3) {
        ok = apply_nlc_lpf_minwise(v);
      } else {
        ok = apply_default(v);
      }
      if (!ok) continue;

      candidates_[u].push_back(v);
    }
    std::sort(candidates_[u].begin(), candidates_[u].end());
  }
}

void LFTJSubIsoGpu::ComputeMatchingOrder() {
  VertexID pn = pattern_.get_num_vertices();
  order_.clear();
  order_.reserve(pn);

  if (disable_matching_order_) {
    for (VertexID u = 0; u < pn; ++u) order_.push_back(u);
  } else {
    std::vector<bool> selected(pn, false);
    VertexID start = 0;
    size_t min_size = std::numeric_limits<size_t>::max();
    for (VertexID u = 0; u < pn; ++u) {
      if (candidates_[u].size() < min_size) {
        min_size = candidates_[u].size();
        start = u;
      }
    }
    order_.push_back(start);
    selected[start] = true;

    const VertexLabel* plabels = pattern_.GetVLabelBasePointer();
    while (order_.size() < pn) {
      int max_edges = -1;
      VertexID chosen = 0;
      size_t chosen_size = std::numeric_limits<size_t>::max();
      int chosen_label_freq = -1;
      for (VertexID u = 0; u < pn; ++u) {
        if (selected[u]) continue;
        int edges_to_selected = 0;
        int label_freq = 0;
        for (VertexID s : order_) {
          if (std::binary_search(pattern_adj_[u].begin(), pattern_adj_[u].end(),
                                 s)) {
            ++edges_to_selected;
          }
          if (plabels[s] == plabels[u]) ++label_freq;
        }
        size_t csize = candidates_[u].size();
        if (edges_to_selected > max_edges ||
            (edges_to_selected == max_edges && csize < chosen_size) ||
            (edges_to_selected == max_edges && csize == chosen_size &&
             label_freq > chosen_label_freq) ||
            (edges_to_selected == max_edges && csize == chosen_size &&
             label_freq == chosen_label_freq &&
             pattern_adj_[u].size() > pattern_adj_[chosen].size())) {
          max_edges = edges_to_selected;
          chosen = u;
          chosen_size = csize;
          chosen_label_freq = label_freq;
        }
      }
      order_.push_back(chosen);
      selected[chosen] = true;
    }
  }

  bn_list_.assign(pn, {});
  for (VertexID d = 0; d < pn; ++d) {
    VertexID u = order_[d];
    for (VertexID prev_d = 0; prev_d < d; ++prev_d) {
      VertexID prev_u = order_[prev_d];
      if (std::binary_search(pattern_adj_[u].begin(), pattern_adj_[u].end(),
                             prev_u)) {
        bn_list_[d].push_back(prev_d);
      }
    }
  }
}

uint64_t LFTJSubIsoGpu::EnumerateOnHost() const {
  VertexID pn = pattern_.get_num_vertices();
  VertexID dn = data_graph_.get_num_vertices();
  if (pn == 0) return 0;

  // Lambda: compute local candidates at depth d given embedding.
  auto compute_local = [&](uint32_t depth, const std::vector<VertexID>& emb,
                           std::vector<VertexID>& out) {
    VertexID u = order_[depth];
    out.clear();
    if (bn_list_[depth].empty()) {
      out = candidates_[u];
      return;
    }
    std::vector<const VertexID*> lists;
    std::vector<size_t> sizes;
    lists.push_back(candidates_[u].data());
    sizes.push_back(candidates_[u].size());
    for (VertexID bn_depth : bn_list_[depth]) {
      VertexID mapped_v = emb[bn_depth];
      const VertexID* nbrs = data_neighbors_.data() + data_offsets_[mapped_v];
      size_t deg = data_offsets_[mapped_v + 1] - data_offsets_[mapped_v];
      lists.push_back(nbrs);
      sizes.push_back(deg);
    }
    std::vector<size_t> list_order(lists.size());
    std::iota(list_order.begin(), list_order.end(), 0);
    std::sort(list_order.begin(), list_order.end(),
              [&sizes](size_t a, size_t b) { return sizes[a] < sizes[b]; });

    size_t base_idx = list_order[0];
    out.assign(lists[base_idx], lists[base_idx] + sizes[base_idx]);
    std::vector<VertexID> tmp;
    for (size_t i = 1; i < list_order.size() && !out.empty(); ++i) {
      size_t idx = list_order[i];
      tmp.clear();
      const VertexID* a = out.data();
      size_t na = out.size();
      const VertexID* b = lists[idx];
      size_t nb = sizes[idx];
      if (na > nb) {
        std::swap(a, b);
        std::swap(na, nb);
      }
      for (size_t j = 0; j < na; ++j) {
        VertexID v = a[j];
        if (std::binary_search(b, b + nb, v)) tmp.push_back(v);
      }
      out.swap(tmp);
    }
  };

  std::vector<VertexID> root;
  std::vector<VertexID> emb(pn);
  compute_local(0, emb, root);
  uint64_t total = 0;

  // Simple single-thread DFS for host reference.
  std::vector<std::vector<VertexID>> cand_stack(pn);
  std::vector<uint32_t> idx(pn, 0);
  std::vector<bool> visited(dn, false);
  cand_stack[0].swap(root);
  int32_t depth = 0;
  while (true) {
    if (depth < 0) break;
    if (idx[depth] >= cand_stack[depth].size()) {
      if (depth > 0) visited[emb[depth - 1]] = false;
      --depth;
      continue;
    }
    VertexID v = cand_stack[depth][idx[depth]++];
    if (visited[v]) continue;
    if (canonical_ && depth > 0 && v <= emb[depth - 1]) continue;
    emb[depth] = v;
    visited[v] = true;
    if (depth == pn - 1) {
      ++total;
      visited[v] = false;
      continue;
    }
    compute_local(depth + 1, emb, cand_stack[depth + 1]);
    if (cand_stack[depth + 1].empty()) {
      visited[v] = false;
      continue;
    }
    idx[depth + 1] = 0;
    ++depth;
  }
  return total;
}

void LFTJSubIsoGpu::Run() {
  auto t0 = std::chrono::high_resolution_clock::now();
  LoadData();
  auto t1 = std::chrono::high_resolution_clock::now();

  BuildUndirectedAdjacency();
  BuildMinWiseFilterCaches();

  // LCF: global label-count pre-check.
  if (enable_lcf_filter_) {
    const int kLcfLabelCap = 32;
    std::array<int, kLcfLabelCap> p_freq = {};
    std::array<int, kLcfLabelCap> g_freq = {};
    const VertexLabel* plabels = pattern_.GetVLabelBasePointer();
    const VertexLabel* glabels = data_graph_.GetVLabelBasePointer();
    for (VertexID u = 0; u < pattern_.get_num_vertices(); ++u) {
      VertexLabel lbl = plabels[u];
      if (lbl < kLcfLabelCap) ++p_freq[lbl];
    }
    for (VertexID v = 0; v < data_graph_.get_num_vertices(); ++v) {
      VertexLabel lbl = glabels[v];
      if (lbl < kLcfLabelCap) ++g_freq[lbl];
    }
    for (int lbl = 0; lbl < kLcfLabelCap; ++lbl) {
      if (g_freq[lbl] < p_freq[lbl]) {
        std::cout << "[LFTJSubIsoGpu] LCF rejected globally (label " << lbl
                  << " count: pattern=" << p_freq[lbl] << " data="
                  << g_freq[lbl] << ")." << std::endl;
        std::cout << "[LFTJSubIsoGpu] Total matches: 0" << std::endl;
        return;
      }
    }
  }

  BuildCandidateSets();
  ComputeMatchingOrder();

  auto t2 = std::chrono::high_resolution_clock::now();

  CUDA_CHECK(cudaSetDevice(
      sics::matrixgraph::core::util::MatrixGraphCudaDevice()));

  uint64_t match_count = LFTJSubIsoKernelWrapper::Enumerate(
      pattern_, data_graph_, data_offsets_, data_neighbors_, pattern_adj_,
      candidates_, order_, bn_list_, canonical_, num_threads_);

  CUDA_CHECK(cudaDeviceSynchronize());

  auto t3 = std::chrono::high_resolution_clock::now();

  double load_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() /
      1000.0;
  double plan_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() /
      1000.0;
  double enum_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() /
      1000.0;

  std::cout << "[LFTJSubIsoGpu] Load+parse time: " << load_ms << " ms"
            << std::endl;
  std::cout << "[LFTJSubIsoGpu] Plan+filter time: " << plan_ms << " ms"
            << std::endl;
  std::cout << "[LFTJSubIsoGpu] Enumeration time: " << enum_ms << " ms"
            << std::endl;
  std::cout << "[LFTJSubIsoGpu] Total matches: " << match_count << std::endl;
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
