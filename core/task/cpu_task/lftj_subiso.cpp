#include "core/task/cpu_task/lftj_subiso.cuh"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stack>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using VertexID = sics::matrixgraph::core::common::VertexID;
using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;

void LFTJSubIso::Run() {
  auto t0 = std::chrono::high_resolution_clock::now();
  LoadData();
  auto t1 = std::chrono::high_resolution_clock::now();

  BuildUndirectedAdjacency();
  BuildMinWiseFilterCaches();
  BuildCandidateSets();
  if (!reject_output_path_.empty()) {
    WriteRejectedPairs();
  }
  ComputeMatchingOrder();

  materialize_ = !output_path_.empty();

  auto t2 = std::chrono::high_resolution_clock::now();
  match_count_ = Enumerate();
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

  std::cout << "[LFTJSubIso] Load+parse time: " << load_ms << " ms" << std::endl;
  std::cout << "[LFTJSubIso] Plan+filter time: " << plan_ms << " ms" << std::endl;
  std::cout << "[LFTJSubIso] Enumeration time: " << enum_ms << " ms" << std::endl;
  std::cout << "[LFTJSubIso] Total matches: " << match_count_ << std::endl;

  std::cout << "=== Filter Counts ===" << std::endl;
  std::cout << "Label Filters:      " << label_filtered_count_ << std::endl;
  std::cout << "Degree Filters:     " << degree_filtered_count_ << std::endl;
  std::cout << "LDF Filters:        " << ldf_filtered_count_ << std::endl;
  std::cout << "NLC Filters:        " << nlc_filtered_count_ << std::endl;
  std::cout << "Min-Wise Filters:   " << min_wise_filtered_count_ << std::endl;
  std::cout << "Intersection Prune: " << intersection_pruned_count_ << std::endl;

  if (materialize_ && match_count_ > 0) {
    VertexID pn = pattern_.get_num_vertices();
    std::ofstream out(output_path_, std::ios::binary);
    uint64_t n_matches = materialized_matches_.size() / pn;
    out.write(reinterpret_cast<const char*>(&pn), sizeof(pn));
    out.write(reinterpret_cast<const char*>(&n_matches), sizeof(n_matches));
    out.write(reinterpret_cast<const char*>(materialized_matches_.data()),
              materialized_matches_.size() * sizeof(VertexID));
    out.close();
    std::cout << "[LFTJSubIso] Materialized " << n_matches
              << " matches to: " << output_path_ << std::endl;
  }
}

void LFTJSubIso::LoadData() {
  pattern_.Read(pattern_path_);
  data_graph_.Read(data_graph_path_);
}

void LFTJSubIso::BuildUndirectedAdjacency() {
  // Pattern graph: vector-of-vectors.
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

  // Data graph: CSR.
  VertexID dn = data_graph_.get_num_vertices();
  data_offsets_.assign(dn + 1, 0);

  // First pass: collect and deduplicate neighbors per vertex, count degrees.
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

void LFTJSubIso::BuildMinWiseFilterCaches() {
  p_min_wise_cache_.clear();
  g_min_wise_cache_.clear();
  // The cache is needed for both min-wise filter and NLC filter.
  if (!enable_min_wise_filter_ && !enable_nlc_filter_) return;

  BuildMinWiseFilterCache(pattern_, p_min_wise_cache_, filter_hop_, filter_k_);
  BuildMinWiseFilterCache(data_graph_, g_min_wise_cache_, filter_hop_,
                          filter_k_);
}

void LFTJSubIso::BuildCandidateSets() {
  VertexID pn = pattern_.get_num_vertices();
  VertexID dn = data_graph_.get_num_vertices();
  const VertexLabel* plabels = pattern_.GetVLabelBasePointer();
  const VertexLabel* dlabels = data_graph_.GetVLabelBasePointer();

  candidates_.assign(pn, {});
  label_filtered_count_ = 0;
  degree_filtered_count_ = 0;
  ldf_filtered_count_ = 0;
  nlc_filtered_count_ = 0;
  min_wise_filtered_count_ = 0;
  for (VertexID u = 0; u < pn; ++u) {
    VertexLabel u_label = plabels[u];
    VertexID u_deg = pattern_adj_[u].size();
    VertexID u_out_deg = pattern_.GetOutDegreeByLocalID(u);
    VertexID u_in_deg = pattern_.GetInDegreeByLocalID(u);
    for (VertexID v = 0; v < dn; ++v) {
      if (dlabels[v] != u_label) {
        ++label_filtered_count_;
        continue;
      }
      VertexID v_deg = data_offsets_[v + 1] - data_offsets_[v];
      if (v_deg < u_deg) {
        ++degree_filtered_count_;
        continue;
      }
      if (enable_ldf_filter_ &&
          (data_graph_.GetOutDegreeByLocalID(v) < u_out_deg ||
           data_graph_.GetInDegreeByLocalID(v) < u_in_deg)) {
        ++ldf_filtered_count_;
        continue;
      }
      if (enable_nlc_filter_ &&
          g_min_wise_cache_[v].all_neighbor_label_count <
              p_min_wise_cache_[u].all_neighbor_label_count) {
        ++nlc_filtered_count_;
        continue;
      }
      if (enable_min_wise_filter_ &&
          !KMinWiseIPFilter(u, v, pattern_, data_graph_, p_min_wise_cache_,
                            g_min_wise_cache_)) {
        ++min_wise_filtered_count_;
        continue;
      }
      candidates_[u].push_back(v);
    }
    // Sort by vertex ID for correct binary-search intersection.
    std::sort(candidates_[u].begin(), candidates_[u].end());
  }
}

void LFTJSubIso::WriteRejectedPairs() {
  VertexID pn = pattern_.get_num_vertices();
  VertexID dn = data_graph_.get_num_vertices();
  const VertexLabel* plabels = pattern_.GetVLabelBasePointer();
  const VertexLabel* dlabels = data_graph_.GetVLabelBasePointer();

  // Group data vertices by label (sorted by construction since we iterate in
  // ascending vertex-id order).
  std::unordered_map<VertexLabel, std::vector<VertexID>> label_to_vertices;
  label_to_vertices.reserve(pn * 2 + 1);
  for (VertexID v = 0; v < dn; ++v) {
    label_to_vertices[dlabels[v]].push_back(v);
  }

  std::ofstream out(reject_output_path_);
  if (!out) {
    std::cerr << "[LFTJSubIso] Failed to open reject output: "
              << reject_output_path_ << std::endl;
    return;
  }
  out << "u,v\n";

  uint64_t rejected_count = 0;
  for (VertexID u = 0; u < pn; ++u) {
    VertexLabel u_label = plabels[u];
    auto it = label_to_vertices.find(u_label);
    if (it == label_to_vertices.end()) continue;

    const std::vector<VertexID>& same_label_vertices = it->second;
    const std::vector<VertexID>& cand = candidates_[u];
    size_t ci = 0;
    for (VertexID v : same_label_vertices) {
      // Advance candidate pointer past any values smaller than v.
      while (ci < cand.size() && cand[ci] < v) ++ci;
      if (ci < cand.size() && cand[ci] == v) continue;  // survived filter
      out << u << ',' << v << '\n';
      ++rejected_count;
    }
  }
  out.close();
  std::cout << "[LFTJSubIso] Wrote " << rejected_count
            << " rejected (u,v) pairs to: " << reject_output_path_ << std::endl;
}

void LFTJSubIso::ComputeMatchingOrder() {
  VertexID pn = pattern_.get_num_vertices();
  order_.clear();
  order_.reserve(pn);

  if (disable_matching_order_) {
    // Natural order: 0, 1, 2, ... (expected to degrade performance).
    for (VertexID u = 0; u < pn; ++u) {
      order_.push_back(u);
    }
  } else {
    std::vector<bool> selected(pn, false);

    // Start with the pattern vertex having the smallest candidate set.
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

    // Greedily pick the vertex with the most edges to already-selected vertices.
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

  // Build backward-neighbor lists and order_pos (shared).
  order_pos_.assign(pn, 0);
  for (VertexID i = 0; i < pn; ++i) order_pos_[order_[i]] = i;

  bn_list_.assign(pn, {});
  for (VertexID d = 0; d < pn; ++d) {
    VertexID u = order_[d];
    for (VertexID prev_d = 0; prev_d < d; ++prev_d) {
      VertexID prev_u = order_[prev_d];
      if (std::binary_search(pattern_adj_[u].begin(), pattern_adj_[u].end(),
                             prev_u)) {
        bn_list_[d].push_back(prev_d);  // store depth of backward neighbor
      }
    }
  }
}

void LFTJSubIso::ComputeLocalCandidates(uint32_t depth,
                                        const VertexID* embedding,
                                        std::vector<VertexID>& out) {
  VertexID u = order_[depth];
  out.clear();

  if (bn_list_[depth].empty()) {
    // First vertex: just use label/degree-filtered candidates.
    out = candidates_[u];
    return;
  }

  // Collect constraint lists: the global candidate set plus the adjacency lists
  // of all mapped backward neighbors. Sort them by size so we iterate over the
  // smallest list and binary-search in the larger ones.
  std::vector<const VertexID*> lists;
  std::vector<size_t> sizes;
  std::vector<size_t> list_order;

  size_t base_candidate_size = candidates_[u].size();
  lists.push_back(candidates_[u].data());
  sizes.push_back(candidates_[u].size());

  for (VertexID bn_depth : bn_list_[depth]) {
    VertexID mapped_v = embedding[bn_depth];
    const VertexID* nbrs = data_neighbors_.data() + data_offsets_[mapped_v];
    size_t deg = data_offsets_[mapped_v + 1] - data_offsets_[mapped_v];
    if (deg == 0) {
      intersection_pruned_count_.fetch_add(base_candidate_size);
      out.clear();
      return;
    }
    lists.push_back(nbrs);
    sizes.push_back(deg);
  }

  list_order.resize(lists.size());
  std::iota(list_order.begin(), list_order.end(), 0);
  std::sort(list_order.begin(), list_order.end(),
            [&sizes](size_t a, size_t b) { return sizes[a] < sizes[b]; });

  size_t base_idx = list_order[0];
  out.assign(lists[base_idx], lists[base_idx] + sizes[base_idx]);

  std::vector<VertexID> tmp;
  for (size_t i = 1; i < list_order.size() && !out.empty(); ++i) {
    size_t idx = list_order[i];
    Intersect(out.data(), out.size(), lists[idx], sizes[idx], tmp);
    out.swap(tmp);
  }

  // The result is always a subset of candidates_[u], so the reduction in size
  // is the number of candidates pruned by backward-neighbor constraints.
  if (base_candidate_size > out.size()) {
    intersection_pruned_count_.fetch_add(base_candidate_size - out.size());
  }
}

uint64_t LFTJSubIso::EnumerateRange(size_t cand_start, size_t cand_end,
                                    uint64_t thread_limit,
                                    std::vector<VertexID>* thread_matches) {
  VertexID pn = pattern_.get_num_vertices();
  VertexID dn = data_graph_.get_num_vertices();

  // Per-thread DFS state.
  std::vector<VertexID> embedding(pn, std::numeric_limits<VertexID>::max());
  std::vector<uint8_t> visited(dn, 0);
  std::vector<std::vector<VertexID>> cand_buffer(pn);
  std::vector<VertexID> local;
  std::vector<uint32_t> idx(pn, 0);

  ComputeLocalCandidates(0, embedding.data(), local);
  cand_buffer[0].swap(local);
  idx[0] = cand_start;

  uint64_t count = 0;
  int32_t depth = 0;
  while (true) {
    if (depth < 0) break;

    if (idx[depth] >= cand_buffer[depth].size() ||
        (depth == 0 && idx[depth] >= cand_end)) {
      // Backtrack from this depth: unmark the parent vertex at depth-1.
      if (depth > 0) {
        visited[embedding[depth - 1]] = 0;
      }
      --depth;
      continue;
    }

    VertexID v = cand_buffer[depth][idx[depth]++];
    if (visited[v]) continue;

    // Canonical mode: enforce strictly increasing data vertices with depth.
    // This avoids automorphic duplicates for clique-like patterns and matches
    // RapidMatch's default unlabeled #Embeddings semantics.
    if (canonical_ && depth > 0 && v <= embedding[depth - 1]) continue;

    embedding[depth] = v;
    visited[v] = 1;

    if (depth == pn - 1) {
      ++count;
      if (thread_matches != nullptr) {
        thread_matches->insert(thread_matches->end(), embedding.begin(),
                               embedding.end());
      }
      if (count >= thread_limit) return count;
      visited[v] = 0;
      continue;
    }

    // Extend.
    ComputeLocalCandidates(depth + 1, embedding.data(), local);
    if (local.empty()) {
      visited[v] = 0;
      continue;
    }
    cand_buffer[depth + 1].swap(local);
    idx[depth + 1] = 0;
    ++depth;
  }

  return count;
}

uint64_t LFTJSubIso::Enumerate() {
  VertexID pn = pattern_.get_num_vertices();
  if (pn == 0) return 0;

  // Compute root candidates once (they are the same for every thread).
  std::vector<VertexID> root_cand;
  ComputeLocalCandidates(0, nullptr, root_cand);
  size_t root_size = root_cand.size();
  if (root_size == 0) return 0;

  int n_threads = num_threads_;
  if (n_threads <= 0) {
    n_threads = static_cast<int>(std::thread::hardware_concurrency());
  }
  n_threads = std::max(1, n_threads);
  n_threads = std::min(n_threads, (int)root_size);

  std::vector<uint64_t> thread_counts(n_threads, 0);
  std::vector<std::thread> workers;

  // Per-thread match buffers, only allocated when materialization is requested.
  std::vector<std::vector<VertexID>> thread_matches;
  if (materialize_) {
    thread_matches.resize(n_threads);
    for (auto& buf : thread_matches) {
      buf.reserve(static_cast<size_t>(root_size / n_threads + 1) * pn);
    }
  }

  // Split root candidates evenly among threads.
  size_t chunk = (root_size + n_threads - 1) / n_threads;
  for (int t = 0; t < n_threads; ++t) {
    size_t start = t * chunk;
    size_t end = std::min(start + chunk, root_size);
    if (start >= end) {
      thread_counts[t] = 0;
      continue;
    }
    workers.emplace_back([this, start, end, &thread_counts, t,
                          &thread_matches]() {
      thread_counts[t] = EnumerateRange(
          start, end, std::numeric_limits<uint64_t>::max(),
          materialize_ ? &thread_matches[t] : nullptr);
    });
  }

  for (auto& w : workers) w.join();

  // Merge per-thread match buffers in deterministic thread order.
  if (materialize_) {
    size_t total_rows = 0;
    for (const auto& buf : thread_matches) total_rows += buf.size() / pn;
    uint64_t keep_rows =
        std::min<uint64_t>(total_rows, output_limit_);
    materialized_matches_.clear();
    materialized_matches_.reserve(static_cast<size_t>(keep_rows) * pn);
    for (const auto& buf : thread_matches) {
      size_t buf_rows = buf.size() / pn;
      size_t need_rows = keep_rows > materialized_matches_.size() / pn
                             ? keep_rows - materialized_matches_.size() / pn
                             : 0;
      size_t copy_rows = std::min(buf_rows, need_rows);
      if (copy_rows == 0) break;
      materialized_matches_.insert(
          materialized_matches_.end(), buf.begin(),
          buf.begin() + copy_rows * pn);
    }
  }

  uint64_t total = 0;
  for (auto c : thread_counts) {
    total += c;
    if (total >= output_limit_) return output_limit_;
  }
  return total;
}

void LFTJSubIso::Intersect(const VertexID* a, size_t na, const VertexID* b,
                           size_t nb, std::vector<VertexID>& dst) {
  dst.clear();
  if (na == 0 || nb == 0) return;
  // Always iterate over the smaller list and binary-search in the larger.
  if (na > nb) {
    std::swap(a, b);
    std::swap(na, nb);
  }
  dst.reserve(na);
  for (size_t i = 0; i < na; ++i) {
    VertexID v = a[i];
    if (Contains(b, nb, v)) dst.push_back(v);
  }
}

bool LFTJSubIso::Contains(const VertexID* arr, size_t n, VertexID v) {
  return std::binary_search(arr, arr + n, v);
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
