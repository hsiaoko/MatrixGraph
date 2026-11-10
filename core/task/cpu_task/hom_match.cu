#include "core/task/cpu_task/hom_match.cuh"

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace cpu_task {

namespace {

// Slot of the same data edges walked in the opposite direction (materialized
// inverse convention: base 0..N-1, inverse N..2N-1).
std::uint32_t FlipSlot(std::uint32_t slot, int num_relations) {
  return slot < static_cast<std::uint32_t>(num_relations)
             ? slot + num_relations
             : slot - num_relations;
}

// nullptr when pattern vertex v is unpinned.
const std::uint32_t* PinOf(
    const std::unordered_map<std::uint32_t, std::uint32_t>& pins,
    std::uint32_t v) {
  const auto it = pins.find(v);
  return it == pins.end() ? nullptr : &it->second;
}

}  // namespace

HomMatchIndex::HomMatchIndex(const GARGraphArrays& g, int num_relations)
    : num_relations_(num_relations), num_slots_(2 * num_relations) {
  pairs_.resize(static_cast<std::size_t>(num_slots_));
  sets_.resize(static_cast<std::size_t>(num_slots_));
  // Base labels only are expected (the GPU-side input folds inverse slots);
  // each edge is materialized into its slot and the mirrored inverse slot.
  for (int e = 0; e < g.n_edges; ++e) {
    const int label = g.e_label_idx[e];
    if (label < 0 || label >= num_relations_) continue;
    const std::uint32_t s = g.e_src[e];
    const std::uint32_t d = g.e_dst[e];
    const std::uint64_t fwd = PackPair(s, d);
    pairs_[static_cast<std::size_t>(label)].push_back(fwd);
    sets_[static_cast<std::size_t>(label)].insert(fwd);
    const std::uint64_t rev = PackPair(d, s);
    pairs_[static_cast<std::size_t>(label + num_relations_)].push_back(rev);
    sets_[static_cast<std::size_t>(label + num_relations_)].insert(rev);
  }
  for (auto& arr : pairs_) std::sort(arr.begin(), arr.end());
}

std::uint64_t HomMatcher::Enumerate(
    const HomMatchIndex& idx, const GARPatternArrays& p,
    std::uint64_t match_cap, std::vector<std::uint32_t>* out, bool* capped,
    const std::unordered_map<std::uint32_t, std::uint32_t>& pins) const {
  *capped = false;
  const int n = p.n_nodes;
  const int m = p.n_edges;
  if (n <= 1 || m == 0) return 0;
  const int R = idx.num_relations();

  // Undirected adjacency over pattern edges.
  struct Adj {
    std::uint32_t edge;
    std::uint32_t other;
    bool is_out;
    std::uint32_t slot;
  };
  std::vector<std::vector<Adj>> adj(static_cast<std::size_t>(n));
  for (int e = 0; e < m; ++e) {
    const std::uint32_t s = static_cast<std::uint32_t>(p.edge_src[e]);
    const std::uint32_t d = static_cast<std::uint32_t>(p.edge_dst[e]);
    const std::uint32_t slot = static_cast<std::uint32_t>(p.edge_label_idx[e]);
    adj[s].push_back({static_cast<std::uint32_t>(e), d, true, slot});
    adj[d].push_back({static_cast<std::uint32_t>(e), s, false, slot});
  }

  // Greedy matching order (seed 0, then most edges back into the bound set).
  std::vector<std::uint32_t> order;
  order.reserve(static_cast<std::size_t>(n));
  {
    std::vector<char> bound(static_cast<std::size_t>(n), 0);
    bound[0] = 1;
    order.push_back(0);
    while (order.size() < static_cast<std::size_t>(n)) {
      std::uint32_t best = static_cast<std::uint32_t>(n);
      std::uint32_t best_back = 0;
      for (int v = 0; v < n; ++v) {
        if (bound[v]) continue;
        std::uint32_t back = 0;
        for (const auto& a : adj[v]) back += bound[a.other];
        if (back == 0) continue;
        if (back > best_back ||
            (back == best_back && static_cast<std::uint32_t>(v) < best)) {
          best = v;
          best_back = back;
        }
      }
      if (best == static_cast<std::uint32_t>(n)) {
        for (int v = 0; v < n; ++v) {
          if (!bound[v]) {
            best = v;
            break;
          }
        }
      }
      bound[best] = 1;
      order.push_back(best);
    }
  }

  // Per ordered vertex (past the seed), the edges to earlier vertices.
  std::vector<std::vector<Adj>> back_edges(static_cast<std::size_t>(n));
  {
    std::vector<char> earlier(static_cast<std::size_t>(n), 0);
    earlier[order[0]] = 1;
    for (std::size_t i = 1; i < order.size(); ++i) {
      const std::uint32_t v = order[i];
      for (const auto& a : adj[v]) {
        if (earlier[a.other]) back_edges[i].push_back(a);
      }
      earlier[v] = 1;
    }
  }
  for (std::size_t i = 1; i < order.size(); ++i) {
    if (back_edges[i].empty()) return 0;  // disconnected remainder unsupported
  }

  auto has_edge = [&](std::uint32_t src, std::uint32_t dst,
                      std::uint32_t slot) {
    return idx.RelationPairSet(static_cast<int>(slot)).count(PackPair(src, dst)) != 0;
  };

  std::vector<std::uint32_t> emb(static_cast<std::size_t>(n), 0);
  std::uint64_t count = 0;

  auto candidates = [&](const Adj& driver, std::vector<std::uint32_t>* out_c) {
    out_c->clear();
    const std::uint32_t slot = driver.is_out
                                   ? FlipSlot(driver.slot, R)
                                   : driver.slot;
    const std::uint32_t anchor = emb[driver.other];
    const auto& pairs = idx.RelationPairs(static_cast<int>(slot));
    const std::uint64_t key = PackPair(anchor, 0);
    for (auto it = std::lower_bound(pairs.begin(), pairs.end(), key);
         it != pairs.end() && (*it >> 32) == anchor; ++it) {
      out_c->push_back(static_cast<std::uint32_t>(*it));
    }
  };

  auto check_back_edges = [&](std::size_t pos) {
    const std::uint32_t v = order[pos];
    for (std::size_t i = 1; i < back_edges[pos].size(); ++i) {
      const auto& a = back_edges[pos][i];
      const std::uint32_t src = a.is_out ? emb[v] : emb[a.other];
      const std::uint32_t dst = a.is_out ? emb[a.other] : emb[v];
      if (!has_edge(src, dst, a.slot)) return false;
    }
    return true;
  };

  // Seed edge binds order[0] and order[1] together.
  const Adj seed_adj = back_edges[1][0];
  const std::uint32_t seed_slot =
      seed_adj.is_out ? FlipSlot(seed_adj.slot, R) : seed_adj.slot;
  const auto& seed_pairs = idx.RelationPairs(static_cast<int>(seed_slot));

  const std::uint32_t* pin0 = PinOf(pins, order[0]);
  const std::uint32_t* pin1 = PinOf(pins, order[1]);
  const std::vector<std::uint64_t>* scan = &seed_pairs;
  bool swapped = false;
  if (pin0 == nullptr && pin1 != nullptr) {
    scan = &idx.RelationPairs(static_cast<int>(FlipSlot(seed_slot, R)));
    swapped = true;
  }
  std::size_t scan_lo = 0, scan_hi = scan->size();
  if (const std::uint32_t* pp = swapped ? pin1 : pin0) {
    const auto& arr = *scan;
    scan_lo = static_cast<std::size_t>(
        std::lower_bound(arr.begin(), arr.end(), PackPair(*pp, 0)) -
        arr.begin());
    scan_hi = static_cast<std::size_t>(
        std::lower_bound(arr.begin(), arr.end(), PackPair(*pp + 1, 0)) -
        arr.begin());
  }

  std::vector<std::vector<std::uint32_t>> cands(static_cast<std::size_t>(n));
  std::vector<std::size_t> cand_pos(static_cast<std::size_t>(n), 0);

  for (std::size_t si = scan_lo; si < scan_hi; ++si) {
    const std::uint64_t sp = (*scan)[si];
    const auto hi = static_cast<std::uint32_t>(sp >> 32);
    const auto lo = static_cast<std::uint32_t>(sp);
    const auto ps = swapped ? lo : hi;
    const auto pd = swapped ? hi : lo;
    if (pin0 != nullptr && ps != *pin0) continue;
    if (pin1 != nullptr && pd != *pin1) continue;
    emb[order[0]] = ps;
    emb[order[1]] = pd;
    bool ok = true;
    for (const auto& a : adj[order[1]]) {
      if (a.other == order[0] && a.edge != seed_adj.edge) {
        const std::uint32_t src = a.is_out ? emb[order[1]] : emb[order[0]];
        const std::uint32_t dst = a.is_out ? emb[order[0]] : emb[order[1]];
        if (!has_edge(src, dst, a.slot)) {
          ok = false;
          break;
        }
      }
    }
    if (!ok) continue;

    std::size_t pos = 2;
    while (pos >= 2) {
      if (pos == order.size()) {
        out->insert(out->end(), emb.begin(), emb.end());
        if (++count >= match_cap) {
          *capped = true;
          return count;
        }
        --pos;
        continue;
      }
      if (cand_pos[pos] == 0) {
        if (const std::uint32_t* pp = PinOf(pins, order[pos])) {
          const Adj& driver = back_edges[pos][0];
          const std::uint32_t src = driver.is_out ? *pp : emb[driver.other];
          const std::uint32_t dst = driver.is_out ? emb[driver.other] : *pp;
          cands[pos].clear();
          if (has_edge(src, dst, driver.slot)) cands[pos].push_back(*pp);
        } else {
          candidates(back_edges[pos][0], &cands[pos]);
        }
      }
      bool advanced = false;
      while (cand_pos[pos] < cands[pos].size()) {
        emb[order[pos]] = cands[pos][cand_pos[pos]++];
        if (check_back_edges(pos)) {
          advanced = true;
          break;
        }
      }
      if (advanced) {
        ++pos;
        if (pos < order.size()) cand_pos[pos] = 0;
      } else {
        cand_pos[pos] = 0;
        cands[pos].clear();
        --pos;
      }
    }
    if (count >= match_cap) {
      *capped = true;
      return count;
    }
  }
  return count;
}

}  // namespace cpu_task
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
