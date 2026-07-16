#ifndef MATRIXGRAPH_CORE_TASK_CPU_MIN_WISE_FILTER_H_
#define MATRIXGRAPH_CORE_TASK_CPU_MIN_WISE_FILTER_H_

#include <algorithm>
#include <atomic>
#include <cstring>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "core/common/consts.h"
#include "core/common/types.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/util/execution_policy.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using VertexID = sics::matrixgraph::core::common::VertexID;
using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;
using sics::matrixgraph::core::common::kDefaultHeapCapacity;
using sics::matrixgraph::core::common::kMaxVertexID;

// CPU-compatible 32-bit bitmap used by the min-wise filter.
class MinWiseBitmap {
 public:
  explicit MinWiseBitmap(unsigned size) { Init(size); }

  void Init(unsigned size) {
    size_ = size;
    data_ = 0;
  }

  void Clear() { data_ = 0; }

  void SetBit(unsigned i) {
    if (i > size_) return;
    data_ |= (1u << i);
  }

  unsigned Count() const {
    unsigned x = data_;
    x = (x & 0x55555555u) + ((x >> 1) & 0x5555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x3333333u);
    x = (x & 0x0f0f0f0fu) + ((x >> 4) & 0x0f0f0f0fu);
    x = (x & 0x00ff00ffu) + ((x >> 8) & 0x00ff00ffu);
    x = (x & 0x0000ffffu) + ((x >> 16) & 0x0000ffffu);
    return x;
  }

  unsigned GetData() const { return data_; }

 private:
  unsigned size_ = 0;
  unsigned data_ = 0;
};

// Label hash table used by the original GPU/CPU min-wise pipeline.
// Maps a label hash bucket index (0..15) to a permutation bucket.
static inline VertexID MinWiseHashTable(VertexID key) {
  switch (key) {
    case 0: return 12;
    case 1: return 3;
    case 2: return 11;
    case 3: return 9;
    case 4: return 15;
    case 5: return 2;
    case 6: return 8;
    case 7: return 4;
    case 8: return 13;
    case 9: return 10;
    case 10: return 5;
    case 11: return 7;
    case 12: return 14;
    case 13: return 6;
    case 14: return 1;
    case 15: return 0;
  }
  return 0;
}

struct MinWiseFilterCache {
  VertexID out_min_hash = kMaxVertexID;
  VertexID in_min_hash = kMaxVertexID;
  uint32_t all_neighbor_label_count = 0;
  uint32_t out_neighbor_label_count = 0;
  uint32_t in_neighbor_label_count = 0;
  uint32_t out_k_min_size = 0;
  uint32_t in_k_min_size = 0;
  uint32_t out_k_min_data[kDefaultHeapCapacity] = {0};
  uint32_t in_k_min_data[kDefaultHeapCapacity] = {0};
  // 64-bit bitmap: bit i is set iff the i-th hash bucket is among the k-min
  // selected labels for out/in neighbors.  Enables O(1) subset check.
  uint64_t out_k_min_bitmap = 0;
  uint64_t in_k_min_bitmap = 0;
};

namespace detail {

inline void EnsureVisitedBuffer(VertexID n,
                                std::vector<uint32_t>& visited,
                                uint32_t& stamp) {
  if (visited.size() < n) {
    visited.assign(n, 0);
  }
  if (__builtin_expect(++stamp == 0, 0)) {
    std::fill(visited.begin(), visited.end(), 0);
    stamp = 1;
  }
}

inline void CollectEllHopOutLabels(const ImmutableCSR& csr, VertexID src,
                                   int hop, MinWiseBitmap& bitmap,
                                   uint32_t hash_freq[16],
                                   VertexID& min_hash,
                                   std::vector<uint32_t>& visited,
                                   uint32_t& stamp) {
  auto n = csr.get_num_vertices();
  EnsureVisitedBuffer(n, visited, stamp);
  const uint32_t current_stamp = stamp;
  std::vector<std::pair<VertexID, int>> queue;
  queue.reserve(64);

  auto init_degree = csr.GetOutDegreeByLocalID(src);
  auto init_edges = csr.GetOutgoingEdgesByLocalID(src);
  for (VertexID i = 0; i < init_degree; ++i) {
    VertexID nbr = init_edges[i];
    if (visited[nbr] != current_stamp) {
      visited[nbr] = current_stamp;
      queue.emplace_back(nbr, 1);
    }
  }

  for (size_t head = 0; head < queue.size(); ++head) {
    VertexID cur = queue[head].first;
    int dist = queue[head].second;

    VertexLabel lbl = csr.GetVLabelBasePointer()[cur];
    bitmap.SetBit(lbl);
    VertexID h = MinWiseHashTable(lbl);
    if (h < 16) {
      hash_freq[h]++;
      min_hash = min_hash < h ? min_hash : h;
    }

    if (dist < hop) {
      auto cur_degree = csr.GetOutDegreeByLocalID(cur);
      auto cur_edges = csr.GetOutgoingEdgesByLocalID(cur);
      for (VertexID i = 0; i < cur_degree; ++i) {
        VertexID nxt = cur_edges[i];
        if (visited[nxt] != current_stamp) {
          visited[nxt] = current_stamp;
          queue.emplace_back(nxt, dist + 1);
        }
      }
    }
  }
}

inline void CollectEllHopInLabels(const ImmutableCSR& csr, VertexID src,
                                  int hop, MinWiseBitmap& bitmap,
                                  uint32_t hash_freq[16],
                                  VertexID& min_hash,
                                  std::vector<uint32_t>& visited,
                                  uint32_t& stamp) {
  auto n = csr.get_num_vertices();
  EnsureVisitedBuffer(n, visited, stamp);
  const uint32_t current_stamp = stamp;
  std::vector<std::pair<VertexID, int>> queue;
  queue.reserve(64);

  auto init_degree = csr.GetInDegreeByLocalID(src);
  auto init_edges = csr.GetIncomingEdgesByLocalID(src);
  for (VertexID i = 0; i < init_degree; ++i) {
    VertexID nbr = init_edges[i];
    if (visited[nbr] != current_stamp) {
      visited[nbr] = current_stamp;
      queue.emplace_back(nbr, 1);
    }
  }

  for (size_t head = 0; head < queue.size(); ++head) {
    VertexID cur = queue[head].first;
    int dist = queue[head].second;

    VertexLabel lbl = csr.GetVLabelBasePointer()[cur];
    bitmap.SetBit(lbl);
    VertexID h = MinWiseHashTable(lbl);
    if (h < 16) {
      hash_freq[h]++;
      min_hash = min_hash < h ? min_hash : h;
    }

    if (dist < hop) {
      auto cur_degree = csr.GetInDegreeByLocalID(cur);
      auto cur_edges = csr.GetIncomingEdgesByLocalID(cur);
      for (VertexID i = 0; i < cur_degree; ++i) {
        VertexID nxt = cur_edges[i];
        if (visited[nxt] != current_stamp) {
          visited[nxt] = current_stamp;
          queue.emplace_back(nxt, dist + 1);
        }
      }
    }
  }
}

}  // namespace detail

inline void BuildMinWiseFilterCache(const ImmutableCSR& csr,
                                    std::vector<MinWiseFilterCache>& cache,
                                    int hop, int k) {
  auto n = csr.get_num_vertices();
  cache.resize(n);

  auto parallelism = std::thread::hardware_concurrency();
  if (parallelism == 0) parallelism = 1;
  std::vector<size_t> worker(parallelism);
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  ParForEach(worker.begin(), worker.end(),
      [step, n, hop, k, &csr, &cache](auto w) {
        // Per-thread reusable BFS state.
        std::vector<uint32_t> visited;
        uint32_t stamp = 0;
        for (VertexID v = w; v < n; v += step) {
          auto& fc = cache[v];

          // Out edges (ell-hop).
          MinWiseBitmap out_bitmap(32);
          out_bitmap.Clear();
          uint32_t hash_freq[16] = {0};
          VertexID out_min = kMaxVertexID;

          detail::CollectEllHopOutLabels(csr, v, hop, out_bitmap, hash_freq,
                                         out_min, visited, stamp);
          fc.out_neighbor_label_count = out_bitmap.Count();
          fc.out_min_hash = out_min;

          uint32_t filled = 0;
          uint64_t out_k_min_bitmap_acc = 0;
          for (uint32_t h = 0; h < 16 && filled < static_cast<uint32_t>(k);
               ++h) {
            if (hash_freq[h] > 0) {
              fc.out_k_min_data[filled++] = h;
              out_k_min_bitmap_acc |= (1ULL << h);
            }
          }
          fc.out_k_min_size = filled;
          fc.out_k_min_bitmap = out_k_min_bitmap_acc;

          // In edges (ell-hop).
          MinWiseBitmap in_bitmap(32);
          in_bitmap.Clear();
          std::memset(hash_freq, 0, sizeof(hash_freq));
          VertexID in_min = kMaxVertexID;

          detail::CollectEllHopInLabels(csr, v, hop, in_bitmap, hash_freq,
                                        in_min, visited, stamp);
          fc.in_neighbor_label_count = in_bitmap.Count();
          fc.in_min_hash = in_min;

          filled = 0;
          uint64_t in_k_min_bitmap_acc = 0;
          for (uint32_t h = 0; h < 16 && filled < static_cast<uint32_t>(k);
               ++h) {
            if (hash_freq[h] > 0) {
              fc.in_k_min_data[filled++] = h;
              in_k_min_bitmap_acc |= (1ULL << h);
            }
          }
          fc.in_k_min_size = filled;
          fc.in_k_min_bitmap = in_k_min_bitmap_acc;

          // All neighbor label count (out | in).
          unsigned all_data = out_bitmap.GetData() | in_bitmap.GetData();
          MinWiseBitmap all_bitmap(32);
          // Directly populate the bitmap from the merged 32-bit mask.
          all_bitmap = MinWiseBitmap(32);
          for (unsigned b = 0; b < 32; ++b) {
            if (all_data & (1u << b)) {
              all_bitmap.SetBit(b);
            }
          }
          fc.all_neighbor_label_count = all_bitmap.Count();
        }
      });
}

inline bool KMinBloomIPFilter(VertexID u_idx, VertexID v_idx,
                              const ImmutableCSR& p, const ImmutableCSR& g,
                              const std::vector<MinWiseFilterCache>& p_cache,
                              const std::vector<MinWiseFilterCache>& g_cache) {
  const auto& u_cache = p_cache[u_idx];
  const auto& v_cache = g_cache[v_idx];

  // Out-edge k-min dominance: equivalent to KMinWiseIPFilter's out-edge loop.
  uint64_t common = u_cache.out_k_min_bitmap & v_cache.out_k_min_bitmap;
  uint64_t u_only = u_cache.out_k_min_bitmap ^ common;
  uint64_t v_only = v_cache.out_k_min_bitmap ^ common;
  if (u_only != 0) {
    if (v_only == 0) return false;
    if (__builtin_ctzll(u_only) < __builtin_ctzll(v_only)) return false;
  }

  // In-edge k-min dominance.
  common = u_cache.in_k_min_bitmap & v_cache.in_k_min_bitmap;
  u_only = u_cache.in_k_min_bitmap ^ common;
  v_only = v_cache.in_k_min_bitmap ^ common;
  if (u_only != 0) {
    if (v_only == 0) return false;
    if (__builtin_ctzll(u_only) < __builtin_ctzll(v_only)) return false;
  }

  // Same final count/degree checks as KMinWiseIPFilter.
  return v_cache.in_neighbor_label_count >= u_cache.in_neighbor_label_count &&
         v_cache.out_neighbor_label_count >= u_cache.out_neighbor_label_count &&
         g.GetOutDegreeByLocalID(v_idx) >= p.GetOutDegreeByLocalID(u_idx) &&
         g.GetInDegreeByLocalID(v_idx) >= p.GetInDegreeByLocalID(u_idx);
}

inline bool KMinWiseIPFilter(VertexID u_idx, VertexID v_idx,
                             const ImmutableCSR& p, const ImmutableCSR& g,
                             const std::vector<MinWiseFilterCache>& p_cache,
                             const std::vector<MinWiseFilterCache>& g_cache) {
  auto u_label = p.GetVLabelBasePointer()[u_idx];
  auto v_label = g.GetVLabelBasePointer()[v_idx];
  if (u_label != v_label) return false;

  const auto& u_cache = p_cache[u_idx];
  const auto& v_cache = g_cache[v_idx];

  // Filter by out edges.
  VertexID min_v_ip_val = kMaxVertexID;
  VertexID min_u_ip_val = kMaxVertexID;

  uint32_t u_k_min_heap_data[kDefaultHeapCapacity];
  uint32_t v_k_min_heap_data[kDefaultHeapCapacity];
  std::memcpy(u_k_min_heap_data, u_cache.out_k_min_data,
              sizeof(uint32_t) * kDefaultHeapCapacity);
  std::memcpy(v_k_min_heap_data, v_cache.out_k_min_data,
              sizeof(uint32_t) * kDefaultHeapCapacity);

  for (uint32_t i = 0; i < v_cache.out_k_min_size; i++) {
    auto v_ip_val = v_k_min_heap_data[i];
    for (uint32_t j = 0; j < u_cache.out_k_min_size; j++) {
      auto u_ip_val = u_k_min_heap_data[j];
      if (v_ip_val == u_ip_val) {
        v_k_min_heap_data[i] = kMaxVertexID;
        u_k_min_heap_data[j] = kMaxVertexID;
        break;
      }
    }
  }

  for (uint32_t i = 0; i < v_cache.out_k_min_size; i++) {
    auto v_ip_val = v_k_min_heap_data[i];
    min_v_ip_val = min_v_ip_val < v_ip_val ? min_v_ip_val : v_ip_val;
  }

  for (uint32_t i = 0; i < u_cache.out_k_min_size; i++) {
    auto u_ip_val = u_k_min_heap_data[i];
    min_u_ip_val = min_u_ip_val < u_ip_val ? min_u_ip_val : u_ip_val;
  }

  if (min_v_ip_val == kMaxVertexID && min_u_ip_val != kMaxVertexID)
    return false;

  for (uint32_t i = 0; i < u_cache.out_k_min_size; i++) {
    if (u_k_min_heap_data[i] < min_v_ip_val) {
      return false;
    }
  }

  // Filter by in edges.
  min_v_ip_val = kMaxVertexID;
  min_u_ip_val = kMaxVertexID;
  std::memcpy(u_k_min_heap_data, u_cache.in_k_min_data,
              sizeof(uint32_t) * kDefaultHeapCapacity);
  std::memcpy(v_k_min_heap_data, v_cache.in_k_min_data,
              sizeof(uint32_t) * kDefaultHeapCapacity);

  for (uint32_t i = 0; i < v_cache.in_k_min_size; i++) {
    auto v_ip_val = v_k_min_heap_data[i];
    for (uint32_t j = 0; j < u_cache.in_k_min_size; j++) {
      auto u_ip_val = u_k_min_heap_data[j];
      if (v_ip_val == u_ip_val) {
        v_k_min_heap_data[i] = kMaxVertexID;
        u_k_min_heap_data[j] = kMaxVertexID;
        break;
      }
    }
  }

  for (uint32_t i = 0; i < v_cache.in_k_min_size; i++) {
    auto v_ip_val = v_k_min_heap_data[i];
    min_v_ip_val = min_v_ip_val < v_ip_val ? min_v_ip_val : v_ip_val;
  }

  for (uint32_t i = 0; i < u_cache.in_k_min_size; i++) {
    auto u_ip_val = u_k_min_heap_data[i];
    min_u_ip_val = min_u_ip_val < u_ip_val ? min_u_ip_val : u_ip_val;
  }

  if (min_v_ip_val == kMaxVertexID && min_u_ip_val != kMaxVertexID)
    return false;

  for (uint32_t i = 0; i < u_cache.in_k_min_size; i++) {
    if (u_k_min_heap_data[i] < min_v_ip_val) {
      return false;
    }
  }

  return v_cache.in_neighbor_label_count >= u_cache.in_neighbor_label_count &&
         v_cache.out_neighbor_label_count >= u_cache.out_neighbor_label_count &&
         g.GetOutDegreeByLocalID(v_idx) >= p.GetOutDegreeByLocalID(u_idx) &&
         g.GetInDegreeByLocalID(v_idx) >= p.GetInDegreeByLocalID(u_idx);
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_CPU_MIN_WISE_FILTER_H_
