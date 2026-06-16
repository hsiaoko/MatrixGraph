#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <unordered_set>
#include "core/util/execution_policy.h"
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "core/common/consts.h"
#include "core/common/types.h"
#include "core/data_structures/exec_plan.cuh"
#include "core/data_structures/heap.cuh"
#include "core/data_structures/host_buffer.cuh"
#include "core/data_structures/matches.cuh"
#include "core/data_structures/matrix.cuh"
#include "core/data_structures/metadata.h"
#include "core/data_structures/mini_kernel_bitmap.cuh"
#include "core/data_structures/unified_buffer.cuh"
#include "core/data_structures/woj_exec_plan.cuh"
#include "core/data_structures/woj_matches.cuh"
#include "core/task/cpu_task/cpu_subiso.cuh"
#include "core/task/gpu_task/kernel/algorithms/hash.cuh"
#include "core/task/gpu_task/kernel/algorithms/sort.cuh"
#include "core/task/gpu_task/matrix_ops.cuh"
#include "core/util/bitmap_no_ownership.h"
#include "core/util/bitmap_ownership.h"
#include "core/util/format_converter.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using sics::matrixgraph::core::task::kernel::MatrixOpsKernelWrapper;
using MinHeap = sics::matrixgraph::core::task::kernel::MinHeap;
using UnifiedOwnedBufferFloat =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<float>;
using BufferFloat = sics::matrixgraph::core::data_structures::Buffer<float>;
using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using VertexID = sics::matrixgraph::core::common::VertexID;
using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;
using Matrix = sics::matrixgraph::core::data_structures::Matrix;
using Edges = sics::matrixgraph::core::data_structures::Edges;
using Edge = sics::matrixgraph::core::data_structures::Edge;
using GraphMetadata = sics::matrixgraph::core::data_structures::GraphMetadata;
using WOJMatches = sics::matrixgraph::core::data_structures::WOJMatches;
using Matches = sics::matrixgraph::core::data_structures::Matches;
using MiniKernelBitmap =
    sics::matrixgraph::core::task::kernel::MiniKernelBitmap;
using sics::matrixgraph::core::task::kernel::HashTable;
using WOJExecutionPlan =
    sics::matrixgraph::core::data_structures::WOJExecutionPlan;
using ExecutionPlan = sics::matrixgraph::core::data_structures::ExecutionPlan;
using BitmapOwnership = sics::matrixgraph::core::util::BitmapOwnership;
using sics::matrixgraph::core::common::kMaxNumLocalWeft;
using sics::matrixgraph::core::common::kMaxNumWeft;
using sics::matrixgraph::core::common::kMaxMatchTableRows;
using sics::matrixgraph::core::common::kMaxVertexID;
using sics::matrixgraph::core::common::kSubIsoMaxBacktrackNodes;
using sics::matrixgraph::core::common::kSubIsoMaxMsPerWeft;
using sics::matrixgraph::core::common::kSubIsoMaxValidateWefts;
using sics::matrixgraph::core::common::kSubIsoValidateMatchingTimeoutSec;
using sics::matrixgraph::core::common::kSubIsoProgressPrintInterval;
using sics::matrixgraph::core::common::kSubIsoLocalMatchesSizeBuffer;
using BitmapNoOwnerShip = sics::matrixgraph::core::util::BitmapNoOwnerShip;
using sics::matrixgraph::core::common::kDefaultHeapCapacity;

static int filter_count = 0;
static int label_filter_count = 0;
static int label_degree_filter_count = 0;
static int gnn_filter_count = 0;
static int nlc_filter_count = 0;
static int ip_filter_count = 0;
static int index_filter_count = 0;

// Global rejection collector for -reject_output.  Empty path means disabled.
// When enabled, ValidateWeft appends rejected (u,v) pairs encoded as
// (uint64_t(u) << 32) | uint64_t(v).  Kept global to avoid threading a pointer
// through every static helper; access is sequential because ValidateMatching
// processes wefts one at a time.
static std::string g_reject_output_path;
static std::vector<uint64_t> g_rejected_pairs;

static inline void RecordRejectedPair(VertexID u, VertexID v) {
  if (g_reject_output_path.empty()) return;
  g_rejected_pairs.push_back((static_cast<uint64_t>(u) << 32) |
                             static_cast<uint64_t>(v));
}

static void WriteRejectedPairs() {
  if (g_reject_output_path.empty()) return;
  std::ofstream out(g_reject_output_path);
  if (!out) {
    std::cerr << "[CPUSubIso] Failed to open reject output: "
              << g_reject_output_path << std::endl;
    return;
  }
  std::sort(g_rejected_pairs.begin(), g_rejected_pairs.end());
  g_rejected_pairs.erase(
      std::unique(g_rejected_pairs.begin(), g_rejected_pairs.end()),
      g_rejected_pairs.end());
  for (auto encoded : g_rejected_pairs) {
    VertexID u = static_cast<VertexID>(encoded >> 32);
    VertexID v = static_cast<VertexID>(encoded & 0xFFFFFFFFu);
    out << u << ',' << v << '\n';
  }
  out.close();
  std::cout << "[CPUSubIso] Wrote " << g_rejected_pairs.size()
            << " rejected (u,v) pairs to " << g_reject_output_path << std::endl;
}

static std::hash<int> hasher;

struct VertexFilterCache {
  VertexID out_min_hash = kMaxVertexID;
  VertexID in_min_hash = kMaxVertexID;
  uint32_t all_neighbor_label_count = 0;
  uint32_t out_neighbor_label_count = 0;
  uint32_t in_neighbor_label_count = 0;
  uint32_t out_k_min_size = 0;
  uint32_t in_k_min_size = 0;
  uint32_t out_k_min_data[kDefaultHeapCapacity] = {0};
  uint32_t in_k_min_data[kDefaultHeapCapacity] = {0};
};

static std::vector<VertexFilterCache> p_filter_cache;
static std::vector<VertexFilterCache> g_filter_cache;

static void BuildFilterCache(const ImmutableCSR& csr,
                             std::vector<VertexFilterCache>& cache) {
  auto n = csr.get_num_vertices();
  cache.resize(n);

  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  ParForEach(worker.begin(), worker.end(),
      [step, n, &csr, &cache](auto w) {
        for (VertexID v = w; v < n; v += step) {
          auto& fc = cache[v];

          // Out edges.
          MiniKernelBitmap out_bitmap(32);
          out_bitmap.Clear();
          uint32_t hash_freq[16] = {0};
          VertexID out_min = kMaxVertexID;

          auto out_degree = csr.GetOutDegreeByLocalID(v);
          auto out_edges = csr.GetOutgoingEdgesByLocalID(v);
          for (VertexID i = 0; i < out_degree; ++i) {
            VertexID nbr = out_edges[i];
            VertexLabel lbl = csr.GetVLabelBasePointer()[nbr];
            out_bitmap.SetBit(lbl);
            VertexID h = HashTable(lbl);
            if (h < 16) {
              hash_freq[h]++;
              out_min = out_min < h ? out_min : h;
            }
          }
          fc.out_neighbor_label_count = out_bitmap.Count();
          fc.out_min_hash = out_min;

          uint32_t filled = 0;
          for (uint32_t h = 0; h < 16 && filled < kDefaultHeapCapacity; ++h) {
            if (hash_freq[h] > 0) {
              fc.out_k_min_data[filled++] = h;
            }
          }
          fc.out_k_min_size = filled;

          // In edges.
          MiniKernelBitmap in_bitmap(32);
          in_bitmap.Clear();
          memset(hash_freq, 0, sizeof(hash_freq));
          VertexID in_min = kMaxVertexID;

          auto in_degree = csr.GetInDegreeByLocalID(v);
          auto in_edges = csr.GetIncomingEdgesByLocalID(v);
          for (VertexID i = 0; i < in_degree; ++i) {
            VertexID nbr = in_edges[i];
            VertexLabel lbl = csr.GetVLabelBasePointer()[nbr];
            in_bitmap.SetBit(lbl);
            VertexID h = HashTable(lbl);
            if (h < 16) {
              hash_freq[h]++;
              in_min = in_min < h ? in_min : h;
            }
          }
          fc.in_neighbor_label_count = in_bitmap.Count();
          fc.in_min_hash = in_min;

          filled = 0;
          for (uint32_t h = 0; h < 16 && filled < kDefaultHeapCapacity; ++h) {
            if (hash_freq[h] > 0) {
              fc.in_k_min_data[filled++] = h;
            }
          }
          fc.in_k_min_size = filled;

          // All neighbor label count (out | in).
          unsigned all_data = out_bitmap.GetData() | in_bitmap.GetData();
          MiniKernelBitmap all_bitmap(32);
          all_bitmap.data_ = all_data;
          fc.all_neighbor_label_count = all_bitmap.Count();
        }
      });
}

struct LocalMatches {
  VertexID* data = nullptr;
  VertexID* size = nullptr;
};

static void SimdSquaredDifference(const float* __restrict v_a,
                                  const float* __restrict v_b,
                                  float* __restrict v_c, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    v_c[i] = (v_a[i] - v_b[i]) * (v_a[i] - v_b[i]);
  }
}

static inline bool LabelFilter(VertexID u_idx, VertexID v_idx,
                               const ImmutableCSR& p, const ImmutableCSR& g) {
  auto u_label = p.GetVLabelBasePointer()[u_idx];
  auto v_label = g.GetVLabelBasePointer()[v_idx];
  return u_label == v_label;
}

static inline bool LabelDegreeFilter(VertexID u_idx, VertexID v_idx,
                                     const ImmutableCSR& p,
                                     const ImmutableCSR& g) {
  auto u_label = p.GetVLabelBasePointer()[u_idx];
  auto v_label = g.GetVLabelBasePointer()[v_idx];
  return u_label == v_label &&
         g.GetOutDegreeByLocalID(v_idx) >= p.GetOutDegreeByLocalID(u_idx) &&
         g.GetInDegreeByLocalID(v_idx) >= p.GetInDegreeByLocalID(u_idx);
}


static bool NeighborLabelCounterFilter(VertexID u_idx, VertexID v_idx,
                                       const ImmutableCSR& p,
                                       const ImmutableCSR& g) {
  return g_filter_cache[v_idx].all_neighbor_label_count >=
         p_filter_cache[u_idx].all_neighbor_label_count;
}

static bool MinWiseIPFilter(VertexID u_idx, VertexID v_idx,
                              const ImmutableCSR& p, const ImmutableCSR& g) {
  auto u_label = p.GetVLabelBasePointer()[u_idx];
  auto v_label = g.GetVLabelBasePointer()[v_idx];
  if (u_label != v_label) return false;

  const auto& u_cache = p_filter_cache[u_idx];
  const auto& v_cache = g_filter_cache[v_idx];

  return u_cache.in_min_hash >= v_cache.in_min_hash &&
         v_cache.all_neighbor_label_count >= u_cache.all_neighbor_label_count &&
         u_cache.out_min_hash >= v_cache.out_min_hash &&
         g.GetOutDegreeByLocalID(v_idx) >= p.GetOutDegreeByLocalID(u_idx) &&
         g.GetInDegreeByLocalID(v_idx) >= p.GetInDegreeByLocalID(u_idx);
}

static bool KMinWiseIPFilter(VertexID u_idx, VertexID v_idx,
                             const ImmutableCSR& p, const ImmutableCSR& g) {
  auto u_label = p.GetVLabelBasePointer()[u_idx];
  auto v_label = g.GetVLabelBasePointer()[v_idx];
  if (u_label != v_label) return false;

  const auto& u_cache = p_filter_cache[u_idx];
  const auto& v_cache = g_filter_cache[v_idx];

  // Filter by out edges.
  VertexID min_v_ip_val = kMaxVertexID;
  VertexID min_u_ip_val = kMaxVertexID;

  uint32_t u_k_min_heap_data[kDefaultHeapCapacity];
  uint32_t v_k_min_heap_data[kDefaultHeapCapacity];
  memcpy(u_k_min_heap_data, u_cache.out_k_min_data,
         sizeof(uint32_t) * kDefaultHeapCapacity);
  memcpy(v_k_min_heap_data, v_cache.out_k_min_data,
         sizeof(uint32_t) * kDefaultHeapCapacity);

  for (VertexID _ = 0; _ < v_cache.out_k_min_size; _++) {
    auto v_ip_val = v_k_min_heap_data[_];
    for (VertexID __ = 0; __ < u_cache.out_k_min_size; __++) {
      auto u_ip_val = u_k_min_heap_data[__];
      if (v_ip_val == u_ip_val) {
        v_k_min_heap_data[_] = kMaxVertexID;
        u_k_min_heap_data[__] = kMaxVertexID;
        break;
      }
    }
  }

  for (VertexID _ = 0; _ < v_cache.out_k_min_size; _++) {
    auto v_ip_val = v_k_min_heap_data[_];
    min_v_ip_val = min_v_ip_val < v_ip_val ? min_v_ip_val : v_ip_val;
  }

  for (VertexID _ = 0; _ < u_cache.out_k_min_size; _++) {
    auto u_ip_val = u_k_min_heap_data[_];
    min_u_ip_val = min_u_ip_val < u_ip_val ? min_u_ip_val : u_ip_val;
  }

  if (min_v_ip_val == kMaxVertexID && min_u_ip_val != kMaxVertexID)
    return false;

  for (VertexID _ = 0; _ < u_cache.out_k_min_size; _++) {
    if (u_k_min_heap_data[_] < min_v_ip_val) {
      return false;
    }
  }

  // Filter by in edges.
  min_v_ip_val = kMaxVertexID;
  min_u_ip_val = kMaxVertexID;
  memcpy(u_k_min_heap_data, u_cache.in_k_min_data,
         sizeof(uint32_t) * kDefaultHeapCapacity);
  memcpy(v_k_min_heap_data, v_cache.in_k_min_data,
         sizeof(uint32_t) * kDefaultHeapCapacity);

  for (VertexID _ = 0; _ < v_cache.in_k_min_size; _++) {
    auto v_ip_val = v_k_min_heap_data[_];
    for (VertexID __ = 0; __ < u_cache.in_k_min_size; __++) {
      auto u_ip_val = u_k_min_heap_data[__];
      if (v_ip_val == u_ip_val) {
        v_k_min_heap_data[_] = kMaxVertexID;
        u_k_min_heap_data[__] = kMaxVertexID;
        break;
      }
    }
  }

  for (VertexID _ = 0; _ < v_cache.in_k_min_size; _++) {
    auto v_ip_val = v_k_min_heap_data[_];
    min_v_ip_val = min_v_ip_val < v_ip_val ? min_v_ip_val : v_ip_val;
  }

  for (VertexID _ = 0; _ < u_cache.in_k_min_size; _++) {
    auto u_ip_val = u_k_min_heap_data[_];
    min_u_ip_val = min_u_ip_val < u_ip_val ? min_u_ip_val : u_ip_val;
  }

  if (min_v_ip_val == kMaxVertexID && min_u_ip_val != kMaxVertexID)
    return false;

  for (VertexID _ = 0; _ < u_cache.in_k_min_size; _++) {
    if (u_k_min_heap_data[_] < min_v_ip_val) {
      return false;
    }
  }

  return v_cache.in_neighbor_label_count >= u_cache.in_neighbor_label_count &&
         v_cache.out_neighbor_label_count >= u_cache.out_neighbor_label_count &&
         g.GetOutDegreeByLocalID(v_idx) >= p.GetOutDegreeByLocalID(u_idx) &&
         g.GetInDegreeByLocalID(v_idx) >= p.GetInDegreeByLocalID(u_idx);
}

static bool Filter(VertexID u_idx, VertexID v_idx, const ImmutableCSR& p,
                   const ImmutableCSR& g,
                   std::vector<uint64_t>* rejected_pairs = nullptr) {
  if (u_idx == kMaxVertexID) return false;
  if (v_idx == kMaxVertexID) return false;
  if (!LabelFilter(u_idx, v_idx, p, g)) {
    __sync_fetch_and_add(&label_filter_count, 1);
    __sync_fetch_and_add(&filter_count, 1);
    if (rejected_pairs) {
      rejected_pairs->push_back(
          (static_cast<uint64_t>(u_idx) << 32) |
          static_cast<uint64_t>(g.GetGloablIDBasePointer()[v_idx]));
    }
    return false;
   }
  if (!LabelDegreeFilter(u_idx, v_idx, p, g)) {
    __sync_fetch_and_add(&label_degree_filter_count, 1);
    __sync_fetch_and_add(&filter_count, 1);
    if (rejected_pairs) {
      rejected_pairs->push_back(
          (static_cast<uint64_t>(u_idx) << 32) |
          static_cast<uint64_t>(g.GetGloablIDBasePointer()[v_idx]));
    }
    return false;
  }
  if (!NeighborLabelCounterFilter(u_idx, v_idx, p, g)) {
    __sync_fetch_and_add(&nlc_filter_count, 1);
    __sync_fetch_and_add(&filter_count, 1);
    if (rejected_pairs) {
      rejected_pairs->push_back(
          (static_cast<uint64_t>(u_idx) << 32) |
          static_cast<uint64_t>(g.GetGloablIDBasePointer()[v_idx]));
    }
    return false;
  }

  return true;
}

static bool MatrixFilter(
    VertexID u_idx, VertexID v_idx, const ImmutableCSR& p,
    const ImmutableCSR& g, const std::vector<Matrix>& m_vec,
    const std::vector<UnifiedOwnedBufferFloat*>& m_unified_buffer_vec,
    std::vector<uint64_t>* rejected_pairs = nullptr) {
  if (0 == m_vec.size()) return true;
  auto vec_len = m_vec[0].get_y();

  /// Init similarity vector.

  float sim_vec[vec_len] = {0};
  MatrixOpsKernelWrapper::CPUSimdSquaredDifference(
      m_vec[0].GetPtr() + u_idx * vec_len, m_vec[1].GetPtr() + v_idx * vec_len,
      sim_vec, vec_len);

  float z1[64] = {0};
  float z2[1] = {0};

  MatrixOpsKernelWrapper::CPUOnlyMatMult(sim_vec, m_vec[2].GetPtr(), z1, 1,
                                         m_vec[2].get_y(), m_vec[2].get_x(),
                                         false, true);

  MatrixOpsKernelWrapper::CPUOnlyMatAdd(z1, m_vec[3].GetPtr(), m_vec[3].get_x(),
                                        m_vec[3].get_y());

  MatrixOpsKernelWrapper::CPURelu(z1, m_vec[3].get_x(), m_vec[3].get_y());

  MatrixOpsKernelWrapper::CPUOnlyMatMult(z1, m_vec[4].GetPtr(), z2, 1,
                                         m_vec[4].get_x(), m_vec[4].get_y(),
                                         false, true);

  MatrixOpsKernelWrapper::CPUOnlyMatAdd(z2, m_vec[5].GetPtr(), m_vec[5].get_x(),
                                        m_vec[5].get_y());

  MatrixOpsKernelWrapper::CPUSigmoid(z2, 1, 1);

  if (z2[0] < 0.1) {
    if (rejected_pairs) {
      rejected_pairs->push_back(
          (static_cast<uint64_t>(u_idx) << 32) |
          static_cast<uint64_t>(g.GetGloablIDBasePointer()[v_idx]));
    }
    return false;
  }

  return true;
}

static bool GPUMatrixFilter(
    VertexID u_idx, VertexID v_idx, const ImmutableCSR& p,
    const ImmutableCSR& g, const std::vector<Matrix>& m_vec,
    const std::vector<UnifiedOwnedBufferFloat*>& m_unified_buffer_vec) {
  return true;
  BufferFloat buffer_m1;
  BufferFloat buffer_m2;

  auto vec_len = m_vec[0].get_y();

  buffer_m1.data = m_vec[0].GetPtr() + u_idx * vec_len;
  buffer_m2.data = m_vec[1].GetPtr() + v_idx * vec_len;

  buffer_m1.size = sizeof(uint64_t) * vec_len;
  buffer_m2.size = sizeof(uint64_t) * vec_len;

  /// Init similarity vector.
  UnifiedOwnedBufferFloat unified_sim_vec;
  unified_sim_vec.Init(vec_len * sizeof(float));

  SimdSquaredDifference(buffer_m1.GetPtr(), buffer_m2.GetPtr(),
                        unified_sim_vec.GetPtr(), vec_len);

  UnifiedOwnedBufferFloat z1;
  z1.Init(sizeof(float) * 8);

  UnifiedOwnedBufferFloat z2;
  z2.Init(sizeof(float) * 1);

  MatrixOps matrix_ops;
  matrix_ops.MatMult(unified_sim_vec.GetPtr(),
                     m_unified_buffer_vec[2]->GetPtr(), z1.GetPtr(), 1, 64, 8,
                     false, true);

  matrix_ops.MatAdd(z1.GetPtr(), m_unified_buffer_vec[3]->GetPtr(), 1, 8);

  matrix_ops.Activate(z1.GetPtr(), 1, 8);

  matrix_ops.MatMult(z1.GetPtr(), m_unified_buffer_vec[4]->GetPtr(),
                     z2.GetPtr(), 1, 8, 1, false, true);

  matrix_ops.MatAdd(z2.GetPtr(), m_unified_buffer_vec[5]->GetPtr(), 1, 1);

  matrix_ops.Activate(z2.GetPtr(), 1, 1, 's');

  // std::cout << *z2.GetPtr() << " ";
  if (*z2.GetPtr() < 0.8) return false;

  return true;
}

static std::vector<WOJMatches*> WOJFilter(
    const WOJExecutionPlan& exec_plan, const ImmutableCSR& p,
    const ImmutableCSR& g, const std::vector<Matrix>& m_vec,
    const std::vector<UnifiedOwnedBufferFloat*>& m_unified_buffer_vec) {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  // Init output vector.
  std::vector<WOJMatches*> woj_matches_vec;
  woj_matches_vec.resize(exec_plan.get_n_edges_p());

  for (VertexID _ = 0; _ < exec_plan.get_n_edges_p(); _++) {
    woj_matches_vec[_] = new WOJMatches();
    woj_matches_vec[_]->Init(exec_plan.get_n_edges_p(), kMaxMatchTableRows);
    woj_matches_vec[_]->SetXOffset(2);
    woj_matches_vec[_]->SetYOffset(0);
    woj_matches_vec[_]->SetHeader(
        0, exec_plan.get_exec_path_in_edges_ptr()[_ * 2]);
    woj_matches_vec[_]->SetHeader(
        1, exec_plan.get_exec_path_in_edges_ptr()[_ * 2 + 1]);
  }

  for (VertexID eid = 0; eid < exec_plan.get_n_edges_p(); eid++) {
    VertexID u_src = exec_plan.get_exec_path_in_edges_ptr()[2 * eid];
    VertexID u_dst = exec_plan.get_exec_path_in_edges_ptr()[2 * eid + 1];
    VertexID src_idx = p.GetLocalIDByGlobalID(u_src);
    VertexID dst_idx = p.GetLocalIDByGlobalID(u_dst);

    ParForEach(worker.begin(), worker.end(),
        [step, &p, &g, eid, u_src, u_dst, src_idx, dst_idx, &exec_plan,
         &woj_matches_vec, &m_vec, &m_unified_buffer_vec](auto w) {
          for (VertexID v_idx = w; v_idx < g.get_num_vertices();
               v_idx += step) {
            auto offset = g.GetOutOffsetByLocalID(v_idx);
            auto degree = g.GetOutDegreeByLocalID(v_idx);
            auto* out_edges = g.GetOutgoingEdgesByLocalID(v_idx);
            VertexID global_id = g.GetGlobalIDByLocalID(v_idx);
            if (MatrixFilter(src_idx, v_idx, p, g, m_vec,
                             m_unified_buffer_vec)) {
              for (VertexID nbr_v_idx = 0; nbr_v_idx < degree; nbr_v_idx++) {
                VertexID nbr_v = out_edges[nbr_v_idx];
                VertexID nbr_localid = g.GetLocalIDByGlobalID(nbr_v);
                if (MatrixFilter(dst_idx, nbr_localid, p, g, m_vec,
                                 m_unified_buffer_vec)) {
                  auto local_offset = __sync_fetch_and_add(
                      woj_matches_vec[eid]->get_y_offset_ptr(), 1);
                  woj_matches_vec[eid]->get_data_ptr()[local_offset * 2] =
                      global_id;
                  woj_matches_vec[eid]->get_data_ptr()[local_offset * 2 + 1] =
                      nbr_v;
                }
              }
            }
          }
        });
  }

  return woj_matches_vec;
}

static inline void Join(VertexID n_vertices_g,
                        const WOJMatches& left_woj_matches,
                        const WOJMatches& right_woj_matches,
                        WOJMatches* output_woj_matches, VertexID left_hash_idx,
                        VertexID right_hash_idx, BitmapOwnership& right_visited,
                        BitmapOwnership& jump_visited) {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  VertexID* global_offset_ptr = output_woj_matches->get_y_offset_ptr();

  VertexID* left_data = left_woj_matches.get_data_ptr();
  VertexID* right_data = right_woj_matches.get_data_ptr();
  VertexID* output_data = output_woj_matches->get_data_ptr();
  VertexID left_x_offset = left_woj_matches.get_x_offset();
  VertexID right_x_offset = right_woj_matches.get_x_offset();
  VertexID output_x_offset = output_woj_matches->get_x_offset();
  VertexID left_y_offset = left_woj_matches.get_y_offset();
  VertexID right_y_offset = right_woj_matches.get_y_offset();
  VertexID* output_y_offset_ptr = output_woj_matches->get_y_offset_ptr();

  ParForEach(worker.begin(), worker.end(),
      [step, &left_woj_matches, &right_woj_matches, &left_data, &right_data,
       &output_data, left_x_offset, right_x_offset, output_x_offset,
       left_y_offset, right_y_offset, left_hash_idx, right_hash_idx,
       &output_y_offset_ptr](auto w) {
        for (VertexID left_data_offset = w;
             left_data_offset < left_woj_matches.get_y_offset();
             left_data_offset += step) {
          VertexID target =
              left_data[left_x_offset * left_data_offset + left_hash_idx];

          VertexID right_data_offset =
              right_woj_matches.BinarySearch(right_hash_idx, target);
          if (right_data_offset != kMaxVertexID &&
              right_data_offset < right_woj_matches.get_y_offset()) {
            VertexID left_walker = right_data_offset - 1;
            VertexID right_walker = right_data_offset;

            while (left_walker >= 0 && left_walker < right_y_offset &&
                   right_data[left_walker * right_x_offset + right_hash_idx] ==
                       target) {
              // Write direct on the global memory.
              auto output_y_offset =
                  __sync_fetch_and_add(output_y_offset_ptr, 1);
              if (output_y_offset > kMaxMatchTableRows / output_x_offset) break;

              memcpy(output_data + output_y_offset * output_x_offset,
                     left_data + left_data_offset * left_x_offset,
                     sizeof(VertexID) * left_x_offset);

              VertexID write_col = 0;
              for (VertexID right_col_idx = 0; right_col_idx < right_x_offset;
                   right_col_idx++) {
                if (right_col_idx == right_hash_idx) continue;
                *(output_data + output_y_offset * output_x_offset +
                  left_x_offset + write_col) =
                    right_data[left_walker * right_x_offset + right_col_idx];
                write_col++;
              }

              left_walker--;
            }

            while (right_walker >= 0 && right_walker < right_y_offset &&
                   right_data[right_walker * right_x_offset + right_hash_idx] ==
                       target) {
              // Write direct on the global memory.
              auto output_y_offset =
                  __sync_fetch_and_add(output_y_offset_ptr, 1);
              if (output_y_offset > kMaxMatchTableRows / output_x_offset) break;

              memcpy(output_data + output_y_offset * output_x_offset,
                     left_data + left_data_offset * left_x_offset,
                     sizeof(VertexID) * left_x_offset);

              VertexID write_col = 0;
              for (VertexID right_col_idx = 0; right_col_idx < right_x_offset;
                   right_col_idx++) {
                if (right_col_idx == right_hash_idx) continue;
                *(output_data + output_y_offset * output_x_offset +
                  left_x_offset + write_col) =
                    right_data[right_walker * right_x_offset + right_col_idx];
                write_col++;
              }

              right_walker++;
            }
          }
        }
      });
}

static WOJMatches* WOJEnumerating(
    const WOJExecutionPlan& exec_plan,
    const std::vector<WOJMatches*>& input_woj_matches_vec) {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  WOJMatches* output_woj_matches = new WOJMatches();
  output_woj_matches->Init(exec_plan.get_n_edges_p(), kMaxMatchTableRows);

  // Sort candidate
  BitmapOwnership header_visited(32);
  for (VertexID _ = 1; _ < input_woj_matches_vec.size(); _++) {
    bool sort_tag = false;
    auto header_ptr = input_woj_matches_vec[_]->get_header_ptr();
    for (VertexID __ = 0; __ < input_woj_matches_vec[_]->get_x_offset(); __++) {
      if (header_visited.GetBit(header_ptr[__]) && sort_tag == false) {
        kernel::MergeSort(0, input_woj_matches_vec[_]->get_data_ptr(), __,
                          input_woj_matches_vec[_]->get_x_offset(),
                          input_woj_matches_vec[_]->get_y_offset(),
                          sizeof(VertexID) * input_woj_matches_vec[_]->get_y() *
                              input_woj_matches_vec[_]->get_x());
        sort_tag = true;
      }
      header_visited.SetBit(header_ptr[__]);
    }
  }

  BitmapOwnership visited(1024);
  BitmapOwnership jump_visited(1024);

  // Join candidates
  auto left_woj_matches = input_woj_matches_vec[0];
  for (VertexID _ = 1; _ < input_woj_matches_vec.size(); _++) {
    auto right_woj_matches = input_woj_matches_vec[_];
    auto join_keys = left_woj_matches->GetJoinKey(*right_woj_matches);

    if (join_keys.first == kMaxVertexID || join_keys.second == kMaxVertexID)
      continue;

    output_woj_matches->SetHeader(left_woj_matches->get_header_ptr(),
                                  left_woj_matches->get_x_offset(),
                                  right_woj_matches->get_header_ptr(),
                                  right_woj_matches->get_x_offset(), join_keys);

    Join(exec_plan.get_n_vertices_g(), *left_woj_matches, *right_woj_matches,
         output_woj_matches, join_keys.first, join_keys.second, visited,
         jump_visited);
    if (output_woj_matches->get_y_offset() == 0) {
      break;
    }
    if (output_woj_matches->get_x_offset() == output_woj_matches->get_x()) {
      break;
    } else {
      std::swap(left_woj_matches, output_woj_matches);
      output_woj_matches->Clear();
    }
  }

  if (input_woj_matches_vec.size() % 2 == 0) {
    return output_woj_matches;
  } else {
    return left_woj_matches;
  }
}

static bool IsFeasible(
    const ImmutableCSR& p, const ImmutableCSR& g,
    const std::vector<Matrix>& m_vec,
    const std::vector<UnifiedOwnedBufferFloat*>& m_unified_buffer_vec,
    VertexID u_src, VertexID u_dst, VertexID v_src, VertexID v_dst,
    LocalMatches* localMatches, std::vector<uint64_t>* rejected_pairs = nullptr) {
  if (u_src == kMaxVertexID && v_src == kMaxVertexID) {
    return true;
  }
  if (u_src == kMaxVertexID && v_src != kMaxVertexID) return false;
  if (v_src == kMaxVertexID && u_src != kMaxVertexID) return false;

  if (!Filter(u_src, v_src, p, g, rejected_pairs)) return false;
  if (!Filter(u_dst, v_dst, p, g, rejected_pairs)) return false;

  if (!KMinWiseIPFilter(u_dst, v_dst, p, g)) {
    __sync_fetch_and_add(&ip_filter_count, 1);
    if (rejected_pairs) {
      rejected_pairs->push_back(
          (static_cast<uint64_t>(u_dst) << 32) |
          static_cast<uint64_t>(g.GetGloablIDBasePointer()[v_dst]));
    }
    return false;
  }
  if (!KMinWiseIPFilter(u_src, v_src, p, g)) {
    __sync_fetch_and_add(&ip_filter_count, 1);
    if (rejected_pairs) {
      rejected_pairs->push_back(
          (static_cast<uint64_t>(u_src) << 32) |
          static_cast<uint64_t>(g.GetGloablIDBasePointer()[v_src]));
    }
    return false;
  }

  return true;
}

static void DFSExtend(
    const ImmutableCSR& p, const ImmutableCSR& g,
    const ExecutionPlan& exec_plan, const std::vector<Matrix>& m_vec,
    const std::vector<UnifiedOwnedBufferFloat*>& m_unified_buffer_vec,
    VertexID level, VertexID pre_v_idx, VertexID v_idx,
    std::vector<std::unordered_set<uint64_t>>& matches_visited_pairs,
    LocalMatches* local_matches, bool match,
    std::vector<uint64_t>* rejected_pairs = nullptr) {
  // 基础检查
  if (level > exec_plan.get_depth()) {
    return;
  }

  bool extend_tag = false;
  VertexID matched_pattern_vertex = kMaxVertexID;

  // 遍历执行计划中的边约束
  for (auto i = 0; i < exec_plan.get_n_edges(); i++) {
    auto u_src =
        exec_plan.get_sequential_exec_path_in_edges_ptr()->GetPtr()[2 * i];
    auto u_dst =
        exec_plan.get_sequential_exec_path_in_edges_ptr()->GetPtr()[2 * i + 1];

    if ((pre_v_idx == kMaxVertexID) ^ (u_src == kMaxVertexID)) continue;

    //if (!LabelFilter(u_dst, v_idx, p, g)) {
    //  __sync_fetch_and_add(&label_filter_count, 1);
    //  continue;
    //}

    if (level == 1) {
      if (!MatrixFilter(u_src, pre_v_idx, p, g, m_vec, m_unified_buffer_vec,
                        rejected_pairs)) {
        __sync_fetch_and_add(&gnn_filter_count, 1);
        continue;
      }
    }

    if (IsFeasible(p, g, m_vec, m_unified_buffer_vec, u_src, u_dst, pre_v_idx,
                   v_idx, local_matches, rejected_pairs)) {
      VertexID offset = local_matches->size[i];
      if (offset >= kMaxNumLocalWeft) {
        return;
      }

      uint64_t pair_key = (static_cast<uint64_t>(pre_v_idx) << 32) |
                          static_cast<uint64_t>(v_idx);
      if (matches_visited_pairs[i].count(pair_key)) {
        return;
      }
      matches_visited_pairs[i].insert(pair_key);
      extend_tag = true;

      local_matches->size[i]++;
      if (pre_v_idx != kMaxVertexID) {
        local_matches->data[kMaxNumLocalWeft * 2 * i + 2 * offset] =
            g.GetGloablIDBasePointer()[pre_v_idx];
      }
      local_matches->data[kMaxNumLocalWeft * 2 * i + 2 * offset + 1] =
          g.GetGloablIDBasePointer()[v_idx];
    }
  }

  if (extend_tag && level < exec_plan.get_depth()) {
    auto v = g.GetVertexByLocalID(v_idx);

    for (VertexID nbr_idx = 0; nbr_idx < v.outdegree; nbr_idx++) {
      VertexID neighbor = v.outgoing_edges[nbr_idx];

      DFSExtend(p, g, exec_plan, m_vec, m_unified_buffer_vec, level + 1, v_idx,
                neighbor, matches_visited_pairs, local_matches, match,
                rejected_pairs);
    }
  }

  if (!extend_tag) return;
}

static inline void Enumerating(
    const ImmutableCSR& p, const ImmutableCSR& g,
    const ExecutionPlan& exec_plan, const std::vector<Matrix>& m_vec,
    const std::vector<UnifiedOwnedBufferFloat*>& m_unified_buffer_vec,
    Matches* matches) {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  std::vector<LocalMatches> local_matches_vec;
  local_matches_vec.resize(parallelism);
  std::generate(
      local_matches_vec.begin(), local_matches_vec.end(), [&p, &exec_plan]() {
        LocalMatches local_matches;
        local_matches.data =
            new VertexID[exec_plan.get_n_edges() * 2 * kMaxNumLocalWeft]();
        local_matches.size =
            new VertexID[exec_plan.get_n_edges() + kSubIsoLocalMatchesSizeBuffer]();
        return local_matches;
      });

  std::vector<std::vector<std::unordered_set<uint64_t>>>
      matches_visited_pairs_vec;
  matches_visited_pairs_vec.resize(parallelism);
  std::generate(matches_visited_pairs_vec.begin(),
                matches_visited_pairs_vec.end(), [&exec_plan]() {
                  return std::vector<std::unordered_set<uint64_t>>(
                      exec_plan.get_n_edges());
                });

  std::vector<std::vector<uint64_t>> rejected_pairs_vec;
  if (!g_reject_output_path.empty()) {
    rejected_pairs_vec.resize(parallelism);
  }

  std::cout << "Enumerating" << std::endl;
  ParForEach(worker.begin(), worker.end(),
      [step, &mtx, &p, &g, &exec_plan, &m_vec, &m_unified_buffer_vec, &matches,
       &local_matches_vec, &matches_visited_pairs_vec, &rejected_pairs_vec](
          auto w) {
        auto& matches_visited_pairs = matches_visited_pairs_vec[w];

        auto& local_matches = local_matches_vec[w];
        std::vector<uint64_t>* thread_rejected_pairs =
            rejected_pairs_vec.empty() ? nullptr : &rejected_pairs_vec[w];

        for (VertexID v_idx = w; v_idx < g.get_num_vertices(); v_idx += step) {
          for (auto _ = 0; _ < exec_plan.get_n_edges(); _++) {
            matches_visited_pairs[_].clear();
          }

          bool match = false;
          DFSExtend(p, g, exec_plan, m_vec, m_unified_buffer_vec, 0,
                    kMaxVertexID, v_idx, matches_visited_pairs, &local_matches,
                    match, thread_rejected_pairs);

          {
            bool is_match = true;
            for (int _ = 0; _ < exec_plan.get_n_edges(); _++) {
              if (local_matches.size[_] == 0) {
                is_match = false;
              }
            }

            std::lock_guard<std::mutex> lock(mtx);
            if (is_match) {
              auto weft_idx =
                  __sync_fetch_and_add(matches->GetWeftCountPtr(), 1);

              if (weft_idx >= kMaxNumWeft - 1) return;
              int weft_size = 0;
              for (int _ = 0; _ < exec_plan.get_n_edges(); _++) {
                weft_size += local_matches.size[_];
                matches->GetVCandidateOffsetPtr()[weft_idx *
                                                      (exec_plan.get_n_edges() + 1) +
                                                  _ + 1] =
                    matches->GetVCandidateOffsetPtr()
                        [weft_idx * (exec_plan.get_n_edges() + 1) + _] +
                    local_matches.size[_];

                memcpy(matches->GetDataPtr() + weft_idx *
                                                   exec_plan.get_n_edges() * 2 *
                                                   kMaxNumLocalWeft,
                       local_matches.data,
                       exec_plan.get_n_edges() * 2 * kMaxNumLocalWeft *
                           sizeof(VertexID));
              }
            }
            memset(local_matches.data, 0,
                   sizeof(VertexID) * exec_plan.get_n_edges() * 2 *
                       kMaxNumLocalWeft);
            memset(local_matches.size, 0,
                   sizeof(VertexID) * exec_plan.get_n_edges());
          }
        }
      });

  if (!rejected_pairs_vec.empty()) {
    size_t total = 0;
    for (const auto& v : rejected_pairs_vec) total += v.size();
    g_rejected_pairs.reserve(g_rejected_pairs.size() + total);
    for (auto& v : rejected_pairs_vec) {
      g_rejected_pairs.insert(g_rejected_pairs.end(), v.begin(), v.end());
    }
  }
}

static bool ValidateWeft(const ImmutableCSR& p, const ImmutableCSR& g,
                         const ExecutionPlan& exec_plan, Matches* matches,
                         VertexID weft_id,
                         size_t max_nodes = kSubIsoMaxBacktrackNodes,
                         size_t max_ms_per_weft = kSubIsoMaxMsPerWeft,
                         std::vector<VertexID>* out_mapping = nullptr) {
  auto n_pattern_vertices = p.get_num_vertices();
  auto n_edges = matches->get_n_vertices();

  // Step 1: Copy valid edge candidates into local structure.
  std::vector<std::vector<std::pair<VertexID, VertexID>>> edge_cands(n_edges);
  for (VertexID e = 0; e < n_edges; ++e) {
    auto offset =
        matches->GetVCandidateOffsetPtr()[weft_id * (n_edges + 1) + e];
    auto size = matches->GetVCandidateOffsetPtr()[weft_id * (n_edges + 1) +
                                                  e + 1] -
                offset;
    edge_cands[e].reserve(size);
    for (VertexID c = 0; c < size; ++c) {
      auto idx = weft_id * n_edges * 2 * matches->get_max_n_local_weft() +
                 e * 2 * matches->get_max_n_local_weft() + 2 * c;
      auto src = matches->get_matches_data_ptr()[idx];
      auto dst = matches->get_matches_data_ptr()[idx + 1];
      if (src != kMaxVertexID && dst != kMaxVertexID &&
          src <= g.get_max_vid() && dst <= g.get_max_vid()) {
        edge_cands[e].emplace_back(src, dst);
      }
    }
    if (edge_cands[e].empty()) return false;
  }

  auto weft_start_time = std::chrono::steady_clock::now();
  auto time_exceeded = [&]() -> bool {
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - weft_start_time).count();
    return ms > (long long)max_ms_per_weft;
  };

  // Step 2: Build initial vertex candidates by intersection.
  std::vector<std::unordered_set<VertexID>> vertex_cands(n_pattern_vertices);
  for (VertexID u_local = 0; u_local < n_pattern_vertices; ++u_local) {
    if (time_exceeded()) return false;
    VertexID u_global = p.GetGlobalIDByLocalID(u_local);
    bool first_edge = true;
    for (VertexID e = 0; e < n_edges; ++e) {
      auto h = matches->GetHeader()[e];
      bool is_first = (h.first == u_global);
      bool is_second = (h.second == u_global);
      if (!is_first && !is_second) continue;

      std::unordered_set<VertexID> ecands;
      ecands.reserve(edge_cands[e].size());
      for (const auto& pr : edge_cands[e]) {
        if (is_first) ecands.insert(pr.first);
        if (is_second) ecands.insert(pr.second);
      }

      if (first_edge) {
        vertex_cands[u_local] = std::move(ecands);
        first_edge = false;
      } else {
        std::unordered_set<VertexID> new_set;
        for (auto v : vertex_cands[u_local]) {
          if (ecands.count(v)) new_set.insert(v);
        }
        vertex_cands[u_local] = std::move(new_set);
      }
      if (vertex_cands[u_local].empty()) return false;
    }
  }

  // Step 3: Arc consistency propagation.
  bool changed = true;
  while (changed) {
    if (time_exceeded()) return false;
    changed = false;
    // Filter edge candidates by current vertex candidates.
    for (VertexID e = 0; e < n_edges; ++e) {
      auto h = matches->GetHeader()[e];
      VertexID u_src_local = (h.first == kMaxVertexID) ? kMaxVertexID : p.GetLocalIDByGlobalID(h.first);
      VertexID u_dst_local = (h.second == kMaxVertexID) ? kMaxVertexID : p.GetLocalIDByGlobalID(h.second);
      auto& ecands = edge_cands[e];
      size_t write_idx = 0;
      for (size_t i = 0; i < ecands.size(); ++i) {
        bool src_ok = (u_src_local == kMaxVertexID) ||
                      vertex_cands[u_src_local].count(ecands[i].first);
        bool dst_ok = (u_dst_local == kMaxVertexID) ||
                      vertex_cands[u_dst_local].count(ecands[i].second);
        if (src_ok && dst_ok) {
          ecands[write_idx++] = ecands[i];
        } else {
          changed = true;
        }
      }
      ecands.resize(write_idx);
      if (ecands.empty()) return false;
    }

    if (!changed) break;

    // Recompute vertex candidates from filtered edge candidates.
    for (VertexID u_local = 0; u_local < n_pattern_vertices; ++u_local) {
      if (time_exceeded()) return false;
      VertexID u_global = p.GetGlobalIDByLocalID(u_local);
      bool first_edge = true;
      for (VertexID e = 0; e < n_edges; ++e) {
        auto h = matches->GetHeader()[e];
        bool is_first = (h.first == u_global);
        bool is_second = (h.second == u_global);
        if (!is_first && !is_second) continue;

        std::unordered_set<VertexID> ecands;
        ecands.reserve(edge_cands[e].size());
        for (const auto& pr : edge_cands[e]) {
          if (is_first) ecands.insert(pr.first);
          if (is_second) ecands.insert(pr.second);
        }

        if (first_edge) {
          vertex_cands[u_local] = std::move(ecands);
          first_edge = false;
        } else {
          std::unordered_set<VertexID> new_set;
          for (auto v : vertex_cands[u_local]) {
            if (ecands.count(v)) new_set.insert(v);
          }
          vertex_cands[u_local] = std::move(new_set);
        }
        if (vertex_cands[u_local].empty()) return false;
      }
    }
  }

  // Step 4: Backtracking search on reduced candidates.
  std::vector<VertexID> order(n_pattern_vertices);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&vertex_cands](VertexID a, VertexID b) {
              return vertex_cands[a].size() < vertex_cands[b].size();
            });

  std::vector<VertexID> mapping(n_pattern_vertices, kMaxVertexID);
  std::vector<bool> used(g.get_num_vertices(), false);

  // Convert to sorted vectors for iteration.
  std::vector<std::vector<VertexID>> cand_vec(n_pattern_vertices);
  for (VertexID i = 0; i < n_pattern_vertices; ++i) {
    cand_vec[i].assign(vertex_cands[i].begin(), vertex_cands[i].end());
  }

  size_t nodes_visited = 0;
  auto dfs_start_time = std::chrono::steady_clock::now();
  std::function<bool(VertexID)> dfs = [&](VertexID depth) -> bool {
    if (++nodes_visited > max_nodes) return false;
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - dfs_start_time).count();
    if (elapsed_ms > (long long)max_ms_per_weft) return false;
    if (depth == n_pattern_vertices) {
      bool ok = true;
      for (VertexID u_local = 0; u_local < n_pattern_vertices; ++u_local) {
        auto u = p.GetVertexByLocalID(u_local);
        for (VertexID nbr_idx = 0; nbr_idx < u.outdegree; ++nbr_idx) {
          VertexID nbr_local = u.outgoing_edges[nbr_idx];
          if (u_local >= nbr_local) continue;
          VertexID v_u_local = g.GetLocalIDByGlobalID(mapping[u_local]);
          VertexID v_nbr_local = g.GetLocalIDByGlobalID(mapping[nbr_local]);
          if (v_u_local >= g.get_num_vertices() || v_nbr_local >= g.get_num_vertices()) {
            ok = false;
            break;
          }
          if (!g.IsConnected(v_u_local, v_nbr_local)) {
            ok = false;
            break;
          }
        }
        if (!ok) break;
      }
      if (ok && out_mapping) *out_mapping = mapping;
      return ok;
    }

    VertexID u_local = order[depth];
    VertexID u_global = p.GetGlobalIDByLocalID(u_local);
    for (VertexID v_global : cand_vec[u_local]) {
      if (v_global == kMaxVertexID) continue;
      VertexID v_local = g.GetLocalIDByGlobalID(v_global);
      if (v_local >= g.get_num_vertices()) {
        RecordRejectedPair(u_global, v_global);
        continue;
      }
      if (used[v_local]) {
        RecordRejectedPair(u_global, v_global);
        continue;
      }

      bool valid = true;
      auto u = p.GetVertexByLocalID(u_local);
      for (VertexID nbr_idx = 0; nbr_idx < u.outdegree; ++nbr_idx) {
        VertexID nbr_local = u.outgoing_edges[nbr_idx];
        bool nbr_mapped = false;
        VertexID v_nbr_local = kMaxVertexID;
        for (VertexID d = 0; d < depth; ++d) {
          if (order[d] == nbr_local) {
            nbr_mapped = true;
            v_nbr_local = g.GetLocalIDByGlobalID(mapping[nbr_local]);
            break;
          }
        }
        if (!nbr_mapped) continue;
        if (v_nbr_local >= g.get_num_vertices()) {
          valid = false;
          break;
        }
        if (!g.IsConnected(v_local, v_nbr_local)) {
          valid = false;
          break;
        }
      }
      if (!valid) {
        RecordRejectedPair(u_global, v_global);
        continue;
      }

      mapping[u_local] = v_global;
      used[v_local] = true;
      if (dfs(depth + 1)) return true;
      used[v_local] = false;
      mapping[u_local] = kMaxVertexID;
    }
    return false;
  };

  return dfs(0);
}

// After a weft is validated, replace its coarse candidate supersets with the
// exact edge mapping found by ValidateWeft. This makes the stored candidates
// correspond one-to-one to the pattern edges under the discovered isomorphism.
static void RewriteWeftWithMapping(const ImmutableCSR& p, Matches* matches,
                                   VertexID weft_id,
                                   const std::vector<VertexID>& mapping) {
  auto n_edges = matches->get_n_vertices();
  auto max_n_local_weft = matches->get_max_n_local_weft();
  auto data_ptr = matches->get_matches_data_ptr();
  auto offset_ptr = matches->GetVCandidateOffsetPtr();
  auto header = matches->GetHeader();

  for (VertexID i = 0; i < n_edges; i++) {
    auto h = header[i];
    VertexID u_src_local = (h.first == kMaxVertexID)
                               ? kMaxVertexID
                               : p.GetLocalIDByGlobalID(h.first);
    VertexID u_dst_local = (h.second == kMaxVertexID)
                               ? kMaxVertexID
                               : p.GetLocalIDByGlobalID(h.second);

    VertexID src = 0;
    VertexID dst = 0;
    if (u_src_local == kMaxVertexID) {
      // Root edge convention: src is unused, dst is the mapped root vertex.
      src = 0;
      dst = mapping[u_dst_local];
    } else {
      src = mapping[u_src_local];
      dst = mapping[u_dst_local];
    }

    size_t base = static_cast<size_t>(weft_id) * n_edges * 2 *
                      max_n_local_weft +
                  i * 2 * max_n_local_weft;
    data_ptr[base] = src;
    data_ptr[base + 1] = dst;

    // Clear any extra candidates that may have been stored for this edge.
    for (VertexID c = 1; c < max_n_local_weft; c++) {
      data_ptr[base + 2 * c] = kMaxVertexID;
      data_ptr[base + 2 * c + 1] = kMaxVertexID;
    }

    offset_ptr[weft_id * (n_edges + 1) + i] = i;
  }
  offset_ptr[weft_id * (n_edges + 1) + n_edges] = n_edges;
}

static void ValidateMatching(const ImmutableCSR& p, const ImmutableCSR& g,
                             const ExecutionPlan& exec_plan, Matches* matches,
                             VertexID max_wefts =
                                 std::numeric_limits<VertexID>::max()) {
  std::cout << "\tValidateMatching (max_wefts=" << max_wefts << ") ..."
            << std::endl;
  VertexID invalid_count = 0;
  VertexID checked_count = 0;
  VertexID total = matches->get_weft_count();
  VertexID limit = std::min(total, max_wefts);
  auto global_start = std::chrono::steady_clock::now();
  for (VertexID weft_id = 0; weft_id < limit; weft_id++) {
    if (matches->get_invalid_match_ptr()->GetBit(weft_id)) continue;
    checked_count++;
    std::vector<VertexID> mapping;
    if (!ValidateWeft(p, g, exec_plan, matches, weft_id,
                      kSubIsoMaxBacktrackNodes, kSubIsoMaxMsPerWeft,
                      &mapping)) {
      matches->get_invalid_match_ptr()->SetBit(weft_id);
      invalid_count++;
    } else {
      RewriteWeftWithMapping(p, matches, weft_id, mapping);
    }
    auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - global_start).count();
    if (elapsed_sec > kSubIsoValidateMatchingTimeoutSec) {
      std::cout << "\tValidateMatching timed out after "
                << kSubIsoValidateMatchingTimeoutSec << " sec at weft "
                << weft_id << "." << std::endl;
      break;
    }
    if ((checked_count) % kSubIsoProgressPrintInterval == 0 ||
        weft_id + 1 == limit) {
      std::cout << "\t  Progress: " << checked_count << "/" << limit
                << " checked, " << invalid_count << " invalid so far."
                << std::endl;
    }
  }
  std::cout << "\tValidateMatching done. Checked " << checked_count
            << " wefts, invalidated " << invalid_count << "." << std::endl;
}

static void Checking(const ImmutableCSR& p, const ImmutableCSR& g,
                     const ExecutionPlan& exec_plan, Matches* matches) {
  return;
  std::cout << "\tChecking ..." << std::endl;
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;

  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  std::queue<VertexID> frontier;
  auto root = p.GetVertexByLocalID(0);
  frontier.push(root.vid);
  BitmapOwnership edges_visited(p.get_num_vertices() * p.get_num_vertices());

  auto header = matches->GetHeader();

  for (VertexID weft_id = 0; weft_id < matches->get_weft_count(); weft_id++) {
    if (matches->get_invalid_match_ptr()->GetBit(weft_id)) continue;

    VertexID delete_count = 0;
    bool fix_point = false;
    while (!fix_point) {
      for (auto i = 0; i < matches->get_n_vertices(); i++) {
        auto v_candidate_offset =
            matches->GetVCandidateOffsetPtr()
                [weft_id * (matches->get_n_vertices() + 1) + i];
        auto v_candidate_size =
            matches->GetVCandidateOffsetPtr()
                [weft_id * (matches->get_n_vertices() + 1) + i + 1] -
            matches->GetVCandidateOffsetPtr()
                [weft_id * (matches->get_n_vertices() + 1) + i];

        for (VertexID candidate_id = 0; candidate_id < v_candidate_size;
             candidate_id++) {
          if (*(matches->get_matches_data_ptr() +
                weft_id * matches->get_n_vertices() * 2 *
                    matches->get_max_n_local_weft() +
                i * 2 * matches->get_max_n_local_weft() + 2 * candidate_id) !=
                  kMaxVertexID &&
              *(matches->get_matches_data_ptr() +
                weft_id * matches->get_n_vertices() * 2 *
                    matches->get_max_n_local_weft() +
                i * 2 * matches->get_max_n_local_weft() + 2 * candidate_id +
                1) != kMaxVertexID) {
            for (auto j = 1; j < matches->get_n_vertices(); j++) {
              if (i == j) continue;
              if (!matches->IsValidCandidate(
                      weft_id, j, matches->get_header_first_by_idx(i),
                      *(matches->get_matches_data_ptr() +
                        weft_id * matches->get_n_vertices() * 2 *
                            matches->get_max_n_local_weft() +
                        i * 2 * matches->get_max_n_local_weft() +
                        2 * candidate_id))) {
                *(matches->get_matches_data_ptr() +
                  weft_id * matches->get_n_vertices() * 2 *
                      matches->get_max_n_local_weft() +
                  i * 2 * matches->get_max_n_local_weft() + 2 * candidate_id) =
                    kMaxVertexID;
                *(matches->get_matches_data_ptr() +
                  weft_id * matches->get_n_vertices() * 2 *
                      matches->get_max_n_local_weft() +
                  i * 2 * matches->get_max_n_local_weft() + 2 * candidate_id +
                  1) = kMaxVertexID;
                matches->GetVDeletedCandidatesCountPtr()
                    [weft_id * (matches->get_n_vertices() + 1) + i]++;
                delete_count++;
              }
              if (!matches->IsValidCandidate(
                      weft_id, j, matches->get_header_second_by_idx(i),
                      *(matches->get_matches_data_ptr() +
                        weft_id * matches->get_n_vertices() * 2 *
                            matches->get_max_n_local_weft() +
                        i * 2 * matches->get_max_n_local_weft() +
                        2 * candidate_id + 1))) {
                *(matches->get_matches_data_ptr() +
                  weft_id * matches->get_n_vertices() * 2 *
                      matches->get_max_n_local_weft() +
                  i * 2 * matches->get_max_n_local_weft() + 2 * candidate_id +
                  1) = kMaxVertexID;
                *(matches->get_matches_data_ptr() +
                  weft_id * matches->get_n_vertices() * 2 *
                      matches->get_max_n_local_weft() +
                  i * 2 * matches->get_max_n_local_weft() + 2 * candidate_id) =
                    kMaxVertexID;
                matches->GetVDeletedCandidatesCountPtr()
                    [weft_id * (matches->get_n_vertices() + 1) + i]++;
                delete_count++;
              }
            }
          }
        }
      }
      if (delete_count == 0) {
        fix_point = true;
      } else {
        delete_count = 0;
        fix_point = false;
      }
    }
  }
}

void CPUSubIso::RecursiveMatching(
    const ImmutableCSR& p, const ImmutableCSR& g,
    const std::vector<Matrix>& m_vec,
    const std::vector<UnifiedOwnedBufferFloat*>& m_unified_buffer_vec) {
  std::cout << "Matching ..." << std::endl;
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;

  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  // Generate Execution Plan...
  ExecutionPlan exec_plan;
  exec_plan.GenerateDFSExecutionPlan(p, g);

  exec_plan.Print();

  Matches matches(exec_plan.get_n_edges(), kMaxNumWeft, kMaxNumLocalWeft,
                  g.get_num_vertices());

  // Set header of matches
  for (auto _ = 0; _ < exec_plan.get_n_edges(); _++) {
    auto src =
        exec_plan.get_sequential_exec_path_in_edges_ptr()->GetPtr()[_ * 2];
    auto dst =
        exec_plan.get_sequential_exec_path_in_edges_ptr()->GetPtr()[_ * 2 + 1];
    matches.SetHeader(_, std::make_pair(src, dst));
  }

  auto start_time_0 = std::chrono::system_clock::now();
  // Enumerating ...
  Enumerating(p, g, exec_plan, m_vec, m_unified_buffer_vec, &matches);

  auto start_time_1 = std::chrono::system_clock::now();

  auto start_time_2 = std::chrono::system_clock::now();

  // Checking ...
  const char* validate_env = std::getenv("MG_VALIDATE_ALL_WEFTS");
  VertexID max_validate_wefts =
      (validate_env && std::string(validate_env) == "1")
          ? std::numeric_limits<VertexID>::max()
          : kSubIsoMaxValidateWefts;
  ValidateMatching(p, g, exec_plan, &matches, max_validate_wefts);
  matches.UpdateInvalidMatches();

  auto start_time_3 = std::chrono::system_clock::now();
  matches.Print(1);
  std::cout << " N Matches: " << matches.ComputeNMatches() << std::endl;

  std::cout << "[RecursiveMatching] Enumerating() elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_1 - start_time_0)
                       .count() /
                   (float)CLOCKS_PER_SEC
            << " sec" << std::endl;

  std::cout << "[RecursiveMatching] Checking() elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_3 - start_time_2)
                       .count() /
                   (float)CLOCKS_PER_SEC
            << " sec" << std::endl;
  if (output_path_ != "") matches.Write(output_path_);
}

void CPUSubIso::WOJMatching(
    const ImmutableCSR& p, const ImmutableCSR& g,
    const std::vector<Matrix>& m_vec,
    const std::vector<UnifiedOwnedBufferFloat*>& m_unified_buffer_vec) {
  WOJExecutionPlan exec_plan;
  exec_plan.GenerateWOJExecutionPlan(p, g);

  auto start_time_0 = std::chrono::system_clock::now();
  auto woj_matches = WOJFilter(exec_plan, p, g, m_vec, m_unified_buffer_vec);

  // for (auto iter : woj_matches) {
  //   iter->Print();
  // }
  auto start_time_1 = std::chrono::system_clock::now();

  auto output = WOJEnumerating(exec_plan, woj_matches);
  auto start_time_2 = std::chrono::system_clock::now();

  output->Print(10);
  std::cout << "[WOJMatching] Filter() elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_1 - start_time_0)
                       .count() /
                   (float)CLOCKS_PER_SEC
            << " sec" << std::endl;
  std::cout << "[WOJMatching] Join() elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_2 - start_time_1)
                       .count() /
                   (float)CLOCKS_PER_SEC
            << " sec" << std::endl;
}

void CPUSubIso::LoadData() {
  std::cout << "[CPUSubIso] LoadData() ..." << std::endl;

  p_.Read(pattern_path_);

  g_.Read(data_graph_path_);

  std::cout << "[CPUSubIso] Building filter caches ..." << std::endl;
  BuildFilterCache(p_, p_filter_cache);
  BuildFilterCache(g_, g_filter_cache);

  auto* g_vlabel = g_.GetVLabelBasePointer();
  auto* p_vlabel = p_.GetVLabelBasePointer();

  // p_.PrintGraph(100);
  // g_.PrintGraph(100);

  if (matrix_path1_ != "" && matrix_path2_ != "" && matrix_path3_ != "" &&
      matrix_path4_ != "" && matrix_path5_ != "" && matrix_path6_ != "") {
    m_vec_.resize(6);
    m_unified_buffer_vec_.resize(6);
    std::generate(m_unified_buffer_vec_.begin(), m_unified_buffer_vec_.end(),
                  []() { return new UnifiedOwnedBufferFloat(); });

    m_vec_[0].Read(matrix_path1_);
    m_vec_[1].Read(matrix_path2_);
    m_vec_[2].Read(matrix_path3_);
    m_vec_[3].Read(matrix_path4_);
    m_vec_[4].Read(matrix_path5_);
    m_vec_[5].Read(matrix_path6_);
    // m_vec_[2].Print(99);
    // m_vec_[3].Print(99);
    // m_vec_[4].Print(99);
    // m_vec_[5].Print(99);

    BufferFloat buffer_m1;
    BufferFloat buffer_m2;
    BufferFloat buffer_m3;
    BufferFloat buffer_m4;
    BufferFloat buffer_m5;
    BufferFloat buffer_m6;
    buffer_m1.data = m_vec_[0].GetPtr();
    buffer_m2.data = m_vec_[1].GetPtr();
    buffer_m3.data = m_vec_[2].GetPtr();
    buffer_m4.data = m_vec_[3].GetPtr();
    buffer_m5.data = m_vec_[4].GetPtr();
    buffer_m6.data = m_vec_[5].GetPtr();
    buffer_m1.size = sizeof(float) * m_vec_[0].get_x() * m_vec_[0].get_y();
    buffer_m2.size = sizeof(float) * m_vec_[1].get_x() * m_vec_[1].get_y();
    buffer_m3.size = sizeof(float) * m_vec_[2].get_x() * m_vec_[2].get_y();
    buffer_m4.size = sizeof(float) * m_vec_[3].get_x() * m_vec_[3].get_y();
    buffer_m5.size = sizeof(float) * m_vec_[4].get_x() * m_vec_[4].get_y();
    buffer_m6.size = sizeof(float) * m_vec_[5].get_x() * m_vec_[5].get_y();

    m_unified_buffer_vec_[0]->Init(buffer_m1);
    m_unified_buffer_vec_[1]->Init(buffer_m2);
    m_unified_buffer_vec_[2]->Init(buffer_m3);
    m_unified_buffer_vec_[3]->Init(buffer_m4);
    m_unified_buffer_vec_[4]->Init(buffer_m5);
    m_unified_buffer_vec_[5]->Init(buffer_m6);
  }
}

void CPUSubIso::Run() {
  auto start_time_0 = std::chrono::system_clock::now();
  LoadData();
  auto start_time_1 = std::chrono::system_clock::now();

  g_reject_output_path = reject_output_path_;
  g_rejected_pairs.clear();

  // WOJMatching(p_, g_, m_vec_);
  RecursiveMatching(p_, g_, m_vec_, m_unified_buffer_vec_);

  WriteRejectedPairs();
  g_reject_output_path.clear();

  std::cout << "=== Filter Counts ===" << std::endl;
  std::cout << "Total Filters:      " << filter_count << std::endl;
  std::cout << "Label Filters:      " << label_filter_count << std::endl;
  std::cout << "Label Degree Filters:      " << label_degree_filter_count
            << std::endl;
  std::cout << "NLC Filters:        " << nlc_filter_count << std::endl;
  std::cout << "IP Filters:        " << ip_filter_count << std::endl;
  std::cout << "GNN Filters:        " << gnn_filter_count << std::endl;
  std::cout << "Index Filters:        " << index_filter_count << std::endl;
  auto end_time = std::chrono::system_clock::now();

  std::cout << "Data loading time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_1 - start_time_0)
                       .count() /
                   1000000.0
            << " sec" << std::endl;

  std::cout << "Matching time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   end_time - start_time_1)
                       .count() /
                   1000000.0
            << " sec" << std::endl;

  std::cout << "Total execution time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   end_time - start_time_0)
                       .count() /
                   1000000.0
            << " sec" << std::endl;
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
