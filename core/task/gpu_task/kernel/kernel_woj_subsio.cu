#include <cuda_runtime.h>

#include <array>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <thread>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "core/common/consts.h"
#include "core/common/host_algorithms.cuh"
#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/heap.cuh"
#include "core/data_structures/host_buffer.cuh"
#include "core/data_structures/immutable_csr_gpu.cuh"
#include "core/data_structures/kernel_bitmap.cuh"
#include "core/data_structures/kernel_bitmap_no_ownership.cuh"
#include "core/data_structures/mini_kernel_bitmap.cuh"
#include "core/data_structures/unified_buffer.cuh"
#include "core/data_structures/woj_matches.cuh"
#include "core/task/gpu_task/kernel/algorithms/hash.cuh"
#include "core/task/gpu_task/kernel/algorithms/sort.cuh"
#include "core/task/gpu_task/kernel/kernel_woj_subiso.cuh"
#include "core/util/bitmap_ownership.h"
#include "core/util/cuda_check.cuh"
#include "core/util/cuda_device.cuh"
#include "core/util/cuda_prefetch.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
using VertexID = sics::matrixgraph::core::common::VertexID;
using BitmapOwnership = sics::matrixgraph::core::util::BitmapOwnership;
using sics::matrixgraph::core::common::kBlockDim;
using sics::matrixgraph::core::common::kGridDim;
using sics::matrixgraph::core::common::kLogWarpSize;
using sics::matrixgraph::core::common::kMaxMatchTableRows;
using sics::matrixgraph::core::common::kMaxVertexID;
using sics::matrixgraph::core::common::kWarpSize;
using sics::matrixgraph::core::task::kernel::HostKernelBitmap;
using sics::matrixgraph::core::task::kernel::HostMiniKernelBitmap;
using sics::matrixgraph::core::task::kernel::KernelBitmap;
using sics::matrixgraph::core::task::kernel::KernelBitmapNoOwnership;
using sics::matrixgraph::core::task::kernel::MiniKernelBitmap;
using WOJExecutionPlan =
    sics::matrixgraph::core::data_structures::WOJExecutionPlan;
using WOJMatches = sics::matrixgraph::core::data_structures::WOJMatches;
using MinHeap = sics::matrixgraph::core::task::kernel::MinHeap;
using BufferUint8 = sics::matrixgraph::core::data_structures::Buffer<uint8_t>;
using BufferUint32 = sics::matrixgraph::core::data_structures::Buffer<uint32_t>;
using BufferUint64 = sics::matrixgraph::core::data_structures::Buffer<uint64_t>;
using BufferVertexID =
    sics::matrixgraph::core::data_structures::Buffer<VertexID>;
using UnifiedOwnedBufferEdgeIndex =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<EdgeIndex>;
using UnifiedOwnedBufferVertexID =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexID>;
using UnifiedOwnedBufferVertexLabel =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexLabel>;
using UnifiedOwnedBufferUint8 =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint8_t>;
using UnifiedOwnedBufferUint64 =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint64_t>;
using BufferVertexLabel =
    sics::matrixgraph::core::data_structures::Buffer<VertexLabel>;
using BufferVertexID =
    sics::matrixgraph::core::data_structures::Buffer<VertexID>;

namespace {

uint32_t WojLaunchTotalThreads() {
  if (const char* s = std::getenv("MG_SUBISO_TOTAL_THREADS")) {
    int v = std::atoi(s);
    if (v > 0) {
      return static_cast<uint32_t>(v);
    }
  }
  return kGridDim * kBlockDim;
}

uint32_t WojLaunchBlockDim() {
  if (const char* s = std::getenv("MG_SUBISO_BLOCK")) {
    int v = std::atoi(s);
    if (v > 0) {
      return static_cast<uint32_t>(v);
    }
  }
  // Default: fixed block size 128, grid derived from total threads.
  return 128;
}

uint32_t WojLaunchGridDim() {
  if (const char* s = std::getenv("MG_SUBISO_GRID")) {
    int v = std::atoi(s);
    if (v > 0) {
      return static_cast<uint32_t>(v);
    }
  }
  // Default: fixed block size 128, grid derived from total threads.
  uint32_t total = WojLaunchTotalThreads();
  uint32_t block = WojLaunchBlockDim();
  return (total + block - 1) / block;
}

// Host-side stripe parallelism for WOJ Filter/Join (std::thread count). Default 4.
size_t WojHostStripeThreads() {
  if (const char* e = std::getenv("MG_WOJ_STRIPE_THREADS")) {
    if (e[0] != '\0') {
      int v = std::atoi(e);
      if (v > 0) {
        return static_cast<size_t>(v);
      }
    }
  }
  return 4;
}

// Join: number of row partitions of table[0]. Defaults to 1 per GPU; increase to
// overlap multiple left-deep pipelines on the same device (MG_WOJ_JOIN_STRIPES_PER_GPU).
size_t WojJoinStripesPerGpu() {
  if (const char* e = std::getenv("MG_WOJ_JOIN_STRIPES_PER_GPU")) {
    if (e[0] != '\0') {
      int v = std::atoi(e);
      if (v > 0) {
        return static_cast<size_t>(v);
      }
    }
  }
  return 1;
}

// Host threads that actually run join stripes (caps oversubscription; override with
// MG_WOJ_JOIN_MAX_THREADS).
size_t WojJoinHostWorkerCount(size_t n_join_stripes) {
  size_t hw = std::thread::hardware_concurrency();
  if (hw == 0) {
    hw = WojHostStripeThreads();
  }
  size_t cap = std::max(hw, WojHostStripeThreads());
  size_t workers = std::min(n_join_stripes, cap);
  if (const char* e = std::getenv("MG_WOJ_JOIN_MAX_THREADS")) {
    if (e[0] != '\0') {
      int v = std::atoi(e);
      if (v > 0) {
        workers = std::min(workers, static_cast<size_t>(v));
      }
    }
  }
  return std::max<size_t>(1, workers);
}

// ParForEach is serial under __CUDACC__ (see execution_policy.h). Use explicit
// host threads so WOJ Filter/Join can drive multiple GPUs concurrently.
template <typename F>
void RunHostStripeParallel(size_t num_stripes, F&& f) {
  const size_t n = std::max<size_t>(1, num_stripes);
  if (n <= 1) {
    f(static_cast<size_t>(0));
    return;
  }
  std::vector<std::thread> threads;
  threads.reserve(n - 1);
  for (size_t w = 1; w < n; ++w) {
    threads.emplace_back([w, &f]() { f(w); });
  }
  f(0);
  for (auto& t : threads) {
    t.join();
  }
}

}  // namespace

struct LocalMatches {
  VertexID* data = nullptr;
  VertexID* size = nullptr;
};

struct ParametersFilter {
  VertexID u_eid;
  VertexID* exec_path_in_edges = nullptr;
  VertexID n_vertices_p;
  EdgeIndex n_edges_p;
  uint8_t* data_p;
  VertexLabel* v_label_p = nullptr;
  VertexID n_vertices_g;
  EdgeIndex n_edges_g;
  uint8_t* data_g = nullptr;
  VertexID* edgelist_g = nullptr;
  VertexLabel* v_label_g = nullptr;
  uint64_t* bloom_signature_p = nullptr;
  uint64_t* bloom_signature_g = nullptr;
  bool enable_bloom_filter = false;
  bool enable_min_wise_filter = false;
  bool enable_min_wise_bloom_filter = false;
  bool enable_label_degree_filter = false;
  bool enable_nlc_filter = false;
  bool enable_lpf_filter = false;
  int filter_order = 0;
  WOJMatches woj_matches;
  uint64_t* test;
};

struct ParametersWedgeFilter {
  VertexID n_vertices_g;
  WOJMatches woj_matches;
  VertexID hash_idx;
  uint64_t* visited_data;
};

struct ParametersJoin {
  VertexID n_vertices_g;
  WOJMatches left_woj_matches;
  WOJMatches right_woj_matches;
  WOJMatches output_woj_matches;
  VertexID left_hash_idx;
  VertexID right_hash_idx;
  uint64_t* right_visited_data;
  uint64_t* jump_visited_data;
  VertexID* jump_count;
};

struct ParametersHashJoin {
  VertexID left_y;
  VertexID left_x;
  VertexID left_x_offset;
  VertexID right_x;
  VertexID right_x_offset;
  VertexID output_x;
  VertexID output_max_y;
  VertexID left_hash_idx;
  VertexID right_hash_idx;
  VertexID* left_data;
  VertexID* right_data;
  VertexID* output_data;
  const VertexID* key_start;
  const VertexID* key_count;
  const VertexID* match_offset;
};

struct SatAdd {
  VertexID cap;
  __host__ __device__ SatAdd(VertexID c) : cap(c) {}
  __host__ __device__ VertexID operator()(VertexID a, VertexID b) const {
    VertexID sum = a + b;
    return sum > cap || sum < a ? cap : sum;
  }
};

static __forceinline__ __device__ bool BloomLabelSetFilter(
    const ParametersFilter& params, VertexID u_idx, VertexID v_idx) {
  if (!params.enable_bloom_filter) return true;

  VertexID* globalid_p = (VertexID*)(params.data_p);
  VertexID* in_degree_p = globalid_p + params.n_vertices_p;
  VertexID* out_degree_p = in_degree_p + params.n_vertices_p;
  EdgeIndex* in_offset_p = (EdgeIndex*)(out_degree_p + params.n_vertices_p);
  EdgeIndex* out_offset_p = (EdgeIndex*)(in_offset_p + params.n_vertices_p + 1);
  EdgeIndex* in_edges_p = (EdgeIndex*)(out_offset_p + params.n_vertices_p + 1);
  VertexID* out_edges_p = in_edges_p + params.n_edges_p;

  uint64_t u_mask = 0;
  EdgeIndex u_offset_base = out_offset_p[u_idx];
  for (VertexID nbr_u_idx = 0; nbr_u_idx < out_degree_p[u_idx]; nbr_u_idx++) {
    VertexID nbr_u = out_edges_p[u_offset_base + nbr_u_idx];
    VertexLabel u_label = params.v_label_p[nbr_u];
    if (u_label < 64) u_mask |= (1ULL << u_label);
  }
  u_offset_base = in_offset_p[u_idx];
  for (VertexID nbr_u_idx = 0; nbr_u_idx < in_degree_p[u_idx]; nbr_u_idx++) {
    VertexID nbr_u = in_edges_p[u_offset_base + nbr_u_idx];
    VertexLabel u_label = params.v_label_p[nbr_u];
    if (u_label < 64) u_mask |= (1ULL << u_label);
  }

  return (params.bloom_signature_g[v_idx] & u_mask) == u_mask;
}

static __forceinline__ __device__ bool LabelFilter(
    const ParametersFilter& params, VertexID u_idx, VertexID v_idx) {
  VertexID* globalid_g = (VertexID*)(params.data_g);
  VertexLabel v_label = params.v_label_g[v_idx];
  VertexLabel u_label = params.v_label_p[u_idx];
  return u_label == v_label;
}

static __forceinline__ __device__ bool LabelDegreeFilter(
    const ParametersFilter& params, VertexID u_idx, VertexID v_idx) {
  VertexID* globalid_p = (VertexID*)(params.data_p);
  VertexID* in_degree_p = globalid_p + params.n_vertices_p;
  VertexID* out_degree_p = in_degree_p + params.n_vertices_p;

  VertexID* globalid_g = (VertexID*)(params.data_g);
  VertexID* in_degree_g = globalid_g + params.n_vertices_g;
  VertexID* out_degree_g = in_degree_g + params.n_vertices_g;

  VertexLabel v_label = params.v_label_g[v_idx];
  VertexLabel u_label = params.v_label_p[u_idx];

  return u_label == v_label && out_degree_g[v_idx] >= out_degree_p[u_idx] &&
         in_degree_g[v_idx] >= in_degree_p[u_idx];
}

static __forceinline__ __device__ bool KMinWiseIPFilter(
    const ParametersFilter& params, VertexID u_idx, VertexID v_idx) {
  VertexID* globalid_p = (VertexID*)(params.data_p);
  VertexID* in_degree_p = globalid_p + params.n_vertices_p;
  VertexID* out_degree_p = in_degree_p + params.n_vertices_p;
  EdgeIndex* in_offset_p = (EdgeIndex*)(out_degree_p + params.n_vertices_p);
  EdgeIndex* out_offset_p = (EdgeIndex*)(in_offset_p + params.n_vertices_p + 1);
  EdgeIndex* in_edges_p = (EdgeIndex*)(out_offset_p + params.n_vertices_p + 1);
  VertexID* out_edges_p = in_edges_p + params.n_edges_p;
  VertexID* edges_globalid_by_localid_p = out_edges_p + params.n_edges_p;

  VertexID* globalid_g = (VertexID*)(params.data_g);
  VertexID* in_degree_g = globalid_g + params.n_vertices_g;
  VertexID* out_degree_g = in_degree_g + params.n_vertices_g;
  EdgeIndex* in_offset_g = (EdgeIndex*)(out_degree_g + params.n_vertices_g);
  EdgeIndex* out_offset_g = (EdgeIndex*)(in_offset_g + params.n_vertices_g + 1);
  EdgeIndex* in_edges_g = (EdgeIndex*)(out_offset_g + params.n_vertices_g + 1);
  VertexID* out_edges_g = in_edges_g + params.n_edges_g;
  VertexID* edges_globalid_by_localid_g = out_edges_g + params.n_edges_g;

  VertexLabel v_label = params.v_label_g[v_idx];
  VertexLabel u_label = params.v_label_p[u_idx];

  if (u_label != v_label) return false;

  VertexID max_v_ip_val = 0;
  VertexID min_v_ip_val = kMaxVertexID;
  VertexID max_u_ip_val = 0;
  VertexID min_u_ip_val = kMaxVertexID;

  MiniKernelBitmap u_label_visited(32);
  MiniKernelBitmap v_label_visited(32);

  MinHeap u_k_min_heap;
  MinHeap v_k_min_heap;

  uint32_t u_k_min_heap_data[kDefaultHeapCapacity];
  uint32_t v_k_min_heap_data[kDefaultHeapCapacity];

  // Filter by out edges.
  EdgeIndex u_offset_base = out_offset_p[u_idx];
  for (VertexID nbr_u_idx = 0; nbr_u_idx < out_degree_p[u_idx]; nbr_u_idx++) {
    VertexID nbr_u = out_edges_p[u_offset_base + nbr_u_idx];
    VertexLabel u_label = params.v_label_p[nbr_u];
    VertexID u_ip_val = HashTable(u_label);
    u_label_visited.SetBit(u_label);
    u_k_min_heap.Insert(u_ip_val);
  }

  EdgeIndex v_offset_base = out_offset_g[v_idx];
  for (VertexID nbr_v_idx = 0; nbr_v_idx < out_degree_g[v_idx]; nbr_v_idx++) {
    VertexID nbr_v = out_edges_g[v_offset_base + nbr_v_idx];
    VertexLabel v_label = params.v_label_g[nbr_v];
    VertexID v_ip_val = HashTable(v_label);
    v_label_visited.SetBit(v_label);
    v_k_min_heap.Insert(v_ip_val);
  }

  u_k_min_heap.CopyData(u_k_min_heap_data);
  v_k_min_heap.CopyData(v_k_min_heap_data);

  for (VertexID _ = 0; _ < v_k_min_heap.get_offset(); _++) {
    auto v_ip_val = v_k_min_heap_data[_];
    for (VertexID __ = 0; __ < u_k_min_heap.get_offset(); __++) {
      auto u_ip_val = u_k_min_heap_data[__];
      if (v_ip_val == u_ip_val) {
        v_k_min_heap_data[_] = kMaxVertexID;
        u_k_min_heap_data[__] = kMaxVertexID;
        break;
      }
    }
  }

  for (VertexID _ = 0; _ < v_k_min_heap.get_offset(); _++) {
    auto v_ip_val = v_k_min_heap_data[_];
    min_v_ip_val = min_v_ip_val < v_ip_val ? min_v_ip_val : v_ip_val;
  }

  for (VertexID _ = 0; _ < u_k_min_heap.get_offset(); _++) {
    auto u_ip_val = u_k_min_heap_data[_];
    min_u_ip_val = min_u_ip_val < u_ip_val ? min_u_ip_val : u_ip_val;
  }

  if (min_v_ip_val == kMaxVertexID && min_u_ip_val != kMaxVertexID)
    return false;

  for (VertexID _ = 0; _ < u_k_min_heap.get_offset(); _++) {
    if (u_k_min_heap_data[_] < min_v_ip_val) {
      return false;
    }
  }

  // Filter by in edges.
  // max_v_ip_val = 0;
  // min_v_ip_val = kMaxVertexID;
  // max_u_ip_val = 0;
  // min_u_ip_val = kMaxVertexID;
  // // u_label_visited.Clear();
  // // v_label_visited.Clear();
  // u_k_min_heap.Clear();
  // v_k_min_heap.Clear();

  // u_offset_base = in_offset_p[u_idx];
  // for (VertexID nbr_u_idx = 0; nbr_u_idx < in_degree_p[u_idx]; nbr_u_idx++) {
  //   VertexID nbr_u = in_edges_p[u_offset_base + nbr_u_idx];
  //   VertexLabel u_label = params.v_label_p[nbr_u];
  //   VertexID u_ip_val = HashTable(u_label);
  //   u_label_visited.SetBit(u_label);
  //   u_k_min_heap.Insert(u_ip_val);
  // }

  // v_offset_base = in_offset_g[v_idx];
  // for (VertexID nbr_v_idx = 0; nbr_v_idx < in_degree_g[v_idx]; nbr_v_idx++) {
  //   VertexID nbr_v = in_edges_g[v_offset_base + nbr_v_idx];
  //   VertexLabel v_label = params.v_label_g[nbr_v];
  //   VertexID v_ip_val = HashTable(v_label);
  //   v_label_visited.SetBit(v_label);
  //   v_k_min_heap.Insert(v_ip_val);
  // }

  // u_k_min_heap.CopyData(u_k_min_heap_data);
  // v_k_min_heap.CopyData(v_k_min_heap_data);

  // for (VertexID _ = 0; _ < v_k_min_heap.get_offset(); _++) {
  //   auto v_ip_val = v_k_min_heap_data[_];
  //   for (VertexID __ = 0; __ < u_k_min_heap.get_offset(); __++) {
  //     auto u_ip_val = u_k_min_heap_data[__];
  //     if (v_ip_val == u_ip_val) {
  //       v_k_min_heap_data[_] = kMaxVertexID;
  //       u_k_min_heap_data[__] = kMaxVertexID;
  //       break;
  //     }
  //   }
  // }

  // for (VertexID _ = 0; _ < v_k_min_heap.get_offset(); _++) {
  //   auto v_ip_val = v_k_min_heap_data[_];
  //   min_v_ip_val = min_v_ip_val < v_ip_val ? min_v_ip_val : v_ip_val;
  // }

  // for (VertexID _ = 0; _ < u_k_min_heap.get_offset(); _++) {
  //   auto u_ip_val = u_k_min_heap_data[_];
  //   min_u_ip_val = min_u_ip_val < u_ip_val ? min_u_ip_val : u_ip_val;
  // }

  // if (min_v_ip_val == kMaxVertexID && min_u_ip_val != kMaxVertexID)
  //   return false;

  // for (VertexID _ = 0; _ < u_k_min_heap.get_offset(); _++) {
  //   if (u_k_min_heap_data[_] < min_v_ip_val) {
  //     return false;
  //   }
  // }

  return v_label_visited.Count() >= u_label_visited.Count() &&
         out_degree_g[v_idx] >= out_degree_p[u_idx] &&
         in_degree_g[v_idx] >= in_degree_p[u_idx];
}

static __forceinline__ __device__ bool MinWiseIPFilter(
    const ParametersFilter& params, VertexID u_idx, VertexID v_idx) {
  VertexID* globalid_p = (VertexID*)(params.data_p);
  VertexID* in_degree_p = globalid_p + params.n_vertices_p;
  VertexID* out_degree_p = in_degree_p + params.n_vertices_p;
  EdgeIndex* in_offset_p = (EdgeIndex*)(out_degree_p + params.n_vertices_p);
  EdgeIndex* out_offset_p = (EdgeIndex*)(in_offset_p + params.n_vertices_p + 1);
  EdgeIndex* in_edges_p = (EdgeIndex*)(out_offset_p + params.n_vertices_p + 1);
  VertexID* out_edges_p = in_edges_p + params.n_edges_p;
  VertexID* edges_globalid_by_localid_p = out_edges_p + params.n_edges_p;

  VertexID* globalid_g = (VertexID*)(params.data_g);
  VertexID* in_degree_g = globalid_g + params.n_vertices_g;
  VertexID* out_degree_g = in_degree_g + params.n_vertices_g;
  EdgeIndex* in_offset_g = (EdgeIndex*)(out_degree_g + params.n_vertices_g);
  EdgeIndex* out_offset_g = (EdgeIndex*)(in_offset_g + params.n_vertices_g + 1);
  EdgeIndex* in_edges_g = (EdgeIndex*)(out_offset_g + params.n_vertices_g + 1);
  VertexID* out_edges_g = in_edges_g + params.n_edges_g;
  VertexID* edges_globalid_by_localid_g = out_edges_g + params.n_edges_g;

  VertexLabel v_label = params.v_label_g[v_idx];
  VertexLabel u_label = params.v_label_p[u_idx];

  if (u_label != v_label) return false;

  VertexID in_min_v_ip_val = kMaxVertexID;
  VertexID in_min_u_ip_val = kMaxVertexID;
  VertexID out_min_v_ip_val = kMaxVertexID;
  VertexID out_min_u_ip_val = kMaxVertexID;
  EdgeIndex u_offset_base;
  EdgeIndex v_offset_base;

  // Filter by edges.

  MiniKernelBitmap out_u_label_visited(32);
  MiniKernelBitmap out_v_label_visited(32);
  out_u_label_visited.Clear();
  out_v_label_visited.Clear();
  u_offset_base = out_offset_p[u_idx];
  for (VertexID nbr_u_idx = 0; nbr_u_idx < out_degree_p[u_idx]; nbr_u_idx++) {
    VertexID nbr_u = out_edges_p[u_offset_base + nbr_u_idx];
    VertexLabel u_label = params.v_label_p[nbr_u];
    VertexID u_ip_val = HashTable(u_label);
    out_u_label_visited.SetBit(u_label);
    out_min_u_ip_val < u_ip_val ? out_min_u_ip_val : u_ip_val;
  }

  v_offset_base = out_offset_g[v_idx];
  for (VertexID nbr_v_idx = 0; nbr_v_idx < out_degree_g[v_idx]; nbr_v_idx++) {
    VertexID nbr_v = out_edges_g[v_offset_base + nbr_v_idx];
    VertexLabel v_label = params.v_label_g[nbr_v];
    VertexID v_ip_val = HashTable(v_label);
    out_min_v_ip_val < v_ip_val ? out_min_v_ip_val : v_ip_val;
    out_v_label_visited.SetBit(v_label);
  }

  // MiniKernelBitmap in_u_label_visited(32);
  // MiniKernelBitmap in_v_label_visited(32);
  // in_u_label_visited.Clear();
  // in_v_label_visited.Clear();
  u_offset_base = in_offset_p[u_idx];
  for (VertexID nbr_u_idx = 0; nbr_u_idx < in_degree_p[u_idx]; nbr_u_idx++) {
    VertexID nbr_u = in_edges_p[u_offset_base + nbr_u_idx];
    VertexLabel u_label = params.v_label_p[nbr_u];
    VertexID u_ip_val = HashTable(u_label);
    in_min_u_ip_val < u_ip_val ? in_min_u_ip_val : u_ip_val;
    out_u_label_visited.SetBit(u_label);
  }

  v_offset_base = in_offset_g[v_idx];
  for (VertexID nbr_v_idx = 0; nbr_v_idx < in_degree_g[v_idx]; nbr_v_idx++) {
    VertexID nbr_v = in_edges_g[v_offset_base + nbr_v_idx];
    VertexLabel v_label = params.v_label_g[nbr_v];
    VertexID v_ip_val = HashTable(v_label);
    in_min_v_ip_val < v_ip_val ? in_min_v_ip_val : v_ip_val;
    out_v_label_visited.SetBit(v_label);
  }

  return  // in_v_label_visited.Count() >= in_u_label_visited.Count() &&
      in_min_u_ip_val >= in_min_v_ip_val &&
      out_v_label_visited.Count() >= out_u_label_visited.Count() &&
      out_min_u_ip_val >= out_min_v_ip_val &&
      out_degree_g[v_idx] >= out_degree_p[u_idx] &&
      in_degree_g[v_idx] >= in_degree_p[u_idx];
}

static __forceinline__ __device__ bool NeighborLabelCounterFilter(
    const ParametersFilter& params, VertexID u_idx, VertexID v_idx) {
  VertexID* globalid_p = (VertexID*)(params.data_p);
  VertexID* in_degree_p = globalid_p + params.n_vertices_p;
  VertexID* out_degree_p = in_degree_p + params.n_vertices_p;
  EdgeIndex* in_offset_p = (EdgeIndex*)(out_degree_p + params.n_vertices_p);
  EdgeIndex* out_offset_p = (EdgeIndex*)(in_offset_p + params.n_vertices_p + 1);
  EdgeIndex* in_edges_p = (EdgeIndex*)(out_offset_p + params.n_vertices_p + 1);
  VertexID* out_edges_p = in_edges_p + params.n_edges_p;

  VertexID* globalid_g = (VertexID*)(params.data_g);
  VertexID* in_degree_g = globalid_g + params.n_vertices_g;
  VertexID* out_degree_g = in_degree_g + params.n_vertices_g;
  EdgeIndex* in_offset_g = (EdgeIndex*)(out_degree_g + params.n_vertices_g);
  EdgeIndex* out_offset_g = (EdgeIndex*)(in_offset_g + params.n_vertices_g + 1);
  EdgeIndex* in_edges_g = (EdgeIndex*)(out_offset_g + params.n_vertices_g + 1);
  VertexID* out_edges_g = in_edges_g + params.n_edges_g;

  MiniKernelBitmap u_out_label_visited(32);
  MiniKernelBitmap u_in_label_visited(32);
  MiniKernelBitmap v_out_label_visited(32);
  MiniKernelBitmap v_in_label_visited(32);

  EdgeIndex u_offset_base = out_offset_p[u_idx];
  for (VertexID i = 0; i < out_degree_p[u_idx]; ++i) {
    VertexLabel lbl = params.v_label_p[out_edges_p[u_offset_base + i]];
    u_out_label_visited.SetBit(lbl);
  }
  u_offset_base = in_offset_p[u_idx];
  for (VertexID i = 0; i < in_degree_p[u_idx]; ++i) {
    VertexLabel lbl = params.v_label_p[in_edges_p[u_offset_base + i]];
    u_in_label_visited.SetBit(lbl);
  }

  EdgeIndex v_offset_base = out_offset_g[v_idx];
  for (VertexID i = 0; i < out_degree_g[v_idx]; ++i) {
    VertexLabel lbl = params.v_label_g[out_edges_g[v_offset_base + i]];
    v_out_label_visited.SetBit(lbl);
  }
  v_offset_base = in_offset_g[v_idx];
  for (VertexID i = 0; i < in_degree_g[v_idx]; ++i) {
    VertexLabel lbl = params.v_label_g[in_edges_g[v_offset_base + i]];
    v_in_label_visited.SetBit(lbl);
  }

  return v_out_label_visited.Count() >= u_out_label_visited.Count() &&
         v_in_label_visited.Count() >= u_in_label_visited.Count();
}

static __forceinline__ __device__ bool SteadyFilter(
    const ParametersFilter& params, VertexID u_idx, VertexID v_idx) {
  VertexID* globalid_p = (VertexID*)(params.data_p);
  VertexID* in_degree_p = globalid_p + params.n_vertices_p;
  VertexID* out_degree_p = in_degree_p + params.n_vertices_p;
  EdgeIndex* in_offset_p = (EdgeIndex*)(out_degree_p + params.n_vertices_p);
  EdgeIndex* out_offset_p = (EdgeIndex*)(in_offset_p + params.n_vertices_p + 1);
  EdgeIndex* in_edges_p = (EdgeIndex*)(out_offset_p + params.n_vertices_p + 1);
  VertexID* out_edges_p = in_edges_p + params.n_edges_p;
  VertexID* edges_globalid_by_localid_p = out_edges_p + params.n_edges_p;

  VertexID* globalid_g = (VertexID*)(params.data_g);
  VertexID* in_degree_g = globalid_g + params.n_vertices_g;
  VertexID* out_degree_g = in_degree_g + params.n_vertices_g;
  EdgeIndex* in_offset_g = (EdgeIndex*)(out_degree_g + params.n_vertices_g);
  EdgeIndex* out_offset_g = (EdgeIndex*)(in_offset_g + params.n_vertices_g + 1);
  EdgeIndex* in_edges_g = (EdgeIndex*)(out_offset_g + params.n_vertices_g + 1);
  VertexID* out_edges_g = in_edges_g + params.n_edges_g;
  VertexID* edges_globalid_by_localid_g = out_edges_g + params.n_edges_g;

  VertexLabel v_label = params.v_label_g[globalid_g[v_idx]];
  VertexLabel u_label = params.v_label_p[u_idx];

  if (u_label != v_label) return false;

  EdgeIndex in_offset_base_v = in_offset_g[v_idx];
  EdgeIndex out_offset_base_v = out_offset_g[v_idx];
  EdgeIndex in_offset_base_u = in_offset_p[u_idx];
  EdgeIndex out_offset_base_u = out_offset_p[u_idx];

  int match_count = 0;
  for (VertexID nbr_u_idx = 0; nbr_u_idx < in_degree_p[u_idx]; nbr_u_idx++) {
    VertexID nbr_u = in_edges_p[in_offset_base_u + nbr_u_idx];

    int matches = 0;
    for (VertexID nbr_v_idx = 0; nbr_v_idx < in_degree_g[v_idx]; nbr_v_idx++) {
      VertexID nbr_v = in_edges_g[in_offset_base_v + nbr_v_idx];

      if (params.v_label_p[nbr_u] == params.v_label_g[nbr_v] &&
          in_degree_p[nbr_u] <= in_degree_g[nbr_v]) {
        matches++;
      }
    }
    if (matches == 0) return false;
  }

  // for (VertexID nbr_u_idx = 0; nbr_u_idx < out_degree_p[u_idx]; nbr_u_idx++)
  // {
  //   VertexID nbr_u = out_edges_p[out_offset_base_u + nbr_u_idx];

  //  int matches = 0;
  //  for (VertexID nbr_v_idx = 0; nbr_v_idx < out_degree_g[v_idx]; nbr_v_idx++)
  //  {
  //    VertexID nbr_v = out_edges_g[out_offset_base_v + nbr_v_idx];

  //    if (params.v_label_p[nbr_u] ==
  //        params.v_label_g[nbr_v] //&& out_degree_p[nbr_u] <
  //                                //        out_degree_g[nbr_v]
  //    ) {
  //      matches++;
  //      // break;
  //    }
  //  }
  //  if (matches == 0)
  //    return false;
  //}

  return true;
}

static __forceinline__ __device__ bool MinWiseBloomFilter(
    const ParametersFilter& params, VertexID u_idx, VertexID v_idx) {
  VertexID* globalid_p = (VertexID*)(params.data_p);
  VertexID* in_degree_p = globalid_p + params.n_vertices_p;
  VertexID* out_degree_p = in_degree_p + params.n_vertices_p;
  EdgeIndex* in_offset_p = (EdgeIndex*)(out_degree_p + params.n_vertices_p);
  EdgeIndex* out_offset_p = (EdgeIndex*)(in_offset_p + params.n_vertices_p + 1);
  EdgeIndex* in_edges_p = (EdgeIndex*)(out_offset_p + params.n_vertices_p + 1);
  VertexID* out_edges_p = in_edges_p + params.n_edges_p;

  VertexID* globalid_g = (VertexID*)(params.data_g);
  VertexID* in_degree_g = globalid_g + params.n_vertices_g;
  VertexID* out_degree_g = in_degree_g + params.n_vertices_g;
  EdgeIndex* in_offset_g = (EdgeIndex*)(out_degree_g + params.n_vertices_g);
  EdgeIndex* out_offset_g = (EdgeIndex*)(in_offset_g + params.n_vertices_g + 1);
  EdgeIndex* in_edges_g = (EdgeIndex*)(out_offset_g + params.n_vertices_g + 1);
  VertexID* out_edges_g = in_edges_g + params.n_edges_g;

  if (params.v_label_p[u_idx] != params.v_label_g[v_idx]) return false;

  uint64_t u_out_bitmap = 0;
  EdgeIndex u_out_base = out_offset_p[u_idx];
  for (VertexID i = 0; i < out_degree_p[u_idx]; ++i) {
    VertexLabel lbl = params.v_label_p[out_edges_p[u_out_base + i]];
    VertexID h = HashTable(lbl);
    if (h < 64) u_out_bitmap |= (1ULL << h);
  }
  uint64_t u_in_bitmap = 0;
  EdgeIndex u_in_base = in_offset_p[u_idx];
  for (VertexID i = 0; i < in_degree_p[u_idx]; ++i) {
    VertexLabel lbl = params.v_label_p[in_edges_p[u_in_base + i]];
    VertexID h = HashTable(lbl);
    if (h < 64) u_in_bitmap |= (1ULL << h);
  }

  uint64_t v_out_bitmap = 0;
  EdgeIndex v_out_base = out_offset_g[v_idx];
  for (VertexID i = 0; i < out_degree_g[v_idx]; ++i) {
    VertexLabel lbl = params.v_label_g[out_edges_g[v_out_base + i]];
    VertexID h = HashTable(lbl);
    if (h < 64) v_out_bitmap |= (1ULL << h);
  }
  uint64_t v_in_bitmap = 0;
  EdgeIndex v_in_base = in_offset_g[v_idx];
  for (VertexID i = 0; i < in_degree_g[v_idx]; ++i) {
    VertexLabel lbl = params.v_label_g[in_edges_g[v_in_base + i]];
    VertexID h = HashTable(lbl);
    if (h < 64) v_in_bitmap |= (1ULL << h);
  }

  uint64_t common = u_out_bitmap & v_out_bitmap;
  uint64_t u_only = u_out_bitmap ^ common;
  uint64_t v_only = v_out_bitmap ^ common;
  if (u_only != 0) {
    if (v_only == 0) return false;
    if ((__ffsll(static_cast<long long>(u_only)) - 1) < (__ffsll(static_cast<long long>(v_only)) - 1)) return false;
  }

  common = u_in_bitmap & v_in_bitmap;
  u_only = u_in_bitmap ^ common;
  v_only = v_in_bitmap ^ common;
  if (u_only != 0) {
    if (v_only == 0) return false;
    if ((__ffsll(static_cast<long long>(u_only)) - 1) < (__ffsll(static_cast<long long>(v_only)) - 1)) return false;
  }

  return out_degree_g[v_idx] >= out_degree_p[u_idx] &&
         in_degree_g[v_idx] >= in_degree_p[u_idx];
}

static __forceinline__ __device__ bool LabelPairFrequencyFilter(
    const ParametersFilter& params, VertexID u_idx, VertexID v_idx) {
  VertexID* globalid_p = (VertexID*)(params.data_p);
  VertexID* in_degree_p = globalid_p + params.n_vertices_p;
  VertexID* out_degree_p = in_degree_p + params.n_vertices_p;
  EdgeIndex* in_offset_p = (EdgeIndex*)(out_degree_p + params.n_vertices_p);
  EdgeIndex* out_offset_p = (EdgeIndex*)(in_offset_p + params.n_vertices_p + 1);
  EdgeIndex* in_edges_p = (EdgeIndex*)(out_offset_p + params.n_vertices_p + 1);
  VertexID* out_edges_p = in_edges_p + params.n_edges_p;

  VertexID* globalid_g = (VertexID*)(params.data_g);
  VertexID* in_degree_g = globalid_g + params.n_vertices_g;
  VertexID* out_degree_g = in_degree_g + params.n_vertices_g;
  EdgeIndex* in_offset_g = (EdgeIndex*)(out_degree_g + params.n_vertices_g);
  EdgeIndex* out_offset_g = (EdgeIndex*)(in_offset_g + params.n_vertices_g + 1);
  EdgeIndex* in_edges_g = (EdgeIndex*)(out_offset_g + params.n_vertices_g + 1);
  VertexID* out_edges_g = in_edges_g + params.n_edges_g;

  // Per-label neighbor frequency, labels capped at 32 to match subiso_cpu.
  int u_freq[32] = {0};
  EdgeIndex u_offset_base = out_offset_p[u_idx];
  for (VertexID i = 0; i < out_degree_p[u_idx]; ++i) {
    VertexLabel lbl = params.v_label_p[out_edges_p[u_offset_base + i]];
    if (lbl < 32) ++u_freq[lbl];
  }
  u_offset_base = in_offset_p[u_idx];
  for (VertexID i = 0; i < in_degree_p[u_idx]; ++i) {
    VertexLabel lbl = params.v_label_p[in_edges_p[u_offset_base + i]];
    if (lbl < 32) ++u_freq[lbl];
  }

  int v_freq[32] = {0};
  EdgeIndex v_offset_base = out_offset_g[v_idx];
  for (VertexID i = 0; i < out_degree_g[v_idx]; ++i) {
    VertexLabel lbl = params.v_label_g[out_edges_g[v_offset_base + i]];
    if (lbl < 32) ++v_freq[lbl];
  }
  v_offset_base = in_offset_g[v_idx];
  for (VertexID i = 0; i < in_degree_g[v_idx]; ++i) {
    VertexLabel lbl = params.v_label_g[in_edges_g[v_offset_base + i]];
    if (lbl < 32) ++v_freq[lbl];
  }

  for (int lbl = 0; lbl < 32; ++lbl) {
    if (v_freq[lbl] < u_freq[lbl]) return false;
  }
  return true;
}

static __forceinline__ __device__ bool Filter(const ParametersFilter& params,
                                              VertexID u_idx, VertexID v_idx) {
  // Label filter is always the first cheap check.
  if (!LabelFilter(params, u_idx, v_idx)) return false;

  auto do_ldf = [&]() {
    return !(params.enable_label_degree_filter &&
             !LabelDegreeFilter(params, u_idx, v_idx));
  };
  auto do_nlc = [&]() {
    return !(params.enable_nlc_filter &&
             !NeighborLabelCounterFilter(params, u_idx, v_idx));
  };
  auto do_lpf = [&]() {
    return !(params.enable_lpf_filter &&
             !LabelPairFrequencyFilter(params, u_idx, v_idx));
  };
  auto do_bloom = [&]() {
    return !(params.enable_bloom_filter &&
             !BloomLabelSetFilter(params, u_idx, v_idx));
  };
  auto do_minwise = [&]() {
    return !(params.enable_min_wise_filter &&
             !KMinWiseIPFilter(params, u_idx, v_idx));
  };
  auto do_minwise_bloom = [&]() {
    return !(params.enable_min_wise_bloom_filter &&
             !MinWiseBloomFilter(params, u_idx, v_idx));
  };

  switch (params.filter_order) {
    case 1: {  // minwise_first
      if (!do_minwise()) return false;
      if (!do_ldf()) return false;
      if (!do_nlc()) return false;
      if (!do_lpf()) return false;
      if (!do_bloom()) return false;
      if (!do_minwise_bloom()) return false;
      break;
    }
    case 2: {  // bloom_first
      if (!do_bloom()) return false;
      if (!do_ldf()) return false;
      if (!do_nlc()) return false;
      if (!do_lpf()) return false;
      if (!do_minwise()) return false;
      if (!do_minwise_bloom()) return false;
      break;
    }
    case 3: {  // lpf_first
      if (!do_lpf()) return false;
      if (!do_ldf()) return false;
      if (!do_nlc()) return false;
      if (!do_bloom()) return false;
      if (!do_minwise()) return false;
      if (!do_minwise_bloom()) return false;
      break;
    }
    case 4: {  // nlc_first
      if (!do_nlc()) return false;
      if (!do_ldf()) return false;
      if (!do_lpf()) return false;
      if (!do_bloom()) return false;
      if (!do_minwise()) return false;
      if (!do_minwise_bloom()) return false;
      break;
    }
    case 5: {  // lpf_nlc_bloom_minwisebloom_ldf
      if (!do_lpf()) return false;
      if (!do_nlc()) return false;
      if (!do_bloom()) return false;
      if (!do_minwise_bloom()) return false;
      if (!do_ldf()) return false;
      if (!do_minwise()) return false;
      break;
    }
    case 6: {  // minwise_nlc_lpf
      if (!do_minwise()) return false;
      if (!do_nlc()) return false;
      if (!do_lpf()) return false;
      if (!do_ldf()) return false;
      if (!do_bloom()) return false;
      if (!do_minwise_bloom()) return false;
      break;
    }
    case 0:
    default: {  // default
      if (!do_ldf()) return false;
      if (!do_nlc()) return false;
      if (!do_lpf()) return false;
      if (!do_bloom()) return false;
      if (!do_minwise()) return false;
      if (!do_minwise_bloom()) return false;
      break;
    }
  }
  return true;
}

static __global__ void WOJFilterVCKernel(ParametersFilter params) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  VertexID* globalid_g = (VertexID*)(params.data_g);
  VertexID* in_degree_g = globalid_g + params.n_vertices_g;
  VertexID* out_degree_g = in_degree_g + params.n_vertices_g;
  EdgeIndex* in_offset_g = (EdgeIndex*)(out_degree_g + params.n_vertices_g);
  EdgeIndex* out_offset_g = (EdgeIndex*)(in_offset_g + params.n_vertices_g + 1);
  EdgeIndex* in_edges_g = (EdgeIndex*)(out_offset_g + params.n_vertices_g + 1);
  VertexID* out_edges_g = in_edges_g + params.n_edges_g;

  VertexID* global_y_offset_ptr = params.woj_matches.get_y_offset_ptr();
  VertexID* data_ptr = params.woj_matches.get_data_ptr();
  const VertexID x_stride = params.woj_matches.get_x();
  const VertexID max_table_rows =
      x_stride ? (params.n_edges_p * kMaxMatchTableRows) / x_stride : 0;

  VertexID u_eid = params.u_eid;
  VertexID u_src = params.exec_path_in_edges[2 * u_eid];
  VertexID u_dst = params.exec_path_in_edges[2 * u_eid + 1];

  // One atomic row reservation per match. The old block-shared staging + mid-kernel
  // flush was racy for blockDim.x > 1; direct global writes are correct for any block.
  for (VertexID v_idx = tid; v_idx < params.n_vertices_g; v_idx += step) {
    if (Filter(params, u_src, v_idx)) {
      EdgeIndex v_offset_base = out_offset_g[v_idx];
      for (VertexID nbr_v_idx = 0; nbr_v_idx < out_degree_g[v_idx];
           nbr_v_idx++) {
        VertexID nbr_v = out_edges_g[v_offset_base + nbr_v_idx];

        if (Filter(params, u_dst, nbr_v)) {
          VertexID row = atomicAdd(global_y_offset_ptr, 1);
          if (row < max_table_rows) {
            data_ptr[x_stride * row] = v_idx;
            data_ptr[x_stride * row + 1] = nbr_v;
          } else {
            atomicSub(global_y_offset_ptr, 1);
          }
        }
      }
    }
  }
}

static __noinline__ __global__ void GetVisitedByKeyKernel(
    ParametersWedgeFilter params) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;
  KernelBitmapNoOwnership visited(params.n_vertices_g, params.visited_data);

  VertexID* data_ptr = params.woj_matches.get_data_ptr();
  VertexID x_offset = params.woj_matches.get_x_offset();
  VertexID y_offset = params.woj_matches.get_y_offset();

  for (VertexID data_offset = tid; data_offset < y_offset;
       data_offset += step) {
    VertexID target = data_ptr[x_offset * data_offset + params.hash_idx];
    visited.SetBit(data_ptr[x_offset * data_offset + params.hash_idx]);
  }
}

static __noinline__ __global__ void WOJJoinKernel(ParametersJoin params) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  auto lane_id = threadIdx.x & (kWarpSize - 1);
  auto warp_id = threadIdx.x >> kLogWarpSize;

  VertexID* left_data = params.left_woj_matches.get_data_ptr();
  VertexID* right_data = params.right_woj_matches.get_data_ptr();
  VertexID* output_data = params.output_woj_matches.get_data_ptr();
  VertexID left_x = params.left_woj_matches.get_x();
  VertexID right_x = params.right_woj_matches.get_x();
  VertexID output_x = params.output_woj_matches.get_x();
  VertexID left_x_offset = params.left_woj_matches.get_x_offset();
  VertexID right_x_offset = params.right_woj_matches.get_x_offset();
  VertexID output_x_offset = params.output_woj_matches.get_x_offset();
  VertexID left_y_offset = params.left_woj_matches.get_y_offset();
  VertexID right_y_offset = params.right_woj_matches.get_y_offset();

  VertexID* global_offset_ptr = params.output_woj_matches.get_y_offset_ptr();
  // KernelBitmapNoOwnership visited(params.n_vertices_g,
  //                                 params.right_visited_data);
  // KernelBitmapNoOwnership jump_visited(params.n_vertices_g,
  //                                      params.jump_visited_data);

  for (VertexID left_data_offset = tid;
       left_data_offset < params.left_woj_matches.get_y_offset();
       left_data_offset += step) {
    VertexID target =
        left_data[left_x * left_data_offset + params.left_hash_idx];

    // if (visited.GetBit(target) == 0) {
    //   atomicAdd(params.jump_count, 1);
    //   jump_visited.SetBit(target);
    //   continue;
    // }

    VertexID right_data_offset =
        params.right_woj_matches.BinarySearch(params.right_hash_idx, target);
    if (right_data_offset != kMaxVertexID &&
        right_data_offset < params.right_woj_matches.get_y_offset()) {
      VertexID left_walker = right_data_offset - 1;
      VertexID right_walker = right_data_offset;

      while (left_walker >= 0 && left_walker < right_y_offset &&
             right_data[left_walker * right_x + params.right_hash_idx] ==
                 target) {
        // Write direct on the global memory.
        auto global_offset = atomicAdd(global_offset_ptr, 1);
        if (global_offset >= params.output_woj_matches.get_y()) break;

        memcpy(output_data + global_offset * output_x,
               left_data + left_data_offset * left_x,
               sizeof(VertexID) * left_x_offset);

        VertexID write_col = 0;
        for (VertexID right_col_idx = 0; right_col_idx < right_x_offset;
             right_col_idx++) {
          if (right_col_idx == params.right_hash_idx) continue;
          *(output_data + global_offset * output_x + left_x_offset +
            write_col) =
              right_data[left_walker * right_x + right_col_idx];
          write_col++;
        }

        left_walker--;
      }

      while (
          right_walker >= 0 && right_walker < right_y_offset &&
          right_data[right_walker * right_x + params.right_hash_idx] ==
              target) {
        // Write direct on the global memory.
        auto global_offset = atomicAdd(global_offset_ptr, 1);
        if (global_offset >= params.output_woj_matches.get_y()) break;

        memcpy(output_data + global_offset * output_x,
               left_data + left_data_offset * left_x,
               sizeof(VertexID) * left_x_offset);

        VertexID write_col = 0;
        for (VertexID right_col_idx = 0; right_col_idx < right_x_offset;
             right_col_idx++) {
          if (right_col_idx == params.right_hash_idx) continue;
          *(output_data + global_offset * output_x + left_x_offset +
            write_col) =
              right_data[right_walker * right_x + right_col_idx];
          write_col++;
        }

        right_walker++;
      }
    }
  }

  // __syncthreads();
  // if (threadIdx.x == 0) {
  //   auto offset = atomicAdd(global_offset_ptr, local_matches_offset);
  //   memcpy(output_data + output_x_offset * offset, local_matches_data,
  //          sizeof(VertexID) * output_x_offset * local_matches_offset);
  // }
}

// ---------------------------------------------------------------------------
// Hash-join kernels for WOJ Join.
// Right table must be sorted by its join-key column.
// We build key_start/key_count arrays indexed by the join-key value (a vertex
// id in [0, n_vertices_g)), then do a two-pass join with a saturating prefix
// sum. This eliminates per-output atomicAdd.
// ---------------------------------------------------------------------------
static __global__ void BuildKeyStartKernel(const VertexID* right_data,
                                           VertexID right_x,
                                           VertexID right_hash_idx,
                                           VertexID right_y,
                                           VertexID* key_start) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;
  for (VertexID i = tid; i < right_y; i += step) {
    VertexID key = right_data[i * right_x + right_hash_idx];
    if (i == 0) {
      key_start[key] = 0;
    } else {
      VertexID prev_key = right_data[(i - 1) * right_x + right_hash_idx];
      if (key != prev_key) {
        key_start[key] = i;
      }
    }
  }
}

static __global__ void BuildKeyCountKernel(const VertexID* right_data,
                                           VertexID right_x,
                                           VertexID right_hash_idx,
                                           VertexID right_y,
                                           const VertexID* key_start,
                                           VertexID* key_count) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;
  for (VertexID i = tid; i < right_y; i += step) {
    VertexID key = right_data[i * right_x + right_hash_idx];
    bool is_last = (i + 1 == right_y);
    if (is_last) {
      VertexID start = key_start[key];
      key_count[key] = i - start + 1;
    } else {
      VertexID next_key = right_data[(i + 1) * right_x + right_hash_idx];
      if (key != next_key) {
        VertexID start = key_start[key];
        key_count[key] = i - start + 1;
      }
    }
  }
}

static __global__ void HashJoinCountKernel(const VertexID* left_data,
                                           VertexID left_x,
                                           VertexID left_hash_idx,
                                           VertexID left_y,
                                           const VertexID* key_count,
                                           VertexID* match_count) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;
  for (VertexID i = tid; i < left_y; i += step) {
    VertexID key = left_data[i * left_x + left_hash_idx];
    match_count[i] = key_count[key];
  }
}

static __global__ void HashJoinWriteKernel(ParametersHashJoin params) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  const VertexID output_x = params.output_x;
  const VertexID output_max_y = params.output_max_y;
  const VertexID left_x = params.left_x;
  const VertexID left_x_offset = params.left_x_offset;
  const VertexID right_x = params.right_x;
  const VertexID right_x_offset = params.right_x_offset;
  const VertexID left_hash_idx = params.left_hash_idx;
  const VertexID right_hash_idx = params.right_hash_idx;
  const VertexID* key_start = params.key_start;
  const VertexID* key_count = params.key_count;
  const VertexID* match_offset = params.match_offset;
  const VertexID* left_data = params.left_data;
  const VertexID* right_data = params.right_data;
  VertexID* output_data = params.output_data;

  for (VertexID i = tid; i < params.left_y; i += step) {
    VertexID key = left_data[i * left_x + left_hash_idx];
    VertexID count = key_count[key];
    if (count == 0) continue;

    VertexID base = match_offset[i];
    if (base >= output_max_y) continue;

    VertexID right_begin = key_start[key];
    for (VertexID j = 0; j < count; ++j) {
      uint64_t out_idx64 = static_cast<uint64_t>(base) + j;
      if (out_idx64 >= output_max_y) break;
      VertexID out_idx = static_cast<VertexID>(out_idx64);

      VertexID* out_row = output_data + out_idx * output_x;
      // Copy left columns.
      for (VertexID c = 0; c < left_x_offset; ++c) {
        out_row[c] = left_data[i * left_x + c];
      }
      // Copy right columns except the join key.
      VertexID write_col = left_x_offset;
      VertexID right_row = right_begin + j;
      for (VertexID c = 0; c < right_x_offset; ++c) {
        if (c == right_hash_idx) continue;
        out_row[write_col++] = right_data[right_row * right_x + c];
      }
    }
  }
}

std::vector<WOJMatches*> WOJSubIsoKernelWrapper::Filter(
    const WOJExecutionPlan& exec_plan, const ImmutableCSR& p,
    const ImmutableCSR& g, bool enable_min_wise_filter,
    bool enable_label_degree_filter, bool enable_nlc_filter,
    bool enable_lpf_filter, bool enable_lcf_filter, bool enable_bloom_filter,
    bool enable_min_wise_bloom_filter) {
  dim3 dimBlock(WojLaunchBlockDim());
  dim3 dimGrid(WojLaunchGridDim());

  const size_t parallelism = WojHostStripeThreads();

  std::cout << "[WOJ Filter] begin: |E_p|=" << exec_plan.get_n_edges_p()
            << " cudaGrid=(" << dimGrid.x << "," << dimGrid.y << ","
            << dimGrid.z << ") block=(" << dimBlock.x << "," << dimBlock.y
            << "," << dimBlock.z << ") host_worker_stripes=" << parallelism
            << " GPUs=" << static_cast<int>(exec_plan.get_n_devices())
            << std::endl;

  // Init Streams
  std::vector<cudaStream_t> p_streams_vec;
  p_streams_vec.resize(p.get_num_outgoing_edges());
  RunHostStripeParallel(parallelism,
      [parallelism, &exec_plan, &p_streams_vec](size_t w) {
        for (VertexID i = static_cast<VertexID>(w); i < p_streams_vec.size();
             i += static_cast<VertexID>(parallelism)) {
          const int logical =
              static_cast<int>(common::hash_function(i) %
                               exec_plan.get_n_devices());
          cudaSetDevice(exec_plan.CudaDeviceId(logical));
          cudaStreamCreate(&p_streams_vec[i]);
        }
      });

  {
    const VertexID n_e = exec_plan.get_n_edges_p();
    const VertexID kMaxEdgeRows = 48;
    std::cout
        << "[WOJ Filter] one cudaStream per pattern edge eid; kernel "
           "WOJFilterVCKernel<<<grid,block,0,stream>>>"
        << std::endl;
    for (VertexID e = 0; e < n_e && e < kMaxEdgeRows; ++e) {
      const VertexID logical_dev =
          common::hash_function(e) % exec_plan.get_n_devices();
      const int cuda_dev =
          exec_plan.CudaDeviceId(static_cast<int>(logical_dev));
      std::cout << "  eid=" << e << " logical_gpu=" << logical_dev
                << " cudaDevice=" << cuda_dev << " stream=0x" << std::hex
                << reinterpret_cast<uintptr_t>(p_streams_vec[e]) << std::dec
                << std::endl;
    }
    if (n_e > kMaxEdgeRows) {
      std::cout << "  ... (" << (n_e - kMaxEdgeRows)
                << " more edges, stream mapping follows same hash rule)"
                << std::endl;
    }
  }

  std::cout << "[WOJ Filter] dispatch WOJFilterVCKernel ..." << std::endl;

  // Init pattern.
  BufferUint8 data_p;
  BufferVertexLabel v_label_p;
  BufferVertexID buffer_exec_path_in_edges;

  data_p.data = p.GetGraphBuffer();
  data_p.size = sizeof(VertexID) * p.get_num_vertices() +
                sizeof(VertexID) * p.get_num_vertices() +
                sizeof(VertexID) * p.get_num_vertices() +
                sizeof(EdgeIndex) * (p.get_num_vertices() + 1) +
                sizeof(EdgeIndex) * (p.get_num_vertices() + 1) +
                sizeof(VertexID) * p.get_num_incoming_edges() +
                sizeof(VertexID) * p.get_num_outgoing_edges() +
                sizeof(VertexID) * (p.get_max_vid() + 1);

  buffer_exec_path_in_edges.data = exec_plan.get_exec_path_in_edges_ptr();
  buffer_exec_path_in_edges.size =
      sizeof(VertexID) * p.get_num_outgoing_edges() * 2;

  v_label_p.data = p.GetVLabelBasePointer();
  v_label_p.size = sizeof(VertexLabel) * p.get_num_vertices();

  // Init data_graph.
  BufferUint8 data_g;
  BufferVertexLabel v_label_g;
  BufferVertexID data_edgelist_g;

  data_g.data = g.GetGraphBuffer();
  data_g.size = sizeof(VertexID) * g.get_num_vertices() +
                sizeof(VertexID) * g.get_num_vertices() +
                sizeof(VertexID) * g.get_num_vertices() +
                sizeof(EdgeIndex) * (g.get_num_vertices() + 1) +
                sizeof(EdgeIndex) * (g.get_num_vertices() + 1) +
                sizeof(VertexID) * g.get_num_incoming_edges() +
                sizeof(VertexID) * g.get_num_outgoing_edges() +
                sizeof(VertexID) * (g.get_max_vid() + 1);

  v_label_g.data = g.GetVLabelBasePointer();
  v_label_g.size = sizeof(VertexLabel) * g.get_num_vertices();

  //  Init output.
  std::vector<WOJMatches*> woj_matches_vec;
  woj_matches_vec.resize(exec_plan.get_n_edges_p());

  // std::vector<ImmutableCSRGPU> data_graph_gpu_vec;
  // data_graph_gpu_vec.resize(exec_plan.get_n_devices());
  // std::vector<ImmutableCSRGPU> pattern_graph_gpu_vec;
  // pattern_graph_gpu_vec.resize(exec_plan.get_n_devices());

  std::vector<UnifiedOwnedBufferVertexID> exec_path_in_edges_vec;
  exec_path_in_edges_vec.resize(exec_plan.get_n_devices());
  std::vector<UnifiedOwnedBufferUint8> data_p_vec;
  data_p_vec.resize(exec_plan.get_n_devices());
  std::vector<UnifiedOwnedBufferVertexLabel> v_label_p_vec;
  v_label_p_vec.resize(exec_plan.get_n_devices());
  std::vector<UnifiedOwnedBufferUint8> data_g_vec;
  data_g_vec.resize(exec_plan.get_n_devices());
  std::vector<UnifiedOwnedBufferVertexLabel> v_label_g_vec;
  v_label_g_vec.resize(exec_plan.get_n_devices());

  // LCF: global label-count pre-check (host side).
  if (enable_lcf_filter) {
    const int kLcfLabelCap = 32;
    std::array<int, kLcfLabelCap> p_freq = {};
    std::array<int, kLcfLabelCap> g_freq = {};
    const VertexLabel* plabels = p.GetVLabelBasePointer();
    const VertexLabel* glabels = g.GetVLabelBasePointer();
    for (VertexID u = 0; u < p.get_num_vertices(); ++u) {
      VertexLabel lbl = plabels[u];
      if (lbl < kLcfLabelCap) ++p_freq[lbl];
    }
    for (VertexID v = 0; v < g.get_num_vertices(); ++v) {
      VertexLabel lbl = glabels[v];
      if (lbl < kLcfLabelCap) ++g_freq[lbl];
    }
    for (int lbl = 0; lbl < kLcfLabelCap; ++lbl) {
      if (g_freq[lbl] < p_freq[lbl]) {
        std::cout << "[WOJ Filter] LCF rejected globally (label " << lbl
                  << " count: pattern=" << p_freq[lbl] << " data="
                  << g_freq[lbl] << ")." << std::endl;
        // Return initialized but empty match tables.
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
        return woj_matches_vec;
      }
    }
  }

  // Bloom label-set signatures for pattern and data graph.
  std::vector<uint64_t> bloom_signature_p_host(p.get_num_vertices(), 0);
  std::vector<uint64_t> bloom_signature_g_host(g.get_num_vertices(), 0);

  std::cout << "[WOJ Filter] enable_min_wise_filter=" << enable_min_wise_filter
            << " enable_label_degree_filter=" << enable_label_degree_filter
            << " enable_nlc_filter=" << enable_nlc_filter
            << " enable_lpf_filter=" << enable_lpf_filter
            << " enable_lcf_filter=" << enable_lcf_filter
            << " enable_bloom_filter=" << enable_bloom_filter
            << " enable_min_wise_bloom_filter=" << enable_min_wise_bloom_filter
            << std::endl;
  if (enable_bloom_filter) {
    const VertexLabel* plabels = p.GetVLabelBasePointer();
    auto pn = p.get_num_vertices();
    for (VertexID u = 0; u < pn; ++u) {
      uint64_t sig = 0;
      auto out_deg = p.GetOutDegreeByLocalID(u);
      auto out_edges = p.GetOutgoingEdgesByLocalID(u);
      for (VertexID i = 0; i < out_deg; ++i) {
        VertexLabel lbl = plabels[out_edges[i]];
        if (lbl < 64) sig |= (1ULL << lbl);
      }
      auto in_deg = p.GetInDegreeByLocalID(u);
      auto in_edges = p.GetIncomingEdgesByLocalID(u);
      for (VertexID i = 0; i < in_deg; ++i) {
        VertexLabel lbl = plabels[in_edges[i]];
        if (lbl < 64) sig |= (1ULL << lbl);
      }
      bloom_signature_p_host[u] = sig;
    }

    const VertexLabel* glabels = g.GetVLabelBasePointer();
    auto gn = g.get_num_vertices();
    for (VertexID v = 0; v < gn; ++v) {
      uint64_t sig = 0;
      auto out_deg = g.GetOutDegreeByLocalID(v);
      auto out_edges = g.GetOutgoingEdgesByLocalID(v);
      for (VertexID i = 0; i < out_deg; ++i) {
        VertexLabel lbl = glabels[out_edges[i]];
        if (lbl < 64) sig |= (1ULL << lbl);
      }
      auto in_deg = g.GetInDegreeByLocalID(v);
      auto in_edges = g.GetIncomingEdgesByLocalID(v);
      for (VertexID i = 0; i < in_deg; ++i) {
        VertexLabel lbl = glabels[in_edges[i]];
        if (lbl < 64) sig |= (1ULL << lbl);
      }
      bloom_signature_g_host[v] = sig;
    }
  }

  std::vector<UnifiedOwnedBufferUint64> bloom_signature_p_vec;
  std::vector<UnifiedOwnedBufferUint64> bloom_signature_g_vec;
  bloom_signature_p_vec.resize(exec_plan.get_n_devices());
  bloom_signature_g_vec.resize(exec_plan.get_n_devices());

  for (VertexID _ = 0; _ < exec_plan.get_n_devices(); _++) {
    // data_graph_gpu_vec[_].Init(g);
    // pattern_graph_gpu_vec[_].Init(p);
    exec_path_in_edges_vec[_].Init(buffer_exec_path_in_edges);
    data_p_vec[_].Init(data_p);
    v_label_p_vec[_].Init(v_label_p);
    data_g_vec[_].Init(data_g);
    v_label_g_vec[_].Init(v_label_g);
    if (enable_bloom_filter) {
      BufferUint64 bloom_p_buf;
      bloom_p_buf.data = bloom_signature_p_host.data();
      bloom_p_buf.size = sizeof(uint64_t) * p.get_num_vertices();
      bloom_signature_p_vec[_].Init(bloom_p_buf);
      BufferUint64 bloom_g_buf;
      bloom_g_buf.data = bloom_signature_g_host.data();
      bloom_g_buf.size = sizeof(uint64_t) * g.get_num_vertices();
      bloom_signature_g_vec[_].Init(bloom_g_buf);
    }
  }

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

  // Filter order override via environment variable.
  int filter_order = 0;
  const char* order_env = std::getenv("MG_FILTER_ORDER");
  if (order_env) {
    std::string order_str(order_env);
    if (order_str == "minwise_first") {
      filter_order = 1;
    } else if (order_str == "bloom_first") {
      filter_order = 2;
    } else if (order_str == "lpf_first") {
      filter_order = 3;
    } else if (order_str == "nlc_first") {
      filter_order = 4;
    } else if (order_str == "lpf_nlc_bloom_minwisebloom_ldf") {
      filter_order = 5;
    } else if (order_str == "minwise_nlc_lpf") {
      filter_order = 6;
    }
  }

  auto time1 = std::chrono::system_clock::now();
  RunHostStripeParallel(parallelism,
      [parallelism, &dimGrid, &dimBlock, &exec_plan, &p, &g,
       &exec_path_in_edges_vec, &data_p_vec, &v_label_p_vec, &data_g_vec,
       &v_label_g_vec, &bloom_signature_p_vec, &bloom_signature_g_vec,
       enable_min_wise_filter, enable_label_degree_filter, enable_nlc_filter,
       enable_lpf_filter, enable_bloom_filter, enable_min_wise_bloom_filter,
       filter_order, &woj_matches_vec, &p_streams_vec](size_t w) {
        for (VertexID _ = static_cast<VertexID>(w);
             _ < exec_plan.get_n_edges_p();
             _ += static_cast<VertexID>(parallelism)) {
          const VertexID logical_dev =
              common::hash_function(_) % exec_plan.get_n_devices();
          cudaSetDevice(
              exec_plan.CudaDeviceId(static_cast<int>(logical_dev)));
          cudaStream_t& stream = p_streams_vec[_];
          ParametersFilter params{
              .u_eid = _,
              .exec_path_in_edges =
                  exec_path_in_edges_vec[logical_dev].GetPtr(),
              .n_vertices_p = p.get_num_vertices(),
              .n_edges_p = p.get_num_outgoing_edges(),
              .data_p = data_p_vec[logical_dev].GetPtr(),
              .v_label_p = v_label_p_vec[logical_dev].GetPtr(),
              .n_vertices_g = g.get_num_vertices(),
              .n_edges_g = g.get_num_outgoing_edges(),
              .data_g = data_g_vec[logical_dev].GetPtr(),
              .edgelist_g = nullptr,
              .v_label_g = v_label_g_vec[logical_dev].GetPtr(),
              .bloom_signature_p = enable_bloom_filter
                                      ? bloom_signature_p_vec[logical_dev].GetPtr()
                                      : nullptr,
              .bloom_signature_g = enable_bloom_filter
                                      ? bloom_signature_g_vec[logical_dev].GetPtr()
                                      : nullptr,
              .enable_bloom_filter = enable_bloom_filter,
              .enable_min_wise_filter = enable_min_wise_filter,
              .enable_min_wise_bloom_filter = enable_min_wise_bloom_filter,
              .enable_label_degree_filter = enable_label_degree_filter,
              .enable_nlc_filter = enable_nlc_filter,
              .enable_lpf_filter = enable_lpf_filter,
              .filter_order = filter_order,
              .woj_matches = *woj_matches_vec[_]};

          WOJFilterVCKernel<<<dimGrid, dimBlock, 0, stream>>>(params);
        }
      });

  std::cout << "[WOJ Filter] cudaDeviceSynchronize() each plan GPU..."
            << std::endl;

  for (VertexID device_id = 0; device_id < exec_plan.get_n_devices();
       device_id++) {
    cudaSetDevice(exec_plan.CudaDeviceId(static_cast<int>(device_id)));
    cudaDeviceSynchronize();
    // pattern_graph_gpu_vec[device_id].Free();
    // data_graph_gpu_vec[device_id].Free();
  }

  auto time2 = std::chrono::system_clock::now();

  std::cout << "[Filter]:"
            << std::chrono::duration_cast<std::chrono::microseconds>(time2 -
                                                                     time1)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;

  for (VertexID _ = 0; _ < exec_plan.get_n_edges_p(); _++) {
    std::cout << "Eid - " << _ << " has " << woj_matches_vec[_]->get_y_offset()
              << " matches." << std::endl;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    CUDA_CHECK(err);
  }

  RunHostStripeParallel(parallelism,
      [parallelism, &exec_plan, &p_streams_vec](size_t w) {
        for (VertexID i = static_cast<VertexID>(w); i < p_streams_vec.size();
             i += static_cast<VertexID>(parallelism)) {
          const int logical =
              static_cast<int>(common::hash_function(i) %
                               exec_plan.get_n_devices());
          cudaSetDevice(exec_plan.CudaDeviceId(logical));
          cudaStreamDestroy(p_streams_vec[i]);
        }
      });

  return woj_matches_vec;
}

namespace {

void SortWojJoinInputs(const std::vector<WOJMatches*>& input_woj_matches_vec) {
  if (input_woj_matches_vec.empty()) {
    return;
  }

  size_t max_row_bytes = 0;
  for (auto* t : input_woj_matches_vec) {
    const size_t b = sizeof(VertexID) * static_cast<size_t>(t->get_x()) *
                     static_cast<size_t>(t->get_y_offset());
    if (b > max_row_bytes) {
      max_row_bytes = b;
    }
  }

  VertexID* sort_scratch = nullptr;
  cudaStream_t sort_stream{};
  CUDA_CHECK(cudaStreamCreate(&sort_stream));
  if (max_row_bytes > 0) {
    CUDA_CHECK(cudaMallocManaged(&sort_scratch, max_row_bytes));
  }

  BitmapOwnership header_visited(32);
  auto header_ptr0 = input_woj_matches_vec[0]->get_header_ptr();
  for (auto _ = 0; _ < input_woj_matches_vec[0]->get_x_offset(); _++) {
    header_visited.SetBit(header_ptr0[_]);
  }

  for (VertexID _ = 1; _ < input_woj_matches_vec.size(); _++) {
    bool sort_tag = false;
    auto header_ptr = input_woj_matches_vec[_]->get_header_ptr();
    const size_t row_bytes =
        sizeof(VertexID) * static_cast<size_t>(input_woj_matches_vec[_]->get_x()) *
        static_cast<size_t>(input_woj_matches_vec[_]->get_y_offset());
    for (VertexID __ = 0; __ < input_woj_matches_vec[_]->get_x_offset(); __++) {
      if (header_visited.GetBit(header_ptr[__]) && sort_tag == false) {
        MergeSort(sort_stream, input_woj_matches_vec[_]->get_data_ptr(), __,
                  input_woj_matches_vec[_]->get_x(),
                  input_woj_matches_vec[_]->get_y_offset(), row_bytes,
                  sort_scratch);
        sort_tag = true;
      }
      header_visited.SetBit(header_ptr[__]);
    }
  }

  if (sort_scratch != nullptr) {
    CUDA_CHECK(cudaFree(sort_scratch));
  }
  CUDA_CHECK(cudaStreamDestroy(sort_stream));
}

bool WojPrefetchJoinInputEnabled() {
  const char* e = std::getenv("MG_WOJ_PREFETCH_JOIN");
  if (e != nullptr && e[0] == '0' && e[1] == '\0') {
    return false;
  }
  return true;
}

void PrefetchWojJoinTables(
    const std::vector<WOJMatches*>& input_woj_matches_vec) {
  if (!WojPrefetchJoinInputEnabled() || input_woj_matches_vec.empty()) {
    return;
  }
  using sics::matrixgraph::core::util::MatrixGraphCudaDeviceList;
  using sics::matrixgraph::core::util::MatrixGraphCudaStreamsPerGpu;
  using sics::matrixgraph::core::util::MatrixGraphPrefetchManagedToDevice;

  const std::vector<int> devices =
      MatrixGraphCudaDeviceList();
  if (devices.empty()) {
    return;
  }
  std::vector<std::pair<void*, size_t>> chunks;
  chunks.reserve(input_woj_matches_vec.size() * 4u);
  for (auto* t : input_woj_matches_vec) {
    const size_t data_bytes =
        sizeof(VertexID) * static_cast<size_t>(t->get_x()) *
        static_cast<size_t>(t->get_y_offset());
    const size_t header_bytes =
        sizeof(VertexID) * static_cast<size_t>(t->get_x());
    chunks.push_back({static_cast<void*>(t->get_data_ptr()), data_bytes});
    chunks.push_back({static_cast<void*>(t->get_header_ptr()), header_bytes});
    chunks.push_back({static_cast<void*>(t->get_y_offset_ptr()), sizeof(VertexID)});
    chunks.push_back({static_cast<void*>(t->get_x_offset_ptr()), sizeof(VertexID)});
  }
  const int n_streams = MatrixGraphCudaStreamsPerGpu();
  int prev_dev = 0;
  CUDA_CHECK(cudaGetDevice(&prev_dev));
  for (int dev : devices) {
    CUDA_CHECK(cudaSetDevice(dev));
    MatrixGraphPrefetchManagedToDevice(dev, n_streams, chunks);
  }
  CUDA_CHECK(cudaSetDevice(prev_dev));
}

bool WojBushyJoinEnabled() {
  const char* e = std::getenv("MG_WOJ_BUSHY_JOIN");
  if (e == nullptr || e[0] == '\0') {
    return false;
  }
  return e[0] == '1';
}

bool AdjacentPairsAllJoinable(const std::vector<WOJMatches*>& cur) {
  const size_t n = cur.size();
  if (n < 2) {
    return false;
  }
  for (size_t i = 0; i + 1 < n; i += 2) {
    auto jk = cur[i]->GetJoinKey(*cur[i + 1]);
    if (jk.first == kMaxVertexID || jk.second == kMaxVertexID) {
      return false;
    }
  }
  return true;
}

// Hash join on two WOJMatches tables. The right table must be sorted by its
// join-key column. This function sets output_woj_matches's header and y_offset.
void RunHashJoinOnDevice(const WOJExecutionPlan& exec_plan,
                         WOJMatches* left_woj_matches,
                         WOJMatches* right_woj_matches,
                         WOJMatches* output_woj_matches, dim3 dimGrid,
                         dim3 dimBlock, int logical_gpu, cudaStream_t stream) {
  cudaSetDevice(exec_plan.CudaDeviceId(logical_gpu));

  auto join_keys = left_woj_matches->GetJoinKey(*right_woj_matches);
  if (join_keys.first == kMaxVertexID || join_keys.second == kMaxVertexID) {
    output_woj_matches->SetYOffset(0);
    return;
  }

  output_woj_matches->SetHeader(
      left_woj_matches->get_header_ptr(),
      left_woj_matches->get_x_offset(), right_woj_matches->get_header_ptr(),
      right_woj_matches->get_x_offset(), join_keys);

  const VertexID n_vertices_g = exec_plan.get_n_vertices_g();
  const VertexID right_y = right_woj_matches->get_y_offset();
  const VertexID left_y = left_woj_matches->get_y_offset();
  const VertexID output_max_y = output_woj_matches->get_y();

  VertexID* d_key_start = nullptr;
  VertexID* d_key_count = nullptr;
  CUDA_CHECK(cudaMallocManaged(&d_key_start,
                               sizeof(VertexID) * (n_vertices_g + 1)));
  CUDA_CHECK(cudaMallocManaged(&d_key_count,
                               sizeof(VertexID) * (n_vertices_g + 1)));
  CUDA_CHECK(cudaMemset(d_key_start, 0xFF,
                        sizeof(VertexID) * (n_vertices_g + 1)));
  CUDA_CHECK(cudaMemset(d_key_count, 0,
                        sizeof(VertexID) * (n_vertices_g + 1)));

  if (right_y > 0) {
    BuildKeyStartKernel<<<dimGrid, dimBlock, 0, stream>>>(
        right_woj_matches->get_data_ptr(), right_woj_matches->get_x(),
        join_keys.second, right_y, d_key_start);
    BuildKeyCountKernel<<<dimGrid, dimBlock, 0, stream>>>(
        right_woj_matches->get_data_ptr(), right_woj_matches->get_x(),
        join_keys.second, right_y, d_key_start, d_key_count);
  }

  VertexID* d_match_count = nullptr;
  VertexID* d_match_offset = nullptr;
  VertexID total_output_rows = 0;

  if (left_y > 0) {
    CUDA_CHECK(cudaMalloc(&d_match_count, sizeof(VertexID) * left_y));
    CUDA_CHECK(cudaMalloc(&d_match_offset, sizeof(VertexID) * left_y));

    HashJoinCountKernel<<<dimGrid, dimBlock, 0, stream>>>(
        left_woj_matches->get_data_ptr(), left_woj_matches->get_x(),
        join_keys.first, left_y, d_key_count, d_match_count);

    thrust::exclusive_scan(thrust::cuda::par.on(stream), d_match_count,
                           d_match_count + left_y, d_match_offset,
                           static_cast<VertexID>(0), SatAdd(output_max_y));

    VertexID h_last_offset = 0;
    VertexID h_last_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_last_offset, d_match_offset + (left_y - 1),
                               sizeof(VertexID), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(&h_last_count, d_match_count + (left_y - 1),
                               sizeof(VertexID), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    uint64_t total64 = static_cast<uint64_t>(h_last_offset) + h_last_count;
    total_output_rows =
        total64 > output_max_y ? output_max_y : static_cast<VertexID>(total64);
    output_woj_matches->SetYOffset(total_output_rows);

    if (total_output_rows > 0) {
      ParametersHashJoin params{
          .left_y = left_y,
          .left_x = left_woj_matches->get_x(),
          .left_x_offset = left_woj_matches->get_x_offset(),
          .right_x = right_woj_matches->get_x(),
          .right_x_offset = right_woj_matches->get_x_offset(),
          .output_x = output_woj_matches->get_x(),
          .output_max_y = output_max_y,
          .left_hash_idx = join_keys.first,
          .right_hash_idx = join_keys.second,
          .left_data = left_woj_matches->get_data_ptr(),
          .right_data = right_woj_matches->get_data_ptr(),
          .output_data = output_woj_matches->get_data_ptr(),
          .key_start = d_key_start,
          .key_count = d_key_count,
          .match_offset = d_match_offset};
      HashJoinWriteKernel<<<dimGrid, dimBlock, 0, stream>>>(params);
    }
  } else {
    output_woj_matches->SetYOffset(0);
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));

  if (d_match_count != nullptr) CUDA_CHECK(cudaFree(d_match_count));
  if (d_match_offset != nullptr) CUDA_CHECK(cudaFree(d_match_offset));
  if (d_key_start != nullptr) CUDA_CHECK(cudaFree(d_key_start));
  if (d_key_count != nullptr) CUDA_CHECK(cudaFree(d_key_count));
}

void RunBinaryJoinOnDevice(const WOJExecutionPlan& exec_plan,
                           WOJMatches* left_woj_matches,
                           WOJMatches* right_woj_matches,
                           WOJMatches* output_woj_matches, dim3 dimGrid,
                           dim3 dimBlock, int logical_gpu) {
  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreate(&stream));
  RunHashJoinOnDevice(exec_plan, left_woj_matches, right_woj_matches,
                      output_woj_matches, dimGrid, dimBlock, logical_gpu,
                      stream);
  CUDA_CHECK(cudaStreamDestroy(stream));
}

std::vector<WOJMatches*> ExecuteBushyPairwiseRound(
    const WOJExecutionPlan& exec_plan, const std::vector<WOJMatches*>& cur,
    dim3 dimGrid, dim3 dimBlock) {
  const size_t n = cur.size();
  const size_t npairs = n / 2;
  const bool has_tail = (n % 2u) == 1u;
  std::vector<WOJMatches*> outs(npairs);
  for (size_t k = 0; k < npairs; ++k) {
    outs[k] = new WOJMatches();
    outs[k]->Init(exec_plan.get_n_vertices_p(), kMaxMatchTableRows);
  }
  const VertexID nd =
      std::max(static_cast<VertexID>(1), exec_plan.get_n_devices());
  const size_t workers = WojJoinHostWorkerCount(npairs);

  RunHostStripeParallel(workers, [&](size_t w) {
    for (size_t k = w; k < npairs; k += workers) {
      const int log_dev = static_cast<int>(static_cast<VertexID>(k) % nd);
      RunBinaryJoinOnDevice(exec_plan, cur[2 * k], cur[2 * k + 1], outs[k],
                            dimGrid, dimBlock, log_dev);
    }
  });

  std::vector<WOJMatches*> next;
  next.reserve(npairs + (has_tail ? 1u : 0u));
  for (size_t k = 0; k < npairs; ++k) {
    next.push_back(outs[k]);
  }
  if (has_tail) {
    next.push_back(cur[n - 1]);
  }
  return next;
}

std::optional<std::vector<WOJMatches*>> TryBushyPairwiseJoin(
    const WOJExecutionPlan& exec_plan, std::vector<WOJMatches*> cur,
    dim3 dimGrid, dim3 dimBlock) {
  int level = 0;
  while (cur.size() > 1) {
    if (!AdjacentPairsAllJoinable(cur)) {
      std::cout << "[WOJ Join] bushy: level " << level
                << " cannot pair adjacent tables (missing join key) -> striped "
                   "fallback\n";
      return std::nullopt;
    }
    std::cout << "[WOJ Join] bushy level " << level << " pairwise joins, tables="
              << cur.size() << std::endl;
    cur = ExecuteBushyPairwiseRound(exec_plan, cur, dimGrid, dimBlock);
    level++;
  }
  return std::vector<WOJMatches*>{cur[0]};
}

}  // namespace

std::vector<WOJMatches*> WOJSubIsoKernelWrapper::Join(
    const WOJExecutionPlan& exec_plan,
    const std::vector<WOJMatches*>& input_woj_matches_vec) {
  std::cout << " --- Join --- " << std::endl;
  auto join_t0 = std::chrono::system_clock::now();

  // Join Tables.
  dim3 dimBlock(WojLaunchBlockDim());
  dim3 dimGrid(WojLaunchGridDim());

  SortWojJoinInputs(input_woj_matches_vec);
  auto join_t1 = std::chrono::system_clock::now();
  PrefetchWojJoinTables(input_woj_matches_vec);
  auto join_t2 = std::chrono::system_clock::now();
  if (WojBushyJoinEnabled() && input_woj_matches_vec.size() >= 2) {
    std::vector<WOJMatches*> bushy_tables;
    bushy_tables.reserve(input_woj_matches_vec.size());
    for (auto* t : input_woj_matches_vec) {
      bushy_tables.push_back(t);
    }
    auto bushy_out =
        TryBushyPairwiseJoin(exec_plan, std::move(bushy_tables), dimGrid, dimBlock);
    if (bushy_out.has_value()) {
      std::cout << "[WOJ Join] bushy pairwise reduction completed -> 1 result "
                   "table\n";
      return std::move(*bushy_out);
    }
  }
  std::cout << "[WOJ Join] striped left-deep path\n";

  const VertexID n_devices = exec_plan.get_n_devices();
  const size_t join_stripes_per_gpu = WojJoinStripesPerGpu();
  const VertexID n_join_stripes = static_cast<VertexID>(std::max<size_t>(
      1, static_cast<size_t>(n_devices) * join_stripes_per_gpu));
  const size_t join_workers = WojJoinHostWorkerCount(
      static_cast<size_t>(n_join_stripes));

  std::cout << "[WOJ Join] begin: input match tables="
            << input_woj_matches_vec.size() << " cudaGrid=(" << dimGrid.x
            << "," << dimGrid.y << "," << dimGrid.z << ") block=("
            << dimBlock.x << "," << dimBlock.y << "," << dimBlock.z
            << ") join_stripes=" << static_cast<int>(n_join_stripes) << " (GPUs="
            << static_cast<int>(n_devices) << " * JOIN_STRIPES_PER_GPU="
            << join_stripes_per_gpu << ") host_workers=" << join_workers
            << std::endl;

  auto src_matches_vec =
      input_woj_matches_vec[0]->SplitAndCopy(n_join_stripes, exec_plan.get_n_vertices_p());

  std::vector<WOJMatches*> output_woj_matches_vec;
  output_woj_matches_vec.resize(static_cast<size_t>(n_join_stripes));

  for (VertexID _ = 0; _ < n_join_stripes; _++) {
    output_woj_matches_vec[_] = new WOJMatches();
    output_woj_matches_vec[_]->Init(exec_plan.get_n_vertices_p(), kMaxMatchTableRows);
  }

  // Init Streams
  std::vector<cudaStream_t> p_streams_vec;
  p_streams_vec.resize(static_cast<size_t>(n_join_stripes));

  std::cout << "[WOJ Join] one stream per split stripe s in [0, "
            << p_streams_vec.size()
            << "); device = CudaDeviceId(s % N); HashJoinWriteKernel on that stripe's stream"
            << std::endl;
  for (size_t s = 0; s < p_streams_vec.size(); ++s) {
    const int logical =
        static_cast<int>(static_cast<VertexID>(s) % exec_plan.get_n_devices());
    const int cuda_dev = exec_plan.CudaDeviceId(logical);
    std::cout << "  stripe s=" << s << " logical_gpu=" << logical
              << " cudaDevice=" << cuda_dev << std::endl;
  }

  // Join candidates: one left-deep chain per stripe; stripes are scheduled across
  // join_workers host threads (see WojJoinHostWorkerCount).
  RunHostStripeParallel(
      join_workers,
      [join_workers, n_join_stripes, &dimBlock, &dimGrid, &exec_plan,
       &p_streams_vec, &src_matches_vec, &input_woj_matches_vec,
       &output_woj_matches_vec](size_t w) {
        for (VertexID s = static_cast<VertexID>(w); s < n_join_stripes;
             s += static_cast<VertexID>(join_workers)) {
          const int logical = static_cast<int>(
              static_cast<VertexID>(s) % exec_plan.get_n_devices());
          cudaSetDevice(exec_plan.CudaDeviceId(logical));
          cudaStream_t& stream = p_streams_vec[s];

          cudaStreamCreate(&stream);

          auto left_woj_matches = src_matches_vec[s];

          for (VertexID _ = 1; _ < input_woj_matches_vec.size(); _++) {
            auto right_woj_matches = input_woj_matches_vec[_];

            RunHashJoinOnDevice(exec_plan, left_woj_matches, right_woj_matches,
                                output_woj_matches_vec[s], dimGrid, dimBlock,
                                logical, stream);

            if (output_woj_matches_vec[s]->get_y_offset() == 0) {
              break;
            }

            bool is_last_iteration =
                (_ == input_woj_matches_vec.size() - 1);
            if (output_woj_matches_vec[s]->get_x_offset() ==
                    output_woj_matches_vec[s]->get_x() ||
                is_last_iteration) {
              // Final result already resides in output_woj_matches_vec[s].
              if (!is_last_iteration) return;
              break;
            } else {
              std::swap(left_woj_matches, output_woj_matches_vec[s]);
              output_woj_matches_vec[s]->SetYOffset(0);
              output_woj_matches_vec[s]->SetXOffset(0);
            }
          }
        }
      });

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    CUDA_CHECK(err);
  }

  std::cout << "[WOJ Join] destroy per-stripe streams ..." << std::endl;

  RunHostStripeParallel(
      join_workers,
      [join_workers, n_join_stripes, &p_streams_vec, &exec_plan](size_t w) {
        for (VertexID i = static_cast<VertexID>(w); i < n_join_stripes;
             i += static_cast<VertexID>(join_workers)) {
          const int logical = static_cast<int>(
              static_cast<VertexID>(i) % exec_plan.get_n_devices());
          cudaSetDevice(exec_plan.CudaDeviceId(logical));
          cudaStreamDestroy(p_streams_vec[i]);
        }
      });

  auto join_t3 = std::chrono::system_clock::now();

  auto us = [](auto a, auto b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a)
               .count() /
           (double)CLOCKS_PER_SEC;
  };
  std::cout << "[WOJ Join] breakdown: sort=" << us(join_t0, join_t1)
            << " prefetch=" << us(join_t1, join_t2)
            << " join_loop=" << us(join_t2, join_t3)
            << " total=" << us(join_t0, join_t3) << std::endl;

  std::cout << "[WOJ Join] finished" << std::endl;

  return output_woj_matches_vec;
}

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics