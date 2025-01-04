#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

#include "core/common/consts.h"
#include "core/common/host_algorithms.cuh"
#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/host_buffer.cuh"
#include "core/data_structures/unified_buffer.cuh"
#include "core/task/kernel/algorithms/hash.cuh"
#include "core/task/kernel/algorithms/sort.cuh"
#include "core/task/kernel/data_structures/heap.cuh"
#include "core/task/kernel/data_structures/immutable_csr_gpu.cuh"
#include "core/task/kernel/data_structures/kernel_bitmap.cuh"
#include "core/task/kernel/data_structures/kernel_bitmap_no_ownership.cuh"
#include "core/task/kernel/data_structures/mini_kernel_bitmap.cuh"
#include "core/task/kernel/data_structures/woj_matches.cuh"
#include "core/task/kernel/kernel_woj_subiso.cuh"
#include "core/util/bitmap.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
using VertexID = sics::matrixgraph::core::common::VertexID;
using Bitmap = sics::matrixgraph::core::util::Bitmap;
using sics::matrixgraph::core::common::kBlockDim;
using sics::matrixgraph::core::common::kGridDim;
using sics::matrixgraph::core::common::kLogWarpSize;
using sics::matrixgraph::core::common::kMaxNumCandidatesPerThread;
using sics::matrixgraph::core::common::kMaxNumWeft;
using sics::matrixgraph::core::common::kMaxVertexID;
using sics::matrixgraph::core::common::kNCUDACoresPerSM;
using sics::matrixgraph::core::common::kNSMsPerGPU;
using sics::matrixgraph::core::common::kNWarpPerCUDACore;
using sics::matrixgraph::core::common::kSharedMemoryCapacity;
using sics::matrixgraph::core::common::kSharedMemorySize;
using sics::matrixgraph::core::common::kWarpSize;
using sics::matrixgraph::core::task::kernel::HostKernelBitmap;
using sics::matrixgraph::core::task::kernel::HostMiniKernelBitmap;
using sics::matrixgraph::core::task::kernel::KernelBitmap;
using sics::matrixgraph::core::task::kernel::KernelBitmapNoOwnership;
using sics::matrixgraph::core::task::kernel::MiniKernelBitmap;
using WOJMatches = sics::matrixgraph::core::task::kernel::WOJMatches;
using MinHeap = sics::matrixgraph::core::task::kernel::MinHeap;
using BufferUint8 = sics::matrixgraph::core::data_structures::Buffer<uint8_t>;
using BufferUint32 = sics::matrixgraph::core::data_structures::Buffer<uint32_t>;
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
using BufferVertexLabel =
    sics::matrixgraph::core::data_structures::Buffer<VertexLabel>;
using BufferVertexID =
    sics::matrixgraph::core::data_structures::Buffer<VertexID>;

struct LocalMatches {
  VertexID *data = nullptr;
  VertexID *size = nullptr;
};

struct ParametersFilter {
  VertexID u_eid;
  VertexID *exec_path_in_edges = nullptr;
  VertexID n_vertices_p;
  EdgeIndex n_edges_p;
  uint8_t *data_p;
  VertexLabel *v_label_p = nullptr;
  VertexID n_vertices_g;
  EdgeIndex n_edges_g;
  uint8_t *data_g = nullptr;
  VertexID *edgelist_g = nullptr;
  VertexLabel *v_label_g = nullptr;
  WOJMatches woj_matches;
  uint64_t *test;
};

struct ParametersWedgeFilter {
  VertexID n_vertices_g;
  WOJMatches woj_matches;
  VertexID hash_idx;
  uint64_t *visited_data;
};

struct ParametersJoin {
  VertexID n_vertices_g;
  WOJMatches left_woj_matches;
  WOJMatches right_woj_matches;
  WOJMatches output_woj_matches;
  VertexID left_hash_idx;
  VertexID right_hash_idx;
  uint64_t *right_visited_data;
  uint64_t *jump_visited_data;
  VertexID *jump_count;
};

static __forceinline__ __device__ bool
LabelFilter(const ParametersFilter &params, VertexID u_idx, VertexID v_idx) {
  VertexID *globalid_g = (VertexID *)(params.data_g);
  VertexLabel v_label = params.v_label_g[v_idx];
  VertexLabel u_label = params.v_label_p[u_idx];
  return u_label == v_label;
}

static __forceinline__ __device__ bool
LabelDegreeFilter(const ParametersFilter &params, VertexID u_idx,
                  VertexID v_idx) {

  VertexID *globalid_p = (VertexID *)(params.data_p);
  VertexID *in_degree_p = globalid_p + params.n_vertices_p;
  VertexID *out_degree_p = in_degree_p + params.n_vertices_p;

  VertexID *globalid_g = (VertexID *)(params.data_g);
  VertexID *in_degree_g = globalid_g + params.n_vertices_g;
  VertexID *out_degree_g = in_degree_g + params.n_vertices_g;

  VertexLabel v_label = params.v_label_g[globalid_g[v_idx]];
  VertexLabel u_label = params.v_label_p[u_idx];

  if (u_label != v_label) {
    return false;
  } else {
    return out_degree_g[v_idx] >= out_degree_p[u_idx] &&
           in_degree_g[v_idx] >= in_degree_p[u_idx];
  }
}

static __forceinline__ __device__ bool
KMinWiseIPFilter(const ParametersFilter &params, VertexID u_idx,
                 VertexID v_idx) {
  VertexID *globalid_p = (VertexID *)(params.data_p);
  VertexID *in_degree_p = globalid_p + params.n_vertices_p;
  VertexID *out_degree_p = in_degree_p + params.n_vertices_p;
  EdgeIndex *in_offset_p = (EdgeIndex *)(out_degree_p + params.n_vertices_p);
  EdgeIndex *out_offset_p =
      (EdgeIndex *)(in_offset_p + params.n_vertices_p + 1);
  EdgeIndex *in_edges_p = (EdgeIndex *)(out_offset_p + params.n_vertices_p + 1);
  VertexID *out_edges_p = in_edges_p + params.n_edges_p;
  VertexID *edges_globalid_by_localid_p = out_edges_p + params.n_edges_p;

  VertexID *globalid_g = (VertexID *)(params.data_g);
  VertexID *in_degree_g = globalid_g + params.n_vertices_g;
  VertexID *out_degree_g = in_degree_g + params.n_vertices_g;
  EdgeIndex *in_offset_g = (EdgeIndex *)(out_degree_g + params.n_vertices_g);
  EdgeIndex *out_offset_g =
      (EdgeIndex *)(in_offset_g + params.n_vertices_g + 1);
  EdgeIndex *in_edges_g = (EdgeIndex *)(out_offset_g + params.n_vertices_g + 1);
  VertexID *out_edges_g = in_edges_g + params.n_edges_g;
  VertexID *edges_globalid_by_localid_g = out_edges_g + params.n_edges_g;

  VertexLabel v_label = params.v_label_g[v_idx];
  VertexLabel u_label = params.v_label_p[u_idx];

  if (u_label != v_label)
    return false;

  VertexID max_v_ip_val = 0;
  VertexID min_v_ip_val = kMaxVertexID;
  VertexID max_u_ip_val = 0;
  VertexID min_u_ip_val = kMaxVertexID;

  MiniKernelBitmap u_label_visited(32);
  MiniKernelBitmap v_label_visited(32);
  MiniKernelBitmap u_ip_val_visited(32);
  MiniKernelBitmap v_ip_val_visited(32);

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
    u_ip_val_visited.SetBit(u_ip_val);
    u_k_min_heap.Insert(u_ip_val);
  }

  EdgeIndex v_offset_base = out_offset_g[v_idx];
  for (VertexID nbr_v_idx = 0; nbr_v_idx < out_degree_g[v_idx]; nbr_v_idx++) {
    VertexID nbr_v = out_edges_g[v_offset_base + nbr_v_idx];
    VertexLabel v_label = params.v_label_g[nbr_v];
    VertexID v_ip_val = HashTable(v_label);
    v_label_visited.SetBit(v_label);
    v_ip_val_visited.SetBit(v_ip_val);
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
  max_v_ip_val = 0;
  min_v_ip_val = kMaxVertexID;
  max_u_ip_val = 0;
  min_u_ip_val = kMaxVertexID;
  u_label_visited.Clear();
  v_label_visited.Clear();
  u_k_min_heap.Clear();
  v_k_min_heap.Clear();

  u_offset_base = in_offset_p[u_idx];
  for (VertexID nbr_u_idx = 0; nbr_u_idx < in_degree_p[u_idx]; nbr_u_idx++) {
    VertexID nbr_u = in_edges_p[u_offset_base + nbr_u_idx];
    VertexLabel u_label = params.v_label_p[nbr_u];
    VertexID u_ip_val = HashTable(u_label);
    u_label_visited.SetBit(u_label);
    u_k_min_heap.Insert(u_ip_val);
  }

  v_offset_base = in_offset_g[v_idx];
  for (VertexID nbr_v_idx = 0; nbr_v_idx < in_degree_g[v_idx]; nbr_v_idx++) {
    VertexID nbr_v = in_edges_g[v_offset_base + nbr_v_idx];
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

  return v_label_visited.Count() >= u_label_visited.Count() &&
         out_degree_g[v_idx] >= out_degree_p[u_idx] &&
         in_degree_g[v_idx] >= in_degree_p[u_idx];
}

static __forceinline__ __device__ bool
MinWiseIPFilter(const ParametersFilter &params, VertexID u_idx,
                VertexID v_idx) {
  VertexID *globalid_p = (VertexID *)(params.data_p);
  VertexID *in_degree_p = globalid_p + params.n_vertices_p;
  VertexID *out_degree_p = in_degree_p + params.n_vertices_p;
  EdgeIndex *in_offset_p = (EdgeIndex *)(out_degree_p + params.n_vertices_p);
  EdgeIndex *out_offset_p =
      (EdgeIndex *)(in_offset_p + params.n_vertices_p + 1);
  EdgeIndex *in_edges_p = (EdgeIndex *)(out_offset_p + params.n_vertices_p + 1);
  VertexID *out_edges_p = in_edges_p + params.n_edges_p;
  VertexID *edges_globalid_by_localid_p = out_edges_p + params.n_edges_p;

  VertexID *globalid_g = (VertexID *)(params.data_g);
  VertexID *in_degree_g = globalid_g + params.n_vertices_g;
  VertexID *out_degree_g = in_degree_g + params.n_vertices_g;
  EdgeIndex *in_offset_g = (EdgeIndex *)(out_degree_g + params.n_vertices_g);
  EdgeIndex *out_offset_g =
      (EdgeIndex *)(in_offset_g + params.n_vertices_g + 1);
  EdgeIndex *in_edges_g = (EdgeIndex *)(out_offset_g + params.n_vertices_g + 1);
  VertexID *out_edges_g = in_edges_g + params.n_edges_g;
  VertexID *edges_globalid_by_localid_g = out_edges_g + params.n_edges_g;

  VertexLabel v_label = params.v_label_g[v_idx];
  VertexLabel u_label = params.v_label_p[u_idx];

  if (u_label != v_label)
    return false;

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

  MiniKernelBitmap in_u_label_visited(32);
  MiniKernelBitmap in_v_label_visited(32);
  in_u_label_visited.Clear();
  in_v_label_visited.Clear();
  u_offset_base = in_offset_p[u_idx];
  for (VertexID nbr_u_idx = 0; nbr_u_idx < in_degree_p[u_idx]; nbr_u_idx++) {
    VertexID nbr_u = in_edges_p[u_offset_base + nbr_u_idx];
    VertexLabel u_label = params.v_label_p[nbr_u];
    VertexID u_ip_val = HashTable(u_label);
    in_min_u_ip_val < u_ip_val ? in_min_u_ip_val : u_ip_val;
    in_u_label_visited.SetBit(u_label);
  }

  v_offset_base = in_offset_g[v_idx];
  for (VertexID nbr_v_idx = 0; nbr_v_idx < in_degree_g[v_idx]; nbr_v_idx++) {
    VertexID nbr_v = in_edges_g[v_offset_base + nbr_v_idx];
    VertexLabel v_label = params.v_label_g[nbr_v];
    VertexID v_ip_val = HashTable(v_label);
    in_min_v_ip_val < v_ip_val ? in_min_v_ip_val : v_ip_val;
    in_v_label_visited.SetBit(v_label);
  }

  return in_v_label_visited.Count() >= in_u_label_visited.Count() &&
         in_min_u_ip_val >= in_min_v_ip_val &&
         out_v_label_visited.Count() >= out_u_label_visited.Count() &&
         out_min_u_ip_val >= out_min_v_ip_val &&
         out_degree_g[v_idx] >= out_degree_p[u_idx] &&
         in_degree_g[v_idx] >= in_degree_p[u_idx];
}

static __forceinline__ __device__ bool
NeighborLabelCounterFilter(const ParametersFilter &params, VertexID u_idx,
                           VertexID v_idx) {
  VertexID *globalid_p = (VertexID *)(params.data_p);
  VertexID *in_degree_p = globalid_p + params.n_vertices_p;
  VertexID *out_degree_p = in_degree_p + params.n_vertices_p;
  EdgeIndex *in_offset_p = (EdgeIndex *)(out_degree_p + params.n_vertices_p);
  EdgeIndex *out_offset_p =
      (EdgeIndex *)(in_offset_p + params.n_vertices_p + 1);
  EdgeIndex *in_edges_p = (EdgeIndex *)(out_offset_p + params.n_vertices_p + 1);
  VertexID *out_edges_p = in_edges_p + params.n_edges_p;
  VertexID *edges_globalid_by_localid_p = out_edges_p + params.n_edges_p;

  VertexID *globalid_g = (VertexID *)(params.data_g);
  VertexID *in_degree_g = globalid_g + params.n_vertices_g;
  VertexID *out_degree_g = in_degree_g + params.n_vertices_g;
  EdgeIndex *in_offset_g = (EdgeIndex *)(out_degree_g + params.n_vertices_g);
  EdgeIndex *out_offset_g =
      (EdgeIndex *)(in_offset_g + params.n_vertices_g + 1);
  EdgeIndex *in_edges_g = (EdgeIndex *)(out_offset_g + params.n_vertices_g + 1);
  VertexID *out_edges_g = in_edges_g + params.n_edges_g;
  VertexID *edges_globalid_by_localid_g = out_edges_g + params.n_edges_g;

  VertexLabel v_label = params.v_label_g[globalid_g[v_idx]];
  VertexLabel u_label = params.v_label_p[u_idx];

  if (u_label != v_label)
    return false;

  MiniKernelBitmap u_label_visited(32);
  MiniKernelBitmap v_label_visited(32);

  // EdgeIndex u_offset_base = out_offset_p[u_idx];
  // for (VertexID nbr_u_idx = 0; nbr_u_idx < out_degree_p[u_idx]; nbr_u_idx++)
  // {
  //   VertexID nbr_u = out_edges_p[u_offset_base + nbr_u_idx];
  //   VertexLabel u_label = params.v_label_p[nbr_u];
  //   u_label_visited.SetBit(u_label);
  // }

  // EdgeIndex v_offset_base = out_offset_g[v_idx];
  // for (VertexID nbr_v_idx = 0; nbr_v_idx < out_degree_g[v_idx]; nbr_v_idx++)
  // {
  //   VertexID nbr_v = out_edges_g[v_offset_base + nbr_v_idx];
  //   VertexLabel v_label = params.v_label_g[nbr_v];
  //   v_label_visited.SetBit(v_label);
  // }

  EdgeIndex u_offset_base = in_offset_p[u_idx];
  for (VertexID nbr_u_idx = 0; nbr_u_idx < in_degree_p[u_idx]; nbr_u_idx++) {
    VertexID nbr_u = in_edges_p[u_offset_base + nbr_u_idx];
    VertexLabel u_label = params.v_label_p[nbr_u];
    u_label_visited.SetBit(u_label);
  }

  EdgeIndex v_offset_base = in_offset_g[v_idx];
  for (VertexID nbr_v_idx = 0; nbr_v_idx < in_degree_g[v_idx]; nbr_v_idx++) {
    VertexID nbr_v = in_edges_g[v_offset_base + nbr_v_idx];
    VertexLabel v_label = params.v_label_g[nbr_v];
    v_label_visited.SetBit(v_label);
  }

  return v_label_visited.Count() >= u_label_visited.Count();
}

static __forceinline__ __device__ bool
SteadyFilter(const ParametersFilter &params, VertexID u_idx, VertexID v_idx) {
  VertexID *globalid_p = (VertexID *)(params.data_p);
  VertexID *in_degree_p = globalid_p + params.n_vertices_p;
  VertexID *out_degree_p = in_degree_p + params.n_vertices_p;
  EdgeIndex *in_offset_p = (EdgeIndex *)(out_degree_p + params.n_vertices_p);
  EdgeIndex *out_offset_p =
      (EdgeIndex *)(in_offset_p + params.n_vertices_p + 1);
  EdgeIndex *in_edges_p = (EdgeIndex *)(out_offset_p + params.n_vertices_p + 1);
  VertexID *out_edges_p = in_edges_p + params.n_edges_p;
  VertexID *edges_globalid_by_localid_p = out_edges_p + params.n_edges_p;

  VertexID *globalid_g = (VertexID *)(params.data_g);
  VertexID *in_degree_g = globalid_g + params.n_vertices_g;
  VertexID *out_degree_g = in_degree_g + params.n_vertices_g;
  EdgeIndex *in_offset_g = (EdgeIndex *)(out_degree_g + params.n_vertices_g);
  EdgeIndex *out_offset_g =
      (EdgeIndex *)(in_offset_g + params.n_vertices_g + 1);
  EdgeIndex *in_edges_g = (EdgeIndex *)(out_offset_g + params.n_vertices_g + 1);
  VertexID *out_edges_g = in_edges_g + params.n_edges_g;
  VertexID *edges_globalid_by_localid_g = out_edges_g + params.n_edges_g;

  VertexLabel v_label = params.v_label_g[globalid_g[v_idx]];
  VertexLabel u_label = params.v_label_p[u_idx];

  if (u_label != v_label)
    return false;

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
    if (matches == 0)
      return false;
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

static __forceinline__ __device__ bool Filter(const ParametersFilter &params,
                                              VertexID u_idx, VertexID v_idx) {

  // return LabelFilter(params, u_idx, v_idx);
  //  return MinWiseIPFilter(params, u_idx, v_idx);
  return NeighborLabelCounterFilter(params, u_idx, v_idx);
  // return LabelDegreeFilter(params, u_idx, v_idx);
  // return SteadyFilter(params, u_idx, v_idx);
  //      if (SteadyFilter(params, u_idx, v_idx)) {
  //        return LabelDegreeFilter(params, u_idx, v_idx);
  //        // return NeighborLabelCounterFilter(params, u_idx, v_idx);
  //        // return MinWiseIPFilter(params, u_idx, v_idx);
  //      }
  //      return false;
}

static __global__ void WOJFilterVCKernel(ParametersFilter params) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  auto lane_id = threadIdx.x & (kWarpSize - 1);
  auto warp_id = threadIdx.x >> kLogWarpSize;

  __shared__ VertexID local_matches_data[kSharedMemorySize];
  __shared__ VertexID local_matches_offset;

  if (threadIdx.x == 0)
    local_matches_offset = 0;
  __syncthreads();

  VertexID *globalid_p = (VertexID *)(params.data_p);
  VertexID *in_degree_p = globalid_p + params.n_vertices_p;
  VertexID *out_degree_p = in_degree_p + params.n_vertices_p;
  EdgeIndex *in_offset_p = (EdgeIndex *)(out_degree_p + params.n_vertices_p);
  EdgeIndex *out_offset_p =
      (EdgeIndex *)(in_offset_p + params.n_vertices_p + 1);
  EdgeIndex *in_edges_p = (EdgeIndex *)(out_offset_p + params.n_vertices_p + 1);
  VertexID *out_edges_p = in_edges_p + params.n_edges_p;
  VertexID *edges_globalid_by_localid_p = out_edges_p + params.n_edges_p;

  VertexID *globalid_g = (VertexID *)(params.data_g);
  VertexID *in_degree_g = globalid_g + params.n_vertices_g;
  VertexID *out_degree_g = in_degree_g + params.n_vertices_g;
  EdgeIndex *in_offset_g = (EdgeIndex *)(out_degree_g + params.n_vertices_g);
  EdgeIndex *out_offset_g =
      (EdgeIndex *)(in_offset_g + params.n_vertices_g + 1);
  EdgeIndex *in_edges_g = (EdgeIndex *)(out_offset_g + params.n_vertices_g + 1);
  VertexID *out_edges_g = in_edges_g + params.n_edges_g;
  VertexID *edges_globalid_by_localid_g = out_edges_g + params.n_edges_g;

  VertexID offset = 0;
  VertexID *global_y_offset_ptr = params.woj_matches.get_y_offset_ptr();
  auto data_ptr = params.woj_matches.get_data_ptr();

  VertexID u_eid = params.u_eid;
  VertexID u_src = params.exec_path_in_edges[2 * u_eid];
  VertexID u_dst = params.exec_path_in_edges[2 * u_eid + 1];

  for (VertexID v_idx = tid; v_idx < params.n_vertices_g; v_idx += step) {
    if (Filter(params, u_src, v_idx)) {
      EdgeIndex v_offset_base = out_offset_g[v_idx];
      for (VertexID nbr_v_idx = 0; nbr_v_idx < out_degree_g[v_idx];
           nbr_v_idx++) {
        VertexID nbr_v = out_edges_g[v_offset_base + nbr_v_idx];

        if (Filter(params, u_dst, nbr_v)) {
          offset = atomicAdd(&local_matches_offset, 1);
          local_matches_data[2 * offset] = v_idx;
          local_matches_data[2 * offset + 1] = nbr_v;
          if (offset > kSharedMemorySize / 2 - 32) {
            VertexID write_y_offset =
                atomicAdd(global_y_offset_ptr, offset + 1);
            memcpy(data_ptr + 2 * write_y_offset, local_matches_data,
                   sizeof(VertexID) * 2 * (offset + 1));
            atomicMin(&local_matches_offset, 0);
          }
        }
      }
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    auto offset = atomicAdd(global_y_offset_ptr, local_matches_offset);
    if (offset > kMaxNumWeft) {
      auto offset = atomicAdd(global_y_offset_ptr, local_matches_offset);
      return;
    }
    memcpy(data_ptr + 2 * offset, local_matches_data,
           sizeof(VertexID) * 2 * local_matches_offset);
  }
}

static __noinline__ __global__ void
GetVisitedByKeyKernel(ParametersWedgeFilter params) {

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;
  KernelBitmapNoOwnership visited(params.n_vertices_g, params.visited_data);

  VertexID *data_ptr = params.woj_matches.get_data_ptr();
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

  VertexID *left_data = params.left_woj_matches.get_data_ptr();
  VertexID *right_data = params.right_woj_matches.get_data_ptr();
  VertexID *output_data = params.output_woj_matches.get_data_ptr();
  VertexID left_x_offset = params.left_woj_matches.get_x_offset();
  VertexID right_x_offset = params.right_woj_matches.get_x_offset();
  VertexID output_x_offset = params.output_woj_matches.get_x_offset();
  VertexID left_y_offset = params.left_woj_matches.get_y_offset();
  VertexID right_y_offset = params.right_woj_matches.get_y_offset();

  VertexID *global_offset_ptr = params.output_woj_matches.get_y_offset_ptr();
  KernelBitmapNoOwnership visited(params.n_vertices_g,
                                  params.right_visited_data);
  KernelBitmapNoOwnership jump_visited(params.n_vertices_g,
                                       params.jump_visited_data);

  for (VertexID left_data_offset = tid;
       left_data_offset < params.left_woj_matches.get_y_offset();
       left_data_offset += step) {
    VertexID target =
        left_data[left_x_offset * left_data_offset + params.left_hash_idx];

    if (visited.GetBit(target) == 0) {
      atomicAdd(params.jump_count, 1);
      jump_visited.SetBit(target);
      continue;
    }

    VertexID right_data_offset =
        params.right_woj_matches.BinarySearch(params.right_hash_idx, target);
    if (right_data_offset != kMaxVertexID &&
        right_data_offset < params.right_woj_matches.get_y_offset()) {

      VertexID left_walker = right_data_offset - 1;
      VertexID right_walker = right_data_offset;

      while (left_walker >= 0 && left_walker < right_y_offset &&
             right_data[left_walker * right_x_offset + params.right_hash_idx] ==
                 target) {

        // Write direct on the global memory.
        auto global_offset = atomicAdd(global_offset_ptr, 1);
        if (global_offset > kMaxNumWeft / output_x_offset)
          break;

        memcpy(output_data + global_offset * output_x_offset,
               left_data + left_data_offset * left_x_offset,
               sizeof(VertexID) * left_x_offset);

        VertexID write_col = 0;
        for (VertexID right_col_idx = 0; right_col_idx < right_x_offset;
             right_col_idx++) {
          if (right_col_idx == params.right_hash_idx)
            continue;
          *(output_data + global_offset * output_x_offset + left_x_offset +
            write_col) =
              right_data[left_walker * right_x_offset + right_col_idx];
          write_col++;
        }

        left_walker--;
      }

      while (
          right_walker >= 0 && right_walker < right_y_offset &&
          right_data[right_walker * right_x_offset + params.right_hash_idx] ==
              target) {
        // Write direct on the global memory.
        auto global_offset = atomicAdd(global_offset_ptr, 1);
        if (global_offset > kMaxNumWeft / output_x_offset)
          break;

        memcpy(output_data + global_offset * output_x_offset,
               left_data + left_data_offset * left_x_offset,
               sizeof(VertexID) * left_x_offset);

        VertexID write_col = 0;
        for (VertexID right_col_idx = 0; right_col_idx < right_x_offset;
             right_col_idx++) {
          if (right_col_idx == params.right_hash_idx)
            continue;
          *(output_data + global_offset * output_x_offset + left_x_offset +
            write_col) =
              right_data[right_walker * right_x_offset + right_col_idx];
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

std::vector<WOJMatches *>
WOJSubIsoKernelWrapper::Filter(const WOJExecutionPlan &exec_plan,
                               const ImmutableCSR &p, const ImmutableCSR &g) {
  dim3 dimBlock(kBlockDim);
  dim3 dimGrid(kGridDim);

  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;

  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  // Init Streams
  std::vector<cudaStream_t> p_streams_vec;
  p_streams_vec.resize(p.get_num_outgoing_edges());
  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [step, &exec_plan, &p_streams_vec, &mtx](auto w) {
                  for (VertexID i = w; i < p_streams_vec.size(); i += step) {
                    cudaSetDevice(common::hash_function(i) %
                                  exec_plan.get_n_devices());
                    cudaStreamCreate(&p_streams_vec[i]);
                  }
                });

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
  std::vector<WOJMatches *> woj_matches_vec;
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
  v_label_g_vec.resize(4);

  for (VertexID _ = 0; _ < exec_plan.get_n_devices(); _++) {
    // data_graph_gpu_vec[_].Init(g);
    // pattern_graph_gpu_vec[_].Init(p);
    exec_path_in_edges_vec[_].Init(buffer_exec_path_in_edges);
    data_p_vec[_].Init(data_p);
    v_label_p_vec[_].Init(v_label_p);
    data_g_vec[_].Init(data_g);
    v_label_g_vec[_].Init(v_label_g);
  }

  for (VertexID _ = 0; _ < exec_plan.get_n_edges_p(); _++) {
    woj_matches_vec[_] = new WOJMatches();
    woj_matches_vec[_]->Init(exec_plan.get_n_edges_p(), kMaxNumWeft);
    woj_matches_vec[_]->SetXOffset(2);
    woj_matches_vec[_]->SetYOffset(0);
    woj_matches_vec[_]->SetHeader(
        0, exec_plan.get_exec_path_in_edges_ptr()[_ * 2]);
    woj_matches_vec[_]->SetHeader(
        1, exec_plan.get_exec_path_in_edges_ptr()[_ * 2 + 1]);
  }

  auto time1 = std::chrono::system_clock::now();
  for (VertexID _ = 0; _ < exec_plan.get_n_edges_p(); _++) {
    VertexID device_id = common::hash_function(_) % exec_plan.get_n_devices();
    cudaSetDevice(device_id);
    cudaStream_t &stream = p_streams_vec[_];
    ParametersFilter params{.u_eid = _,
                            .exec_path_in_edges =
                                exec_path_in_edges_vec[device_id].GetPtr(),
                            .n_vertices_p = p.get_num_vertices(),
                            .n_edges_p = p.get_num_outgoing_edges(),
                            .data_p = data_p_vec[device_id].GetPtr(),
                            .v_label_p = v_label_p_vec[device_id].GetPtr(),
                            .n_vertices_g = g.get_num_vertices(),
                            .n_edges_g = g.get_num_outgoing_edges(),
                            .data_g = data_g_vec[device_id].GetPtr(),
                            .edgelist_g = nullptr,
                            .v_label_g = v_label_g_vec[device_id].GetPtr(),
                            .woj_matches = *woj_matches_vec[_]};

    WOJFilterVCKernel<<<dimGrid, dimBlock, 0, stream>>>(params);

    // cudaDeviceSynchronize();
  }

  for (VertexID device_id = 0; device_id < exec_plan.get_n_devices();
       device_id++) {
    cudaSetDevice(device_id);
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

  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [step, &p_streams_vec, &mtx](auto w) {
                  for (VertexID i = w; i < p_streams_vec.size(); i += step) {
                    cudaStreamDestroy(p_streams_vec[i]);
                  }
                });

  return woj_matches_vec;
}

std::vector<WOJMatches *> WOJSubIsoKernelWrapper::Join(
    const WOJExecutionPlan &exec_plan,
    const std::vector<WOJMatches *> &input_woj_matches_vec) {
  std::cout << " --- Join --- " << std::endl;

  // Join Tables.
  dim3 dimBlock(kBlockDim);
  dim3 dimGrid(kGridDim);

  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;

  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  auto src_matches_vec =
      input_woj_matches_vec[0]->SplitAndCopy(exec_plan.get_n_devices());

  std::vector<WOJMatches *> output_woj_matches_vec;
  output_woj_matches_vec.resize(exec_plan.get_n_devices());

  for (auto _ = 0; _ < exec_plan.get_n_devices(); _++) {
    output_woj_matches_vec[_] = new WOJMatches();
    output_woj_matches_vec[_]->Init(exec_plan.get_n_edges_p(), kMaxNumWeft);
  }

  // Init Streams
  std::vector<cudaStream_t> p_streams_vec;
  p_streams_vec.resize(exec_plan.get_n_devices());

  std::vector<HostKernelBitmap> visited_bm_vec;
  visited_bm_vec.resize(p_streams_vec.size() * exec_plan.get_n_edges_p());

  std::vector<HostKernelBitmap> jump_visited_bm_vec;
  jump_visited_bm_vec.resize(p_streams_vec.size() * exec_plan.get_n_edges_p());

  VertexID *jump_count_ptr;
  CUDA_CHECK(cudaMallocManaged(&jump_count_ptr, sizeof(VertexID) *
                                                    p_streams_vec.size() *
                                                    exec_plan.get_n_edges_p()));
  CUDA_CHECK(cudaMemset(jump_count_ptr, 0,
                        sizeof(VertexID) * p_streams_vec.size() *
                            exec_plan.get_n_edges_p()));

  // Sort candidates.
  Bitmap candidates_visited(input_woj_matches_vec.size());

  Bitmap header_visited(32);
  auto header_ptr = input_woj_matches_vec[0]->get_header_ptr();
  for (auto _ = 0; _ < input_woj_matches_vec[0]->get_x_offset(); _++)
    header_visited.SetBit(header_ptr[_]);

  for (VertexID _ = 1; _ < input_woj_matches_vec.size(); _++) {
    bool sort_tag = false;
    auto header_ptr = input_woj_matches_vec[_]->get_header_ptr();
    for (VertexID __ = 0; __ < input_woj_matches_vec[_]->get_x_offset(); __++) {
      if (header_visited.GetBit(header_ptr[__]) && sort_tag == false) {
        MergeSort(0, input_woj_matches_vec[_]->get_data_ptr(), __,
                  input_woj_matches_vec[_]->get_x_offset(),
                  input_woj_matches_vec[_]->get_y_offset(),
                  sizeof(VertexID) * input_woj_matches_vec[_]->get_y() *
                      input_woj_matches_vec[_]->get_x());
        sort_tag = true;
      }
      header_visited.SetBit(header_ptr[__]);
    }
  }

  // Join candidates
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [step, &dimBlock, &dimGrid, &exec_plan, &p_streams_vec, &mtx,
       &src_matches_vec, &input_woj_matches_vec, &output_woj_matches_vec,
       &visited_bm_vec, &jump_visited_bm_vec, &jump_count_ptr](auto w) {
        for (VertexID s = w; s < p_streams_vec.size(); s += step) {
          cudaSetDevice(common::hash_function(s) % exec_plan.get_n_devices());
          cudaStream_t &stream = p_streams_vec[s];

          cudaStreamCreate(&stream);

          auto left_woj_matches = src_matches_vec[s];

          for (VertexID _ = 1; _ < input_woj_matches_vec.size(); _++) {

            auto right_woj_matches = input_woj_matches_vec[_];

            auto join_keys = left_woj_matches->GetJoinKey(*right_woj_matches);

            visited_bm_vec[s * exec_plan.get_n_edges_p() + _].Init(
                exec_plan.get_n_vertices_g());
            visited_bm_vec[s * exec_plan.get_n_edges_p() + _].ClearAsync(
                stream);
            uint64_t *visited_data =
                visited_bm_vec[s * exec_plan.get_n_edges_p() + _].GetPtr();

            jump_visited_bm_vec[s * exec_plan.get_n_edges_p() + _].Init(
                exec_plan.get_n_vertices_g());
            jump_visited_bm_vec[s * exec_plan.get_n_edges_p() + _].ClearAsync(
                stream);
            uint64_t *jump_visited_data =
                jump_visited_bm_vec[s * exec_plan.get_n_edges_p() + _].GetPtr();

            ParametersWedgeFilter wedge_filter_params{
                .n_vertices_g = exec_plan.get_n_vertices_g(),
                .woj_matches = *right_woj_matches,
                .hash_idx = join_keys.second,
                .visited_data = visited_data};

            GetVisitedByKeyKernel<<<dimGrid, dimBlock, 0, stream>>>(
                wedge_filter_params);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
              CUDA_CHECK(err);
            }
            auto local_matches = new WOJMatches();
            // local_matches->CopyDataAsync(*right_woj_matches, stream);

            if (join_keys.first == kMaxVertexID ||
                join_keys.second == kMaxVertexID)
              continue;

            output_woj_matches_vec[s]->SetHeader(
                left_woj_matches->get_header_ptr(),
                left_woj_matches->get_x_offset(),
                right_woj_matches->get_header_ptr(),
                right_woj_matches->get_x_offset(), join_keys);

            auto *jump_count = jump_count_ptr + s * p_streams_vec.size() + _;

            ParametersJoin params{
                .n_vertices_g = exec_plan.get_n_vertices_g(),
                .left_woj_matches = *left_woj_matches,
                //                    .right_woj_matches = *local_matches,
                .right_woj_matches = *right_woj_matches,
                .output_woj_matches = *output_woj_matches_vec[s],
                .left_hash_idx = join_keys.first,
                .right_hash_idx = join_keys.second,
                .right_visited_data = visited_data,
                .jump_visited_data = jump_visited_data,
                .jump_count = jump_count};

            WOJJoinKernel<<<dimGrid, dimBlock, 0, stream>>>(params);

            cudaStreamSynchronize(stream);

            err = cudaGetLastError();
            if (err != cudaSuccess)
              CUDA_CHECK(err);
            // output_woj_matches_vec[s]->Print();

            if (output_woj_matches_vec[s]->get_y_offset() == 0) {
              break;
            }

            if (output_woj_matches_vec[s]->get_x_offset() ==
                output_woj_matches_vec[s]->get_x()) {
              return;
            } else {

              std::swap(left_woj_matches, output_woj_matches_vec[s]);
              output_woj_matches_vec[s]->Clear();
            }
          }
        }
      });

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    CUDA_CHECK(err);
  }

  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [step, &p_streams_vec, &mtx](auto w) {
                  for (VertexID i = w; i < p_streams_vec.size(); i += step) {
                    cudaStreamDestroy(p_streams_vec[i]);
                  }
                });

  for (VertexID eid = 1; eid < (exec_plan.get_n_edges_p() - 1); eid++) {
    VertexID count = 0;
    VertexID element_count = 0;
    for (VertexID _ = 0; _ < p_streams_vec.size(); _++) {
      count += jump_count_ptr[_ * exec_plan.get_n_edges_p() + eid];
      element_count =
          jump_visited_bm_vec[_ * exec_plan.get_n_edges_p() + eid].Count();
    }
    std::cout << " join eid: " << eid << " jump compute " << count
              << " jump n_elements " << element_count << std::endl;
  }

  cudaFree(jump_count_ptr);
  return output_woj_matches_vec;
}

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics