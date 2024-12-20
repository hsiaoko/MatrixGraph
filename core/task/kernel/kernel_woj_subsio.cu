#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

#include "core/common/consts.h"
#include "core/common/host_algorithms.cuh"
#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/host_buffer.cuh"
#include "core/data_structures/unified_buffer.cuh"
#include "core/task/kernel/data_structures/hash_buckets.cuh"
#include "core/task/kernel/data_structures/immutable_csr_gpu.cuh"
#include "core/task/kernel/data_structures/kernel_bitmap.cuh"
#include "core/task/kernel/data_structures/mini_kernel_bitmap.cuh"
#include "core/task/kernel/kernel_woj_subiso.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
using VertexID = sics::matrixgraph::core::common::VertexID;
using sics::matrixgraph::core::common::kMaxNumCandidatesPerThread;
using sics::matrixgraph::core::common::kMaxNumWeft;
using sics::matrixgraph::core::common::kMaxVertexID;
using sics::matrixgraph::core::task::kernel::HostKernelBitmap;
using sics::matrixgraph::core::task::kernel::HostMiniKernelBitmap;
using sics::matrixgraph::core::task::kernel::KernelBitmap;
using sics::matrixgraph::core::task::kernel::MiniKernelBitmap;
using HashBuckets = sics::matrixgraph::core::task::kernel::HashBuckets;
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
  HashBuckets hash_buckets;
};

__forceinline__ __device__ unsigned lane_id() {
  unsigned ret;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
  return ret;
}

__forceinline__ __device__ unsigned warp_id() {
  unsigned ret;
  asm volatile("mov.u32 %0, %warpid;" : "=r"(ret));
  return ret;
}

static __forceinline__ __device__ bool
LabelFilter(const ParametersFilter &params, VertexID u_idx, VertexID v_idx) {
  VertexID *globalid_g = (VertexID *)(params.data_g);
  VertexLabel v_label = params.v_label_g[globalid_g[v_idx]];
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
    return out_degree_g[v_idx] >= out_degree_p[u_idx];
  }
}

static __forceinline__ __device__ bool Filter(const ParametersFilter &params,
                                              VertexID u_idx, VertexID v_idx) {

  return LabelFilter(params, u_idx, v_idx);

  // return NeighborLabelCounterFilter(params, u_idx, v_idx);

  // return LabelDegreeFilter(params, u_idx, v_idx);
}

static __noinline__ __global__ void WOJExtendKernel(ParametersFilter params) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  enum : unsigned { warp_size = 32, log_warp_size = 5 };
  auto lane_id = threadIdx.x & (warp_size - 1);
  auto warp_id = threadIdx.x >> log_warp_size;

  __shared__ VertexID local_matches_data[2048];
  __shared__ VertexID local_matches_offset;

  VertexID u_eid = params.u_eid;
  VertexID u_src = params.exec_path_in_edges[2 * u_eid];
  VertexID u_dst = params.exec_path_in_edges[2 * u_eid + 1];
  for (VertexID e_idx = tid; e_idx < params.n_edges_g; e_idx += step) {
    VertexID v_src = params.edgelist_g[2 * e_idx];
    VertexID v_dst = params.edgelist_g[2 * e_idx + 1];
    bool src_tag = true;
    bool dst_tag = true;

    if (u_src != -1) {
      src_tag = Filter(params, u_src, v_src);
    }
    if (u_dst != -1) {
      dst_tag = Filter(params, u_dst, v_dst);
    }
    if (src_tag && dst_tag) {
      VertexID offset = atomicAdd(&local_matches_offset, 1);
      local_matches_data[2 * offset] = v_src;
      local_matches_data[2 * offset + 1] = v_dst;
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    auto offset =
        atomicAdd(&params.hash_buckets.offset_[u_eid], local_matches_offset);
    memcpy(params.hash_buckets.data_[u_eid] + 2 * offset, local_matches_data,
           sizeof(VertexID) * 2 * local_matches_offset);
  }
}

void WOJSubIsoKernelWrapper::Filter(const ImmutableCSR &p,
                                    const ImmutableCSR &g, const Edges &e,
                                    const ExecutionPlan &exec_plan) {
  std::cout << "Filter" << std::endl;
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;

  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  // Init Streams
  std::vector<cudaStream_t> p_streams_vec;
  p_streams_vec.resize(p.get_num_outgoing_edges());
  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [step, &p_streams_vec, &mtx](auto w) {
                  for (VertexID i = w; i < p_streams_vec.size(); i += step) {
                    //                 cudaSetDevice(common::hash_function(i) %
                    //                 4);
                    cudaStreamCreate(&p_streams_vec[i]);
                  }
                });

  // Init pattern.
  BufferUint8 data_p;
  BufferVertexLabel v_label_p;
  BufferVertexID buffer_exec_path;
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

  buffer_exec_path.data = exec_plan.get_exec_path_ptr();
  buffer_exec_path.size = sizeof(VertexID) * p.get_num_vertices();

  buffer_exec_path_in_edges.data = exec_plan.get_exec_path_in_edges_ptr();
  buffer_exec_path_in_edges.size = sizeof(VertexID) * p.get_num_vertices() * 2;

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

  data_edgelist_g.data = (VertexID *)e.get_base_ptr();
  data_edgelist_g.size = sizeof(VertexID) * e.get_metadata().num_edges * 2;

  //  Init output.
  std::vector<HashBuckets> hash_buckets_vec;
  hash_buckets_vec.resize(4);

  std::vector<ImmutableCSRGPU> data_graph_gpu_vec;
  data_graph_gpu_vec.resize(4);
  std::vector<ImmutableCSRGPU> pattern_graph_gpu_vec;
  pattern_graph_gpu_vec.resize(4);

  std::vector<UnifiedOwnedBufferVertexID> exec_path_in_edges_vec;
  exec_path_in_edges_vec.resize(4);
  std::vector<UnifiedOwnedBufferUint8> data_p_vec;
  data_p_vec.resize(4);
  std::vector<UnifiedOwnedBufferVertexLabel> v_label_p_vec;
  v_label_p_vec.resize(4);
  std::vector<UnifiedOwnedBufferUint8> data_g_vec;
  data_g_vec.resize(4);
  std::vector<UnifiedOwnedBufferVertexID> edgelist_g_vec;
  edgelist_g_vec.resize(4);
  std::vector<UnifiedOwnedBufferVertexLabel> v_label_g_vec;
  v_label_g_vec.resize(4);

  for (VertexID _ = 0; _ < 4; _++) {
    data_graph_gpu_vec[_].Init(g);
    pattern_graph_gpu_vec[_].Init(p);
    exec_path_in_edges_vec[_].Init(buffer_exec_path_in_edges);
    data_p_vec[_].Init(data_p);
    v_label_p_vec[_].Init(v_label_p);
    data_g_vec[_].Init(data_g);
    edgelist_g_vec[_].Init(data_edgelist_g);
    v_label_g_vec[_].Init(v_label_g);
    hash_buckets_vec[_].Init(p.get_num_outgoing_edges(), kMaxNumWeft);
  }

  dim3 dimBlock(1);
  dim3 dimGrid(1);

  auto time1 = std::chrono::system_clock::now();
  for (VertexID _ = 0; _ < p.get_num_outgoing_edges(); _++) {
    VertexID device_id = common::hash_function(_) % 4;
    //  cudaSetDevice(device_id);
    cudaStream_t &stream = p_streams_vec[_];
    ParametersFilter params{
        .u_eid = _,
        .exec_path_in_edges = exec_path_in_edges_vec[device_id].GetPtr(),
        .n_vertices_p = p.get_num_vertices(),
        .n_edges_p = p.get_num_outgoing_edges(),
        .data_p = data_p_vec[device_id].GetPtr(),
        .v_label_p = v_label_p_vec[device_id].GetPtr(),
        .n_vertices_g = g.get_num_vertices(),
        .n_edges_g = g.get_num_outgoing_edges(),
        .data_g = data_g_vec[device_id].GetPtr(),
        .edgelist_g = edgelist_g_vec[device_id].GetPtr(),
        .v_label_g = v_label_g_vec[device_id].GetPtr(),
        .hash_buckets = hash_buckets_vec[0],
    };
    // for (auto __ = 0; __ < p.get_num_outgoing_edges(); __++) {
    //   std::cout << params.exec_path_in_edges[2 * __]
    //             << params.exec_path_in_edges[2 * __ + 1] << std::endl;
    // }
    WOJExtendKernel<<<dimGrid, dimBlock, 0, stream>>>(params);
  }

  cudaDeviceSynchronize();

  auto time2 = std::chrono::system_clock::now();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    CUDA_CHECK(err);
  }

  for (auto j = 0; j < p.get_num_vertices(); j++) {
    std::cout << ", offset: " << hash_buckets_vec[0].offset_[j] << std::endl;
    for (VertexID eid = 0; eid < hash_buckets_vec[0].offset_[j]; eid++) {
      std::cout << hash_buckets_vec[0].data_[j][2 * eid] << " -> "
                << hash_buckets_vec[0].data_[j][2 * eid + 1] << std::endl;
    }
  }
  std::cout << "-------------" << std::endl;

  std::cout << "[Filter]:"
            << std::chrono::duration_cast<std::chrono::microseconds>(time2 -
                                                                     time1)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;

  for (VertexID _ = 0; _ < 4; _++) {
    pattern_graph_gpu_vec[_].Free();
    data_graph_gpu_vec[_].Free();
  }
}

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics