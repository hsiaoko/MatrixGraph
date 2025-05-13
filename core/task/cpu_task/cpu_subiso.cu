#include <algorithm>
#include <chrono>
#include <execution>
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
using WOJExecutionPlan =
    sics::matrixgraph::core::data_structures::WOJExecutionPlan;
using ExecutionPlan = sics::matrixgraph::core::data_structures::ExecutionPlan;
using BitmapOwnership = sics::matrixgraph::core::util::BitmapOwnership;
using sics::matrixgraph::core::common::kLogWarpSize;
using sics::matrixgraph::core::common::kMaxNumLocalWeft;
using sics::matrixgraph::core::common::kMaxNumWeft;
using sics::matrixgraph::core::common::kMaxVertexID;
using sics::matrixgraph::core::common::kNCUDACoresPerSM;
using sics::matrixgraph::core::common::kNSMsPerGPU;
using sics::matrixgraph::core::common::kNWarpPerCUDACore;
using sics::matrixgraph::core::common::kSharedMemoryCapacity;
using sics::matrixgraph::core::common::kSharedMemorySize;
using sics::matrixgraph::core::common::kWarpSize;
using BitmapOwnership = sics::matrixgraph::core::util::BitmapOwnership;
using BitmapNoOwnerShip = sics::matrixgraph::core::util::BitmapNoOwnerShip;
using sics::matrixgraph::core::common::kDefaultHeapCapacity;

static int filter_count = 0;
static int label_filter_count = 0;
static int label_degree_filter_count = 0;
static int gnn_filter_count = 0;
static int nlc_filter_count = 0;
static int ip_filter_count = 0;
static int index_filter_count = 0;

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

static inline bool LabelDegreeFilter(VertexID u_idx, VertexID v_idx,
                                     const ImmutableCSR& p,
                                     const ImmutableCSR& g) {
  auto u_label = p.GetVLabelBasePointer()[u_idx];
  auto v_label = g.GetVLabelBasePointer()[v_idx];
  if (u_label == v_label)
    return true;
  else
    return false;
  return u_label == v_label &&
         g.GetOutDegreeByLocalID(v_idx) >= p.GetOutDegreeByLocalID(u_idx) &&
         g.GetInDegreeByLocalID(v_idx) >= p.GetInDegreeByLocalID(u_idx);
}

static inline bool LabelFilter(VertexID u_idx, VertexID v_idx,
                               const ImmutableCSR& p, const ImmutableCSR& g) {
  auto u_label = p.GetVLabelBasePointer()[u_idx];
  auto v_label = g.GetVLabelBasePointer()[v_idx];
  return u_label == v_label;
}

static bool NeighborLabelCounterFilter(VertexID u_idx, VertexID v_idx,
                                       const ImmutableCSR& p,
                                       const ImmutableCSR& g) {
  auto u_label = p.GetVLabelBasePointer()[u_idx];
  auto v_label = g.GetVLabelBasePointer()[v_idx];
  if (u_label != v_label) return false;

  MiniKernelBitmap u_label_visited(32);
  MiniKernelBitmap v_label_visited(32);

  auto u = p.GetVertexByLocalID(u_idx);
  auto v = g.GetVertexByLocalID(v_idx);

  auto u_label_ptr = p.GetVLabelBasePointer();
  auto v_label_ptr = g.GetVLabelBasePointer();

  for (VertexID nbr_u_idx = 0; nbr_u_idx < u.outdegree; nbr_u_idx++) {
    VertexID nbr_u = u.incoming_edges[nbr_u_idx];
    VertexLabel u_label = v_label_ptr[nbr_u];
    u_label_visited.SetBit(u_label);
  }

  for (VertexID nbr_v_idx = 0; nbr_v_idx < v.outdegree; nbr_v_idx++) {
    VertexID nbr_v = v.incoming_edges[nbr_v_idx];
    VertexLabel v_label = v_label_ptr[nbr_v];
    v_label_visited.SetBit(v_label);
  }

  return v_label_visited.Count() >= u_label_visited.Count();
}

static bool KMinWiseIPFilter(VertexID u_idx, VertexID v_idx,
                             const ImmutableCSR& p, const ImmutableCSR& g) {
  VertexID max_v_ip_val = 0;
  VertexID min_v_ip_val = kMaxVertexID;
  VertexID max_u_ip_val = 0;
  VertexID min_u_ip_val = kMaxVertexID;

  MiniKernelBitmap u_label_visited(32);
  MiniKernelBitmap v_label_visited(32);

  auto u = p.GetVertexByLocalID(u_idx);
  auto v = g.GetVertexByLocalID(v_idx);

  MinHeap u_k_min_heap;
  MinHeap v_k_min_heap;

  auto u_label_ptr = p.GetVLabelBasePointer();
  auto v_label_ptr = g.GetVLabelBasePointer();

  VertexLabel u_k_min_heap_data[kDefaultHeapCapacity];
  VertexLabel v_k_min_heap_data[kDefaultHeapCapacity];

  for (VertexID nbr_u_idx = 0; nbr_u_idx < u.outdegree; nbr_u_idx++) {
    VertexID nbr_u = u.incoming_edges[nbr_u_idx];
    VertexLabel u_label = v_label_ptr[nbr_u];
    VertexID u_ip_val = kernel::HashTable(u_label);
    u_label_visited.SetBit(u_label);
    u_k_min_heap.Insert(u_ip_val);
  }

  for (VertexID nbr_v_idx = 0; nbr_v_idx < v.outdegree; nbr_v_idx++) {
    VertexID nbr_v = v.incoming_edges[nbr_v_idx];
    VertexLabel v_label = v_label_ptr[nbr_v];
    VertexID v_ip_val = kernel::HashTable(v_label);
    v_label_visited.SetBit(v_label);
    v_k_min_heap.Insert(v_ip_val);
  }

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

  return true;
}

static bool Filter(VertexID u_idx, VertexID v_idx, const ImmutableCSR& p,
                   const ImmutableCSR& g) {
  if (u_idx == kMaxVertexID) return false;
  if (v_idx == kMaxVertexID) return false;
  if (!LabelFilter(u_idx, v_idx, p, g)) {
    __sync_fetch_and_add(&label_filter_count, 1);
    __sync_fetch_and_add(&filter_count, 1);
    return false;
  }
  if (!LabelDegreeFilter(u_idx, v_idx, p, g)) {
    __sync_fetch_and_add(&label_degree_filter_count, 1);
    __sync_fetch_and_add(&filter_count, 1);
    return false;
  }
  if (!NeighborLabelCounterFilter(u_idx, v_idx, p, g)) {
    __sync_fetch_and_add(&nlc_filter_count, 1);
    __sync_fetch_and_add(&filter_count, 1);
    return false;
  }

  return true;
}

static bool MatrixFilter(
    VertexID u_idx, VertexID v_idx, const ImmutableCSR& p,
    const ImmutableCSR& g, const std::vector<Matrix>& m_vec,
    const std::vector<UnifiedOwnedBufferFloat*>& m_unified_buffer_vec) {
  return true;
  auto vec_len = m_vec[0].get_y();

  /// Init similarity vector.

  float sim_vec[vec_len] = {0};

  MatrixOpsKernelWrapper::CPUSimdSquaredDifference(
      m_vec[0].GetPtr() + u_idx * vec_len, m_vec[1].GetPtr() + v_idx * vec_len,
      sim_vec, vec_len);

  float z1[8] = {0};
  float z2[1] = {0};

  MatrixOpsKernelWrapper::CPUMatMult(sim_vec, m_vec[2].GetPtr(), z1, 1,
                                     m_vec[2].get_x(), m_vec[2].get_y(), false,
                                     true);

  MatrixOpsKernelWrapper::CPUMatAdd(z1, m_vec[3].GetPtr(), m_vec[3].get_x(),
                                    m_vec[3].get_y());

  MatrixOpsKernelWrapper::CPURelu(z1, m_vec[3].get_x(), m_vec[3].get_y());

  MatrixOpsKernelWrapper::CPUMatMult(z1, m_vec[4].GetPtr(), z2, 1,
                                     m_vec[4].get_x(), m_vec[4].get_y(), false,
                                     true);
  MatrixOpsKernelWrapper::CPUMatAdd(z2, m_vec[5].GetPtr(), m_vec[5].get_x(),
                                    m_vec[5].get_y());

  MatrixOpsKernelWrapper::CPUSigmoid(z2, 1, 1);

  // std::cout << z2[0] << std::endl;
  if (z2[0] < 0.8) {
    return false;
  }

  return true;
}

static bool GPUMatrixFilter(
    VertexID u_idx, VertexID v_idx, const ImmutableCSR& p,
    const ImmutableCSR& g, const std::vector<Matrix>& m_vec,
    const std::vector<UnifiedOwnedBufferFloat*>& m_unified_buffer_vec) {
  std::cout << "u_idx:" << u_idx << " v_idx: " << v_idx << std::endl;
  // return true;
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

  if (*z2.GetPtr() < 0.5) return false;

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
    woj_matches_vec[_]->Init(exec_plan.get_n_edges_p(), kMaxNumWeft);
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

    std::for_each(
        // std::execution::par,
        worker.begin(), worker.end(),
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

  std::for_each(
      // std::execution::par,
      worker.begin(), worker.end(),
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
              if (output_y_offset > kMaxNumWeft / output_x_offset) break;

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
              if (output_y_offset > kMaxNumWeft / output_x_offset) break;

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
  output_woj_matches->Init(exec_plan.get_n_edges_p(), kMaxNumWeft);

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
    LocalMatches* localMatches) {
  bool src_tag = false;
  bool dst_tag = false;

  if (u_src == kMaxVertexID && v_src == kMaxVertexID) {
    src_tag = true;
  } else if (u_src != kMaxVertexID && v_src != kMaxVertexID) {
    src_tag = Filter(u_src, v_src, p, g);
    if (!src_tag) return false;

    // src_tag = KMinWiseIPFilter(u_src, v_src, p, g);
    // if (!src_tag) {
    //   __sync_fetch_and_add(&ip_filter_count, 1);
    //   return false;
    // }
  }

  dst_tag = Filter(u_dst, v_dst, p, g);
  if (!dst_tag) return false;

  // dst_tag = KMinWiseIPFilter(u_dst, v_dst, p, g);
  // if (!dst_tag) {
  //   __sync_fetch_and_add(&ip_filter_count, 1);
  //   return false;
  // }
  // if (!dst_tag) return false;

  return true;
}

static void DFSExtend(
    const ImmutableCSR& p, const ImmutableCSR& g,
    const ExecutionPlan& exec_plan, const std::vector<Matrix>& m_vec,
    const std::vector<UnifiedOwnedBufferFloat*>& m_unified_buffer_vec,
    VertexID level, VertexID pre_v_idx, VertexID v_idx,
    std::vector<BitmapOwnership>& matches_visited_vec,
    LocalMatches* local_matches, bool match) {
  if (level > exec_plan.get_depth()) {
    return;
  }

  bool extend_tag = false;

  for (auto _ = 0; _ < exec_plan.get_n_vertices(); _++) {
    auto u_src = exec_plan.get_exec_path_in_edges_ptr()[2 * _];
    auto u_dst = exec_plan.get_exec_path_in_edges_ptr()[2 * _ + 1];

    if ((pre_v_idx == kMaxVertexID) ^ (u_src == kMaxVertexID)) continue;

    // if (u_src != kMaxVertexID) {
    // if (!matches_visited_vec[u_src].GetBit(pre_v_idx)) {
    //   continue;
    // }
    // }

    if (IsFeasible(p, g, m_vec, m_unified_buffer_vec, u_src, u_dst, pre_v_idx,
                   v_idx, local_matches)) {
      if (level == 1) {
        if (!MatrixFilter(u_src, pre_v_idx, p, g, m_vec,
                          m_unified_buffer_vec)) {
          __sync_fetch_and_add(&gnn_filter_count, 1);
          return;
        }
      }
      VertexID offset = local_matches->size[_];
      if (offset > kMaxNumLocalWeft - 1) return;
      local_matches->data[kMaxNumLocalWeft * 2 * _ + 2 * offset + 1] =
          g.GetGloablIDBasePointer()[v_idx];
      local_matches->size[_]++;
      if (pre_v_idx != kMaxVertexID) {
        local_matches->data[kMaxNumLocalWeft * 2 * _ + 2 * offset] =
            g.GetGloablIDBasePointer()[pre_v_idx];
      }
      // if (u_src != kMaxVertexID) {
      //   matches_visited_vec[u_src].SetBit(pre_v_idx);
      // }
      // matches_visited_vec[u_dst].SetBit(v_idx);

      extend_tag = true;
    }
  }

  if (extend_tag) {
    auto v = g.GetVertexByLocalID(v_idx);
    for (VertexID nbr_idx = 0; nbr_idx < g.GetOutDegreeByLocalID(v_idx);
         nbr_idx++) {
      DFSExtend(p, g, exec_plan, m_vec, m_unified_buffer_vec, level + 1, v_idx,
                v.outgoing_edges[nbr_idx], matches_visited_vec, local_matches,
                match);
    };
  }
}

static inline void Enumerating(
    const ImmutableCSR& p, const ImmutableCSR& g,
    const ExecutionPlan& exec_plan, const std::vector<Matrix>& m_vec,
    const std::vector<UnifiedOwnedBufferFloat*>& m_unified_buffer_vec,
    Matches* matches) {
  // auto parallelism = std::thread::hardware_concurrency();
  auto parallelism = 1;
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  std::vector<LocalMatches> local_matches_vec;
  local_matches_vec.resize(parallelism);
  std::generate(local_matches_vec.begin(), local_matches_vec.end(), [&p]() {
    LocalMatches local_matches;
    local_matches.data =
        new VertexID[p.get_num_vertices() * 2 * kMaxNumLocalWeft]();
    local_matches.size = new VertexID[p.get_num_vertices()]();
    return local_matches;
  });

  std::vector<std::vector<BitmapOwnership>> matches_visited_vec_vec;
  matches_visited_vec_vec.resize(parallelism);
  std::generate(matches_visited_vec_vec.begin(), matches_visited_vec_vec.end(),
                [&p, &g]() {
                  std::vector<BitmapOwnership> visited_vec;
                  visited_vec.resize(p.get_num_vertices(),
                                     g.get_num_vertices());
                  return visited_vec;
                });

  std::cout << "Enumerating" << std::endl;
  std::for_each(
      // std::execution::par,
      worker.begin(), worker.end(),
      [step, &mtx, &p, &g, &exec_plan, &m_vec, &m_unified_buffer_vec, &matches,
       &local_matches_vec, matches_visited_vec_vec](auto w) {
        auto matches_visited_vec = matches_visited_vec_vec[w];
        auto local_matches = local_matches_vec[w];

        for (VertexID v_idx = w; v_idx < g.get_num_vertices(); v_idx += step) {
          bool match = false;
          DFSExtend(p, g, exec_plan, m_vec, m_unified_buffer_vec, 0,
                    kMaxVertexID, v_idx, matches_visited_vec, &local_matches,
                    match);
          {
            std::lock_guard<std::mutex> lock(mtx);
            if (local_matches.size[p.get_num_vertices() - 1] != 0) {
              auto weft_idx =
                  __sync_fetch_and_add(matches->GetWeftCountPtr(), 1);

              if (weft_idx >= kMaxNumWeft - 1) return;
              int weft_size = 0;
              for (int _ = 0; _ < p.get_num_vertices(); _++) {
                weft_size += local_matches.size[_];
                matches->GetVCandidateOffsetPtr()
                    [weft_idx * (p.get_num_vertices() + 1) + _ + 1] =
                    matches->GetVCandidateOffsetPtr()
                        [weft_idx * (p.get_num_vertices() + 1) + _] +
                    local_matches.size[_];

                memcpy(matches->GetDataPtr() + weft_idx * p.get_num_vertices() *
                                                   2 * kMaxNumLocalWeft,
                       local_matches.data,
                       p.get_num_vertices() * 2 * kMaxNumLocalWeft *
                           sizeof(VertexID));
              }
            }
            memset(local_matches.data, 0,
                   sizeof(VertexID) * p.get_num_vertices() * kMaxNumLocalWeft);
            memset(local_matches.size, 0,
                   sizeof(VertexID) * p.get_num_vertices());
            // for (auto _ = 0; _ < matches_visited_vec.size(); _++) {
            //   matches_visited_vec[_].Clear();
            // }
          }
        }
      });
}

static void Checking(const ImmutableCSR& p, const ImmutableCSR& g,
                     const ExecutionPlan& exec_plan, Matches* matches) {
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

  auto visited_edges_count = 0;
  while (!frontier.empty()) {
    auto u_vid = frontier.front();
    frontier.pop();
    if (edges_visited.Count() == p.get_num_outgoing_edges()) {
      break;
    }

    auto u = p.GetVertexByLocalID(u_vid);
    for (auto nbr_idx = 0; nbr_idx < u.outdegree; nbr_idx++) {
      auto nbr_vid = u.outgoing_edges[nbr_idx];

      if (edges_visited.GetBit(u_vid * p.get_num_vertices() + nbr_vid)) {
        continue;
      } else {
        edges_visited.SetBit(u_vid * p.get_num_vertices() + nbr_vid);
      }

      frontier.push(nbr_vid);

      if (exec_plan.IsInExecPathInEdges(u_vid, nbr_vid)) {
        continue;
      }

      for (auto h_src_idx = 0; h_src_idx < header.size(); h_src_idx++) {
        if (header[h_src_idx].first == u_vid &&
            header[h_src_idx].second == nbr_vid)
          continue;
        auto bias_src = 0;
        auto head_src = kMaxVertexID;
        if (header[h_src_idx].first == u_vid) {
          head_src = h_src_idx;
        } else if (header[h_src_idx].second == u_vid) {
          head_src = h_src_idx;
          bias_src = 1;
        }
        if (head_src == kMaxVertexID) continue;

        for (auto h_dst_idx = 0; h_dst_idx < header.size(); h_dst_idx++) {
          if (header[h_dst_idx].first == u_vid &&
              header[h_dst_idx].second == nbr_vid)
            continue;
          auto bias_dst = 0;
          auto head_dst = kMaxVertexID;
          if (header[h_dst_idx].first == nbr_vid) {
            head_dst = h_dst_idx;
          } else if (header[h_dst_idx].second == nbr_vid) {
            head_dst = h_dst_idx;
            bias_dst = 1;
          }
          if (head_dst == kMaxVertexID) continue;

          /// Check whether candidates at head_src and candidates at head_dst
          /// have an edge in graph g.
          for (VertexID weft_id = 0; weft_id < matches->get_weft_count();
               weft_id++) {
            auto src_candidate_offset =
                matches->GetVCandidateOffsetPtr()
                    [weft_id * (matches->get_n_vertices() + 1) + head_src];

            auto src_candidate_size =
                matches->GetVCandidateOffsetPtr()
                    [weft_id * (matches->get_n_vertices() + 1) + head_src + 1] -
                matches->GetVCandidateOffsetPtr()
                    [weft_id * (matches->get_n_vertices() + 1) + head_src];

            auto dst_candidate_offset =
                matches->GetVCandidateOffsetPtr()
                    [weft_id * (matches->get_n_vertices() + 1) + head_dst];
            auto dst_candidate_size =
                matches->GetVCandidateOffsetPtr()
                    [weft_id * (matches->get_n_vertices() + 1) + head_dst + 1] -
                matches->GetVCandidateOffsetPtr()
                    [weft_id * (matches->get_n_vertices() + 1) + head_dst];

            for (auto candidates_idx_src = 0;
                 candidates_idx_src < src_candidate_size;
                 candidates_idx_src++) {
              auto candidate_src =
                  *(matches->GetDataPtr() +
                    weft_id * matches->get_n_vertices() * 2 *
                        matches->get_max_n_local_weft() +
                    h_src_idx * 2 * matches->get_max_n_local_weft() +
                    2 * candidates_idx_src + bias_src);
              auto src_connect_count = 0;
              if (candidate_src == kMaxVertexID) continue;
              for (auto candidates_idx_dst = 0;
                   candidates_idx_dst < dst_candidate_size;
                   candidates_idx_dst++) {
                auto candidate_dst =
                    *(matches->GetDataPtr() +
                      weft_id * matches->get_n_vertices() * 2 *
                          matches->get_max_n_local_weft() +
                      h_dst_idx * 2 * matches->get_max_n_local_weft() +
                      2 * candidates_idx_dst + bias_dst);
                if (candidate_dst == kMaxVertexID) continue;

                if (g.IsConnected(candidate_src, candidate_dst)) {
                  src_connect_count++;
                  break;
                }
              }

              if (src_connect_count == 0) {
                *(matches->GetDataPtr() +
                  weft_id * matches->get_n_vertices() * 2 *
                      matches->get_max_n_local_weft() +
                  h_src_idx * 2 * matches->get_max_n_local_weft() +
                  2 * candidates_idx_src) = kMaxVertexID;
                *(matches->GetDataPtr() +
                  weft_id * matches->get_n_vertices() * 2 *
                      matches->get_max_n_local_weft() +
                  h_src_idx * 2 * matches->get_max_n_local_weft() +
                  2 * candidates_idx_src + 1) = kMaxVertexID;
              }
            }
          }
        }
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

  std::cout << p.get_num_vertices() << " " << g.get_num_vertices() << std::endl;
  Matches matches(p.get_num_vertices(), kMaxNumWeft, kMaxNumLocalWeft,
                  g.get_num_vertices());

  // Generate Execution Plan...
  ExecutionPlan exec_plan;
  exec_plan.GenerateDFSExecutionPlan(p, g);

  exec_plan.Print();

  // Set header of matches
  for (auto _ = 0; _ < p.get_num_vertices(); _++) {
    auto src =
        exec_plan.get_sequential_exec_path_in_edges_ptr()->GetPtr()[_ * 2];
    auto dst =
        exec_plan.get_sequential_exec_path_in_edges_ptr()->GetPtr()[_ * 2 + 1];
    matches.SetHeader(_, std::make_pair(src, dst));
  }

  auto start_time_0 = std::chrono::system_clock::now();
  // Enumerating ...
  Enumerating(p, g, exec_plan, m_vec, m_unified_buffer_vec, &matches);
  std::cout << "\t weft count: " << matches.get_weft_count() << std::endl;

  auto start_time_1 = std::chrono::system_clock::now();

  auto start_time_2 = std::chrono::system_clock::now();

  // Checking ...
  Checking(p, g, exec_plan, &matches);
  std::cout << "update" << std::endl;
  matches.UpdateInvalidMatches();
  auto start_time_3 = std::chrono::system_clock::now();
  matches.Print(3);
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
  ;
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

  auto* g_vlabel = g_.GetVLabelBasePointer();
  auto* p_vlabel = p_.GetVLabelBasePointer();

  p_.PrintGraph(10);
  g_.PrintGraph(0);

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
    m_vec_[2].Print(99);
    m_vec_[3].Print(99);
    m_vec_[4].Print(99);
    m_vec_[5].Print(99);

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

  // WOJMatching(p_, g_, m_vec_);
  RecursiveMatching(p_, g_, m_vec_, m_unified_buffer_vec_);
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
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   start_time_1 - start_time_0)
                       .count() /
                   (float)CLOCKS_PER_SEC
            << " sec" << std::endl;

  std::cout << "Matching time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   end_time - start_time_1)
                       .count() /
                   (float)CLOCKS_PER_SEC
            << " sec" << std::endl;

  std::cout << "Total execution time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   end_time - start_time_0)
                       .count() /
                   (float)CLOCKS_PER_SEC
            << " sec" << std::endl;
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
