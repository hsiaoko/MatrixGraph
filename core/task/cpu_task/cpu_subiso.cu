#include <algorithm>
#include <chrono>
#include <execution>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "core/common/consts.h"
#include "core/common/types.h"
#include "core/data_structures/exec_plan.cuh"
#include "core/data_structures/matches.cuh"
#include "core/data_structures/metadata.h"
#include "core/data_structures/woj_exec_plan.cuh"
#include "core/data_structures/woj_matches.cuh"
#include "core/task/cpu_task/cpu_subiso.cuh"
#include "core/task/gpu_task/kernel/algorithms/sort.cuh"
#include "core/util/bitmap_ownership.h"
#include "core/util/format_converter.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using VertexID = sics::matrixgraph::core::common::VertexID;
using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;
using Edges = sics::matrixgraph::core::data_structures::Edges;
using Edge = sics::matrixgraph::core::data_structures::Edge;
using GraphMetadata = sics::matrixgraph::core::data_structures::GraphMetadata;
using WOJMatches = sics::matrixgraph::core::data_structures::WOJMatches;
using Matches = sics::matrixgraph::core::data_structures::Matches;
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

struct LocalMatches {
  VertexID* data = nullptr;
  VertexID* size = nullptr;
};

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

static inline bool Filter(VertexID u_idx, VertexID v_idx, const ImmutableCSR& p,
                          const ImmutableCSR& g) {
  // return LabelFilter(u_idx, v_idx, p, g);
  return LabelDegreeFilter(u_idx, v_idx, p, g);
}

static std::vector<WOJMatches*> WOJFilter(const WOJExecutionPlan& exec_plan,
                                          const ImmutableCSR& p,
                                          const ImmutableCSR& g) {
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
         &woj_matches_vec](auto w) {
          for (VertexID v_idx = w; v_idx < g.get_num_vertices();
               v_idx += step) {
            auto offset = g.GetOutOffsetByLocalID(v_idx);
            auto degree = g.GetOutDegreeByLocalID(v_idx);
            auto* out_edges = g.GetOutgoingEdgesByLocalID(v_idx);
            VertexID global_id = g.GetGlobalIDByLocalID(v_idx);
            if (Filter(src_idx, v_idx, p, g)) {
              for (VertexID nbr_v_idx = 0; nbr_v_idx < degree; nbr_v_idx++) {
                VertexID nbr_v = out_edges[nbr_v_idx];
                VertexID nbr_localid = g.GetLocalIDByGlobalID(nbr_v);
                if (Filter(dst_idx, nbr_localid, p, g)) {
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

static inline void DFSExtend(const ImmutableCSR& p, const ImmutableCSR& g,
                             const ExecutionPlan& exec_plan, VertexID level,
                             VertexID pre_v_idx, VertexID v_idx,
                             BitmapOwnership& global_visited,
                             BitmapOwnership local_visited,
                             BitmapOwnership* level_visited_ptr_array,
                             LocalMatches* local_matches, bool match) {
  if (level > exec_plan.get_depth()) {
    return;
  }

  unsigned exec_plan_idx = local_visited.Count();
  unsigned global_exec_plan_idx = global_visited.Count();

  if (global_exec_plan_idx == p.get_num_vertices()) {
    return;
  }
  if (local_matches->size[exec_plan_idx] >= kMaxNumLocalWeft) {
    return;
  }
  if (local_matches->size[global_exec_plan_idx] >= kMaxNumLocalWeft) {
    return;
  }

  VertexID u = exec_plan.get_exec_path_ptr()[exec_plan_idx];

  VertexLabel u_label = p.GetVLabelBasePointer()[exec_plan_idx];

  VertexID global_u = exec_plan.get_exec_path_ptr()[global_exec_plan_idx];
  VertexLabel global_u_label = p.GetVLabelBasePointer()[global_exec_plan_idx];

  VertexLabel v_label = g.GetVLabelBasePointer()[v_idx];

  VertexID global_pre_u = exec_plan.get_sequential_exec_path_in_edges_ptr()
                              ->GetPtr()[2 * global_exec_plan_idx];
  VertexID local_pre_u = exec_plan.get_sequential_exec_path_in_edges_ptr()
                             ->GetPtr()[2 * exec_plan_idx];

  bool local_match_tag = false;
  bool global_match_tag = false;
  bool extend_tag = false;

  // If true, we need Check both global u and local u. it false we only need to
  // Check local u.
  local_match_tag = Filter(u, v_idx, p, g);
  global_match_tag = Filter(global_u, v_idx, p, g);

  if (global_u == u) {
    if (local_match_tag) {
      local_visited.SetBit(u);
      global_visited.SetBit(u);
      VertexID offset = local_matches->size[exec_plan_idx];
      local_matches->size[exec_plan_idx]++;
      if (pre_v_idx != kMaxVertexID) {
        local_matches->data[kMaxNumWeft * 2 * exec_plan_idx + 2 * offset] =
            g.GetGloablIDBasePointer()[pre_v_idx];
      }
      local_matches->data[kMaxNumWeft * 2 * exec_plan_idx + 2 * offset + 1] =
          g.GetGloablIDBasePointer()[v_idx];

      if (!level_visited_ptr_array[exec_plan_idx].GetBit(v_idx)) {
        level_visited_ptr_array[exec_plan_idx].SetBit(v_idx);
        extend_tag = true;
      }
    }
  } else {
    if (local_match_tag) {
      local_visited.SetBit(u);
      VertexID offset = local_matches->size[exec_plan_idx];
      local_matches->size[exec_plan_idx]++;
      if (pre_v_idx != kMaxVertexID) {
        local_matches->data[kMaxNumWeft * 2 * exec_plan_idx + 2 * offset] =
            g.GetGloablIDBasePointer()[pre_v_idx];
      }
      local_matches->data[kMaxNumWeft * 2 * exec_plan_idx + 2 * offset + 1] =
          g.GetGloablIDBasePointer()[v_idx];

      if (!level_visited_ptr_array[exec_plan_idx].GetBit(v_idx)) {
        level_visited_ptr_array[exec_plan_idx].SetBit(v_idx);
        extend_tag = true;
      }
    }

    if (global_match_tag) {
      local_visited.SetBit(global_u);
      VertexID offset = local_matches->size[global_exec_plan_idx];
      if (pre_v_idx != kMaxVertexID) {
        local_matches
            ->data[kMaxNumWeft * 2 * global_exec_plan_idx + 2 * offset] =
            g.GetGloablIDBasePointer()[pre_v_idx];
      }
      local_matches
          ->data[kMaxNumWeft * 2 * global_exec_plan_idx + 2 * offset + 1] =
          g.GetGloablIDBasePointer()[v_idx];
      local_matches->size[global_exec_plan_idx]++;
      if (!level_visited_ptr_array[global_exec_plan_idx].GetBit(v_idx)) {
        level_visited_ptr_array[global_exec_plan_idx].SetBit(v_idx);
        extend_tag = true;
      }
    }
  }

  if (extend_tag) {
    auto v = g.GetVertexByLocalID(v_idx);
    for (VertexID nbr_idx = 0; nbr_idx < g.GetOutDegreeByLocalID(v_idx);
         nbr_idx++) {
      DFSExtend(p, g, exec_plan, level + 1, v_idx, v.outgoing_edges[nbr_idx],
                global_visited, local_visited, level_visited_ptr_array,
                local_matches, match);
    };
  }
}

static inline void Enumerating(const ImmutableCSR& p, const ImmutableCSR& g,
                               const ExecutionPlan& exec_plan,
                               Matches* matches) {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  std::for_each(
      // std::execution::par,
      worker.begin(), worker.end(),
      [step, &p, &g, &exec_plan, &matches](auto w) {
        BitmapOwnership visited(p.get_num_vertices());

        auto level_visited_ptr_array =
            new BitmapOwnership[p.get_num_vertices()](1 << 15);

        LocalMatches local_matches;
        local_matches.data =
            new VertexID[p.get_num_vertices() * 2 * kMaxNumLocalWeft]();
        local_matches.size = new VertexID[p.get_num_vertices()]();

        for (VertexID v_idx = w; v_idx < g.get_num_vertices(); v_idx += step) {
          visited.Clear();
          bool match = false;

          DFSExtend(p, g, exec_plan, 0, kMaxVertexID, v_idx, visited, visited,
                    level_visited_ptr_array, &local_matches, match);
          if (local_matches.size[p.get_num_vertices() - 1] != 0) {
            auto weft_idx = __sync_fetch_and_add(matches->GetWeftCountPtr(), 1);

            int weft_size = 0;
            for (int _ = 0; _ < p.get_num_vertices(); _++) {
              weft_size += local_matches.size[_];
              matches->GetVCandidateOffsetPtr()[weft_idx *
                                                    (p.get_num_vertices() + 1) +
                                                _ + 1] =
                  matches->GetVCandidateOffsetPtr()
                      [weft_idx * (p.get_num_vertices() + 1) + _] +
                  local_matches.size[_];
            }
            memcpy(
                matches->GetDataPtr() +
                    weft_idx * p.get_num_vertices() * 2 * kMaxNumLocalWeft,
                local_matches.data,
                p.get_num_vertices() * 2 * kMaxNumLocalWeft * sizeof(VertexID));
          }
          memset(local_matches.data, 0,
                 sizeof(VertexID) * p.get_num_vertices() * kMaxNumLocalWeft);
          memset(local_matches.size, 0,
                 sizeof(VertexID) * p.get_num_vertices());
        }
      });

  matches->Print();
}

void CPUSubIso::RecursiveMatching(const ImmutableCSR& p,
                                  const ImmutableCSR& g) {
  std::cout << "Matching ..." << std::endl;
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;

  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  Matches matches(p.get_num_vertices(), kMaxNumWeft, kMaxNumLocalWeft);

  // Generate Execution Plan...
  ExecutionPlan exec_plan;
  exec_plan.GenerateDFSExecutionPlan(p, g);

  exec_plan.Print();

  // Enumerating...
  Enumerating(p, g, exec_plan, &matches);
}

void CPUSubIso::WOJMatching(const ImmutableCSR& p, const ImmutableCSR& g) {
  WOJExecutionPlan exec_plan;
  exec_plan.GenerateWOJExecutionPlan(p, g);

  auto start_time_0 = std::chrono::system_clock::now();
  auto woj_matches = WOJFilter(exec_plan, p, g);

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
                   (double)CLOCKS_PER_SEC
            << " sec" << std::endl;
  std::cout << "[WOJMatching] Join() elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_2 - start_time_1)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << " sec" << std::endl;
}

void CPUSubIso::LoadData() {
  std::cout << "[CPUSubIso] LoadData() ..." << std::endl;

  p_.Read(pattern_path_);

  g_.Read(data_graph_path_);

  auto* g_vlabel = g_.GetVLabelBasePointer();
  auto* p_vlabel = p_.GetVLabelBasePointer();

  p_vlabel[0] = 0;
  p_vlabel[1] = 1;
  p_vlabel[2] = 2;
  p_vlabel[3] = 3;
  p_vlabel[4] = 4;
  p_vlabel[5] = 3;
  g_vlabel[0] = 0;
  g_vlabel[1] = 1;
  g_vlabel[2] = 2;
  g_vlabel[3] = 3;
  g_vlabel[4] = 3;
  g_vlabel[4] = 3;
  g_vlabel[5] = 4;
  g_vlabel[6] = 4;
  g_vlabel[7] = 1;
  g_vlabel[8] = 0;
  p_.PrintGraph(10);
  g_.PrintGraph();
}

void CPUSubIso::Run() {
  auto start_time_0 = std::chrono::system_clock::now();
  LoadData();
  auto start_time_1 = std::chrono::system_clock::now();

  std::cout << "Data loading time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   start_time_1 - start_time_0)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << " sec" << std::endl;

  // WOJMatching(p_, g_);
  RecursiveMatching(p_, g_);

  auto end_time = std::chrono::system_clock::now();
  std::cout << "Total execution time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   end_time - start_time_0)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << " sec" << std::endl;
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
