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
using WOJExecutionPlan =
    sics::matrixgraph::core::data_structures::WOJExecutionPlan;
using ExecutionPlan = sics::matrixgraph::core::data_structures::ExecutionPlan;
using BitmapOwnership = sics::matrixgraph::core::util::BitmapOwnership;
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

static inline bool FilterCore(VertexID u_idx, VertexID v_idx,
                              const ImmutableCSR& p, const ImmutableCSR& g) {
  auto u_label = p.GetVLabelBasePointer()[u_idx];
  auto v_label = g.GetVLabelBasePointer()[v_idx];
  // printf("check %d(%d), %d(%d)\n", u_idx, u_label, v_idx, v_label);
  if (u_label == v_label)
    return true;
  else
    return false;
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

    // std::cout << u_src << "(" << src_idx << ")"
    //           << "=>" << u_dst << "(" << dst_idx << ")" << std::endl;
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
            // std::cout << global_id << ", degree: " << degree
            //           << ", label: " << std::endl;
            if (FilterCore(src_idx, v_idx, p, g)) {
              for (VertexID nbr_v_idx = 0; nbr_v_idx < degree; nbr_v_idx++) {
                VertexID nbr_v = out_edges[nbr_v_idx];
                VertexID nbr_localid = g.GetLocalIDByGlobalID(nbr_v);
                // std::cout << global_id << "->" << nbr_v << ": " << nbr_v_idx
                //           << "/" << degree << std::endl;
                if (FilterCore(dst_idx, nbr_localid, p, g)) {
                  //                 std::cout << global_id << "->" << nbr_v <<
                  //                 " #" << std::endl;

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

static inline void JoinCore(VertexID n_vertices_g,
                            const WOJMatches& left_woj_matches,
                            const WOJMatches& right_woj_matches,
                            WOJMatches* output_woj_matches,
                            VertexID left_hash_idx, VertexID right_hash_idx,
                            BitmapOwnership& right_visited,
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

static WOJMatches* WOJJoin(
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

    JoinCore(exec_plan.get_n_vertices_g(), *left_woj_matches,
             *right_woj_matches, output_woj_matches, join_keys.first,
             join_keys.second, visited, jump_visited);
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

  output_woj_matches->Print(99);
  return output_woj_matches;
}

void CPUSubIso::LoadData() {
  std::cout << "[CPUSubIso] LoadData() ..." << std::endl;

  p_.Read(pattern_path_);
  p_.PrintGraph(99);

  g_.Read(data_graph_path_);
  g_.PrintGraph(99);
}

void CPUSubIso::Matching(const ImmutableCSR& p, const ImmutableCSR& g) {}

void CPUSubIso::ParallelMatching(const ImmutableCSR& p, const ImmutableCSR& g,
                                 size_t thread_id, size_t total_threads) {}

void CPUSubIso::WOJMatching(const ImmutableCSR& p, const ImmutableCSR& g) {
  WOJExecutionPlan exec_plan;
  exec_plan.GenerateWOJExecutionPlan(p, g);

  auto start_time_0 = std::chrono::system_clock::now();
  auto woj_matches = WOJFilter(exec_plan, p, g);

  for (auto iter : woj_matches) {
    iter->Print();
  }
  auto start_time_1 = std::chrono::system_clock::now();

  WOJJoin(exec_plan, woj_matches);
  auto start_time_2 = std::chrono::system_clock::now();

  std::cout << "[WOJMatching] Filter() elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_1 - start_time_0)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;
  std::cout << "[WOJMatching] Join() elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_2 - start_time_1)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;
}

void CPUSubIso::Run() {
  auto start_time_0 = std::chrono::system_clock::now();
  LoadData();
  auto start_time_1 = std::chrono::system_clock::now();

  std::cout << "Data loading time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   start_time_1 - start_time_0)
                   .count()
            << " ms" << std::endl;

  WOJMatching(p_, g_);

  auto end_time = std::chrono::system_clock::now();
  std::cout << "Total execution time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   end_time - start_time_0)
                   .count()
            << " ms" << std::endl;
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
