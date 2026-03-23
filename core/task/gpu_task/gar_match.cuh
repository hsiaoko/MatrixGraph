#ifndef MATRIXGRAPH_CORE_TASK_GAR_MATCH_CUH_
#define MATRIXGRAPH_CORE_TASK_GAR_MATCH_CUH_

#include <cstdint>
#include <string>

#include "core/common/types.h"
#include "core/data_structures/gar_graph_arrays.h"
#include "core/data_structures/gar_match_arrays.h"
#include "core/data_structures/gar_pattern_arrays.h"
#include "core/task/gpu_task/kernel/kernel_gar_match.cuh"
#include "core/task/gpu_task/task_base.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

class GARMatch : public TaskBase {
 private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using GARGraphArrays =
      sics::matrixgraph::core::data_structures::GARGraphArrays;
  using GARPatternArrays =
      sics::matrixgraph::core::data_structures::GARPatternArrays;
  using GARMatchArrays =
      sics::matrixgraph::core::data_structures::GARMatchArrays;

 public:
  GARMatch(const std::string& config_path, const std::string& output_path)
      : config_path_(config_path), output_path_(output_path) {}

  GARMatch(const GARGraphArrays& g, const GARPatternArrays& p,
           GARMatchArrays* out)
      : g_(g), p_(p), out_(out) {}

  // C-style entry for go_api: accepts serialized g/p arrays and writes match arrays.
  __host__ static int Run(
      const uint32_t* g_v_id,
      const int32_t* g_v_label_idx,
      int g_n_vertices,
      const uint32_t* g_e_src,
      const uint32_t* g_e_dst,
      const uint32_t* g_e_id,
      const int32_t* g_e_label_idx,
      int g_n_edges,
      const int32_t* p_node_label_idx,
      int p_n_nodes,
      const int32_t* p_edge_src,
      const int32_t* p_edge_dst,
      const int32_t* p_edge_label_idx,
      int p_n_edges,
      int* out_num_conditions,
      uint32_t* out_row_pivot_id,
      int32_t* out_row_cond_j,
      int32_t* out_row_pos,
      int32_t* out_row_offset,
      int32_t* out_row_count,
      int out_row_capacity,
      int* out_row_size,
      uint32_t* out_matched_v_ids,
      int out_match_capacity,
      int* out_match_size);

  __host__ void Run();

  __host__ int Status() const { return status_; }

 private:
  std::string config_path_;
  std::string output_path_;
  GARGraphArrays g_{};
  GARPatternArrays p_{};
  GARMatchArrays* out_ = nullptr;
  int status_ = 1;
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_GAR_MATCH_CUH_
