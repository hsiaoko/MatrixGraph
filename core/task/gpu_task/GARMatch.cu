#include "core/task/gpu_task/GARMatch.cuh"
#include "core/task/gpu_task/kernel/kernel_gar_match.cuh"
#include <iostream>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using GARMatchKernelWrapper =
    sics::matrixgraph::core::task::kernel::GARMatchKernelWrapper;

__host__ int GARMatch::SubIso(
    const uint32_t* g_v_id, const int32_t* g_v_label_idx, int g_n_vertices,
    const uint32_t* g_e_src, const uint32_t* g_e_dst, const uint32_t* g_e_id,
    const int32_t* g_e_label_idx, int g_n_edges,
    const int32_t* p_node_label_idx, int p_n_nodes, const int32_t* p_edge_src,
    const int32_t* p_edge_dst, const int32_t* p_edge_label_idx, int p_n_edges,
    int* out_num_conditions, uint32_t* out_row_pivot_id,
    int32_t* out_row_cond_j, int32_t* out_row_pos, int32_t* out_row_offset,
    int32_t* out_row_count, int out_row_capacity, int* out_row_size,
    uint32_t* out_matched_v_ids, int out_match_capacity, int* out_match_size) {
  std::cout << "[GARMatch] SubIso() ..." << std::endl;
  GARGraphParams g{
      .v_id = g_v_id,
      .v_label_idx = g_v_label_idx,
      .n_vertices = g_n_vertices,
      .e_src = g_e_src,
      .e_dst = g_e_dst,
      .e_id = g_e_id,
      .e_label_idx = g_e_label_idx,
      .n_edges = g_n_edges,
  };
  GARPatternParams p{
      .node_label_idx = p_node_label_idx,
      .n_nodes = p_n_nodes,
      .edge_src = p_edge_src,
      .edge_dst = p_edge_dst,
      .edge_label_idx = p_edge_label_idx,
      .n_edges = p_n_edges,
  };
  GARMatchOutput out{
      .num_conditions = out_num_conditions,
      .row_pivot_id = out_row_pivot_id,
      .row_cond_j = out_row_cond_j,
      .row_pos = out_row_pos,
      .row_offset = out_row_offset,
      .row_count = out_row_count,
      .row_capacity = out_row_capacity,
      .row_size = out_row_size,
      .matched_v_ids = out_matched_v_ids,
      .match_capacity = out_match_capacity,
      .match_size = out_match_size,
  };
  return GARMatchKernelWrapper::GARMatch(g, p, &out);
}

__host__ void GARMatch::Run() {
  std::cout << "[GARMatch] Run() ..." << std::endl;
  status_ = GARMatchKernelWrapper::GARMatch(g_, p_, out_);
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
