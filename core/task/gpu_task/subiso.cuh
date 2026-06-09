#ifndef MATRIXGRAPH_CORE_TASK_SUBISO_CUH_
#define MATRIXGRAPH_CORE_TASK_SUBISO_CUH_

#include <string>

#include "core/common/types.h"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/exec_plan.cuh"
#include "core/data_structures/grid_csr_tiled_matrix.cuh"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/unified_buffer.cuh"
#include "core/data_structures/woj_exec_plan.cuh"
#include "core/task/gpu_task/task_base.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

class SubIso : public TaskBase {
 private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using GraphID = sics::matrixgraph::core::common::GraphID;
  using TileIndex = sics::matrixgraph::core::common::TileIndex;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using GridCSRTiledMatrix =
      sics::matrixgraph::core::data_structures::GridCSRTiledMatrix;
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;
  using Edges = sics::matrixgraph::core::data_structures::Edges;
  using GridGraphMetadata =
      sics::matrixgraph::core::data_structures::GridGraphMetadata;
  using UnifiedOwnedBufferUint32 =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint32_t>;
  using UnifiedOwnedBufferUint64 =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint64_t>;
  using UnifiedOwnedBufferVertexID =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexID>;
  using ExecutionPlan = sics::matrixgraph::core::data_structures::ExecutionPlan;
  using WOJExecutionPlan =
      sics::matrixgraph::core::data_structures::WOJExecutionPlan;

 public:
  SubIso(const std::string& pattern_path, const std::string& data_graph_path,
         const std::string& data_graph_edgelist_path,
         const std::string& output_path)
      : pattern_path_(pattern_path),
        data_graph_path_(data_graph_path),
        data_graph_edgelist_path_(data_graph_edgelist_path),
        output_path_(output_path) {}

  __host__ void Run();

  // C API helper: run subiso from flat CSR buffers + labels.
  // Returns 0 on success, non-zero on error.
  // Output is written into caller-allocated flat buffers; if a table
  // exceeds max_result_rows it is truncated.
  __host__ static int Run(
      // Pattern graph
      uint32_t p_num_vertices, uint32_t p_num_in_edges, uint32_t p_num_out_edges,
      uint32_t p_max_vid, uint32_t p_min_vid, const uint8_t* p_csr_data,
      uint64_t p_csr_data_size, const uint32_t* p_labels,
      // Data graph
      uint32_t g_num_vertices, uint32_t g_num_in_edges, uint32_t g_num_out_edges,
      uint32_t g_max_vid, uint32_t g_min_vid, const uint8_t* g_csr_data,
      uint64_t g_csr_data_size, const uint32_t* g_labels,
      // Output capacity
      int max_result_tables, int max_result_rows, int max_result_cols,
      // Output buffers (caller-allocated)
      uint32_t* out_table_cols, uint32_t* out_table_rows,
      uint32_t* out_headers_flat, uint32_t* out_data_flat,
      int* out_num_tables);

 private:
  __host__ void LoadData();

  __host__ void InitLabel(VertexLabel* label_p, VertexLabel* label_g);

  __host__ void InitLabel();

  __host__ void AllocMappingBuf();

  __host__ void Matching(const ImmutableCSR& p, const ImmutableCSR& g);

  __host__ void WOJMatching(const ImmutableCSR& p, const ImmutableCSR& g);

  ImmutableCSR p_;

  ImmutableCSR g_;

  Edges e_;

  UnifiedOwnedBufferUint32 m_;

  const std::string pattern_path_;
  const std::string data_graph_path_;
  const std::string data_graph_edgelist_path_;
  const std::string output_path_;

  VertexLabel* label_p_ = nullptr;

  VertexLabel* label_g_ = nullptr;
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_COMPONENTS_SubIso_CUH_