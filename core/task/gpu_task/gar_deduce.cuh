#ifndef MATRIXGRAPH_CORE_TASK_GPU_TASK_GAR_DEDUCE_CUH_
#define MATRIXGRAPH_CORE_TASK_GPU_TASK_GAR_DEDUCE_CUH_

#include <cstdint>
#include <memory>
#include <unordered_map>

#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/gar_deduce_arrays.h"
#include "core/data_structures/gar_graph_arrays.h"
#include "core/data_structures/gar_pattern_arrays.h"
#include "core/task/gpu_task/task_base.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

// GARDeduce: GPU match + X-evaluation + head emission for one GAR rule,
// the device-side half of the chase (logical closure). Semantics mirror the
// CPU gar::reason::ChaseEngine:
//
//   * homomorphism matching (no injectivity), vertex labels enforced at
//     candidate time (equivalent to the CPU post-filter for connected
//     patterns), pattern edges / edge predicates carry base relation ids
//     (inverse slots folded by the caller);
//   * X predicates evaluated per embedding against vertex attribute columns
//     (double-normalized) and the resident edge arrays;
//   * x-satisfied embeddings emit their (head_u, head_v) vertex pair; dedup
//     and graph merging stay on the host.
//
// The graph is uploaded once per chase round via SetGraph; attribute columns
// are static across rounds and cached on device (keyed by the caller's
// col_ids). Run() executes one rule against the resident graph.
class GARDeduce : public TaskBase {
 private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using GARGraphArrays =
      sics::matrixgraph::core::data_structures::GARGraphArrays;
  using GARPatternArrays =
      sics::matrixgraph::core::data_structures::GARPatternArrays;
  using GARPredicate = sics::matrixgraph::core::data_structures::GARPredicate;
  using GARAttrColumnSet =
      sics::matrixgraph::core::data_structures::GARAttrColumnSet;
  using GARDeduceArrays =
      sics::matrixgraph::core::data_structures::GARDeduceArrays;
  template <typename T>
  using DeviceOwnedBuffer =
      sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<T>;

 public:
  GARDeduce() = default;
  ~GARDeduce() = default;

  // Uploads the graph. e_src/e_dst must be sorted by (src, dst) — the edge
  // predicate evaluation binary-searches them. Call once per chase round.
  __host__ void SetGraph(const GARGraphArrays& g);

  // Runs one rule. frontier_cap bounds the device embedding frontier (and
  // its truncation sets match_capped); eval_cap bounds the embeddings that
  // are counted / X-evaluated / head-emitted (the CPU chase's
  // max_matches_per_rule). head_u/head_v < 0 means no head (query).
  __host__ int Run(const GARPatternArrays& p, const GARPredicate* preds,
                   int n_preds, const GARAttrColumnSet& attrs, int head_u,
                   int head_v, int frontier_cap, int eval_cap,
                   GARDeduceArrays* out);

  __host__ int Status() const { return status_; }

 private:
  // Resident graph (per round).
  DeviceOwnedBuffer<uint32_t> d_e_src_;
  DeviceOwnedBuffer<uint32_t> d_e_dst_;
  DeviceOwnedBuffer<int32_t> d_e_label_;
  DeviceOwnedBuffer<int32_t> d_v_label_;
  int n_vertices_ = 0;
  int n_edges_ = 0;
  bool has_v_label_ = false;

  // Static attribute columns, uploaded on first use and cached across rules
  // and rounds (keyed by GARAttrColumnSet::col_ids).
  std::unordered_map<int32_t, std::unique_ptr<DeviceOwnedBuffer<double>>>
      attr_vals_cache_;
  std::unordered_map<int32_t, std::unique_ptr<DeviceOwnedBuffer<uint8_t>>>
      attr_valid_cache_;

  // Per-rule scratch, grow-only: allocated on first use and reused across
  // rules and chase rounds (large frontier buffers make cudaMalloc churn
  // the dominant overhead otherwise).
  DeviceOwnedBuffer<int32_t> d_frontier_a_;
  DeviceOwnedBuffer<int32_t> d_frontier_b_;
  DeviceOwnedBuffer<uint32_t> d_cand_src_;
  DeviceOwnedBuffer<uint32_t> d_cand_dst_;
  DeviceOwnedBuffer<uint64_t> d_fwd_in_;
  DeviceOwnedBuffer<uint64_t> d_bwd_in_;
  DeviceOwnedBuffer<uint64_t> d_fwd_;
  DeviceOwnedBuffer<uint64_t> d_bwd_;
  DeviceOwnedBuffer<char> d_sort_temp_;
  DeviceOwnedBuffer<uint32_t> d_head_src_;
  DeviceOwnedBuffer<uint32_t> d_head_dst_;
  DeviceOwnedBuffer<uint8_t> d_x_ok_;

  template <typename T>
  static void EnsureCapacity(DeviceOwnedBuffer<T>& buf, size_t bytes) {
    if (buf.GetSize() < bytes) buf.Init(bytes);
  }

  int status_ = 0;
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_GPU_TASK_GAR_DEDUCE_CUH_
