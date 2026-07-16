#ifndef MATRIXGRAPH_CORE_TASK_GPU_LFTJ_SUBISO_GPU_CUH_
#define MATRIXGRAPH_CORE_TASK_GPU_LFTJ_SUBISO_GPU_CUH_

#include <stdint.h>
#include <string>
#include <vector>

#include "core/common/types.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/task/cpu_task/min_wise_filter.h"
#include "core/task/gpu_task/task_base.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

// Single-GPU, count-only LFTJ-style subgraph isomorphism.
// Host side reuses the CPU LFTJ preprocessing (undirected adjacency,
// filters, greedy matching order, candidate sets).  Device side performs
// per-thread DFS enumeration over the matching order and returns the total
// match count.
class LFTJSubIsoGpu : public TaskBase {
 private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;

 public:
  LFTJSubIsoGpu(const std::string& pattern_path,
                const std::string& data_graph_path,
                const std::string& output_path, int num_threads,
                bool canonical = false, bool enable_min_wise_filter = true,
                int filter_hop = 1, int filter_k = 3,
                bool disable_matching_order = false,
                bool enable_ldf_filter = false, bool enable_nlc_filter = true,
                bool enable_lpf_filter = true, bool enable_lcf_filter = true,
                bool enable_bloom_filter = false,
                bool enable_min_wise_bloom_filter = false)
      : pattern_path_(pattern_path),
        data_graph_path_(data_graph_path),
        output_path_(output_path),
        num_threads_(num_threads),
        canonical_(canonical),
        enable_min_wise_filter_(enable_min_wise_filter),
        filter_hop_(filter_hop),
        filter_k_(filter_k),
        disable_matching_order_(disable_matching_order),
        enable_ldf_filter_(enable_ldf_filter),
        enable_nlc_filter_(enable_nlc_filter),
        enable_lpf_filter_(enable_lpf_filter),
        enable_lcf_filter_(enable_lcf_filter),
        enable_bloom_filter_(enable_bloom_filter),
        enable_min_wise_bloom_filter_(enable_min_wise_bloom_filter) {}

  void Run();

 private:
  void LoadData();
  void BuildUndirectedAdjacency();
  void BuildMinWiseFilterCaches();
  void BuildCandidateSets();
  void ComputeMatchingOrder();
  uint64_t EnumerateOnHost() const;  // for validation

  ImmutableCSR pattern_;
  ImmutableCSR data_graph_;

  // Unified undirected adjacency for data graph (CSR).
  std::vector<EdgeIndex> data_offsets_;
  std::vector<VertexID> data_neighbors_;

  // Unified undirected adjacency for pattern graph.
  std::vector<std::vector<VertexID>> pattern_adj_;

  // Candidate sets for pattern vertices.
  std::vector<std::vector<VertexID>> candidates_;

  // Matching order and backward-neighbor lists.
  std::vector<VertexID> order_;
  std::vector<std::vector<VertexID>> bn_list_;

  // Min-wise / NLC / Bloom filter state.
  std::vector<MinWiseFilterCache> p_min_wise_cache_;
  std::vector<MinWiseFilterCache> g_min_wise_cache_;
  std::vector<uint64_t> p_bloom_signature_;
  std::vector<uint64_t> g_bloom_signature_;

  bool canonical_ = false;
  bool enable_min_wise_filter_ = true;
  bool enable_nlc_filter_ = true;
  bool enable_lpf_filter_ = true;
  bool enable_lcf_filter_ = true;
  bool enable_bloom_filter_ = false;
  bool enable_min_wise_bloom_filter_ = false;
  bool enable_ldf_filter_ = false;
  bool disable_matching_order_ = false;
  int filter_hop_ = 1;
  int filter_k_ = 3;
  int filter_order_ = 0;  // 0=default, 1=minwise_nlc_lpf

  const std::string pattern_path_;
  const std::string data_graph_path_;
  const std::string output_path_;
  const int num_threads_;
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_GPU_LFTJ_SUBISO_GPU_CUH_
