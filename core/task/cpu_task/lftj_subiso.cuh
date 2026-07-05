#ifndef MATRIXGRAPH_CORE_TASK_CPU_LFTJ_SUBISO_CUH_
#define MATRIXGRAPH_CORE_TASK_CPU_LFTJ_SUBISO_CUH_

#include <atomic>
#include <string>
#include <vector>

#include "core/common/types.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/task/cpu_task/cpu_task_base.h"
#include "core/task/cpu_task/min_wise_filter.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

// CPU-only Leapfrog-Trie-Join-style subgraph isomorphism counter.
// By default it only counts exact subgraph isomorphisms (injective vertex
// mappings that preserve edges and labels), producing a number directly
// comparable to RapidMatch's #Embeddings. If output_path_ is non-empty,
// all embeddings are also materialized and written to a binary file.
class LFTJSubIso : public CPUTaskBase {
 private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;

 public:
  LFTJSubIso(const std::string& pattern_path,
             const std::string& data_graph_path,
             const std::string& output_path, int num_threads,
             uint64_t output_limit = std::numeric_limits<uint64_t>::max(),
             bool canonical = false, bool enable_min_wise_filter = true,
             int filter_hop = 1, int filter_k = 3,
             bool disable_matching_order = false,
             bool enable_ldf_filter = true, bool enable_nlc_filter = true,
             bool enable_bloom_filter = true,
             bool enable_min_wise_bloom_filter = true,
             const std::string& reject_output_path = "")
      : pattern_path_(pattern_path),
        data_graph_path_(data_graph_path),
        output_path_(output_path),
        reject_output_path_(reject_output_path),
        num_threads_(num_threads),
        output_limit_(output_limit),
        canonical_(canonical),
        enable_min_wise_filter_(enable_min_wise_filter),
        filter_hop_(filter_hop),
        filter_k_(filter_k),
        disable_matching_order_(disable_matching_order),
        enable_ldf_filter_(enable_ldf_filter),
        enable_nlc_filter_(enable_nlc_filter),
        enable_bloom_filter_(enable_bloom_filter),
        enable_min_wise_bloom_filter_(enable_min_wise_bloom_filter) {}

  void Run();

 private:
  void LoadData();

  // Build unified undirected adjacency for pattern and data graphs.
  void BuildUndirectedAdjacency();

  // Compute candidate sets C(u) for each pattern vertex based on label,
  // degree and (optionally) k-min-wise constraints.
  void BuildCandidateSets();

  // Precompute min-wise label-hash signatures for pattern and data graphs.
  void BuildMinWiseFilterCaches();

  // Greedy matching order: maximize backward neighbors, tie-break by smaller
  // candidate set / larger pattern degree.
  void ComputeMatchingOrder();

  // Write filtered (u,v) pairs: data vertices with the same label as pattern
  // vertex u that did not survive into candidates_[u].
  void WriteRejectedPairs();

  // Depth-first LFTJ enumeration. Returns the number of complete embeddings.
  // When materialize_ is true, also stores every complete embedding in
  // materialized_matches_ (row-major, each row has |V_pattern| vertices).
  uint64_t Enumerate();

  // Single-thread DFS over a sub-range of root candidates. If
  // thread_matches is non-null, append each complete embedding to it.
  uint64_t EnumerateRange(size_t cand_start, size_t cand_end,
                          uint64_t thread_limit,
                          std::vector<VertexID>* thread_matches);

  // Intersect the neighbor lists of all backward neighbors of order[depth]
  // and store valid, unvisited candidates in out. The input base is the
  // candidate set of order[depth] (already filtered by label/degree).
  void ComputeLocalCandidates(uint32_t depth, const VertexID* embedding,
                              std::vector<VertexID>& out);

  // Sorted set intersection: dst = a ∩ b.
  static void Intersect(const VertexID* a, size_t na, const VertexID* b,
                        size_t nb, std::vector<VertexID>& dst);

  // Binary search helper.
  static bool Contains(const VertexID* arr, size_t n, VertexID v);

  ImmutableCSR pattern_;
  ImmutableCSR data_graph_;

  // Unified undirected adjacency for data graph (CSR).
  std::vector<EdgeIndex> data_offsets_;
  std::vector<VertexID> data_neighbors_;

  // Unified undirected adjacency for pattern graph.
  std::vector<std::vector<VertexID>> pattern_adj_;

  // Candidate sets for pattern vertices (original local IDs).
  std::vector<std::vector<VertexID>> candidates_;

  // Matching order and auxiliary structures.
  std::vector<VertexID> order_;                // order[depth] = pattern local id
  std::vector<VertexID> order_pos_;            // order_pos[u] = depth of u
  std::vector<std::vector<VertexID>> bn_list_; // backward neighbors per depth

  // DFS state.
  std::vector<VertexID> embedding_;
  std::vector<uint8_t> visited_;
  std::vector<std::vector<VertexID>> cand_buffer_;

  uint64_t match_count_ = 0;
  uint64_t output_limit_ = std::numeric_limits<uint64_t>::max();
  bool canonical_ = false;
  bool materialize_ = false;

  // Filter statistics (reported at the end of Run).
  std::atomic<uint64_t> label_filtered_count_{0};
  std::atomic<uint64_t> degree_filtered_count_{0};
  std::atomic<uint64_t> ldf_filtered_count_{0};
  std::atomic<uint64_t> nlc_filtered_count_{0};
  std::atomic<uint64_t> bloom_filtered_count_{0};
  std::atomic<uint64_t> min_wise_filtered_count_{0};
  std::atomic<uint64_t> min_wise_bloom_filtered_count_{0};
  std::atomic<uint64_t> intersection_pruned_count_{0};

  // Min-wise / NLC filter state.
  bool enable_min_wise_filter_ = true;
  bool enable_nlc_filter_ = true;
  bool enable_bloom_filter_ = true;
  bool enable_min_wise_bloom_filter_ = true;
  int filter_hop_ = 1;
  int filter_k_ = 3;
  std::vector<MinWiseFilterCache> p_min_wise_cache_;
  std::vector<MinWiseFilterCache> g_min_wise_cache_;

  // Bloom filter state.
  std::vector<uint64_t> p_bloom_signature_;
  std::vector<uint64_t> g_bloom_signature_;

  // If true, use the natural vertex order (0,1,2,...) instead of the greedy
  // matching order. This is expected to degrade performance.
  bool disable_matching_order_ = false;

  // Label-degree filter (directed out/in degree check, like subiso_cpu).
  // Disabled by default: LFTJ treats the graph as undirected, so this filter
  // is only sound when the CSR is symmetric (undirected stored as bidirectional).
  bool enable_ldf_filter_ = false;

  // Materialized embeddings, stored row-major. Only populated when
  // output_path_ is non-empty.
  std::vector<VertexID> materialized_matches_;

  const std::string pattern_path_;
  const std::string data_graph_path_;
  const std::string output_path_;
  const std::string reject_output_path_;
  const int num_threads_;
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_CPU_LFTJ_SUBISO_CUH_
