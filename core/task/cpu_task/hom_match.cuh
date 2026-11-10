#ifndef MATRIXGRAPH_CORE_TASK_CPU_TASK_HOM_MATCH_CUH_
#define MATRIXGRAPH_CORE_TASK_CPU_TASK_HOM_MATCH_CUH_

// HomMatch: CPU homomorphism matcher over the flat GAR arrays. This is the
// CPU counterpart of GARMatch (which is injective); it reproduces the greedy
// backtracking of GARAgent's PatternMatcher so the reasoning layers
// (miner / chase / agent) keep byte-identical match sets.
//
// Inverse convention (scheme A, materialized): the input GARGraphArrays
// carries BASE labels 0..num_relations-1; HomMatchIndex materializes the
// reversed direction into slot label + num_relations (r^-1(s,d) == r(d,s)).
// A pattern edge with label >= num_relations therefore walks the flipped
// slot's pair array, exactly like the old TripleGraph.

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/data_structures/gar_graph_arrays.h"
#include "core/data_structures/gar_pattern_arrays.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace cpu_task {

using GARGraphArrays =
    sics::matrixgraph::core::data_structures::GARGraphArrays;
using GARPatternArrays =
    sics::matrixgraph::core::data_structures::GARPatternArrays;

// Packs (src, dst) into one uint64 keyed by src for the sorted pair arrays.
inline std::uint64_t PackPair(std::uint32_t src, std::uint32_t dst) {
  return (static_cast<std::uint64_t>(src) << 32) | dst;
}

// Per-slot sorted (src,dst) pair arrays + hash sets, built once from a flat
// graph. Slot ids run 0..2*num_relations-1 (base + materialized inverses).
class HomMatchIndex {
 public:
  HomMatchIndex(const GARGraphArrays& g, int num_relations);

  int num_relations() const { return num_relations_; }
  int num_slots() const { return num_slots_; }

  const std::vector<std::uint64_t>& RelationPairs(int slot) const {
    return pairs_[static_cast<std::size_t>(slot)];
  }
  const std::unordered_set<std::uint64_t>& RelationPairSet(int slot) const {
    return sets_[static_cast<std::size_t>(slot)];
  }

 private:
  int num_relations_ = 0;
  int num_slots_ = 0;
  std::vector<std::vector<std::uint64_t>> pairs_;
  std::vector<std::unordered_set<std::uint64_t>> sets_;
};

// HomMatcher: enumerates homomorphism embeddings (distinct pattern vertices
// may map to the same data vertex). `out` receives embeddings flat
// (n_nodes per embedding); stops at match_cap and sets *capped. `pins` maps
// pattern vertex -> graph vertex (pinned variant). node_label_idx is not
// consulted — edge labels only, matching the old PatternMatcher.
class HomMatcher {
 public:
  std::uint64_t Enumerate(
      const HomMatchIndex& idx, const GARPatternArrays& p,
      std::uint64_t match_cap, std::vector<std::uint32_t>* out, bool* capped,
      const std::unordered_map<std::uint32_t, std::uint32_t>& pins) const;

  std::uint64_t Enumerate(const HomMatchIndex& idx, const GARPatternArrays& p,
                          std::uint64_t match_cap,
                          std::vector<std::uint32_t>* out, bool* capped) const {
    return Enumerate(idx, p, match_cap, out, capped, {});
  }
};

}  // namespace cpu_task
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_CPU_TASK_HOM_MATCH_CUH_
