#ifndef MATRIXGRAPH_CORE_TASK_GPU_TASK_KERNEL_KERNEL_GAR_DEDUCE_CUH_
#define MATRIXGRAPH_CORE_TASK_GPU_TASK_KERNEL_KERNEL_GAR_DEDUCE_CUH_

#include <cstdint>
#include <cuda_runtime.h>

#include "core/data_structures/gar_deduce_arrays.h"

// Kernels for the GPU deduce (chase) task: pattern matching under
// HOMOMORPHISM semantics (distinct pattern vertices may map to the same data
// vertex — the same counting convention as the CPU miner::PatternMatcher,
// unlike kernel_gar_match which enforces injectivity), followed by X
// predicate evaluation and head-fact emission, all on device.
//
// Pipeline per rule (driven by task::GARDeduce):
//   1. candidate counting + fill over all pattern edges (two full scans);
//   2. embedding join: init from the first edge (BFS order over shared
//      nodes), expand over the rest — consistency checks only;
//   3. per-embedding X evaluation (const/var/exists against attribute
//      columns, edge via binary search over the sorted edge arrays) and
//      head-pair emission for satisfied embeddings.
namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

using GARPredicate = sics::matrixgraph::core::data_structures::GARPredicate;

constexpr int kGARDeduceMaxPreds = 16;
constexpr int kGARDeduceMaxCols = 16;
constexpr int kGARDeduceMaxNodes = 16;

// Device view of a folded pattern (inverse relations already folded to base
// relations with swapped endpoints; -1 labels are wildcards).
struct GARDeducePatternDev {
  const int32_t* node_label;  // [n_nodes], -1 = wildcard
  const int32_t* edge_src;    // [n_edges], pattern positions
  const int32_t* edge_dst;
  const int32_t* edge_label;  // [n_edges], base relation ids
  int n_nodes;
  int n_edges;
};

struct GARDeduceFilterParams {
  const uint32_t* g_e_src;
  const uint32_t* g_e_dst;
  const int32_t* g_e_label;
  const int32_t* g_v_label;  // nullptr = every vertex matches any label
  int g_n_edges;
  GARDeducePatternDev p;
  int* cand_counts;              // [p.n_edges]         (pass 1 out)
  const int* cand_offsets;       // [p.n_edges + 1]     (pass 2 in, read-only)
  uint32_t* cand_src;            // [cand_offsets[n]]   (pass 2 out)
  uint32_t* cand_dst;            // [cand_offsets[n]]   (pass 2 out)
};

// Fill cursors: per-pattern-edge write positions, initialized by the caller
// to cand_offsets[0..n-1]; the fill kernel advances them with atomicAdd, so
// cand_offsets itself stays intact for the join phase.

struct GARDeduceEvalParams {
  const int32_t* emb;
  int n_emb;
  int n_nodes;
  // Graph edges sorted by (src, dst) — for edge-predicate binary search.
  const uint32_t* g_e_src;
  const uint32_t* g_e_dst;
  const int32_t* g_e_label;
  int g_n_edges;
  // X predicates, passed by value.
  int n_preds;
  GARPredicate preds[kGARDeduceMaxPreds];
  // Attribute column device pointers (nullptr valid = dense column).
  int n_cols;
  const double* attr_vals[kGARDeduceMaxCols];
  const uint8_t* attr_valid[kGARDeduceMaxCols];
  // Head emission (positions refer to pattern nodes).
  int has_head;
  int head_u;
  int head_v;
  uint32_t* head_src;
  uint32_t* head_dst;
  int* head_count;
  int head_cap;
  // Stats / optional per-embedding x_ok dump.
  unsigned long long* x_count;
  uint8_t* x_ok;  // [n_emb] or nullptr
};

void LaunchGARDeduceCountCandidates(const GARDeduceFilterParams& params);
void LaunchGARDeduceFillCandidates(const GARDeduceFilterParams& params,
                                   int* cursors);
// Packs candidate (src, dst) pairs into sortable uint64 keys:
// fwd = src<<32|dst (sorted by src, then dst), bwd = dst<<32|src.
void LaunchGARDeducePackCandidates(const uint32_t* cand_src,
                                   const uint32_t* cand_dst, int n_cand,
                                   uint64_t* fwd_keys, uint64_t* bwd_keys);
// Init the frontier from the (sorted, duplicate-free) fwd keys of the first
// join edge. Homomorphism: p_u == p_v requires a data self-loop, otherwise
// gs == gd is allowed (the CPU PatternMatcher maps distinct pattern
// vertices to one data vertex).
void LaunchGARDeduceInitEmbeddings(const uint64_t* fwd_keys, int n_cand,
                                   int p_u, int p_v, int n_nodes,
                                   int32_t* out_emb, int* out_count,
                                   int cap);
// Expand the frontier by one pattern edge, adjacency-driven: per frontier
// row a binary search locates the candidate segment of the bound endpoint
// (fwd keys for a bound src, bwd keys for a bound dst) and only that
// segment is scanned — no n_curr x n_cand pair enumeration.
void LaunchGARDeduceExpandEmbeddings(const int32_t* curr, int n_curr,
                                     int n_nodes, int p_u, int p_v,
                                     const uint64_t* fwd_keys,
                                     const uint64_t* bwd_keys, int n_cand,
                                     int32_t* next, int* next_count,
                                     int cap);
void LaunchGARDeduceEvalEmit(const GARDeduceEvalParams& params);

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_GPU_TASK_KERNEL_KERNEL_GAR_DEDUCE_CUH_
