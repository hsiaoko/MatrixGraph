#include "core/data_structures/attributes.h"
#include "core/task/gpu_task/block_agg_prims.cuh"
#include "core/task/gpu_task/kernel/kernel_compute_features.cuh"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

/**
 * @file kernel_compute_features.cu
 * @brief Device interpreter for flat feature-expression plans.
 *
 * The interpreter walks a DAG of expression nodes stored in a flat array.
 * Each CUDA block is responsible for a single pivot vertex; all threads in the
 * block cooperate to:
 *   1. Resolve navigators (Self, Neighbor, Filter) into binding vertex ids.
 *   2. Recursively evaluate sub-expressions for those bindings.
 *   3. Apply aggregation primitives over the resulting values.
 *
 * Workspace layout inside one block:
 *   - [0, workspace_capacity)                  : current navigator bindings.
 *   - [workspace_capacity, 2*workspace_capacity): scratch for nested
 * evaluation.
 *   - Deeper nesting consumes additional slices of size workspace_capacity.
 */

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

using VertexID = sics::matrixgraph::core::common::VertexID;
using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using ValueType = sics::matrixgraph::core::data_structures::ValueType;
using AttributeName = sics::matrixgraph::core::data_structures::AttributeName;
using Attributes = sics::matrixgraph::core::data_structures::Attributes;
using Attribute = sics::matrixgraph::core::data_structures::Attribute;
using sics::matrixgraph::core::data_structures::GetBool;
using sics::matrixgraph::core::data_structures::GetFloat64;
using sics::matrixgraph::core::data_structures::GetInt;

// Sentinel vertex id used to mark filtered-out bindings.
constexpr uint32_t kInvalidVid = 0xffffffffu;

// ---------------------------------------------------------------------------
// CSR buffer parsing (matches ImmutableCSR layout).
// ---------------------------------------------------------------------------

/**
 * @brief ImmutableCSR layout seen by the device interpreter.
 *
 * The fields are exactly the contiguous arrays written by ImmutableCSR::Write()
 * and uploaded by ComputeFeaturesTask::TransferGraphDataToDevice().
 */
struct CSRView {
  const VertexID* globalid;
  const VertexID* indegree;
  const VertexID* outdegree;
  const EdgeIndex* in_offset;
  const EdgeIndex* out_offset;
  const VertexID* incoming_edges;
  const VertexID* outgoing_edges;
};

/**
 * @brief Parse the contiguous CSR buffer into a CSRView.
 *
 * The offsets assume the same ordering used by ImmutableCSR: globalid,
 * indegree, outdegree, in_offset, out_offset, incoming_edges, outgoing_edges.
 */
__device__ inline CSRView ParseCSR(const uint8_t* graph_data,
                                   uint32_t n_vertices, uint32_t n_in_edges,
                                   uint32_t n_out_edges) {
  CSRView csr;
  csr.globalid = reinterpret_cast<const VertexID*>(graph_data);
  csr.indegree = csr.globalid + n_vertices;
  csr.outdegree = csr.indegree + n_vertices;
  csr.in_offset =
      reinterpret_cast<const EdgeIndex*>(csr.outdegree + n_vertices);
  csr.out_offset = csr.in_offset + n_vertices + 1;
  csr.incoming_edges =
      reinterpret_cast<const VertexID*>(csr.out_offset + n_vertices + 1);
  csr.outgoing_edges = csr.incoming_edges + n_in_edges;
  return csr;
}

// ---------------------------------------------------------------------------
// Attribute access.
// ---------------------------------------------------------------------------

/**
 * @brief Read a per-vertex attribute by name.
 *
 * Looks up the attribute in the per-vertex map built by LoadAttributes().
 * Returns an invalid value if the key is missing.  The attribute's own type
 * tag is preserved so that a TIME or FLOAT32 column keeps its identity when it
 * flows straight through to an output slot.
 */
__device__ inline FeatureValue ReadAttribute(const Attributes* attrs,
                                             uint32_t vid, const char* key) {
  const AttributeName name(key);
  const Attribute* attr = attrs[vid].attr_map.find(name);
  if (attr == nullptr) return MakeInvalidValue();

  FeatureValue v = MakeInvalidValue();
  v.type = attr->type;
  switch (attr->type) {
    case ValueType::kInt:
    case ValueType::kTime:
      v.i64 = GetInt(*attr, 0);
      return v;
    case ValueType::kFloat64:
    case ValueType::kFloat32:
      v.f64 = GetFloat64(*attr, 0);
      return v;
    case ValueType::kBool:
      v.b = GetBool(*attr, 0);
      return v;
    default:
      return MakeInvalidValue();
  }
}

// ---------------------------------------------------------------------------
// Label matching for NeighborNav.
// ---------------------------------------------------------------------------

/**
 * @brief Check whether @p vid satisfies a navigator label filter.
 *
 * An empty target_label means "match all".  The current implementation stores
 * a single character code in target_label[0]; this will be extended once label
 * strings are supported.
 */
__device__ inline bool LabelMatches(const uint32_t* labels, uint32_t vid,
                                    const MGPlanNode& nav) {
  if (nav.target_label[0] == '\0') return true;
  if (labels == nullptr) return false;
  return labels[vid] == static_cast<uint32_t>(nav.target_label[0]);
}

// ---------------------------------------------------------------------------
// NeighborNav edge resolution.
// ---------------------------------------------------------------------------

/**
 * @brief Resolve a NeighborNav to an edge pointer and edge count.
 *
 * direction == 0 selects outgoing edges, 1 selects incoming edges, and 2 would
 * select both (not implemented yet).
 */
__device__ inline void ResolveNeighborNav(const MGPlanNode* navs,
                                          int32_t nav_idx, const CSRView& csr,
                                          uint32_t pivot_vid,
                                          const VertexID*& out_edges,
                                          uint32_t& out_count) {
  const MGPlanNode& nav = navs[nav_idx];
  if (nav.direction == 0) {
    out_edges = csr.outgoing_edges + csr.out_offset[pivot_vid];
    out_count = csr.outdegree[pivot_vid];
  } else if (nav.direction == 1) {
    out_edges = csr.incoming_edges + csr.in_offset[pivot_vid];
    out_count = csr.indegree[pivot_vid];
  } else {
    // BOTH: not supported yet.
    out_edges = nullptr;
    out_count = 0;
  }
}

// ---------------------------------------------------------------------------
// Aggregation dispatch.
// ---------------------------------------------------------------------------

/**
 * @brief Apply an aggregation primitive to the collected binding values.
 *
 * percent_true and count_gt_mean are computed here from per-thread predicates
 * summed with BlockSum.  The canonical implementations use a block-wide
 * exclusive scan whose buffers are sized for a common::kBlockDim (1024) thread
 * block; this interpreter runs one 256-thread block per pivot, and under that
 * block size the scan's total came back 0 even though the per-thread count was
 * correct (measured on the parity ring).  BlockSum needs one double per thread
 * and has no such block-size contract.  Everything else dispatches to
 * block_agg_prims unchanged.
 */
__device__ inline FeatureValue ApplyAggPrim(AggPrim prim, FeatureValue* values,
                                            uint32_t n,
                                            const BlockWorkspace& ws) {
  if (prim == AggPrim::kPercentTrue) {
    if (n == 0) return MakeInvalidValue();
    double local = 0.0;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
      if (values[i].type == ValueType::kBool && values[i].b) local += 1.0;
    }
    return MakeFloat64Value(BlockSum(local, ws.sum_buf) /
                            static_cast<double>(n));
  }

  if (prim == AggPrim::kCountGreaterThanMean) {
    if (n == 0) return MakeInvalidValue();
    double sum = 0.0;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
      sum += values[i].ToDouble();
    }
    const double mean = BlockSum(sum, ws.sum_buf) / static_cast<double>(n);
    double local = 0.0;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
      if (values[i].ToDouble() > mean) local += 1.0;
    }
    return MakeIntValue(static_cast<int64_t>(BlockSum(local, ws.sum_buf)));
  }

  return BlockApplyAggPrim(prim, values, n, ws);
}

// ---------------------------------------------------------------------------
// Forward declarations.
// ---------------------------------------------------------------------------

// Forward declaration so EvalCond() can evaluate expression sub-trees.
__device__ FeatureValue EvalExpr(const MGPlanNode* plan, const MGPlanNode* navs,
                                 const MGCondNode* conds,
                                 const Attributes* vertex_attrs,
                                 const uint8_t* graph_data, uint32_t n_vertices,
                                 uint32_t n_in_edges, uint32_t n_out_edges,
                                 const uint32_t* labels, uint32_t pivot_vid,
                                 int32_t expr_idx, FeatureValue* workspace,
                                 uint32_t workspace_capacity,
                                 const BlockWorkspace& ws);

// ---------------------------------------------------------------------------
// Condition evaluation for FilterNav.
// ---------------------------------------------------------------------------

/**
 * @brief Evaluate a condition against a single navigator binding.
 *
 * Both the left and right expressions are evaluated with binding_vid as the
 * pivot, converted to double, and compared using MatrixGraphCondType.
 */
__device__ inline bool EvalCond(const MGCondNode* conds, int32_t cond_idx,
                                const MGPlanNode* plan, const MGPlanNode* navs,
                                const Attributes* vertex_attrs,
                                const uint8_t* graph_data, uint32_t n_vertices,
                                uint32_t n_in_edges, uint32_t n_out_edges,
                                const uint32_t* labels, uint32_t binding_vid,
                                FeatureValue* workspace,
                                uint32_t workspace_capacity,
                                const BlockWorkspace& ws) {
  const MGCondNode& c = conds[cond_idx];
  FeatureValue a =
      EvalExpr(plan, navs, conds, vertex_attrs, graph_data, n_vertices,
               n_in_edges, n_out_edges, labels, binding_vid, c.left_expr,
               workspace + workspace_capacity, workspace_capacity, ws);
  FeatureValue b =
      EvalExpr(plan, navs, conds, vertex_attrs, graph_data, n_vertices,
               n_in_edges, n_out_edges, labels, binding_vid, c.right_expr,
               workspace + workspace_capacity, workspace_capacity, ws);
  double da = a.ToDouble();
  double db = b.ToDouble();
  bool keep;
  switch (c.op) {
    case MG_COND_EQ:
      keep = (da == db);
      break;
    case MG_COND_NE:
      keep = (da != db);
      break;
    case MG_COND_LT:
      keep = (da < db);
      break;
    case MG_COND_LE:
      keep = (da <= db);
      break;
    case MG_COND_GT:
      keep = (da > db);
      break;
    case MG_COND_GE:
      keep = (da >= db);
      break;
    default:
      keep = false;
      break;
  }
  return keep;
}

// ---------------------------------------------------------------------------
// Collect bindings for a navigator.
// Writes binding vertex ids into workspace[0..count-1].i64 and returns count.
// May use workspace + workspace_capacity as temporary scratch.
// ---------------------------------------------------------------------------

/**
 * @brief Materialize the bindings produced by navigator @p nav_idx.
 *
 * For SelfNav this is a single binding (the pivot).
 * For NeighborNav this is the set of outgoing/incoming neighbors that pass the
 * optional label filter, compacted into the start of the workspace.
 * For FilterNav this recursively collects the inner bindings, evaluates the
 * attached condition for each one, and compacts the kept bindings.
 */
__device__ inline uint32_t CollectBindings(
    const MGPlanNode* plan, const MGPlanNode* navs, const MGCondNode* conds,
    const Attributes* vertex_attrs, const uint8_t* graph_data,
    uint32_t n_vertices, uint32_t n_in_edges, uint32_t n_out_edges,
    const uint32_t* labels, uint32_t pivot_vid, int32_t nav_idx,
    FeatureValue* workspace, uint32_t workspace_capacity,
    const BlockWorkspace& ws) {
  const MGPlanNode& nav = navs[nav_idx];
  const MatrixGraphNavType nav_type = static_cast<MatrixGraphNavType>(nav.type);

  if (nav_type == MG_NAV_SELF) {
    if (threadIdx.x == 0) workspace[0].i64 = static_cast<int64_t>(pivot_vid);
    __syncthreads();
    return 1;
  }

  if (nav_type == MG_NAV_NEIGHBOR) {
    CSRView csr = ParseCSR(graph_data, n_vertices, n_in_edges, n_out_edges);
    const VertexID* edges = nullptr;
    uint32_t count = 0;
    ResolveNeighborNav(navs, nav_idx, csr, pivot_vid, edges, count);
    uint32_t capped = count < workspace_capacity ? count : workspace_capacity;

    if (threadIdx.x < capped) {
      uint32_t vid = edges[threadIdx.x];
      workspace[threadIdx.x].i64 = LabelMatches(labels, vid, nav)
                                       ? static_cast<int64_t>(vid)
                                       : static_cast<int64_t>(kInvalidVid);
    }
    __syncthreads();

    __shared__ uint32_t s_count;
    if (threadIdx.x == 0) {
      uint32_t write = 0;
      for (uint32_t i = 0; i < capped; ++i) {
        if (static_cast<uint32_t>(workspace[i].i64) != kInvalidVid) {
          workspace[write++].i64 = workspace[i].i64;
        }
      }
      s_count = write;
    }
    __syncthreads();
    return s_count;
  }

  if (nav_type == MG_NAV_FILTER) {
    // Collect inner bindings into the scratch area.
    FeatureValue* scratch = workspace + workspace_capacity;
    uint32_t inner_count =
        CollectBindings(plan, navs, conds, vertex_attrs, graph_data, n_vertices,
                        n_in_edges, n_out_edges, labels, pivot_vid,
                        nav.inner_nav_idx, scratch, workspace_capacity, ws);

    __shared__ uint32_t s_broadcast;
    __shared__ uint32_t s_count;
    uint32_t kept = 0;

    for (uint32_t i = 0; i < inner_count; ++i) {
      if (threadIdx.x == i) {
        s_broadcast = static_cast<uint32_t>(scratch[i].i64);
      }
      __syncthreads();
      uint32_t binding = s_broadcast;
      __syncthreads();

      bool keep = EvalCond(conds, nav.cond_idx, plan, navs, vertex_attrs,
                           graph_data, n_vertices, n_in_edges, n_out_edges,
                           labels, binding, scratch, workspace_capacity, ws);

      if (threadIdx.x == 0 && keep) {
        workspace[kept++].i64 = static_cast<int64_t>(binding);
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) s_count = kept;
    __syncthreads();
    return s_count;
  }

  // PatternNav is not supported yet.
  return 0;
}

// ---------------------------------------------------------------------------
// Expression evaluation.
// ---------------------------------------------------------------------------

/**
 * @brief Evaluate expression @p expr_idx for @p pivot_vid.
 *
 * The interpreter handles the five expression node kinds:
 *   - ATTR: read attribute.
 *   - CONST: return the typed constant (only the active union member is set).
 *   - PATTERN_ATTR: currently returns invalid; reserved for pattern matching.
 *   - TRANS: evaluate one or two children and apply the operator.
 *   - AGG: collect bindings from the referenced navigator, evaluate the source
 *          expression for every binding, and apply an aggregation primitive.
 */
__device__ FeatureValue EvalExpr(const MGPlanNode* plan, const MGPlanNode* navs,
                                 const MGCondNode* conds,
                                 const Attributes* vertex_attrs,
                                 const uint8_t* graph_data, uint32_t n_vertices,
                                 uint32_t n_in_edges, uint32_t n_out_edges,
                                 const uint32_t* labels, uint32_t pivot_vid,
                                 int32_t expr_idx, FeatureValue* workspace,
                                 uint32_t workspace_capacity,
                                 const BlockWorkspace& ws) {
  const MGPlanNode& node = plan[expr_idx];
  const MatrixGraphExprType type = static_cast<MatrixGraphExprType>(node.type);

  if (type == MG_EXPR_ATTR) {
    return ReadAttribute(vertex_attrs, pivot_vid, node.key);
  }

  if (type == MG_EXPR_CONST) {
    FeatureValue v = MakeInvalidValue();
    v.type = node.const_type;
    switch (v.type) {
      case ValueType::kInt:
      case ValueType::kTime:
        v.i64 = node.const_i64;
        break;
      case ValueType::kFloat64:
      case ValueType::kFloat32:
        v.f64 = node.const_f64;
        break;
      case ValueType::kBool:
        v.b = node.const_b != 0;
        break;
      default:
        return MakeInvalidValue();
    }
    return v;
  }

  if (type == MG_EXPR_PATTERN_ATTR) {
    return MakeInvalidValue();
  }

  if (type == MG_EXPR_TRANS) {
    const MatrixGraphTransOp op = node.trans_op;

    // Unary operators only need child_a.
    if (op == MG_TRANS_NEG || op == MG_TRANS_ABS || op == MG_TRANS_SQRT) {
      FeatureValue a =
          EvalExpr(plan, navs, conds, vertex_attrs, graph_data, n_vertices,
                   n_in_edges, n_out_edges, labels, pivot_vid, node.child_a,
                   workspace + workspace_capacity, workspace_capacity, ws);
      double da = a.ToDouble();
      switch (op) {
        case MG_TRANS_NEG:
          return MakeFloat64Value(-da);
        case MG_TRANS_ABS:
          return MakeFloat64Value(fabs(da));
        case MG_TRANS_SQRT:
          return MakeFloat64Value(sqrt(da));
        default:
          return MakeInvalidValue();
      }
    }

    // Binary operators.
    FeatureValue a =
        EvalExpr(plan, navs, conds, vertex_attrs, graph_data, n_vertices,
                 n_in_edges, n_out_edges, labels, pivot_vid, node.child_a,
                 workspace + workspace_capacity, workspace_capacity, ws);
    FeatureValue b =
        EvalExpr(plan, navs, conds, vertex_attrs, graph_data, n_vertices,
                 n_in_edges, n_out_edges, labels, pivot_vid, node.child_b,
                 workspace + workspace_capacity, workspace_capacity, ws);
    double da = a.ToDouble();
    double db = b.ToDouble();
    switch (op) {
      case MG_TRANS_ADD:
        return MakeFloat64Value(da + db);
      case MG_TRANS_SUB:
        return MakeFloat64Value(da - db);
      case MG_TRANS_MUL:
        return MakeFloat64Value(da * db);
      case MG_TRANS_DIV:
        return (db == 0.0) ? MakeInvalidValue() : MakeFloat64Value(da / db);
      case MG_TRANS_POW:
        return MakeFloat64Value(pow(da, db));
      default:
        return MakeInvalidValue();
    }
  }

  if (type == MG_EXPR_AGG) {
    // Collect bindings (Self / Neighbor / Filter) into workspace[0..count-1].
    uint32_t binding_count =
        CollectBindings(plan, navs, conds, vertex_attrs, graph_data, n_vertices,
                        n_in_edges, n_out_edges, labels, pivot_vid,
                        node.nav_idx, workspace, workspace_capacity, ws);

    // Evaluate the source expression for every binding.  All threads
    // participate so that nested aggregations remain cooperative.
    for (uint32_t i = 0; i < binding_count; ++i) {
      uint32_t binding_vid = static_cast<uint32_t>(workspace[i].i64);
      FeatureValue v =
          EvalExpr(plan, navs, conds, vertex_attrs, graph_data, n_vertices,
                   n_in_edges, n_out_edges, labels, binding_vid, node.src_idx,
                   workspace + workspace_capacity, workspace_capacity, ws);
      if (threadIdx.x == 0) workspace[i] = v;
      __syncthreads();
    }

    return ApplyAggPrim(node.agg_op, workspace, binding_count, ws);
  }

  return MakeInvalidValue();
}

// ---------------------------------------------------------------------------
// Main kernel.
// ---------------------------------------------------------------------------

/**
 * @brief One block per pivot, one output feature per thread-block iteration.
 *
 * Each block selects its pivot from pivot_vertex_ids, computes each requested
 * output expression, and writes the value to outputs[pivot_idx * n_outputs +
 * o].
 */
__global__ void ComputeFeaturesKernel(
    const uint8_t* graph_data, uint32_t n_vertices, uint32_t n_in_edges,
    uint32_t n_out_edges, const uint32_t* labels,
    const Attributes* vertex_attrs, const uint32_t* pivot_vertex_ids,
    uint32_t n_pivots, const MGPlanNode* plan, uint32_t n_plan_nodes,
    const MGPlanNode* navs, uint32_t n_navs, const MGCondNode* conds,
    uint32_t n_conds, const int32_t* output_expr_indices, uint32_t n_outputs,
    FeatureValue* workspace, uint32_t workspace_capacity,
    uint32_t per_pivot_workspace, FeatureValue* outputs) {
  (void)n_plan_nodes;
  (void)n_navs;
  (void)n_conds;

  uint32_t pivot_idx = blockIdx.x;
  if (pivot_idx >= n_pivots) return;

  uint32_t pivot_vid = pivot_vertex_ids[pivot_idx];
  FeatureValue* my_workspace = workspace + pivot_idx * per_pivot_workspace;

  // The block workspace (reduction/scan buffers) is the tail of this pivot's
  // slice, so no dynamic shared memory is needed.
  const BlockWorkspace ws = GetBlockWorkspace(
      reinterpret_cast<uint8_t*>(my_workspace + per_pivot_workspace -
                                 kPrimsWorkspaceValues),
      kComputeFeaturesMaxNeighbors);
  for (uint32_t o = 0; o < n_outputs; ++o) {
    int32_t expr_idx = output_expr_indices[o];
    FeatureValue v =
        EvalExpr(plan, navs, conds, vertex_attrs, graph_data, n_vertices,
                 n_in_edges, n_out_edges, labels, pivot_vid, expr_idx,
                 my_workspace, workspace_capacity, ws);
    // Order-statistic primitives compute on thread 0 only and return
    // placeholders elsewhere. With every thread storing, the placeholders
    // raced the real value (num_unique read back as 0). Thread 0 is the
    // single writer; cooperative primitives already agree across threads.
    if (threadIdx.x == 0) {
      outputs[pivot_idx * n_outputs + o] = v;
    }
  }
}

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
