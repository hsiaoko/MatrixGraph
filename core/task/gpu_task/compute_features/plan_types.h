#ifndef MATRIXGRAPH_CORE_TASK_GPU_TASK_COMPUTE_FEATURES_PLAN_TYPES_H_
#define MATRIXGRAPH_CORE_TASK_GPU_TASK_COMPUTE_FEATURES_PLAN_TYPES_H_

#include "core/data_structures/attributes.h"
#include "core/task/gpu_task/execute_agg_prim.cuh"

#include <cstdint>

/**
 * @file plan_types.h
 * @brief Flat plan types for ComputeFeaturesTask.
 *
 * A feature expression is compiled into a flat array of MatrixGraphPlanNode
 * expression nodes; navigators and conditions live in their own flat arrays.
 * The device interpreter walks these arrays, so the types deliberately reuse
 * the library's own vocabularies instead of inventing parallel numbering:
 * aggregation ids are AggPrim, value tags are data_structures::ValueType.
 *
 * Memory ownership:
 *   - plan / navs / conds: host-owned, copied to device by ComputeFeaturesTask.
 *   - ComputeFeaturesAttributeColumn.values: host-owned pointer; the caller
 *     must keep it alive until LoadAttributes() returns.
 *
 * These are internal types: nothing in the public matrixgraph/ headers exposes
 * them, so fields may change without an ABI commitment.
 */

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using data_structures::ValueType;

/**
 * @brief Expression node types.
 *
 * Consumed by the interpreter's dispatch in EvalExpr().  New kinds go at the
 * end so that stored plans keep their meaning.
 */
enum MatrixGraphExprType {
  MG_EXPR_ATTR = 0,         /**< Read a per-vertex attribute by key. */
  MG_EXPR_CONST = 1,        /**< Literal constant (int/float/bool/time). */
  MG_EXPR_AGG = 2,          /**< Aggregation over a navigator. */
  MG_EXPR_TRANS = 3,        /**< Arithmetic/logical transformation. */
  MG_EXPR_PATTERN_ATTR = 4, /**< Pattern-matched attribute (not implemented). */
};

/**
 * @brief Navigator / binding generator types.
 *
 * A navigator produces a list of "binding" vertex ids for a given pivot.
 * Aggregations consume this list; FilterNav wraps another navigator and keeps
 * only the bindings that satisfy a condition.
 */
enum MatrixGraphNavType {
  MG_NAV_SELF = 0,     /**< Single binding: the pivot vertex itself. */
  MG_NAV_NEIGHBOR = 1, /**< Outgoing/incoming neighbors (see direction). */
  MG_NAV_FILTER = 2,   /**< Filtered wrapper around another navigator. */
  MG_NAV_PATTERN = 3,  /**< Pattern-based bindings (not implemented). */
};

/**
 * @brief Comparison operators used by FilterNav conditions.
 *
 * Comparisons are performed in double precision after converting both operands
 * with FeatureValue::ToDouble().
 */
enum MatrixGraphCondType {
  MG_COND_EQ = 0, /**< Equal. */
  MG_COND_NE = 1, /**< Not equal. */
  MG_COND_LT = 2, /**< Less than. */
  MG_COND_LE = 3, /**< Less than or equal. */
  MG_COND_GT = 4, /**< Greater than. */
  MG_COND_GE = 5, /**< Greater than or equal. */
};

/**
 * @brief Transformation operators for TransExpr.
 *
 * NEG, ABS and SQRT are unary and only use child_a.  All others are binary
 * and use both child_a and child_b.  Transforms are evaluated in double
 * precision and produce float64 values.
 */
enum MatrixGraphTransOp {
  MG_TRANS_ADD = 0,
  MG_TRANS_SUB = 1,
  MG_TRANS_MUL = 2,
  MG_TRANS_DIV = 3,
  MG_TRANS_NEG = 4,
  MG_TRANS_ABS = 5,
  MG_TRANS_SQRT = 6,
  MG_TRANS_POW = 7,
};

/**
 * @brief A single filter condition.
 *
 * left_expr and right_expr are indices into the expression plan array.
 * The condition is evaluated against the current navigator binding, so both
 * expressions are evaluated with the binding vertex as pivot_vid.
 */
struct MatrixGraphCondNode {
  MatrixGraphCondType op; /**< Comparison operator. */
  int32_t left_expr;      /**< Index of the left-hand expression in the plan. */
  int32_t right_expr; /**< Index of the right-hand expression in the plan. */
};

/**
 * @brief A flat plan node.
 *
 * A node is polymorphic: the interpretation of each field depends on `type`.
 * The same struct is reused for both expression nodes and navigator nodes, but
 * they live in two separate arrays (`plan` and `navs`), so callers must read
 * `type` with the matching enum.  Only one of agg_op / trans_op applies to a
 * given node.
 *
 * Expression nodes (MatrixGraphExprType):
 *   - ATTR:         type=MG_EXPR_ATTR, key
 *   - CONST:        type=MG_EXPR_CONST, const_type, const_*
 *   - AGG:          type=MG_EXPR_AGG, agg_op, src_idx, nav_idx
 *   - TRANS:        type=MG_EXPR_TRANS, trans_op, child_a, child_b
 *   - PATTERN_ATTR: type=MG_EXPR_PATTERN_ATTR, key
 *
 * Navigator nodes (MatrixGraphNavType):
 *   - SELF:     type=MG_NAV_SELF
 *   - NEIGHBOR: type=MG_NAV_NEIGHBOR, direction, target_label
 *   - FILTER:   type=MG_NAV_FILTER, inner_nav_idx, cond_idx
 *   - PATTERN:  type=MG_NAV_PATTERN, pattern_ref
 */
struct MatrixGraphPlanNode {
  int32_t type; /**< MatrixGraphExprType for expressions; MatrixGraphNavType for
                   navigators. */

  /* Expression-specific fields. */
  AggPrim agg_op;              /**< Aggregation primitive (AggExpr). */
  MatrixGraphTransOp trans_op; /**< Transform operator (TransExpr). */
  int32_t src_idx;             /**< Child expression index (AggExpr). */
  int32_t nav_idx;             /**< Navigator plan index (AggExpr). */
  int32_t child_a;             /**< First child (TransExpr / binary ops). */
  int32_t child_b;             /**< Second child. */

  /* For AttrExpr / PatternAttrExpr. */
  char key[64];

  /* For ConstExpr.  Only the member matching const_type is valid. */
  ValueType const_type;
  int64_t const_i64;
  double const_f64;
  bool const_b;

  /* For Navigators. */
  int32_t direction;     /**< 0=out, 1=in, 2=both (NeighborNav; unsupported). */
  char target_label[64]; /**< Empty = no label filter (NeighborNav). */
  int32_t cond_idx;      /**< Filter condition index (FilterNav). */
  int32_t inner_nav_idx; /**< Wrapped navigator index (FilterNav/PatternNav). */
  int32_t pattern_ref;   /**< Pattern reference (PatternNav). */
};

/**
 * @brief A single column of per-vertex attribute values.
 *
 * `values` is a host pointer to n_values contiguous entries of the declared
 * type.  It must remain valid for the duration of LoadAttributes().
 */
struct ComputeFeaturesAttributeColumn {
  char key[64];         /**< Attribute name used by AttrExpr. */
  ValueType value_type; /**< Element type of `values`. */
  uint32_t n_values;    /**< Must equal the number of vertices in the graph. */
  void* values; /**< Host-owned; n_values entries of the declared type. */
};

/* Shorter aliases for the interpreter signatures. */
using MGPlanNode = MatrixGraphPlanNode;
using MGCondNode = MatrixGraphCondNode;

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_GPU_TASK_COMPUTE_FEATURES_PLAN_TYPES_H_