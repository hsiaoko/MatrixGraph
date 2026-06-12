#ifndef MATRIXGRAPH_CORE_TASK_GPU_TASK_COMPUTE_FEATURES_TYPES_H_
#define MATRIXGRAPH_CORE_TASK_GPU_TASK_COMPUTE_FEATURES_TYPES_H_

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file compute_features_types.h
 * @brief C-compatible flat plan types for the ComputeFeaturesTask.
 *
 * This header defines the data structures that cross the host/device and C/Go
 * boundaries.  A feature expression is compiled into a flat array of
 * MatrixGraphPlanNode nodes; navigator and condition arrays live in separate
 * flat arrays.  The host (or Go code) builds these arrays, uploads them to the
 * GPU, and the device interpreter evaluates them.
 *
 * Memory ownership:
 *   - plan / navs / conds: host-owned, copied to device by ComputeFeaturesTask.
 *   - ComputeFeaturesAttributeColumn.values: host-owned pointer; the caller
 *     must keep it alive until LoadAttributes() returns.
 *   - MatrixGraphFeatureValue is a discriminated union.  Only the member that
 *     corresponds to `type` is valid; writing all members at once will clobber
 *     each other because they share storage.
 */

/**
 * @brief Expression node types.
 *
 * These values are consumed by the device interpreter switch in EvalExpr().
 * New expression kinds must be added at the end to keep existing serialized
 * plans stable.
 */
enum MatrixGraphExprType {
  MG_EXPR_ATTR = 0,          /**< Read a per-vertex attribute by key. */
  MG_EXPR_CONST = 1,         /**< Literal constant (int/float/bool/time). */
  MG_EXPR_AGG = 2,           /**< Aggregation over a navigator. */
  MG_EXPR_TRANS = 3,         /**< Arithmetic/logical transformation. */
  MG_EXPR_PATTERN_ATTR = 4,  /**< Pattern-matched attribute (not implemented). */
};

/**
 * @brief Navigator / binding generator types.
 *
 * A navigator produces a list of "binding" vertex ids for a given pivot.
 * Aggregations consume this list; FilterNav wraps another navigator and keeps
 * only bindings that satisfy a condition.
 */
enum MatrixGraphNavType {
  MG_NAV_SELF = 0,      /**< Single binding: the pivot vertex itself. */
  MG_NAV_NEIGHBOR = 1,  /**< Outgoing/incoming neighbors (see direction). */
  MG_NAV_FILTER = 2,    /**< Filtered wrapper around another navigator. */
  MG_NAV_PATTERN = 3,   /**< Pattern-based bindings (not implemented). */
};

/**
 * @brief Comparison operators used by FilterNav conditions.
 *
 * Comparisons are performed in double precision after converting both operands
 * with ToDouble().
 */
enum MatrixGraphCondType {
  MG_COND_EQ = 0,  /**< Equal. */
  MG_COND_NE = 1,  /**< Not equal. */
  MG_COND_LT = 2,  /**< Less than. */
  MG_COND_LE = 3,  /**< Less than or equal. */
  MG_COND_GT = 4,  /**< Greater than. */
  MG_COND_GE = 5,  /**< Greater than or equal. */
};

/**
 * @brief Value type tags for MatrixGraphFeatureValue.
 *
 * Only INT, FLOAT64, BOOL and TIME are fully supported by the current device
 * interpreter.  STRING is accepted during attribute loading but not returned by
 * aggregation primitives.
 */
enum MatrixGraphValueType {
  MG_VALUE_INVALID = 0,
  MG_VALUE_INT = 1,
  MG_VALUE_FLOAT64 = 2,
  MG_VALUE_BOOL = 3,
  MG_VALUE_STRING = 4,
  MG_VALUE_TIME = 5,
  MG_VALUE_FLOAT32 = 6,
};

/**
 * @brief Aggregation primitive ids.
 *
 * These intentionally match the AggPrim ordering used by GraphAggregate so that
 * callers can reuse primitive constants across the two tasks.
 */
enum MatrixGraphAggPrim {
  MG_AGG_COUNT = 0,
  MG_AGG_COUNT_GREATER_THAN_MEAN = 1,
  MG_AGG_NUM_UNIQUE = 2,
  MG_AGG_SUM = 3,
  MG_AGG_MEAN = 4,
  MG_AGG_VARIANCE = 5,
  MG_AGG_STD = 6,
  MG_AGG_MODE = 7,
  MG_AGG_MIN = 8,
  MG_AGG_MAX = 9,
  MG_AGG_MEDIAN = 10,
  MG_AGG_QUARTER = 11,
  MG_AGG_QUARTILE3 = 12,
  MG_AGG_ENTROPY = 13,
  MG_AGG_PERCENT_TRUE = 14,
  MG_AGG_SKEW = 15,
};

/**
 * @brief Transformation operators for TransExpr.
 *
 * NEG, ABS and SQRT are unary and only use child_a.  All others are binary
 * and use both child_a and child_b.
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
  int32_t op;          /**< MatrixGraphCondType. */
  int32_t left_expr;   /**< Index of the left-hand expression in the plan. */
  int32_t right_expr;  /**< Index of the right-hand expression in the plan. */
};

/**
 * @brief A flat plan node.
 *
 * A node is polymorphic: the interpretation of each field depends on `type`.
 * The same C struct is reused for both expression nodes and navigator nodes,
 * but they live in two separate arrays (`plan` and `navs`).  Callers must
 * therefore use the correct enum when reading `type`.
 *
 * Expression nodes (MatrixGraphExprType):
 *   - ATTR:         type=MG_EXPR_ATTR, key
 *   - CONST:        type=MG_EXPR_CONST, const_type, const_*
 *   - AGG:          type=MG_EXPR_AGG, op=MG_AGG_*, src_idx, nav_idx
 *   - TRANS:        type=MG_EXPR_TRANS, op=MG_TRANS_*, child_a, child_b
 *   - PATTERN_ATTR: type=MG_EXPR_PATTERN_ATTR, key
 *
 * Navigator nodes (MatrixGraphNavType):
 *   - SELF:     type=MG_NAV_SELF
 *   - NEIGHBOR: type=MG_NAV_NEIGHBOR, direction, target_label
 *   - FILTER:   type=MG_NAV_FILTER, inner_nav_idx, cond_idx
 *   - PATTERN:  type=MG_NAV_PATTERN, pattern_ref
 */
struct MatrixGraphPlanNode {
  int32_t type;  /**< MatrixGraphExprType for expressions; negative nav tag. */

  /* Expression-specific fields. */
  int32_t op;       /**< Aggregation primitive id (for AggExpr). */
  int32_t src_idx;  /**< Child expression index (for AggExpr). */
  int32_t nav_idx;  /**< Navigator plan index (for AggExpr). */
  int32_t child_a;  /**< First child (for TransExpr / binary ops). */
  int32_t child_b;  /**< Second child. */

  /* For AttrExpr / PatternAttrExpr. */
  char key[64];

  /* For ConstExpr.  Only the member matching const_type is valid. */
  int32_t const_type;
  int64_t const_i64;
  double const_f64;
  int32_t const_b;

  /* For Navigators. */
  int32_t direction;       /**< 0=out, 1=in, 2=both (NeighborNav; 2 unsupported). */
  char target_label[64];   /**< Empty = no label filter (NeighborNav). */
  int32_t cond_idx;        /**< Filter condition index (FilterNav). */
  int32_t inner_nav_idx;   /**< Wrapped navigator index (FilterNav/PatternNav). */
  int32_t pattern_ref;     /**< Pattern reference (PatternNav). */
};

/**
 * @brief A typed feature value.
 *
 * This is a discriminated union.  The member that corresponds to `type` is the
 * only valid member.  Because i64/f64/b share storage, callers must not assign
 * all three fields sequentially; doing so overwrites earlier values.
 */
struct MatrixGraphFeatureValue {
  int32_t type;  /**< MatrixGraphValueType. */
  union {
    int64_t i64;  /**< Valid when type == MG_VALUE_INT or MG_VALUE_TIME. */
    double f64;   /**< Valid when type == MG_VALUE_FLOAT64 or MG_VALUE_FLOAT32. */
    int32_t b;    /**< Valid when type == MG_VALUE_BOOL. */
  };
};

/**
 * @brief A single column of per-vertex attribute values.
 *
 * `values` is a host pointer to n_values contiguous entries of the declared
 * type.  It must remain valid for the duration of LoadAttributes().
 */
struct ComputeFeaturesAttributeColumn {
  char key[64];       /**< Attribute name used by AttrExpr. */
  int32_t value_type; /**< MatrixGraphValueType. */
  uint32_t n_values;  /**< Must equal the number of vertices in the graph. */
  void* values;       /**< Host-owned; n_values entries of the declared type. */
};

/* Shorter aliases for internal C++ code. */
typedef struct MatrixGraphPlanNode MGPlanNode;
typedef struct MatrixGraphFeatureValue MGFeatureValue;
typedef struct MatrixGraphCondNode MGCondNode;

#ifdef __cplusplus
}
#endif

#endif  // MATRIXGRAPH_CORE_TASK_GPU_TASK_COMPUTE_FEATURES_TYPES_H_
