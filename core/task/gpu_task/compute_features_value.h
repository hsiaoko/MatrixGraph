/**
 * @file compute_features_value.h
 * @brief Bridge between the plan wire structs and the canonical FeatureValue.
 *
 * compute_features_types.h is an extern "C" wire header: it fixes the plan
 * layout and the MG_* numbering that external callers write. The interpreter
 * runs on task::FeatureValue (the same container the aggregation primitives
 * and GraphAggregate use), so every crossing needs an explicit, total
 * conversion. Unknown wire tags are programming errors and abort the
 * conversion; callers check the invalid tag.
 */

#ifndef MATRIXGRAPH_CORE_TASK_GPU_TASK_COMPUTE_FEATURES_VALUE_H_
#define MATRIXGRAPH_CORE_TASK_GPU_TASK_COMPUTE_FEATURES_VALUE_H_

#include "core/data_structures/attributes.h"
#include "core/task/gpu_task/compute_features_types.h"
#include "core/task/gpu_task/execute_agg_prim.cuh"

#include <cstdint>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using data_structures::ValueType;

/** @brief Map a wire value tag (MG_VALUE_*) to the internal ValueType. */
__device__ __host__ inline ValueType ToValueType(int32_t mg_tag) {
  switch (mg_tag) {
    case MG_VALUE_INT:
      return ValueType::kInt;
    case MG_VALUE_FLOAT64:
      return ValueType::kFloat64;
    case MG_VALUE_FLOAT32:
      return ValueType::kFloat32;
    case MG_VALUE_BOOL:
      return ValueType::kBool;
    case MG_VALUE_TIME:
      return ValueType::kTime;
    case MG_VALUE_STRING:
      return ValueType::kString;
    default:
      return ValueType::kInvalid;
  }
}

/** @brief Map an internal ValueType to the wire value tag (MG_VALUE_*). */
__device__ __host__ inline int32_t ToMgTag(ValueType type) {
  switch (type) {
    case ValueType::kInt:
      return MG_VALUE_INT;
    case ValueType::kFloat64:
      return MG_VALUE_FLOAT64;
    case ValueType::kFloat32:
      return MG_VALUE_FLOAT32;
    case ValueType::kBool:
      return MG_VALUE_BOOL;
    case ValueType::kTime:
      return MG_VALUE_TIME;
    case ValueType::kString:
      return MG_VALUE_STRING;
    default:
      return MG_VALUE_INVALID;
  }
}

/** @brief Copy a wire value into the canonical container. */
__device__ __host__ inline FeatureValue ToFeatureValue(
    const MatrixGraphFeatureValue& wire) {
  FeatureValue out = MakeInvalidValue();
  out.type = ToValueType(wire.type);
  switch (out.type) {
    case ValueType::kInt:
    case ValueType::kTime:
      out.i64 = wire.i64;
      break;
    case ValueType::kFloat64:
    case ValueType::kFloat32:
      out.f64 = wire.f64;
      break;
    case ValueType::kBool:
      out.b = wire.b != 0;
      break;
    default:
      break;
  }
  return out;
}

/** @brief Copy a canonical value into the wire struct for the C-facing API. */
__device__ __host__ inline MatrixGraphFeatureValue ToWireValue(
    const FeatureValue& value) {
  MatrixGraphFeatureValue wire{};
  wire.type = ToMgTag(value.type);
  switch (value.type) {
    case ValueType::kInt:
    case ValueType::kTime:
      wire.i64 = value.i64;
      break;
    case ValueType::kFloat64:
    case ValueType::kFloat32:
      wire.f64 = value.f64;
      break;
    case ValueType::kBool:
      wire.b = value.b ? 1 : 0;
      break;
    default:
      break;
  }
  return wire;
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_GPU_TASK_COMPUTE_FEATURES_VALUE_H_