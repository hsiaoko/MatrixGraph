#ifndef MATRIXGRAPH_CORE_TASK_GPU_TASK_KERNEL_GRAPH_AGGREGATE_CUH_
#define MATRIXGRAPH_CORE_TASK_GPU_TASK_KERNEL_GRAPH_AGGREGATE_CUH_

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#include "core/data_structures/attributes.h"
#include "core/data_structures/immutable_csr.cuh"

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

// ---------------------------------------------------------------------------
// FeatureValue: unified value container for feature computation
// ---------------------------------------------------------------------------
struct FeatureValue {
  ValueType type = ValueType::kInvalid;
  union {
    int64_t i64;
    double f64;
    bool b;
  };

  __device__ __host__ bool IsValid() const { return type != ValueType::kInvalid; }

  __host__ __device__ double ToDouble() const {
    switch (type) {
      case ValueType::kInt:
      case ValueType::kTime:    return static_cast<double>(i64);
      case ValueType::kFloat64: return f64;
      case ValueType::kFloat32: return static_cast<double>(f64); // stored in f64 field for simplicity
      case ValueType::kBool:    return b ? 1.0 : 0.0;
      default:                  return 0.0;
    }
  }

  __device__ int Compare(const FeatureValue& other) const {
    double a = ToDouble();
    double d = other.ToDouble();
    if (a < d) return -1;
    if (a > d) return 1;
    return 0;
  }
};

__device__ __host__ inline FeatureValue MakeIntValue(int64_t v) {
  FeatureValue r; r.type = ValueType::kInt; r.i64 = v; return r;
}
__device__ __host__ inline FeatureValue MakeFloatValue(double v) {
  FeatureValue r; r.type = ValueType::kFloat64; r.f64 = v; return r;
}
__device__ __host__ inline FeatureValue MakeBoolValue(bool v) {
  FeatureValue r; r.type = ValueType::kBool; r.b = v; return r;
}
__device__ __host__ inline FeatureValue MakeInvalidValue() {
  FeatureValue r; r.type = ValueType::kInvalid; return r;
}

// ---------------------------------------------------------------------------
// Aggregation primitives
// ---------------------------------------------------------------------------
enum class AggPrim : uint8_t {
  kCount = 0,
  kCountGreaterThanMean,
  kNumUnique,
  kSum,
  kMean,
  kVariance,
  kStd,
  kMode,
  kMin,
  kMax,
  kMedian,
  kQuarter,
  kQuartile3,
  kEntropy,
  kPercentTrue,
  kSkew
};

// ---------------------------------------------------------------------------
// Aggregation operators (device)
// ---------------------------------------------------------------------------
__device__ inline FeatureValue AggCount(const FeatureValue* values, uint32_t n);
__device__ inline FeatureValue AggSum(const FeatureValue* values, uint32_t n);
__device__ inline FeatureValue AggMean(const FeatureValue* values, uint32_t n);
__device__ inline FeatureValue AggMin(FeatureValue* values, uint32_t n);
__device__ inline FeatureValue AggMax(FeatureValue* values, uint32_t n);
__device__ inline FeatureValue AggVariance(FeatureValue* values, uint32_t n);
__device__ inline FeatureValue AggStd(FeatureValue* values, uint32_t n);
__device__ inline FeatureValue AggMedian(FeatureValue* values, uint32_t n);
__device__ inline FeatureValue AggMode(FeatureValue* values, uint32_t n);
__device__ inline FeatureValue AggNumUnique(FeatureValue* values, uint32_t n);
__device__ inline FeatureValue AggEntropy(FeatureValue* values, uint32_t n);
__device__ inline FeatureValue AggQuarter(FeatureValue* values, uint32_t n);
__device__ inline FeatureValue AggQuartile3(FeatureValue* values, uint32_t n);
__device__ inline FeatureValue AggPercentTrue(const FeatureValue* values, uint32_t n);
__device__ inline FeatureValue AggSkew(FeatureValue* values, uint32_t n);
__device__ inline FeatureValue AggCountGreaterThanMean(FeatureValue* values, uint32_t n);

__device__ inline void DeviceSortValues(FeatureValue* arr, uint32_t n);
__device__ inline FeatureValue ApplyAggPrim(AggPrim prim, FeatureValue* values, uint32_t n);

// ---------------------------------------------------------------------------
// AllFeatures: fused output container for compute_all
// ---------------------------------------------------------------------------
struct AllFeatures {
  FeatureValue count;
  FeatureValue count_greater_than_mean;
  FeatureValue num_unique;
  FeatureValue sum;
  FeatureValue mean;
  FeatureValue variance;
  FeatureValue std;
  FeatureValue mode;
  FeatureValue min;
  FeatureValue max;
  FeatureValue median;
  FeatureValue quarter;
  FeatureValue quartile3;
  FeatureValue entropy;
  FeatureValue percent_true;
  FeatureValue skew;
};

// ---------------------------------------------------------------------------
// Fused aggregation: compute all primitives in one pass over the value list.
// The caller owns the input buffer; values may be reordered (sorted in-place).
// ---------------------------------------------------------------------------
__device__ inline AllFeatures ComputeAllFeaturesFromValues(FeatureValue* values,
                                                           uint32_t n);

// ---------------------------------------------------------------------------
// FeatureRequest
// ---------------------------------------------------------------------------
struct FeatureRequest {
  AttributeName attr_name;    // target attribute to aggregate
  uint32_t neighbor_label;    // edge label filter (0 = any)
  bool use_outgoing;          // true = outgoing neighbors, false = incoming
  AggPrim prim;               // aggregation primitive
};

// ---------------------------------------------------------------------------
// Kernel declaration
// ---------------------------------------------------------------------------
__global__ void ComputeFeaturesKernel(
    const uint8_t* const* graph_data_buffers,   // [n_graphs]
    const uint32_t* graph_n_vertices,           // [n_graphs]
    const uint32_t* graph_n_in_edges,           // [n_graphs]
    const uint32_t* graph_n_out_edges,          // [n_graphs]
    const Attributes* const* vertex_attrs,      // [n_graphs][n_vertices]
    const uint32_t* pivot_graph_id,             // [n_pivots]
    const uint32_t* pivot_vertex_id,            // [n_pivots]
    uint32_t n_pivots,
    const FeatureRequest* requests,             // [n_requests]
    uint32_t n_requests,
    FeatureValue* d_workspace,                  // [n_pivots * max_neighbors]
    uint32_t max_neighbors,
    FeatureValue* d_outputs);                   // [n_pivots * n_requests]

// Fused kernel: one pass over neighbors produces all aggregation primitives.
__global__ void ComputeAllFeaturesKernel(
    const uint8_t* const* graph_data_buffers,   // [n_graphs]
    const uint32_t* graph_n_vertices,           // [n_graphs]
    const uint32_t* graph_n_in_edges,           // [n_graphs]
    const uint32_t* graph_n_out_edges,          // [n_graphs]
    const Attributes* const* vertex_attrs,      // [n_graphs][n_vertices]
    const uint32_t* pivot_graph_id,             // [n_pivots]
    const uint32_t* pivot_vertex_id,            // [n_pivots]
    uint32_t n_pivots,
    AttributeName attr_name,                    // target attribute to aggregate
    bool use_outgoing,                          // true = outgoing neighbors
    FeatureValue* d_workspace,                  // [n_pivots * max_neighbors]
    uint32_t max_neighbors,
    AllFeatures* d_outputs);                    // [n_pivots]

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_GPU_TASK_KERNEL_GRAPH_AGGREGATE_CUH_
