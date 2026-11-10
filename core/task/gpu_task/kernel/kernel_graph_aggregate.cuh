#ifndef MATRIXGRAPH_CORE_TASK_GPU_TASK_KERNEL_KERNEL_GRAPH_AGGREGATE_CUH_
#define MATRIXGRAPH_CORE_TASK_GPU_TASK_KERNEL_KERNEL_GRAPH_AGGREGATE_CUH_

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#include "core/data_structures/attributes.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/task/gpu_task/block_agg_prims.cuh"
#include "core/task/gpu_task/execute_agg_prim.cuh"

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
// FeatureValue and its helpers are shared with execute_agg_prim.cuh (the
// canonical task::FeatureValue).  A local redefinition here created a second,
// distinct FeatureValue type, so the BlockAggCount defined in this TU and the
// one in execute_agg_prim.cu returned different types — nvlink's "prototype
// doesn't match" at device-link time.
//
// AggPrim and AllFeatures below are intentionally LOCAL to GraphAggregate:
// their value ordering differs from task::AggPrim / task::AllFeatures and the
// Go C API depends on this ordering, so they must not be aliased to the task
// versions.
// ---------------------------------------------------------------------------
using sics::matrixgraph::core::task::FeatureValue;
using sics::matrixgraph::core::task::MakeIntValue;
using sics::matrixgraph::core::task::MakeBoolValue;
using sics::matrixgraph::core::task::MakeInvalidValue;
using sics::matrixgraph::core::task::MakeFloat64Value;

// ---------------------------------------------------------------------------
// Aggregation primitives (GraphAggregate ordering — do not reorder).
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
// Aggregation operators (device). BlockAgg* primitives live in
// block_agg_prims.cuh; the dispatcher is defined here because it switches over
// this TU's AggPrim ordering.
// ---------------------------------------------------------------------------
__device__ inline FeatureValue BlockApplyAggPrim(AggPrim prim,
                                                 FeatureValue* values,
                                                 uint32_t n,
                                                 const BlockWorkspace& ws);

// ---------------------------------------------------------------------------
// AllFeatures: fused output container for compute_all.
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

// Fused compute-all: compute all primitives in one pass over the value list.
// The caller owns the input buffer; values may be reordered (sorted in-place).
__device__ inline AllFeatures BlockComputeAllFeaturesFromValues(
    FeatureValue* values, uint32_t n, const BlockWorkspace& ws);

// Returns the dynamic shared memory bytes required for one block given the
// per-pivot neighbor cap.  Layout: [values buf][sum buf][minmax buf][scan buf]
// (see GetBlockWorkspace in block_agg_prims.cuh).
__host__ size_t ComputeGraphAggregateSharedMemSize(uint32_t max_neighbors);

// ---------------------------------------------------------------------------
// FeatureRequest
// ---------------------------------------------------------------------------
struct FeatureRequest {
  AttributeName attr_name;    // target attribute to aggregate
  uint32_t neighbor_label;    // edge label filter (0 = any)
  bool use_outgoing;          // true = outgoing neighbors, false = incoming
  AggPrim prim;               // aggregation primitive
  bool use_edge_attrs = false; // true = read attr from edge, false = from neighbor vertex
};

// ---------------------------------------------------------------------------
// Kernel declaration
//
// Single-graph variant: the task now owns one ImmutableCSR, so the kernels no
// longer need graph-id indexing.  All pointers are device memory for that one
// graph.
// ---------------------------------------------------------------------------
__global__ void ComputeFeaturesKernel(
    const uint8_t* graph_data,                  // single CSR buffer
    uint32_t n_vertices,
    uint32_t n_in_edges,
    uint32_t n_out_edges,
    const Attributes* vertex_attrs,             // [n_vertices]
    const Attributes* edge_attrs_out,           // shared table for outgoing edges
    const Attributes* edge_attrs_in,            // shared table for incoming edges
    const uint32_t* pivot_vertex_ids,           // [n_pivots]
    uint32_t n_pivots,
    const FeatureRequest* requests,             // [n_requests]
    uint32_t n_requests,
    uint32_t max_neighbors,
    FeatureValue* d_outputs);                   // [n_pivots * n_requests]

// Fused kernel: one pass over neighbors produces all aggregation primitives.
__global__ void ComputeAllFeaturesKernel(
    const uint8_t* graph_data,                  // single CSR buffer
    uint32_t n_vertices,
    uint32_t n_in_edges,
    uint32_t n_out_edges,
    const Attributes* vertex_attrs,             // [n_vertices]
    const Attributes* edge_attrs_out,           // shared table for outgoing edges
    const Attributes* edge_attrs_in,            // shared table for incoming edges
    const uint32_t* pivot_vertex_ids,           // [n_pivots]
    uint32_t n_pivots,
    AttributeName attr_name,                    // target attribute to aggregate
    bool use_outgoing,                          // true = outgoing neighbors
    bool use_edge_attrs,                        // true = read attr from edge
    uint32_t max_neighbors,
    AllFeatures* d_outputs);                    // [n_pivots]

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_GPU_TASK_KERNEL_KERNEL_GRAPH_AGGREGATE_CUH_
