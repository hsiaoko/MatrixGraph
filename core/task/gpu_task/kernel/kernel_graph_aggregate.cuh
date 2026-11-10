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
// FeatureValue, its helpers, AggPrim, AllFeatures and the BlockAgg* primitives
// are all shared with execute_agg_prim (the canonical task:: definitions).
// GraphAggregate and ExecuteAggPrim therefore expose one and the same primitive
// ordering (aligned with featurelib/primitive/prim_mapping.go).
// ---------------------------------------------------------------------------
using sics::matrixgraph::core::task::FeatureValue;
using sics::matrixgraph::core::task::MakeIntValue;
using sics::matrixgraph::core::task::MakeBoolValue;
using sics::matrixgraph::core::task::MakeInvalidValue;
using sics::matrixgraph::core::task::MakeFloat64Value;
using sics::matrixgraph::core::task::AggPrim;
using sics::matrixgraph::core::task::AllFeatures;

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
