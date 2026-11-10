/**
 * @file kernel_graph_aggregate.cu
 * @brief Device kernels for GraphAggregate: per-pivot neighbor feature
 *        collection and fused aggregation.
 *
 * The generic BlockAgg* primitives live in block_agg_prims.cuh and are shared
 * with execute_agg_prim.  This TU owns only what is specific to graph
 * aggregation:
 *   - ReadAttributeValue / CollectNeighborValuesBlock: gather a pivot's
 *     neighbor attribute values into shared memory.
 *   - BlockApplyAggPrim / BlockComputeAllFeaturesFromValues: the dispatchers,
 *     which switch over this TU's own AggPrim / AllFeatures ordering.
 *   - ComputeFeaturesKernel / ComputeAllFeaturesKernel and the host-side
 *     shared-memory size helper.
 */

#include "core/task/gpu_task/kernel/kernel_graph_aggregate.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cfloat>

#include "core/common/consts.h"

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
using sics::matrixgraph::core::data_structures::GetInt;
using sics::matrixgraph::core::data_structures::GetFloat64;
using sics::matrixgraph::core::data_structures::GetBool;

// Attribute access helpers
// ---------------------------------------------------------------------------
__device__ inline bool ReadAttributeValue(const Attribute* attr,
                                          uint32_t row,
                                          FeatureValue* out) {
  if (!attr) return false;
  out->type = attr->type;
  switch (attr->type) {
    case ValueType::kInt:
    case ValueType::kTime:
      out->i64 = GetInt(*attr, row);
      break;
    case ValueType::kFloat64:
      out->f64 = GetFloat64(*attr, row);
      break;
    case ValueType::kFloat32:
      out->f64 = static_cast<double>(*(static_cast<const float*>(attr->data) + row));
      break;
    case ValueType::kBool:
      out->b = GetBool(*attr, row);
      break;
    default:
      out->type = ValueType::kInvalid;
      return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Cooperative neighbor value collection
// ---------------------------------------------------------------------------
__device__ inline uint32_t CollectNeighborValuesBlock(
    const uint8_t* graph_data,
    uint32_t n_vertices,
    uint32_t n_in_edges,
    uint32_t n_out_edges,
    const Attributes* vertex_attrs,
    const Attributes* edge_attrs_out,
    const Attributes* edge_attrs_in,
    uint32_t pivot_vid,
    const AttributeName& attr_name,
    bool use_outgoing,
    bool use_edge_attrs,
    FeatureValue* shared_buf,
    uint32_t max_neighbors,
    uint32_t* scratch) {
  // Parse CSR layout (matches ImmutableCSR::ParseBasePtr)
  const VertexID* globalid = reinterpret_cast<const VertexID*>(graph_data);
  const VertexID* indegree = globalid + n_vertices;
  const VertexID* outdegree = indegree + n_vertices;
  const EdgeIndex* in_offset = reinterpret_cast<const EdgeIndex*>(outdegree + n_vertices);
  const EdgeIndex* out_offset = in_offset + n_vertices + 1;
  const VertexID* incoming_edges = reinterpret_cast<const VertexID*>(out_offset + n_vertices + 1);
  const VertexID* outgoing_edges = incoming_edges + n_in_edges;

  uint32_t deg = use_outgoing ? outdegree[pivot_vid] : indegree[pivot_vid];
  const VertexID* edges = use_outgoing
      ? outgoing_edges + out_offset[pivot_vid]
      : incoming_edges + in_offset[pivot_vid];
  const uint32_t edge_base = use_outgoing
      ? out_offset[pivot_vid]
      : in_offset[pivot_vid];
  const Attributes* edge_attrs = use_outgoing ? edge_attrs_out : edge_attrs_in;

  // Single-pass cooperative collection with a shared atomic counter.
  if (threadIdx.x == 0) scratch[0] = 0;
  __syncthreads();

  for (uint32_t i = threadIdx.x; i < deg; i += blockDim.x) {
    const Attribute* attr = nullptr;
    uint32_t row = 0;
    if (use_edge_attrs) {
      attr = edge_attrs ? edge_attrs->attr_map.find(attr_name) : nullptr;
      row = edge_base + i;
    } else {
      VertexID neighbor = edges[i];
      attr = vertex_attrs ? vertex_attrs[neighbor].attr_map.find(attr_name) : nullptr;
      row = 0;
    }
    if (!attr) continue;
    FeatureValue v;
    if (!ReadAttributeValue(attr, row, &v)) continue;
    uint32_t pos = atomicAdd(&scratch[0], 1);
    if (pos < max_neighbors) shared_buf[pos] = v;
  }
  __syncthreads();

  uint32_t total_count = scratch[0];
  return min(total_count, max_neighbors);
}

// Kernels: one block per pivot.
// ---------------------------------------------------------------------------
__global__ void ComputeFeaturesKernel(
    const uint8_t* graph_data,
    uint32_t n_vertices,
    uint32_t n_in_edges,
    uint32_t n_out_edges,
    const Attributes* vertex_attrs,
    const Attributes* edge_attrs_out,
    const Attributes* edge_attrs_in,
    const uint32_t* pivot_vertex_ids,
    uint32_t n_pivots,
    const FeatureRequest* requests,
    uint32_t n_requests,
    uint32_t max_neighbors,
    FeatureValue* d_outputs) {
  extern __shared__ uint8_t shared_mem[];
  FeatureValue* shared_buf = reinterpret_cast<FeatureValue*>(shared_mem);
  uint32_t* scratch = reinterpret_cast<uint32_t*>(
      shared_mem + max_neighbors * sizeof(FeatureValue));
  BlockWorkspace ws = GetBlockWorkspace(shared_mem, max_neighbors);

  uint32_t pivot_idx = blockIdx.x;
  if (pivot_idx >= n_pivots) return;

  uint32_t vid = pivot_vertex_ids[pivot_idx];

  for (uint32_t req_idx = 0; req_idx < n_requests; ++req_idx) {
    uint32_t n_collected = CollectNeighborValuesBlock(
        graph_data, n_vertices, n_in_edges, n_out_edges,
        vertex_attrs, edge_attrs_out, edge_attrs_in, vid,
        requests[req_idx].attr_name,
        requests[req_idx].use_outgoing,
        requests[req_idx].use_edge_attrs,
        shared_buf, max_neighbors, scratch);

    __syncthreads();
    FeatureValue result = BlockApplyAggPrim(
        requests[req_idx].prim, shared_buf, n_collected, ws);

    if (threadIdx.x == 0) {
      d_outputs[pivot_idx * n_requests + req_idx] = result;
    }
    __syncthreads();
  }
}

__launch_bounds__(common::kAllFeaturesBlockDim)
__global__ void ComputeAllFeaturesKernel(
    const uint8_t* graph_data,
    uint32_t n_vertices,
    uint32_t n_in_edges,
    uint32_t n_out_edges,
    const Attributes* vertex_attrs,
    const Attributes* edge_attrs_out,
    const Attributes* edge_attrs_in,
    const uint32_t* pivot_vertex_ids,
    uint32_t n_pivots,
    AttributeName attr_name,
    bool use_outgoing,
    bool use_edge_attrs,
    uint32_t max_neighbors,
    AllFeatures* d_outputs) {
  extern __shared__ uint8_t shared_mem[];
  FeatureValue* shared_buf = reinterpret_cast<FeatureValue*>(shared_mem);
  uint32_t* scratch = reinterpret_cast<uint32_t*>(
      shared_mem + max_neighbors * sizeof(FeatureValue));
  BlockWorkspace ws = GetBlockWorkspace(shared_mem, max_neighbors);

  uint32_t pivot_idx = blockIdx.x;
  if (pivot_idx >= n_pivots) return;

  uint32_t vid = pivot_vertex_ids[pivot_idx];

  FeatureRequest req;
  req.attr_name = attr_name;
  req.neighbor_label = 0;
  req.use_outgoing = use_outgoing;
  req.use_edge_attrs = use_edge_attrs;

  uint32_t n_collected = CollectNeighborValuesBlock(
      graph_data, n_vertices, n_in_edges, n_out_edges,
      vertex_attrs, edge_attrs_out, edge_attrs_in, vid,
      req.attr_name, req.use_outgoing, req.use_edge_attrs,
      shared_buf, max_neighbors, scratch);

  __syncthreads();
  AllFeatures result = BlockComputeAllFeaturesFromValues(
      shared_buf, n_collected, ws);

  if (threadIdx.x == 0) {
    d_outputs[pivot_idx] = result;
  }
}

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

__host__ size_t ComputeGraphAggregateSharedMemSize(uint32_t max_neighbors) {
  // FeatureValue buffer for collected neighbors + scratch for reductions/scans.
  return static_cast<size_t>(max_neighbors) * sizeof(FeatureValue) +
         static_cast<size_t>(common::kBlockDim) * sizeof(double) +
         static_cast<size_t>(common::kBlockDim) * sizeof(FeatureValue) +
         static_cast<size_t>(common::kBlockDim) * sizeof(uint32_t);
}

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
