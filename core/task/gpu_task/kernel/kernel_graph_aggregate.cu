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

__device__ inline FeatureValue BlockApplyAggPrim(AggPrim prim,
                                                 FeatureValue* values,
                                                 uint32_t n,
                                                 const BlockWorkspace& ws) {
  switch (prim) {
    case AggPrim::kCount:                 return BlockAggCount(n);
    case AggPrim::kSum:                   return BlockAggSum(values, n, ws);
    case AggPrim::kMean:                  return BlockAggMean(values, n, ws);
    case AggPrim::kMin:                   return BlockAggMin(values, n, ws);
    case AggPrim::kMax:                   return BlockAggMax(values, n, ws);
    case AggPrim::kVariance:              return BlockAggVariance(values, n, ws);
    case AggPrim::kStd:                   return BlockAggStd(values, n, ws);
    case AggPrim::kMedian:                return BlockAggMedian(values, n);
    case AggPrim::kMode:                  return BlockAggMode(values, n);
    case AggPrim::kNumUnique:             return BlockAggNumUnique(values, n);
    case AggPrim::kEntropy:               return BlockAggEntropy(values, n);
    case AggPrim::kQuarter:               return BlockAggQuarter(values, n);
    case AggPrim::kQuartile3:             return BlockAggQuartile3(values, n);
    case AggPrim::kPercentTrue:           return BlockAggPercentTrue(values, n, ws);
    case AggPrim::kSkew:                  return BlockAggSkew(values, n, ws);
    case AggPrim::kCountGreaterThanMean:  return BlockAggCountGreaterThanMean(values, n, ws);
    default:                              return MakeInvalidValue();
  }
}

// ---------------------------------------------------------------------------
// Fused compute-all using block-level parallelism.
// ---------------------------------------------------------------------------
__device__ inline AllFeatures BlockComputeAllFeaturesFromValues(
    FeatureValue* values,
    uint32_t n,
    const BlockWorkspace& ws) {
  AllFeatures r;
  if (n == 0) {
    FeatureValue invalid = MakeInvalidValue();
    r.count = MakeIntValue(0);
    r.count_greater_than_mean = invalid;
    r.num_unique = invalid;
    r.sum = invalid;
    r.mean = invalid;
    r.variance = invalid;
    r.std = invalid;
    r.mode = invalid;
    r.min = invalid;
    r.max = invalid;
    r.median = invalid;
    r.quarter = invalid;
    r.quartile3 = invalid;
    r.entropy = invalid;
    r.percent_true = invalid;
    r.skew = invalid;
    return r;
  }

  // Pass 1: count, sum, min, max, percent_true in parallel.
  double local_sum = 0.0;
  uint32_t local_true = 0;
  FeatureValue local_min = values[0];
  FeatureValue local_max = values[0];
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    double x = values[i].ToDouble();
    local_sum += x;
    if (values[i].Compare(local_min) < 0) local_min = values[i];
    if (values[i].Compare(local_max) > 0) local_max = values[i];
    if (values[i].type == ValueType::kBool && values[i].b) ++local_true;
  }
  double sum = BlockSum(local_sum, ws.sum_buf);
  FeatureValue minv = BlockMin(local_min, ws.minmax_buf);
  FeatureValue maxv = BlockMax(local_max, ws.minmax_buf);
  uint32_t true_total = 0;
  BlockExclusiveScan(local_true, ws.scan_buf, &true_total);
  double mean = sum / static_cast<double>(n);

  // Pass 2: variance and count-greater-than-mean.
  double local_var = 0.0;
  uint32_t local_gtm = 0;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    double diff = values[i].ToDouble() - mean;
    local_var += diff * diff;
    if (values[i].ToDouble() > mean) ++local_gtm;
  }
  double variance = BlockSum(local_var, ws.sum_buf) / static_cast<double>(n);
  double stdv = sqrt(variance);
  uint32_t gtm_total = 0;
  BlockExclusiveScan(local_gtm, ws.scan_buf, &gtm_total);

  // Pass 3: skew.
  double local_skew = 0.0;
  bool skew_valid = (stdv != 0.0);
  if (skew_valid) {
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
      double z = (values[i].ToDouble() - mean) / stdv;
      local_skew += z * z * z;
    }
  }
  double skew_sum = BlockSum(local_skew, ws.sum_buf);

  // Pass 4: sort once for order-dependent primitives.
  BlockBitonicSort(values, n);

  if (threadIdx.x == 0) {
    FeatureValue median = values[n / 2];

    FeatureValue mode = values[0];
    uint32_t max_run = 1;
    uint32_t curr_run = 1;
    uint32_t uniq = 1;
    double entropy = 0.0;
    for (uint32_t i = 1; i < n; ++i) {
      if (values[i].Compare(values[i - 1]) == 0) {
        ++curr_run;
      } else {
        double p = static_cast<double>(curr_run) / static_cast<double>(n);
        entropy -= p * log2(p);
        ++uniq;
        curr_run = 1;
      }
      if (curr_run > max_run) {
        max_run = curr_run;
        mode = values[i];
      }
    }
    double p = static_cast<double>(curr_run) / static_cast<double>(n);
    entropy -= p * log2(p);

    FeatureValue q1;
    FeatureValue q3;
    if (n == 1) {
      q1 = values[0];
      q3 = values[0];
    } else {
      double pos1 = 0.25 * static_cast<double>(n - 1);
      uint32_t l1 = static_cast<uint32_t>(floor(pos1));
      uint32_t u1 = static_cast<uint32_t>(ceil(pos1));
      double w1 = pos1 - static_cast<double>(l1);
      double v1 = values[l1].ToDouble() * (1.0 - w1) +
                  values[u1].ToDouble() * w1;
      q1 = MakeFloat64Value(v1);

      double pos3 = 0.75 * static_cast<double>(n - 1);
      uint32_t l3 = static_cast<uint32_t>(floor(pos3));
      uint32_t u3 = static_cast<uint32_t>(ceil(pos3));
      double w3 = pos3 - static_cast<double>(l3);
      double v3 = values[l3].ToDouble() * (1.0 - w3) +
                  values[u3].ToDouble() * w3;
      q3 = MakeFloat64Value(v3);
    }

    r.count = MakeIntValue(static_cast<int64_t>(n));
    r.count_greater_than_mean = MakeIntValue(static_cast<int64_t>(gtm_total));
    r.num_unique = MakeIntValue(static_cast<int64_t>(uniq));
    r.sum = MakeFloat64Value(sum);
    r.mean = MakeFloat64Value(mean);
    r.variance = MakeFloat64Value(variance);
    r.std = MakeFloat64Value(stdv);
    r.mode = mode;
    r.min = minv;
    r.max = maxv;
    r.median = median;
    r.quarter = q1;
    r.quartile3 = q3;
    r.entropy = MakeFloat64Value(entropy);
    r.percent_true = MakeFloat64Value(static_cast<double>(true_total) /
                                    static_cast<double>(n));
    r.skew = skew_valid ? MakeFloat64Value(skew_sum / static_cast<double>(n))
                        : MakeInvalidValue();
  }
  __syncthreads();
  return r;
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
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
