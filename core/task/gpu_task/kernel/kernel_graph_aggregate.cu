#include "core/task/gpu_task/kernel/kernel_graph_aggregate.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

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

// ---------------------------------------------------------------------------
// Device sort (insertion sort, suitable for small n)
// ---------------------------------------------------------------------------
__device__ inline void DeviceSortValues(FeatureValue* arr, uint32_t n) {
  for (uint32_t i = 1; i < n; ++i) {
    FeatureValue key = arr[i];
    int j = static_cast<int>(i) - 1;
    while (j >= 0 && arr[j].Compare(key) > 0) {
      arr[j + 1] = arr[j];
      --j;
    }
    arr[j + 1] = key;
  }
}

// ---------------------------------------------------------------------------
// Aggregation primitives
// ---------------------------------------------------------------------------
__device__ inline FeatureValue AggCount(const FeatureValue* values, uint32_t n) {
  (void)values;
  return MakeIntValue(static_cast<int64_t>(n));
}

__device__ inline FeatureValue AggSum(const FeatureValue* values, uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  double sum = 0.0;
  for (uint32_t i = 0; i < n; ++i) sum += values[i].ToDouble();
  return MakeFloatValue(sum);
}

__device__ inline FeatureValue AggMean(const FeatureValue* values, uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  double sum = 0.0;
  for (uint32_t i = 0; i < n; ++i) sum += values[i].ToDouble();
  return MakeFloatValue(sum / static_cast<double>(n));
}

__device__ inline FeatureValue AggMin(FeatureValue* values, uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  FeatureValue minv = values[0];
  for (uint32_t i = 1; i < n; ++i) {
    if (values[i].Compare(minv) < 0) minv = values[i];
  }
  return minv;
}

__device__ inline FeatureValue AggMax(FeatureValue* values, uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  FeatureValue maxv = values[0];
  for (uint32_t i = 1; i < n; ++i) {
    if (values[i].Compare(maxv) > 0) maxv = values[i];
  }
  return maxv;
}

__device__ inline FeatureValue AggVariance(FeatureValue* values, uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  FeatureValue mean = AggMean(values, n);
  double m = mean.ToDouble();
  double sum = 0.0;
  for (uint32_t i = 0; i < n; ++i) {
    double diff = values[i].ToDouble() - m;
    sum += diff * diff;
  }
  return MakeFloatValue(sum / static_cast<double>(n));
}

__device__ inline FeatureValue AggStd(FeatureValue* values, uint32_t n) {
  FeatureValue var = AggVariance(values, n);
  return MakeFloatValue(sqrt(var.ToDouble()));
}

__device__ inline FeatureValue AggMedian(FeatureValue* values, uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  DeviceSortValues(values, n);
  return values[n / 2];
}

__device__ inline FeatureValue AggMode(FeatureValue* values, uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  DeviceSortValues(values, n);
  uint32_t max_count = 1;
  uint32_t curr_count = 1;
  FeatureValue mode = values[0];
  for (uint32_t i = 1; i < n; ++i) {
    if (values[i].Compare(values[i - 1]) == 0) {
      ++curr_count;
    } else {
      curr_count = 1;
    }
    if (curr_count > max_count) {
      max_count = curr_count;
      mode = values[i];
    }
  }
  return mode;
}

__device__ inline FeatureValue AggNumUnique(FeatureValue* values, uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  DeviceSortValues(values, n);
  uint32_t uniq = 1;
  for (uint32_t i = 1; i < n; ++i) {
    if (values[i].Compare(values[i - 1]) != 0) ++uniq;
  }
  return MakeIntValue(static_cast<int64_t>(uniq));
}

__device__ inline FeatureValue AggEntropy(FeatureValue* values, uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  DeviceSortValues(values, n);
  double entropy = 0.0;
  uint32_t i = 0;
  while (i < n) {
    uint32_t j = i + 1;
    while (j < n && values[j].Compare(values[i]) == 0) ++j;
    double p = static_cast<double>(j - i) / static_cast<double>(n);
    entropy -= p * log2(p);
    i = j;
  }
  return MakeFloatValue(entropy);
}

__device__ inline FeatureValue AggQuarter(FeatureValue* values, uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  if (n == 1) return values[0];
  DeviceSortValues(values, n);
  double pos = 0.25 * static_cast<double>(n - 1);
  uint32_t lower = static_cast<uint32_t>(floor(pos));
  uint32_t upper = static_cast<uint32_t>(ceil(pos));
  double w = pos - static_cast<double>(lower);
  double v = values[lower].ToDouble() * (1.0 - w) + values[upper].ToDouble() * w;
  return MakeFloatValue(v);
}

__device__ inline FeatureValue AggQuartile3(FeatureValue* values, uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  if (n == 1) return values[0];
  DeviceSortValues(values, n);
  double pos = 0.75 * static_cast<double>(n - 1);
  uint32_t lower = static_cast<uint32_t>(floor(pos));
  uint32_t upper = static_cast<uint32_t>(ceil(pos));
  double w = pos - static_cast<double>(lower);
  double v = values[lower].ToDouble() * (1.0 - w) + values[upper].ToDouble() * w;
  return MakeFloatValue(v);
}

__device__ inline FeatureValue AggPercentTrue(const FeatureValue* values, uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  uint32_t cnt = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if (values[i].type == ValueType::kBool && values[i].b) ++cnt;
  }
  return MakeFloatValue(static_cast<double>(cnt) / static_cast<double>(n));
}

__device__ inline FeatureValue AggSkew(FeatureValue* values, uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  FeatureValue mean = AggMean(values, n);
  FeatureValue stdv = AggStd(values, n);
  double m = mean.ToDouble();
  double s = stdv.ToDouble();
  if (s == 0.0) return MakeInvalidValue();
  double sum = 0.0;
  for (uint32_t i = 0; i < n; ++i) {
    double z = (values[i].ToDouble() - m) / s;
    sum += z * z * z;
  }
  return MakeFloatValue(sum / static_cast<double>(n));
}

__device__ inline FeatureValue AggCountGreaterThanMean(FeatureValue* values, uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  FeatureValue mean = AggMean(values, n);
  double thr = mean.ToDouble();
  uint32_t cnt = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if (values[i].ToDouble() > thr) ++cnt;
  }
  return MakeIntValue(static_cast<int64_t>(cnt));
}

__device__ inline FeatureValue ApplyAggPrim(AggPrim prim, FeatureValue* values, uint32_t n) {
  switch (prim) {
    case AggPrim::kCount:                 return AggCount(values, n);
    case AggPrim::kSum:                   return AggSum(values, n);
    case AggPrim::kMean:                  return AggMean(values, n);
    case AggPrim::kMin:                   return AggMin(values, n);
    case AggPrim::kMax:                   return AggMax(values, n);
    case AggPrim::kVariance:              return AggVariance(values, n);
    case AggPrim::kStd:                   return AggStd(values, n);
    case AggPrim::kMedian:                return AggMedian(values, n);
    case AggPrim::kMode:                  return AggMode(values, n);
    case AggPrim::kNumUnique:             return AggNumUnique(values, n);
    case AggPrim::kEntropy:               return AggEntropy(values, n);
    case AggPrim::kQuarter:               return AggQuarter(values, n);
    case AggPrim::kQuartile3:             return AggQuartile3(values, n);
    case AggPrim::kPercentTrue:           return AggPercentTrue(values, n);
    case AggPrim::kSkew:                  return AggSkew(values, n);
    case AggPrim::kCountGreaterThanMean:  return AggCountGreaterThanMean(values, n);
    default:                              return MakeInvalidValue();
  }
}

// ---------------------------------------------------------------------------
// Fused aggregation: compute all primitives in one pass over the value list.
// Reuses the shared sort for all order-dependent primitives.
// The input buffer is reordered (sorted in-place) when n > 0.
// ---------------------------------------------------------------------------
__device__ inline AllFeatures ComputeAllFeaturesFromValues(FeatureValue* values,
                                                           uint32_t n) {
  AllFeatures r;
  if (n == 0) {
    FeatureValue invalid = MakeInvalidValue();
    r.count = MakeIntValue(0);            // Count on empty list is 0.
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

  // Pass 1: count, sum, min, max, percent_true in one traversal.
  double sum = 0.0;
  uint32_t true_count = 0;
  FeatureValue minv = values[0];
  FeatureValue maxv = values[0];
  for (uint32_t i = 0; i < n; ++i) {
    double x = values[i].ToDouble();
    sum += x;
    if (values[i].Compare(minv) < 0) minv = values[i];
    if (values[i].Compare(maxv) > 0) maxv = values[i];
    if (values[i].type == ValueType::kBool && values[i].b) ++true_count;
  }
  double mean = sum / static_cast<double>(n);

  // Pass 2: variance and count-greater-than-mean.
  double var_sum = 0.0;
  uint32_t count_gtm = 0;
  for (uint32_t i = 0; i < n; ++i) {
    double diff = values[i].ToDouble() - mean;
    var_sum += diff * diff;
    if (values[i].ToDouble() > mean) ++count_gtm;
  }
  double variance = var_sum / static_cast<double>(n);
  double stdv = sqrt(variance);

  // Pass 3: skew (requires mean and std).  Match AggSkew: undefined when
  // the standard deviation is zero.
  double skew_sum = 0.0;
  bool skew_valid = (stdv != 0.0);
  if (skew_valid) {
    for (uint32_t i = 0; i < n; ++i) {
      double z = (values[i].ToDouble() - mean) / stdv;
      skew_sum += z * z * z;
    }
  }

  // Pass 4: single sort for all order-dependent primitives.
  DeviceSortValues(values, n);

  // Median: upper median (values[n/2]), matching featurelib.
  FeatureValue median = values[n / 2];

  // Mode, num_unique, entropy from sorted run-length scan.
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
  // Account for the final run.
  double p = static_cast<double>(curr_run) / static_cast<double>(n);
  entropy -= p * log2(p);

  // Quartiles: linear interpolation at position 0.25*(n-1) and 0.75*(n-1).
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
    double v1 = values[l1].ToDouble() * (1.0 - w1) + values[u1].ToDouble() * w1;
    q1 = MakeFloatValue(v1);

    double pos3 = 0.75 * static_cast<double>(n - 1);
    uint32_t l3 = static_cast<uint32_t>(floor(pos3));
    uint32_t u3 = static_cast<uint32_t>(ceil(pos3));
    double w3 = pos3 - static_cast<double>(l3);
    double v3 = values[l3].ToDouble() * (1.0 - w3) + values[u3].ToDouble() * w3;
    q3 = MakeFloatValue(v3);
  }

  r.count = MakeIntValue(static_cast<int64_t>(n));
  r.count_greater_than_mean = MakeIntValue(static_cast<int64_t>(count_gtm));
  r.num_unique = MakeIntValue(static_cast<int64_t>(uniq));
  r.sum = MakeFloatValue(sum);
  r.mean = MakeFloatValue(mean);
  r.variance = MakeFloatValue(variance);
  r.std = MakeFloatValue(stdv);
  r.mode = mode;
  r.min = minv;
  r.max = maxv;
  r.median = median;
  r.quarter = q1;
  r.quartile3 = q3;
  r.entropy = MakeFloatValue(entropy);
  r.percent_true = MakeFloatValue(static_cast<double>(true_count) /
                                  static_cast<double>(n));
  r.skew = skew_valid ? MakeFloatValue(skew_sum / static_cast<double>(n))
                      : MakeInvalidValue();
  return r;
}

// ---------------------------------------------------------------------------
// Collect neighbor attribute values for a pivot
// ---------------------------------------------------------------------------
__device__ inline uint32_t CollectNeighborValues(
    const uint8_t* graph_data,
    uint32_t n_vertices,
    uint32_t n_in_edges,
    uint32_t n_out_edges,
    const Attributes* vertex_attrs,
    uint32_t pivot_vid,
    const FeatureRequest& req,
    FeatureValue* out_buffer,
    uint32_t max_n) {
  // Parse CSR layout (matches ImmutableCSR::ParseBasePtr)
  const VertexID* globalid = reinterpret_cast<const VertexID*>(graph_data);
  const VertexID* indegree = globalid + n_vertices;
  const VertexID* outdegree = indegree + n_vertices;
  const EdgeIndex* in_offset = reinterpret_cast<const EdgeIndex*>(outdegree + n_vertices);
  const EdgeIndex* out_offset = in_offset + n_vertices + 1;
  const VertexID* incoming_edges = reinterpret_cast<const VertexID*>(out_offset + n_vertices + 1);
  const VertexID* outgoing_edges = incoming_edges + n_in_edges;

  uint32_t deg = req.use_outgoing ? outdegree[pivot_vid] : indegree[pivot_vid];
  const VertexID* edges = req.use_outgoing
      ? outgoing_edges + out_offset[pivot_vid]
      : incoming_edges + in_offset[pivot_vid];

  uint32_t count = 0;
  for (uint32_t i = 0; i < deg && count < max_n; ++i) {
    VertexID neighbor = edges[i];

    // TODO: edge_label filter (requires edge label array in CSR)
    // TODO: FilterExecutor (attribute condition filtering)

    const Attribute* attr = vertex_attrs[neighbor].attr_map.find(req.attr_name);
    if (!attr) continue;

    FeatureValue v;
    v.type = attr->type;
    switch (attr->type) {
      case ValueType::kInt:
      case ValueType::kTime:
        v.i64 = GetInt(*attr, 0);
        break;
      case ValueType::kFloat64:
        v.f64 = GetFloat64(*attr, 0);
        break;
      case ValueType::kFloat32:
        v.f64 = static_cast<double>(*reinterpret_cast<const float*>(attr->data));
        break;
      case ValueType::kBool:
        v.b = GetBool(*attr, 0);
        break;
      default:
        v.type = ValueType::kInvalid;
        break;
    }
    out_buffer[count++] = v;
  }
  return count;
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__global__ void ComputeFeaturesKernel(
    const uint8_t* const* graph_data_buffers,
    const uint32_t* graph_n_vertices,
    const uint32_t* graph_n_in_edges,
    const uint32_t* graph_n_out_edges,
    const Attributes* const* vertex_attrs,
    const uint32_t* pivot_graph_id,
    const uint32_t* pivot_vertex_id,
    uint32_t n_pivots,
    const FeatureRequest* requests,
    uint32_t n_requests,
    FeatureValue* d_workspace,
    uint32_t max_neighbors,
    FeatureValue* d_outputs) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_pivots) return;

  uint32_t gid = pivot_graph_id[tid];
  uint32_t vid = pivot_vertex_id[tid];

  const uint8_t* graph_data = graph_data_buffers[gid];
  uint32_t n_vertices = graph_n_vertices[gid];
  const Attributes* vattrs = vertex_attrs[gid];

  FeatureValue* my_workspace = d_workspace + tid * max_neighbors;

  for (uint32_t req_idx = 0; req_idx < n_requests; ++req_idx) {
    uint32_t n_collected = CollectNeighborValues(
        graph_data, n_vertices,
        graph_n_in_edges[gid], graph_n_out_edges[gid],
        vattrs, vid,
        requests[req_idx], my_workspace, max_neighbors);

    FeatureValue result = ApplyAggPrim(
        requests[req_idx].prim, my_workspace, n_collected);

    d_outputs[tid * n_requests + req_idx] = result;
  }
}

// ---------------------------------------------------------------------------
// Fused kernel: compute all aggregation primitives in one launch.
// ---------------------------------------------------------------------------
__global__ void ComputeAllFeaturesKernel(
    const uint8_t* const* graph_data_buffers,
    const uint32_t* graph_n_vertices,
    const uint32_t* graph_n_in_edges,
    const uint32_t* graph_n_out_edges,
    const Attributes* const* vertex_attrs,
    const uint32_t* pivot_graph_id,
    const uint32_t* pivot_vertex_id,
    uint32_t n_pivots,
    AttributeName attr_name,
    bool use_outgoing,
    FeatureValue* d_workspace,
    uint32_t max_neighbors,
    AllFeatures* d_outputs) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_pivots) return;

  uint32_t gid = pivot_graph_id[tid];
  uint32_t vid = pivot_vertex_id[tid];

  const uint8_t* graph_data = graph_data_buffers[gid];
  uint32_t n_vertices = graph_n_vertices[gid];
  const Attributes* vattrs = vertex_attrs[gid];

  FeatureValue* my_workspace = d_workspace + tid * max_neighbors;

  FeatureRequest req;
  req.attr_name = attr_name;
  req.neighbor_label = 0;
  req.use_outgoing = use_outgoing;

  uint32_t n_collected = CollectNeighborValues(
      graph_data, n_vertices,
      graph_n_in_edges[gid], graph_n_out_edges[gid],
      vattrs, vid, req, my_workspace, max_neighbors);

  d_outputs[tid] = ComputeAllFeaturesFromValues(my_workspace, n_collected);
}

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
