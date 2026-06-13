#include "core/task/gpu_task/kernel/kernel_graph_aggregate.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cfloat>

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
// Block-level reductions / scans (assumes blockDim.x <= 256)
// ---------------------------------------------------------------------------
__device__ inline double BlockSum(double val) {
  __shared__ double sdata[256];
  sdata[threadIdx.x] = val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  return sdata[0];
}

__device__ inline FeatureValue BlockMin(FeatureValue val) {
  __shared__ FeatureValue sdata[256];
  sdata[threadIdx.x] = val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      if (sdata[threadIdx.x + s].Compare(sdata[threadIdx.x]) < 0)
        sdata[threadIdx.x] = sdata[threadIdx.x + s];
    }
    __syncthreads();
  }
  return sdata[0];
}

__device__ inline FeatureValue BlockMax(FeatureValue val) {
  __shared__ FeatureValue sdata[256];
  sdata[threadIdx.x] = val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      if (sdata[threadIdx.x + s].Compare(sdata[threadIdx.x]) > 0)
        sdata[threadIdx.x] = sdata[threadIdx.x + s];
    }
    __syncthreads();
  }
  return sdata[0];
}

// Exclusive scan of a per-thread uint32_t value.  sdata must have at least
// blockDim.x entries.  The total sum is returned via *total.
__device__ inline uint32_t BlockExclusiveScan(uint32_t val,
                                              uint32_t* sdata,
                                              uint32_t* total) {
  sdata[threadIdx.x] = val;
  __syncthreads();
  for (int offset = 1; offset < blockDim.x; offset <<= 1) {
    uint32_t t = (threadIdx.x >= offset) ? sdata[threadIdx.x - offset] : 0;
    __syncthreads();
    sdata[threadIdx.x] += t;
    __syncthreads();
  }
  if (total) *total = sdata[blockDim.x - 1];
  uint32_t result = (threadIdx.x == 0) ? 0 : sdata[threadIdx.x - 1];
  __syncthreads();
  return result;
}

// ---------------------------------------------------------------------------
// Block-level bitonic sort in shared memory.  Sorts the first n valid entries;
// pads to the next power of two with +DBL_MAX so padding ends up at the tail.
// ---------------------------------------------------------------------------
__device__ inline void BlockBitonicSort(FeatureValue* s, uint32_t n) {
  // For very small arrays a serial insertion sort by one thread is cheaper
  // than paying the __syncthreads overhead of a block-wide bitonic network.
  constexpr uint32_t kSerialSortThreshold = 64;
  if (n <= kSerialSortThreshold) {
    if (threadIdx.x == 0) {
      for (uint32_t i = 1; i < n; ++i) {
        FeatureValue key = s[i];
        int j = static_cast<int>(i) - 1;
        while (j >= 0 && s[j].Compare(key) > 0) {
          s[j + 1] = s[j];
          --j;
        }
        s[j + 1] = key;
      }
    }
    __syncthreads();
    return;
  }

  uint32_t N = 1;
  while (N < n) N <<= 1;

  // Initialize padding to +inf.
  for (uint32_t i = threadIdx.x + n; i < N; i += blockDim.x) {
    s[i].type = ValueType::kFloat64;
    s[i].f64 = DBL_MAX;
  }
  __syncthreads();

  for (uint32_t k = 2; k <= N; k <<= 1) {
    for (uint32_t j = k >> 1; j > 0; j >>= 1) {
      for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) {
        uint32_t partner = i ^ j;
        if (partner > i) {
          bool up = (i & k) == 0;
          int cmp = s[i].Compare(s[partner]);
          bool swap = up ? (cmp > 0) : (cmp < 0);
          if (swap) {
            FeatureValue tmp = s[i];
            s[i] = s[partner];
            s[partner] = tmp;
          }
        }
      }
      __syncthreads();
    }
  }
}

// ---------------------------------------------------------------------------
// Attribute access helpers
// ---------------------------------------------------------------------------
__device__ inline bool ReadAttributeValue(const Attribute* attr,
                                          FeatureValue* out) {
  if (!attr) return false;
  out->type = attr->type;
  switch (attr->type) {
    case ValueType::kInt:
    case ValueType::kTime:
      out->i64 = GetInt(*attr, 0);
      break;
    case ValueType::kFloat64:
      out->f64 = GetFloat64(*attr, 0);
      break;
    case ValueType::kFloat32:
      out->f64 = static_cast<double>(*reinterpret_cast<const float*>(attr->data));
      break;
    case ValueType::kBool:
      out->b = GetBool(*attr, 0);
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
    uint32_t pivot_vid,
    const AttributeName& attr_name,
    bool use_outgoing,
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

  // Single-pass cooperative collection with a shared atomic counter.
  if (threadIdx.x == 0) scratch[0] = 0;
  __syncthreads();

  for (uint32_t i = threadIdx.x; i < deg; i += blockDim.x) {
    VertexID neighbor = edges[i];
    const Attribute* attr = vertex_attrs[neighbor].attr_map.find(attr_name);
    if (!attr) continue;
    FeatureValue v;
    if (!ReadAttributeValue(attr, &v)) continue;
    uint32_t pos = atomicAdd(&scratch[0], 1);
    if (pos < max_neighbors) shared_buf[pos] = v;
  }
  __syncthreads();

  uint32_t total_count = scratch[0];
  return min(total_count, max_neighbors);
}

// ---------------------------------------------------------------------------
// Block-level aggregation primitives
// ---------------------------------------------------------------------------
__device__ inline FeatureValue BlockAggCount(uint32_t n) {
  return MakeIntValue(static_cast<int64_t>(n));
}

__device__ inline FeatureValue BlockAggSum(const FeatureValue* values,
                                           uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  double local = 0.0;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    local += values[i].ToDouble();
  }
  return MakeFloatValue(BlockSum(local));
}

__device__ inline FeatureValue BlockAggMean(const FeatureValue* values,
                                            uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  FeatureValue sum = BlockAggSum(values, n);
  return MakeFloatValue(sum.ToDouble() / static_cast<double>(n));
}

__device__ inline FeatureValue BlockAggMin(const FeatureValue* values,
                                           uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  FeatureValue local = values[0];
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    if (values[i].Compare(local) < 0) local = values[i];
  }
  return BlockMin(local);
}

__device__ inline FeatureValue BlockAggMax(const FeatureValue* values,
                                           uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  FeatureValue local = values[0];
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    if (values[i].Compare(local) > 0) local = values[i];
  }
  return BlockMax(local);
}

__device__ inline FeatureValue BlockAggVariance(const FeatureValue* values,
                                                uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  double mean = BlockAggMean(values, n).ToDouble();
  double local = 0.0;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    double d = values[i].ToDouble() - mean;
    local += d * d;
  }
  return MakeFloatValue(BlockSum(local) / static_cast<double>(n));
}

__device__ inline FeatureValue BlockAggStd(const FeatureValue* values,
                                           uint32_t n) {
  FeatureValue var = BlockAggVariance(values, n);
  return MakeFloatValue(sqrt(var.ToDouble()));
}

__device__ inline FeatureValue BlockAggMedian(FeatureValue* values,
                                              uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  BlockBitonicSort(values, n);
  return values[n / 2];
}

__device__ inline FeatureValue BlockAggMode(FeatureValue* values,
                                            uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  BlockBitonicSort(values, n);
  if (threadIdx.x == 0) {
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
    // Stash result in values[0] so all threads can see it if needed.
    values[0] = mode;
  }
  __syncthreads();
  return values[0];
}

__device__ inline FeatureValue BlockAggNumUnique(FeatureValue* values,
                                                 uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  BlockBitonicSort(values, n);
  if (threadIdx.x == 0) {
    uint32_t uniq = 1;
    for (uint32_t i = 1; i < n; ++i) {
      if (values[i].Compare(values[i - 1]) != 0) ++uniq;
    }
    values[0].type = ValueType::kInt;
    values[0].i64 = static_cast<int64_t>(uniq);
  }
  __syncthreads();
  FeatureValue r = values[0];
  r.type = ValueType::kInt;
  return r;
}

__device__ inline FeatureValue BlockAggEntropy(FeatureValue* values,
                                               uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  BlockBitonicSort(values, n);
  if (threadIdx.x == 0) {
    double entropy = 0.0;
    uint32_t i = 0;
    while (i < n) {
      uint32_t j = i + 1;
      while (j < n && values[j].Compare(values[i]) == 0) ++j;
      double p = static_cast<double>(j - i) / static_cast<double>(n);
      entropy -= p * log2(p);
      i = j;
    }
    values[0] = MakeFloatValue(entropy);
  }
  __syncthreads();
  return values[0];
}

__device__ inline FeatureValue BlockAggQuarter(FeatureValue* values,
                                               uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  if (n == 1) return values[0];
  BlockBitonicSort(values, n);
  if (threadIdx.x == 0) {
    double pos = 0.25 * static_cast<double>(n - 1);
    uint32_t lower = static_cast<uint32_t>(floor(pos));
    uint32_t upper = static_cast<uint32_t>(ceil(pos));
    double w = pos - static_cast<double>(lower);
    double v = values[lower].ToDouble() * (1.0 - w) +
               values[upper].ToDouble() * w;
    values[0] = MakeFloatValue(v);
  }
  __syncthreads();
  return values[0];
}

__device__ inline FeatureValue BlockAggQuartile3(FeatureValue* values,
                                                 uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  if (n == 1) return values[0];
  BlockBitonicSort(values, n);
  if (threadIdx.x == 0) {
    double pos = 0.75 * static_cast<double>(n - 1);
    uint32_t lower = static_cast<uint32_t>(floor(pos));
    uint32_t upper = static_cast<uint32_t>(ceil(pos));
    double w = pos - static_cast<double>(lower);
    double v = values[lower].ToDouble() * (1.0 - w) +
               values[upper].ToDouble() * w;
    values[0] = MakeFloatValue(v);
  }
  __syncthreads();
  return values[0];
}

__device__ inline FeatureValue BlockAggPercentTrue(const FeatureValue* values,
                                                   uint32_t n,
                                                   uint32_t* scratch) {
  if (n == 0) return MakeInvalidValue();
  uint32_t local = 0;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    if (values[i].type == ValueType::kBool && values[i].b) ++local;
  }
  uint32_t total = 0;
  BlockExclusiveScan(local, scratch, &total);
  return MakeFloatValue(static_cast<double>(total) / static_cast<double>(n));
}

__device__ inline FeatureValue BlockAggSkew(const FeatureValue* values,
                                            uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  double mean = BlockAggMean(values, n).ToDouble();
  double stdv = BlockAggStd(values, n).ToDouble();
  if (stdv == 0.0) return MakeInvalidValue();
  double local = 0.0;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    double z = (values[i].ToDouble() - mean) / stdv;
    local += z * z * z;
  }
  return MakeFloatValue(BlockSum(local) / static_cast<double>(n));
}

__device__ inline FeatureValue BlockAggCountGreaterThanMean(FeatureValue* values,
                                                            uint32_t n,
                                                            uint32_t* scratch) {
  if (n == 0) return MakeInvalidValue();
  double mean = BlockAggMean(values, n).ToDouble();
  uint32_t local = 0;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    if (values[i].ToDouble() > mean) ++local;
  }
  uint32_t total = 0;
  BlockExclusiveScan(local, scratch, &total);
  return MakeIntValue(static_cast<int64_t>(total));
}

__device__ inline FeatureValue BlockApplyAggPrim(AggPrim prim,
                                                 FeatureValue* values,
                                                 uint32_t n,
                                                 uint32_t* scratch) {
  switch (prim) {
    case AggPrim::kCount:                 return BlockAggCount(n);
    case AggPrim::kSum:                   return BlockAggSum(values, n);
    case AggPrim::kMean:                  return BlockAggMean(values, n);
    case AggPrim::kMin:                   return BlockAggMin(values, n);
    case AggPrim::kMax:                   return BlockAggMax(values, n);
    case AggPrim::kVariance:              return BlockAggVariance(values, n);
    case AggPrim::kStd:                   return BlockAggStd(values, n);
    case AggPrim::kMedian:                return BlockAggMedian(values, n);
    case AggPrim::kMode:                  return BlockAggMode(values, n);
    case AggPrim::kNumUnique:             return BlockAggNumUnique(values, n);
    case AggPrim::kEntropy:               return BlockAggEntropy(values, n);
    case AggPrim::kQuarter:               return BlockAggQuarter(values, n);
    case AggPrim::kQuartile3:             return BlockAggQuartile3(values, n);
    case AggPrim::kPercentTrue:           return BlockAggPercentTrue(values, n, scratch);
    case AggPrim::kSkew:                  return BlockAggSkew(values, n);
    case AggPrim::kCountGreaterThanMean:  return BlockAggCountGreaterThanMean(values, n, scratch);
    default:                              return MakeInvalidValue();
  }
}

// ---------------------------------------------------------------------------
// Fused compute-all using block-level parallelism.
// ---------------------------------------------------------------------------
__device__ inline AllFeatures BlockComputeAllFeaturesFromValues(
    FeatureValue* values,
    uint32_t n,
    uint32_t* scratch) {
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
  double sum = BlockSum(local_sum);
  FeatureValue minv = BlockMin(local_min);
  FeatureValue maxv = BlockMax(local_max);
  uint32_t true_total = 0;
  BlockExclusiveScan(local_true, scratch, &true_total);
  double mean = sum / static_cast<double>(n);

  // Pass 2: variance and count-greater-than-mean.
  double local_var = 0.0;
  uint32_t local_gtm = 0;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    double diff = values[i].ToDouble() - mean;
    local_var += diff * diff;
    if (values[i].ToDouble() > mean) ++local_gtm;
  }
  double variance = BlockSum(local_var) / static_cast<double>(n);
  double stdv = sqrt(variance);
  uint32_t gtm_total = 0;
  BlockExclusiveScan(local_gtm, scratch, &gtm_total);

  // Pass 3: skew.
  double local_skew = 0.0;
  bool skew_valid = (stdv != 0.0);
  if (skew_valid) {
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
      double z = (values[i].ToDouble() - mean) / stdv;
      local_skew += z * z * z;
    }
  }
  double skew_sum = BlockSum(local_skew);

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
      q1 = MakeFloatValue(v1);

      double pos3 = 0.75 * static_cast<double>(n - 1);
      uint32_t l3 = static_cast<uint32_t>(floor(pos3));
      uint32_t u3 = static_cast<uint32_t>(ceil(pos3));
      double w3 = pos3 - static_cast<double>(l3);
      double v3 = values[l3].ToDouble() * (1.0 - w3) +
                  values[u3].ToDouble() * w3;
      q3 = MakeFloatValue(v3);
    }

    r.count = MakeIntValue(static_cast<int64_t>(n));
    r.count_greater_than_mean = MakeIntValue(static_cast<int64_t>(gtm_total));
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
    r.percent_true = MakeFloatValue(static_cast<double>(true_total) /
                                    static_cast<double>(n));
    r.skew = skew_valid ? MakeFloatValue(skew_sum / static_cast<double>(n))
                        : MakeInvalidValue();
  }
  __syncthreads();
  return r;
}

// ---------------------------------------------------------------------------
// Kernels: one block per pivot.
// ---------------------------------------------------------------------------
__global__ void ComputeFeaturesKernel(
    const uint8_t* graph_data,
    uint32_t n_vertices,
    uint32_t n_in_edges,
    uint32_t n_out_edges,
    const Attributes* vertex_attrs,
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

  uint32_t pivot_idx = blockIdx.x;
  if (pivot_idx >= n_pivots) return;

  uint32_t vid = pivot_vertex_ids[pivot_idx];

  for (uint32_t req_idx = 0; req_idx < n_requests; ++req_idx) {
    uint32_t n_collected = CollectNeighborValuesBlock(
        graph_data, n_vertices, n_in_edges, n_out_edges,
        vertex_attrs, vid,
        requests[req_idx].attr_name,
        requests[req_idx].use_outgoing,
        shared_buf, max_neighbors, scratch);

    __syncthreads();
    FeatureValue result = BlockApplyAggPrim(
        requests[req_idx].prim, shared_buf, n_collected, scratch);

    if (threadIdx.x == 0) {
      d_outputs[pivot_idx * n_requests + req_idx] = result;
    }
    __syncthreads();
  }
}

__global__ void ComputeAllFeaturesKernel(
    const uint8_t* graph_data,
    uint32_t n_vertices,
    uint32_t n_in_edges,
    uint32_t n_out_edges,
    const Attributes* vertex_attrs,
    const uint32_t* pivot_vertex_ids,
    uint32_t n_pivots,
    AttributeName attr_name,
    bool use_outgoing,
    uint32_t max_neighbors,
    AllFeatures* d_outputs) {
  extern __shared__ uint8_t shared_mem[];
  FeatureValue* shared_buf = reinterpret_cast<FeatureValue*>(shared_mem);
  uint32_t* scratch = reinterpret_cast<uint32_t*>(
      shared_mem + max_neighbors * sizeof(FeatureValue));

  uint32_t pivot_idx = blockIdx.x;
  if (pivot_idx >= n_pivots) return;

  uint32_t vid = pivot_vertex_ids[pivot_idx];

  FeatureRequest req;
  req.attr_name = attr_name;
  req.neighbor_label = 0;
  req.use_outgoing = use_outgoing;

  uint32_t n_collected = CollectNeighborValuesBlock(
      graph_data, n_vertices, n_in_edges, n_out_edges,
      vertex_attrs, vid,
      req.attr_name, req.use_outgoing,
      shared_buf, max_neighbors, scratch);

  __syncthreads();
  AllFeatures result = BlockComputeAllFeaturesFromValues(
      shared_buf, n_collected, scratch);

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
  constexpr uint32_t kBlockSize = 256;
  // FeatureValue buffer for collected neighbors + uint32 scratch for scans.
  return static_cast<size_t>(max_neighbors) * sizeof(FeatureValue) +
         static_cast<size_t>(kBlockSize) * sizeof(uint32_t);
}

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
