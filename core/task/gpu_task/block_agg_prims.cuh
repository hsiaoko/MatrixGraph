#ifndef MATRIXGRAPH_CORE_TASK_GPU_TASK_BLOCK_AGG_PRIMS_CUH_
#define MATRIXGRAPH_CORE_TASK_GPU_TASK_BLOCK_AGG_PRIMS_CUH_

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdint>

#include "core/common/consts.h"
#include "core/task/gpu_task/execute_agg_prim.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

// ---------------------------------------------------------------------------
// Block-level aggregation primitives and their shared-memory reduction
// helpers.  Shared by execute_agg_prim (single-list aggregation) and
// kernel_graph_aggregate (fused per-pivot aggregation over neighbor lists).
//
// These live in one header because the two consumers used to carry separate
// copies; that duplication let the copies drift apart (different value types /
// helper signatures), which produced an nvlink "prototype doesn't match" error
// at device-link time.
// ---------------------------------------------------------------------------

struct BlockWorkspace {
  double* sum_buf;
  FeatureValue* minmax_buf;
  uint32_t* scan_buf;
};

#ifdef __CUDACC__

__device__ inline BlockWorkspace GetBlockWorkspace(uint8_t* base,
                                                   uint32_t values_capacity) {
  double* sum_buf = reinterpret_cast<double*>(
      base + values_capacity * sizeof(FeatureValue));
  FeatureValue* minmax_buf = reinterpret_cast<FeatureValue*>(
      base + values_capacity * sizeof(FeatureValue) +
      common::kBlockDim * sizeof(double));
  uint32_t* scan_buf = reinterpret_cast<uint32_t*>(
      base + values_capacity * sizeof(FeatureValue) +
      common::kBlockDim * sizeof(double) +
      common::kBlockDim * sizeof(FeatureValue));
  return {sum_buf, minmax_buf, scan_buf};
}

__device__ __forceinline__ double BlockSum(double val, double* sdata) {
  sdata[threadIdx.x] = val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  return sdata[0];
}
// Warp-shuffle block sum, broadcast to all threads via s[0]. s must hold
// >= blockDim/32 doubles. Unlike BlockSum above (shared-memory tree reduce),
// this reduces within each warp first, so it needs only one slot per warp.
__device__ __forceinline__ double BlockSumAll(double v, double* s) {
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if (lane == 0) s[wid] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    int nwarps = (blockDim.x + 31) >> 5;
    double r = 0.0;
    for (int i = 0; i < nwarps; ++i) r += s[i];
    s[0] = r;
  }
  __syncthreads();
  double r = s[0];
  __syncthreads();
  return r;
}

__device__ __forceinline__ FeatureValue BlockMin(FeatureValue val, FeatureValue* sdata) {
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

__device__ __forceinline__ FeatureValue BlockMax(FeatureValue val, FeatureValue* sdata) {
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

__device__ __forceinline__ uint32_t BlockExclusiveScan(uint32_t val,
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

// -----------------------------------------------------------------------------
// Block-level bitonic sort
// -----------------------------------------------------------------------------
__device__ inline void BlockBitonicSort(FeatureValue* s, uint32_t n) {
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

// -----------------------------------------------------------------------------
// Aggregation primitives
// -----------------------------------------------------------------------------
__device__ inline FeatureValue BlockAggCount(uint32_t n) {
  return MakeIntValue(static_cast<int64_t>(n));
}

__device__ __forceinline__ FeatureValue BlockAggSum(const FeatureValue* values,
                                           uint32_t n,
                                           const BlockWorkspace& ws) {
  if (n == 0) return MakeInvalidValue();
  double local = 0.0;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    local += values[i].ToDouble();
  }
  return MakeFloat64Value(BlockSum(local, ws.sum_buf));
}

__device__ __forceinline__ FeatureValue BlockAggMean(const FeatureValue* values,
                                            uint32_t n,
                                            const BlockWorkspace& ws) {
  if (n == 0) return MakeInvalidValue();
  FeatureValue sum = BlockAggSum(values, n, ws);
  return MakeFloat64Value(sum.ToDouble() / static_cast<double>(n));
}

__device__ __forceinline__ FeatureValue BlockAggMin(const FeatureValue* values,
                                           uint32_t n,
                                           const BlockWorkspace& ws) {
  if (n == 0) return MakeInvalidValue();
  FeatureValue local = values[0];
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    if (values[i].Compare(local) < 0) local = values[i];
  }
  return BlockMin(local, ws.minmax_buf);
}

__device__ __forceinline__ FeatureValue BlockAggMax(const FeatureValue* values,
                                           uint32_t n,
                                           const BlockWorkspace& ws) {
  if (n == 0) return MakeInvalidValue();
  FeatureValue local = values[0];
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    if (values[i].Compare(local) > 0) local = values[i];
  }
  return BlockMax(local, ws.minmax_buf);
}

__device__ __forceinline__ FeatureValue BlockAggVariance(const FeatureValue* values,
                                                uint32_t n,
                                                const BlockWorkspace& ws) {
  if (n == 0) return MakeInvalidValue();
  double mean = BlockAggMean(values, n, ws).ToDouble();
  double local = 0.0;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    double d = values[i].ToDouble() - mean;
    local += d * d;
  }
  return MakeFloat64Value(BlockSum(local, ws.sum_buf) / static_cast<double>(n));
}

__device__ __forceinline__ FeatureValue BlockAggStd(const FeatureValue* values,
                                           uint32_t n,
                                           const BlockWorkspace& ws) {
  FeatureValue var = BlockAggVariance(values, n, ws);
  if (!var.IsValid()) return var;
  return MakeFloat64Value(sqrt(var.ToDouble()));
}

__device__ inline FeatureValue BlockAggMedian(FeatureValue* values, uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  BlockBitonicSort(values, n);
  return values[n / 2];
}

__device__ inline FeatureValue BlockAggMode(FeatureValue* values, uint32_t n) {
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
      // Use > to match featurelib's "first record-breaking" semantics.
      if (curr_count > max_count) {
        max_count = curr_count;
        mode = values[i];
      }
    }
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
    values[0] = MakeIntValue(static_cast<int64_t>(uniq));
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
    values[0] = MakeFloat64Value(entropy);
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
    values[0] = MakeFloat64Value(v);
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
    values[0] = MakeFloat64Value(v);
  }
  __syncthreads();
  return values[0];
}

__device__ __forceinline__ FeatureValue BlockAggPercentTrue(const FeatureValue* values,
                                                   uint32_t n,
                                                   const BlockWorkspace& ws) {
  if (n == 0) return MakeInvalidValue();
  uint32_t local = 0;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    if (values[i].type == ValueType::kBool && values[i].b) ++local;
  }
  uint32_t total = 0;
  BlockExclusiveScan(local, ws.scan_buf, &total);
  return MakeFloat64Value(static_cast<double>(total) / static_cast<double>(n));
}

__device__ __forceinline__ FeatureValue BlockAggSkew(const FeatureValue* values,
                                            uint32_t n,
                                            const BlockWorkspace& ws) {
  if (n == 0) return MakeInvalidValue();
  double mean = BlockAggMean(values, n, ws).ToDouble();
  double stdv = BlockAggStd(values, n, ws).ToDouble();
  if (stdv == 0.0) return MakeInvalidValue();
  double local = 0.0;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    double z = (values[i].ToDouble() - mean) / stdv;
    local += z * z * z;
  }
  return MakeFloat64Value(BlockSum(local, ws.sum_buf) / static_cast<double>(n));
}

__device__ __forceinline__ FeatureValue BlockAggCountGreaterThanMean(
    const FeatureValue* values, uint32_t n, const BlockWorkspace& ws) {
  if (n == 0) return MakeInvalidValue();
  double mean = BlockAggMean(values, n, ws).ToDouble();
  uint32_t local = 0;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    if (values[i].ToDouble() > mean) ++local;
  }
  uint32_t total = 0;
  BlockExclusiveScan(local, ws.scan_buf, &total);
  return MakeIntValue(static_cast<int64_t>(total));
}

__device__ inline FeatureValue BlockAggDFeat(const FeatureValue* values,
                                             uint32_t n) {
  if (n != 1) return MakeInvalidValue();
  return values[0];
}


__device__ __forceinline__ FeatureValue BlockApplyAggPrim(AggPrim prim,
                                                 FeatureValue* values,
                                                 uint32_t n,
                                                 const BlockWorkspace& ws) {
  switch (prim) {
    case AggPrim::kCount:
      return BlockAggCount(n);
    case AggPrim::kSum:
      return BlockAggSum(values, n, ws);
    case AggPrim::kMean:
      return BlockAggMean(values, n, ws);
    case AggPrim::kMedian:
      return BlockAggMedian(values, n);
    case AggPrim::kMode:
      return BlockAggMode(values, n);
    case AggPrim::kMax:
      return BlockAggMax(values, n, ws);
    case AggPrim::kMin:
      return BlockAggMin(values, n, ws);
    case AggPrim::kVariance:
      return BlockAggVariance(values, n, ws);
    case AggPrim::kStd:
      return BlockAggStd(values, n, ws);
    case AggPrim::kSkew:
      return BlockAggSkew(values, n, ws);
    case AggPrim::kEntropy:
      return BlockAggEntropy(values, n);
    case AggPrim::kNumUnique:
      return BlockAggNumUnique(values, n);
    case AggPrim::kPercentTrue:
      return BlockAggPercentTrue(values, n, ws);
    case AggPrim::kQuarter:
      return BlockAggQuarter(values, n);
    case AggPrim::kQuartile3:
      return BlockAggQuartile3(values, n);
    case AggPrim::kCountGreaterThanMean:
      return BlockAggCountGreaterThanMean(values, n, ws);
    case AggPrim::kDFeat:
      return BlockAggDFeat(values, n);
    default:
      return MakeInvalidValue();
  }
}

// -----------------------------------------------------------------------------
// Fused compute-all
// -----------------------------------------------------------------------------
__device__ __forceinline__ AllFeatures BlockComputeAllFeaturesFromValues(
    FeatureValue* values, uint32_t n, const BlockWorkspace& ws) {
  AllFeatures r;
  FeatureValue invalid = MakeInvalidValue();
  r.count = MakeIntValue(static_cast<int64_t>(n));
  r.sum = invalid;
  r.mean = invalid;
  r.variance = invalid;
  r.std = invalid;
  r.skew = invalid;
  r.count_greater_than_mean = invalid;
  r.percent_true = invalid;
  r.min = invalid;
  r.max = invalid;
  r.median = invalid;
  r.mode = invalid;
  r.num_unique = invalid;
  r.entropy = invalid;
  r.quarter = invalid;
  r.quartile3 = invalid;
  r.dfeat = (n == 1) ? values[0] : invalid;

  if (n == 0) {
    r.count = MakeIntValue(0);
    return r;
  }

  // Pass 1: sum, min, max, percent_true.
  double local_sum = 0.0;
  uint32_t local_true = 0;
  FeatureValue local_min = values[0];
  FeatureValue local_max = values[0];
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    local_sum += values[i].ToDouble();
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

// -----------------------------------------------------------------------------

#endif  // __CUDACC__

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_GPU_TASK_BLOCK_AGG_PRIMS_CUH_
