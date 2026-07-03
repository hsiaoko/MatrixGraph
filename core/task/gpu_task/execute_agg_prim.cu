#include "core/task/gpu_task/execute_agg_prim.cuh"

#include <cmath>
#include <cfloat>
#include <cstring>
#include <iostream>

#include "core/common/consts.h"
#include "core/util/cuda_check.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

// -----------------------------------------------------------------------------
// FeatureValue helpers
// -----------------------------------------------------------------------------
__host__ __device__ double FeatureValue::ToDouble() const {
  switch (type) {
    case ValueType::kInt:
    case ValueType::kTime:
      return static_cast<double>(i64);
    case ValueType::kFloat64:
      return f64;
    case ValueType::kFloat32:
      return static_cast<double>(f32);
    case ValueType::kBool:
      return b ? 1.0 : 0.0;
    default:
      return 0.0;
  }
}

__host__ __device__ int FeatureValue::Compare(const FeatureValue& other) const {
  // Both invalid -> equal.
  if (type == ValueType::kInvalid && other.type == ValueType::kInvalid)
    return 0;
  // Invalid sorts after valid.
  if (type == ValueType::kInvalid) return 1;
  if (other.type == ValueType::kInvalid) return -1;

  // String comparison.
  if (type == ValueType::kString || other.type == ValueType::kString) {
    if (type != ValueType::kString || other.type != ValueType::kString) {
      // Mixed string/non-string: order by type ordinal for deterministic sort.
      return static_cast<int>(type) < static_cast<int>(other.type) ? -1 : 1;
    }
    uint32_t len = str.len < other.str.len ? str.len : other.str.len;
    for (uint32_t i = 0; i < len; ++i) {
      if (str.data[i] < other.str.data[i]) return -1;
      if (str.data[i] > other.str.data[i]) return 1;
    }
    if (str.len < other.str.len) return -1;
    if (str.len > other.str.len) return 1;
    return 0;
  }

  // Numeric / bool / time compare via double.
  double a = ToDouble();
  double d = other.ToDouble();
  if (a < d) return -1;
  if (a > d) return 1;

  // For full deterministic order, break ties by type ordinal.
  if (type != other.type) {
    return static_cast<int>(type) < static_cast<int>(other.type) ? -1 : 1;
  }
  return 0;
}

__host__ __device__ uint32_t NextPowerOfTwo(uint32_t v) {
  if (v == 0) return 1;
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

namespace kernel {

namespace {

struct BlockWorkspace {
  double* sum_buf;
  FeatureValue* minmax_buf;
  uint32_t* scan_buf;
};

__device__ inline BlockWorkspace GetBlockWorkspace(uint8_t* base, uint32_t n) {
  uint32_t padded_n = NextPowerOfTwo(n);
  FeatureValue* values_buf = reinterpret_cast<FeatureValue*>(base);
  (void)values_buf;
  double* sum_buf = reinterpret_cast<double*>(base + padded_n * sizeof(FeatureValue));
  FeatureValue* minmax_buf = reinterpret_cast<FeatureValue*>(
      base + padded_n * sizeof(FeatureValue) + common::kBlockDim * sizeof(double));
  uint32_t* scan_buf = reinterpret_cast<uint32_t*>(
      base + padded_n * sizeof(FeatureValue) + common::kBlockDim * sizeof(double) +
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

} // namespace

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
// Kernels
// -----------------------------------------------------------------------------
__global__ void ComputeAggPrimKernel(AggPrim prim,
                                     const FeatureValue* d_values,
                                     const uint32_t* d_offsets,
                                     uint32_t list_offset,
                                     uint32_t total_n_lists,
                                     FeatureValue* d_outputs) {
  extern __shared__ uint8_t shared_mem[];
  FeatureValue* shared_buf = reinterpret_cast<FeatureValue*>(shared_mem);

  uint32_t list_idx = blockIdx.x + list_offset;
  if (list_idx >= total_n_lists) return;

  uint32_t begin = d_offsets[list_idx];
  uint32_t end = d_offsets[list_idx + 1];
  uint32_t n = end - begin;

  BlockWorkspace ws = GetBlockWorkspace(shared_mem, n);

  // Load values into shared memory cooperatively.
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    shared_buf[i] = d_values[begin + i];
  }
  __syncthreads();

  FeatureValue result = BlockApplyAggPrim(prim, shared_buf, n, ws);

  if (threadIdx.x == 0) {
    d_outputs[list_idx] = result;
  }
}

__global__ void ComputeBatchMultiPrimKernel(const FeatureValue* d_values,
                                            const uint32_t* d_offsets,
                                            const AggPrim* d_prims,
                                            uint32_t n_prims,
                                            uint32_t list_offset,
                                            uint32_t total_n_lists,
                                            FeatureValue* d_outputs) {
  extern __shared__ uint8_t shared_mem[];
  FeatureValue* shared_buf = reinterpret_cast<FeatureValue*>(shared_mem);

  uint32_t list_idx = blockIdx.x + list_offset;
  if (list_idx >= total_n_lists) return;

  uint32_t begin = d_offsets[list_idx];
  uint32_t end = d_offsets[list_idx + 1];
  uint32_t n = end - begin;

  BlockWorkspace ws = GetBlockWorkspace(shared_mem, n);

  // Load values into shared memory once for all primitives.
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    shared_buf[i] = d_values[begin + i];
  }
  __syncthreads();

  FeatureValue* out_row = d_outputs + list_idx * n_prims;
  for (uint32_t p = 0; p < n_prims; ++p) {
    // Synchronize before reusing shared_buf for order-dependent primitives.
    __syncthreads();
    FeatureValue result = BlockApplyAggPrim(d_prims[p], shared_buf, n, ws);
    if (threadIdx.x == 0) {
      out_row[p] = result;
    }
  }
}

__launch_bounds__(common::kAllFeaturesBlockDim)
__global__ void ComputeAllAggPrimsKernel(const FeatureValue* d_values,
                                         const uint32_t* d_offsets,
                                         uint32_t list_offset,
                                         uint32_t total_n_lists,
                                         AllFeatures* d_outputs) {
  extern __shared__ uint8_t shared_mem[];
  FeatureValue* shared_buf = reinterpret_cast<FeatureValue*>(shared_mem);

  uint32_t list_idx = blockIdx.x + list_offset;
  if (list_idx >= total_n_lists) return;

  uint32_t begin = d_offsets[list_idx];
  uint32_t end = d_offsets[list_idx + 1];
  uint32_t n = end - begin;

  BlockWorkspace ws = GetBlockWorkspace(shared_mem, n);

  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    shared_buf[i] = d_values[begin + i];
  }
  __syncthreads();

  AllFeatures result = BlockComputeAllFeaturesFromValues(shared_buf, n, ws);

  if (threadIdx.x == 0) {
    d_outputs[list_idx] = result;
  }
}

// -----------------------------------------------------------------------------
// Streaming (shared-free) path: warp-shuffle reductions over global memory.
// -----------------------------------------------------------------------------
namespace {

// Block sum, broadcast to all threads via s[0]. s must hold >= blockDim/32.
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

// Block argmin/argmax over (key, index); returns winning index to all threads.
__device__ __forceinline__ int BlockArgReduce(double key, int idx, double* sk,
                                               int* si, bool is_max) {
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    double ok = __shfl_down_sync(0xffffffffu, key, o);
    int oi = __shfl_down_sync(0xffffffffu, idx, o);
    bool take = is_max ? (ok > key || (ok == key && oi < idx))
                       : (ok < key || (ok == key && oi < idx));
    if (take) { key = ok; idx = oi; }
  }
  if (lane == 0) { sk[wid] = key; si[wid] = idx; }
  __syncthreads();
  if (threadIdx.x == 0) {
    int nwarps = (blockDim.x + 31) >> 5;
    double bk = sk[0];
    int bi = si[0];
    for (int i = 1; i < nwarps; ++i) {
      bool take = is_max ? (sk[i] > bk || (sk[i] == bk && si[i] < bi))
                         : (sk[i] < bk || (sk[i] == bk && si[i] < bi));
      if (take) { bk = sk[i]; bi = si[i]; }
    }
    si[0] = bi;
  }
  __syncthreads();
  int r = si[0];
  __syncthreads();
  return r;
}

// Reduce-class primitive over a single list, read directly from global memory.
// Result is valid in thread 0.
__device__ FeatureValue StreamReduce(AggPrim prim, const FeatureValue* vals,
                                     uint32_t n, double* sd, int* si) {
  uint32_t tid = threadIdx.x;
  uint32_t bd = blockDim.x;
  if (prim == AggPrim::kCount) return MakeIntValue(static_cast<int64_t>(n));
  if (n == 0) return MakeInvalidValue();

  switch (prim) {
    case AggPrim::kSum: {
      double s = 0.0;
      for (uint32_t i = tid; i < n; i += bd) s += vals[i].ToDouble();
      return MakeFloat64Value(BlockSumAll(s, sd));
    }
    case AggPrim::kMean: {
      double s = 0.0;
      for (uint32_t i = tid; i < n; i += bd) s += vals[i].ToDouble();
      s = BlockSumAll(s, sd);
      return MakeFloat64Value(s / static_cast<double>(n));
    }
    case AggPrim::kMin:
    case AggPrim::kMax: {
      bool is_max = (prim == AggPrim::kMax);
      double key = is_max ? -DBL_MAX : DBL_MAX;
      int idx = 0;
      bool found = false;
      for (uint32_t i = tid; i < n; i += bd) {
        double d = vals[i].ToDouble();
        if (!found || (is_max ? d > key : d < key) ||
            (d == key && static_cast<int>(i) < idx)) {
          key = d;
          idx = static_cast<int>(i);
          found = true;
        }
      }
      idx = BlockArgReduce(key, idx, sd, si, is_max);
      return vals[idx];
    }
    case AggPrim::kVariance:
    case AggPrim::kStd: {
      double s = 0.0;
      for (uint32_t i = tid; i < n; i += bd) s += vals[i].ToDouble();
      double mean = BlockSumAll(s, sd) / static_cast<double>(n);
      double v = 0.0;
      for (uint32_t i = tid; i < n; i += bd) {
        double d = vals[i].ToDouble() - mean;
        v += d * d;
      }
      v = BlockSumAll(v, sd) / static_cast<double>(n);
      return MakeFloat64Value(prim == AggPrim::kStd ? sqrt(v) : v);
    }
    case AggPrim::kSkew: {
      double s = 0.0;
      for (uint32_t i = tid; i < n; i += bd) s += vals[i].ToDouble();
      double mean = BlockSumAll(s, sd) / static_cast<double>(n);
      double v = 0.0;
      for (uint32_t i = tid; i < n; i += bd) {
        double d = vals[i].ToDouble() - mean;
        v += d * d;
      }
      double stdv = sqrt(BlockSumAll(v, sd) / static_cast<double>(n));
      if (stdv == 0.0) return MakeInvalidValue();
      double sk = 0.0;
      for (uint32_t i = tid; i < n; i += bd) {
        double z = (vals[i].ToDouble() - mean) / stdv;
        sk += z * z * z;
      }
      sk = BlockSumAll(sk, sd);
      return MakeFloat64Value(sk / static_cast<double>(n));
    }
    case AggPrim::kPercentTrue: {
      double t = 0.0;
      for (uint32_t i = tid; i < n; i += bd) {
        if (vals[i].type == ValueType::kBool && vals[i].b) t += 1.0;
      }
      t = BlockSumAll(t, sd);
      return MakeFloat64Value(t / static_cast<double>(n));
    }
    case AggPrim::kCountGreaterThanMean: {
      double s = 0.0;
      for (uint32_t i = tid; i < n; i += bd) s += vals[i].ToDouble();
      double mean = BlockSumAll(s, sd) / static_cast<double>(n);
      double c = 0.0;
      for (uint32_t i = tid; i < n; i += bd) {
        if (vals[i].ToDouble() > mean) c += 1.0;
      }
      c = BlockSumAll(c, sd);
      return MakeIntValue(static_cast<int64_t>(c + 0.5));
    }
    case AggPrim::kDFeat:
      return (n == 1) ? vals[0] : MakeInvalidValue();
    default:
      return MakeInvalidValue();
  }
}

// 64-bit key for hashing a FeatureValue (numeric/time/bool).
__device__ __forceinline__ unsigned long long FVKey(const FeatureValue& v) {
  switch (v.type) {
    case ValueType::kInt:
    case ValueType::kTime:
      return static_cast<unsigned long long>(v.i64);
    case ValueType::kFloat64:
      return static_cast<unsigned long long>(__double_as_longlong(v.f64));
    case ValueType::kFloat32:
      return static_cast<unsigned long long>(
          __double_as_longlong(static_cast<double>(v.f32)));
    case ValueType::kBool:
      return v.b ? 1ull : 0ull;
    default:
      return 0ull;
  }
}

__device__ __forceinline__ unsigned long long HashMix(unsigned long long x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

}  // namespace

__launch_bounds__(common::kStreamingAggBlockDim)
__global__ void ComputeAggStreamingKernel(const FeatureValue* d_values,
                                          const uint32_t* d_offsets,
                                          const uint32_t* d_list_ids,
                                          const AggPrim* d_list_prims,
                                          uint32_t n_group,
                                          FeatureValue* d_outputs) {
  uint32_t g = blockIdx.x;
  if (g >= n_group) return;
  __shared__ double sd[32];
  __shared__ int si[32];
  uint32_t list = d_list_ids[g];
  AggPrim prim = d_list_prims[g];
  uint32_t begin = d_offsets[list];
  uint32_t n = d_offsets[list + 1] - begin;
  FeatureValue r = StreamReduce(prim, d_values + begin, n, sd, si);
  if (threadIdx.x == 0) d_outputs[list] = r;
}

__launch_bounds__(common::kNumUniqueHashBlockDim)
__global__ void ComputeNumUniqueHashKernel(const FeatureValue* d_values,
                                           const uint32_t* d_offsets,
                                           const uint32_t* d_list_ids,
                                           uint32_t n_group,
                                           unsigned long long* d_hash,
                                           const uint32_t* d_hash_offsets,
                                           FeatureValue* d_outputs) {
  uint32_t g = blockIdx.x;
  if (g >= n_group) return;
  uint32_t list = d_list_ids[g];
  uint32_t begin = d_offsets[list];
  uint32_t n = d_offsets[list + 1] - begin;

  const unsigned long long kHashEmpty = 0xFFFFFFFFFFFFFFFFull;
  __shared__ unsigned int s_count;
  __shared__ int s_has_empty;
  if (threadIdx.x == 0) { s_count = 0u; s_has_empty = 0; }

  unsigned long long* table = d_hash + d_hash_offsets[g];
  uint32_t cap = d_hash_offsets[g + 1] - d_hash_offsets[g];
  for (uint32_t i = threadIdx.x; i < cap; i += blockDim.x) table[i] = kHashEmpty;
  __syncthreads();

  if (n == 0) {
    if (threadIdx.x == 0) d_outputs[list] = MakeInvalidValue();
    return;
  }

  uint32_t mask = cap - 1u;  // cap is a power of two
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    unsigned long long key = FVKey(d_values[begin + i]);
    if (key == kHashEmpty) {
      atomicExch(&s_has_empty, 1);
      continue;
    }
    uint32_t slot = static_cast<uint32_t>(HashMix(key) & mask);
    while (true) {
      unsigned long long old = atomicCAS(&table[slot], kHashEmpty, key);
      if (old == kHashEmpty) { atomicAdd(&s_count, 1u); break; }
      if (old == key) break;
      slot = (slot + 1u) & mask;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    unsigned int uniq = s_count + (s_has_empty ? 1u : 0u);
    d_outputs[list] = MakeIntValue(static_cast<int64_t>(uniq));
  }
}

}  // namespace kernel

// -----------------------------------------------------------------------------
// Host API helpers
// -----------------------------------------------------------------------------
namespace {

__host__ size_t ComputeSharedMemSize(uint32_t max_n) {
  // BlockBitonicSort may round n up to the next power of two for padding.
  uint32_t padded_n = NextPowerOfTwo(max_n);
  return static_cast<size_t>(padded_n) * sizeof(FeatureValue) +
         static_cast<size_t>(common::kBlockDim) * sizeof(double) +
         static_cast<size_t>(common::kBlockDim) * sizeof(FeatureValue) +
         static_cast<size_t>(common::kBlockDim) * sizeof(uint32_t);
}

__host__ void BuildOffsets(const std::vector<std::vector<FeatureValue>>& lists,
                           std::vector<FeatureValue>& flat_values,
                           std::vector<uint32_t>& offsets) {
  offsets.clear();
  offsets.reserve(lists.size() + 1);
  offsets.push_back(0);
  size_t total = 0;
  for (const auto& list : lists) {
    total += list.size();
    offsets.push_back(static_cast<uint32_t>(total));
  }
  flat_values.resize(total);
  size_t pos = 0;
  for (const auto& list : lists) {
    std::memcpy(flat_values.data() + pos, list.data(),
                list.size() * sizeof(FeatureValue));
    pos += list.size();
  }
}

__host__ uint32_t ComputeMaxN(const std::vector<std::vector<FeatureValue>>& lists) {
  uint32_t max_n = 0;
  for (const auto& list : lists) {
    max_n = std::max(max_n, static_cast<uint32_t>(list.size()));
  }
  return max_n;
}

}  // namespace

// -----------------------------------------------------------------------------
// ExecuteAggPrim host methods
// -----------------------------------------------------------------------------
__host__ ExecuteAggPrim::ExecuteAggPrim() = default;

__host__ ExecuteAggPrim::~ExecuteAggPrim() {
  DestroyStreams();
  FreeBuffers();
}

__host__ ExecuteAggPrim::ExecuteAggPrim(ExecuteAggPrim&& other) noexcept
    : n_streams_(other.n_streams_), streams_(std::move(other.streams_)),
      h_values_(other.h_values_), h_values_cap_(other.h_values_cap_),
      d_values_(other.d_values_), d_values_cap_(other.d_values_cap_),
      d_offsets_(other.d_offsets_), d_offsets_cap_(other.d_offsets_cap_),
      d_outputs_(other.d_outputs_), d_outputs_cap_(other.d_outputs_cap_),
      d_ids_(other.d_ids_), d_ids_cap_(other.d_ids_cap_),
      d_prims_(other.d_prims_), d_prims_cap_(other.d_prims_cap_),
      d_hoff_(other.d_hoff_), d_hoff_cap_(other.d_hoff_cap_),
      d_hash_(other.d_hash_), d_hash_cap_(other.d_hash_cap_) {
  other.n_streams_ = 1;
  other.h_values_ = nullptr; other.h_values_cap_ = 0;
  other.d_values_ = nullptr; other.d_values_cap_ = 0;
  other.d_offsets_ = nullptr; other.d_offsets_cap_ = 0;
  other.d_outputs_ = nullptr; other.d_outputs_cap_ = 0;
  other.d_ids_ = nullptr; other.d_ids_cap_ = 0;
  other.d_prims_ = nullptr; other.d_prims_cap_ = 0;
  other.d_hoff_ = nullptr; other.d_hoff_cap_ = 0;
  other.d_hash_ = nullptr; other.d_hash_cap_ = 0;
}

__host__ ExecuteAggPrim& ExecuteAggPrim::operator=(
    ExecuteAggPrim&& other) noexcept {
  if (this != &other) {
    DestroyStreams();
    FreeBuffers();
    n_streams_ = other.n_streams_;
    streams_ = std::move(other.streams_);
    h_values_ = other.h_values_; h_values_cap_ = other.h_values_cap_;
    d_values_ = other.d_values_; d_values_cap_ = other.d_values_cap_;
    d_offsets_ = other.d_offsets_; d_offsets_cap_ = other.d_offsets_cap_;
    d_outputs_ = other.d_outputs_; d_outputs_cap_ = other.d_outputs_cap_;
    d_ids_ = other.d_ids_; d_ids_cap_ = other.d_ids_cap_;
    d_prims_ = other.d_prims_; d_prims_cap_ = other.d_prims_cap_;
    d_hoff_ = other.d_hoff_; d_hoff_cap_ = other.d_hoff_cap_;
    d_hash_ = other.d_hash_; d_hash_cap_ = other.d_hash_cap_;
    other.n_streams_ = 1;
    other.h_values_ = nullptr; other.h_values_cap_ = 0;
    other.d_values_ = nullptr; other.d_values_cap_ = 0;
    other.d_offsets_ = nullptr; other.d_offsets_cap_ = 0;
    other.d_outputs_ = nullptr; other.d_outputs_cap_ = 0;
    other.d_ids_ = nullptr; other.d_ids_cap_ = 0;
    other.d_prims_ = nullptr; other.d_prims_cap_ = 0;
    other.d_hoff_ = nullptr; other.d_hoff_cap_ = 0;
    other.d_hash_ = nullptr; other.d_hash_cap_ = 0;
  }
  return *this;
}

__host__ void ExecuteAggPrim::FreeBuffers() {
  if (h_values_) cudaFreeHost(h_values_);
  if (d_values_) cudaFree(d_values_);
  if (d_offsets_) cudaFree(d_offsets_);
  if (d_outputs_) cudaFree(d_outputs_);
  if (d_ids_) cudaFree(d_ids_);
  if (d_prims_) cudaFree(d_prims_);
  if (d_hoff_) cudaFree(d_hoff_);
  if (d_hash_) cudaFree(d_hash_);
  h_values_ = nullptr; h_values_cap_ = 0;
  d_values_ = nullptr; d_values_cap_ = 0;
  d_offsets_ = nullptr; d_offsets_cap_ = 0;
  d_outputs_ = nullptr; d_outputs_cap_ = 0;
  d_ids_ = nullptr; d_ids_cap_ = 0;
  d_prims_ = nullptr; d_prims_cap_ = 0;
  d_hoff_ = nullptr; d_hoff_cap_ = 0;
  d_hash_ = nullptr; d_hash_cap_ = 0;
}

__host__ FeatureValue* ExecuteAggPrim::EnsurePinnedValues(uint32_t n) {
  if (static_cast<size_t>(n) > h_values_cap_) {
    if (h_values_) cudaFreeHost(h_values_);
    CUDA_CHECK(cudaMallocHost(&h_values_, sizeof(FeatureValue) * n));
    h_values_cap_ = n;
  }
  return h_values_;
}

__host__ void ExecuteAggPrim::SetNumStreams(uint32_t n_streams) {
  if (n_streams == 0) n_streams = 1;
  if (n_streams == n_streams_ && !streams_.empty()) return;
  DestroyStreams();
  n_streams_ = n_streams;
}

__host__ void ExecuteAggPrim::EnsureStreams() {
  if (streams_.size() == n_streams_) return;
  DestroyStreams();
  streams_.resize(n_streams_);
  for (uint32_t i = 0; i < n_streams_; ++i) {
    CUDA_CHECK(cudaStreamCreate(&streams_[i]));
  }
}

__host__ void ExecuteAggPrim::DestroyStreams() {
  for (cudaStream_t stream : streams_) {
    if (stream) cudaStreamDestroy(stream);
  }
  streams_.clear();
}

__host__ FeatureValue ExecuteAggPrim::Compute(AggPrim prim,
                                              const FeatureValue* host_values,
                                              uint32_t n) {
  if (n == 0) {
    return MakeInvalidValue();
  }

  FeatureValue* d_values = nullptr;
  uint32_t* d_offsets = nullptr;
  FeatureValue* d_output = nullptr;

  CUDA_CHECK(cudaMalloc(&d_values, sizeof(FeatureValue) * n));
  CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(uint32_t) * 2));
  CUDA_CHECK(cudaMalloc(&d_output, sizeof(FeatureValue)));

  uint32_t h_offsets[2] = {0, n};
  CUDA_CHECK(cudaMemcpy(d_values, host_values, sizeof(FeatureValue) * n,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets, sizeof(uint32_t) * 2,
                        cudaMemcpyHostToDevice));

  size_t shared_mem = ComputeSharedMemSize(n);
  CUDA_CHECK(cudaFuncSetAttribute(kernel::ComputeAggPrimKernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(shared_mem)));
  kernel::ComputeAggPrimKernel<<<1, common::kBlockDim, shared_mem>>>(
      prim, d_values, d_offsets, 0, 1, d_output);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  FeatureValue result;
  CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(FeatureValue),
                        cudaMemcpyDeviceToHost));

  cudaFree(d_values);
  cudaFree(d_offsets);
  cudaFree(d_output);

  return result;
}

__host__ std::vector<FeatureValue> ExecuteAggPrim::ComputeBatch(
    AggPrim prim,
    const std::vector<std::vector<FeatureValue>>& host_value_lists) {
  if (host_value_lists.empty()) return {};

  std::vector<FeatureValue> flat_values;
  std::vector<uint32_t> offsets;
  BuildOffsets(host_value_lists, flat_values, offsets);
  uint32_t n_lists = static_cast<uint32_t>(host_value_lists.size());
  uint32_t max_n = ComputeMaxN(host_value_lists);

  // Delegate to the flat path. flat_values here is pageable; the async copies
  // remain correct (just without true overlap). The bridge supplies pinned
  // memory for the real hot path.
  return ComputeBatchFlat(prim, flat_values.data(), offsets.data(), n_lists,
                          static_cast<uint32_t>(flat_values.size()), max_n);
}

__host__ std::vector<FeatureValue> ExecuteAggPrim::ComputeBatchFlat(
    AggPrim prim, const FeatureValue* h_flat, const uint32_t* offsets,
    uint32_t n_lists, uint32_t total_values, uint32_t max_n) {
  std::vector<FeatureValue> results;
  if (n_lists == 0) return results;

  EnsureStreams();

  FeatureValue* d_values = nullptr;
  uint32_t* d_offsets = nullptr;
  FeatureValue* d_outputs = nullptr;

  if (total_values > 0) {
    CUDA_CHECK(cudaMalloc(&d_values, sizeof(FeatureValue) * total_values));
  }
  CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(uint32_t) * (n_lists + 1)));
  CUDA_CHECK(cudaMalloc(&d_outputs, sizeof(FeatureValue) * n_lists));

  // Pinned host results buffer so the per-stream D2H is truly asynchronous.
  FeatureValue* h_results = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_results, sizeof(FeatureValue) * n_lists));

  // offsets is small; copy once on the default stream. A synchronous copy on
  // the legacy default stream blocks the host until complete, so every
  // per-stream kernel launched afterward sees the offsets in place.
  CUDA_CHECK(cudaMemcpy(d_offsets, offsets, sizeof(uint32_t) * (n_lists + 1),
                        cudaMemcpyHostToDevice));

  size_t shared_mem = ComputeSharedMemSize(max_n);
  CUDA_CHECK(cudaFuncSetAttribute(kernel::ComputeAggPrimKernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(shared_mem)));

  // Per stream: async H2D of just this stream's value slice -> kernel -> async
  // D2H, all on the same stream. Different streams overlap transfer/compute.
  uint32_t chunk_size = (n_lists + n_streams_ - 1) / n_streams_;
  for (uint32_t s = 0; s < n_streams_; ++s) {
    uint32_t begin = s * chunk_size;
    if (begin >= n_lists) break;
    uint32_t end = std::min(begin + chunk_size, n_lists);
    uint32_t count = end - begin;

    uint32_t vb = offsets[begin];
    uint32_t vc = offsets[end] - vb;
    if (vc > 0) {
      CUDA_CHECK(cudaMemcpyAsync(d_values + vb, h_flat + vb,
                                 sizeof(FeatureValue) * vc,
                                 cudaMemcpyHostToDevice, streams_[s]));
    }
    kernel::ComputeAggPrimKernel<<<count, common::kBlockDim,
                                   shared_mem, streams_[s]>>>(
        prim, d_values, d_offsets, begin, n_lists, d_outputs);
    CUDA_CHECK(cudaMemcpyAsync(h_results + begin, d_outputs + begin,
                               sizeof(FeatureValue) * count,
                               cudaMemcpyDeviceToHost, streams_[s]));
  }
  CUDA_CHECK(cudaGetLastError());

  for (cudaStream_t stream : streams_) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  results.assign(h_results, h_results + n_lists);

  cudaFreeHost(h_results);
  if (d_values) cudaFree(d_values);
  cudaFree(d_offsets);
  cudaFree(d_outputs);

  return results;
}

__host__ AllFeatures ExecuteAggPrim::ComputeAll(const FeatureValue* host_values,
                                                uint32_t n) {
  AllFeatures result{};
  if (n == 0) {
    result.count = MakeIntValue(0);
    return result;
  }

  FeatureValue* d_values = nullptr;
  uint32_t* d_offsets = nullptr;
  AllFeatures* d_output = nullptr;

  CUDA_CHECK(cudaMalloc(&d_values, sizeof(FeatureValue) * n));
  CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(uint32_t) * 2));
  CUDA_CHECK(cudaMalloc(&d_output, sizeof(AllFeatures)));

  uint32_t h_offsets[2] = {0, n};
  CUDA_CHECK(cudaMemcpy(d_values, host_values, sizeof(FeatureValue) * n,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets, sizeof(uint32_t) * 2,
                        cudaMemcpyHostToDevice));

  size_t shared_mem = ComputeSharedMemSize(n);
  CUDA_CHECK(cudaFuncSetAttribute(kernel::ComputeAllAggPrimsKernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(shared_mem)));
  kernel::ComputeAllAggPrimsKernel<<<1, common::kAllFeaturesBlockDim, shared_mem>>>(
      d_values, d_offsets, 0, 1, d_output);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(AllFeatures),
                        cudaMemcpyDeviceToHost));

  cudaFree(d_values);
  cudaFree(d_offsets);
  cudaFree(d_output);

  return result;
}

__host__ std::vector<AllFeatures> ExecuteAggPrim::ComputeAllBatch(
    const std::vector<std::vector<FeatureValue>>& host_value_lists) {
  std::vector<AllFeatures> results;
  if (host_value_lists.empty()) return results;

  std::vector<FeatureValue> flat_values;
  std::vector<uint32_t> offsets;
  BuildOffsets(host_value_lists, flat_values, offsets);
  uint32_t n_lists = static_cast<uint32_t>(host_value_lists.size());
  uint32_t max_n = ComputeMaxN(host_value_lists);

  EnsureStreams();

  FeatureValue* d_values = nullptr;
  uint32_t* d_offsets = nullptr;
  AllFeatures* d_outputs = nullptr;

  CUDA_CHECK(cudaMalloc(&d_values, sizeof(FeatureValue) * flat_values.size()));
  CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(uint32_t) * offsets.size()));
  CUDA_CHECK(cudaMalloc(&d_outputs, sizeof(AllFeatures) * n_lists));

  CUDA_CHECK(cudaMemcpy(d_values, flat_values.data(),
                        sizeof(FeatureValue) * flat_values.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(),
                        sizeof(uint32_t) * offsets.size(),
                        cudaMemcpyHostToDevice));

  size_t shared_mem = ComputeSharedMemSize(max_n);

  CUDA_CHECK(cudaFuncSetAttribute(kernel::ComputeAllAggPrimsKernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(shared_mem)));

  uint32_t chunk_size = (n_lists + n_streams_ - 1) / n_streams_;
  for (uint32_t s = 0; s < n_streams_; ++s) {
    uint32_t begin = s * chunk_size;
    if (begin >= n_lists) break;
    uint32_t end = std::min(begin + chunk_size, n_lists);
    uint32_t count = end - begin;
    kernel::ComputeAllAggPrimsKernel<<<count, common::kAllFeaturesBlockDim,
                                       shared_mem, streams_[s]>>>(
        d_values, d_offsets, begin, n_lists, d_outputs);
  }
  CUDA_CHECK(cudaGetLastError());

  results.resize(n_lists);
  for (uint32_t s = 0; s < n_streams_; ++s) {
    uint32_t begin = s * chunk_size;
    if (begin >= n_lists) break;
    uint32_t end = std::min(begin + chunk_size, n_lists);
    uint32_t count = end - begin;
    CUDA_CHECK(cudaMemcpyAsync(results.data() + begin, d_outputs + begin,
                               sizeof(AllFeatures) * count,
                               cudaMemcpyDeviceToHost, streams_[s]));
  }

  for (cudaStream_t stream : streams_) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  cudaFree(d_values);
  cudaFree(d_offsets);
  cudaFree(d_outputs);

  return results;
}

__host__ std::vector<std::vector<FeatureValue>>
ExecuteAggPrim::ComputeBatchMultiPrim(
    const std::vector<std::vector<FeatureValue>>& inputs,
    const std::vector<AggPrim>& prims) {
  if (inputs.empty() || prims.empty()) return {};

  std::vector<FeatureValue> flat_values;
  std::vector<uint32_t> offsets;
  BuildOffsets(inputs, flat_values, offsets);
  uint32_t n_lists = static_cast<uint32_t>(inputs.size());
  uint32_t max_n = ComputeMaxN(inputs);

  return ComputeBatchMultiPrimFlat(flat_values.data(), offsets.data(), n_lists,
                                   static_cast<uint32_t>(flat_values.size()),
                                   max_n, prims);
}

__host__ std::vector<std::vector<FeatureValue>>
ExecuteAggPrim::ComputeBatchMultiPrimFlat(
    const FeatureValue* h_flat, const uint32_t* offsets, uint32_t n_lists,
    uint32_t total_values, uint32_t max_n, const std::vector<AggPrim>& prims) {
  std::vector<std::vector<FeatureValue>> results;
  if (n_lists == 0 || prims.empty()) return results;
  uint32_t n_prims = static_cast<uint32_t>(prims.size());

  EnsureStreams();

  FeatureValue* d_values = nullptr;
  uint32_t* d_offsets = nullptr;
  AggPrim* d_prims = nullptr;
  FeatureValue* d_outputs = nullptr;

  if (total_values > 0) {
    CUDA_CHECK(cudaMalloc(&d_values, sizeof(FeatureValue) * total_values));
  }
  CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(uint32_t) * (n_lists + 1)));
  CUDA_CHECK(cudaMalloc(&d_prims, sizeof(AggPrim) * n_prims));
  CUDA_CHECK(cudaMalloc(&d_outputs, sizeof(FeatureValue) * n_lists * n_prims));

  // Pinned host results buffer for truly async D2H.
  FeatureValue* h_results = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_results,
                            sizeof(FeatureValue) * n_lists * n_prims));

  // Small metadata copies once on the default stream (host-blocking).
  CUDA_CHECK(cudaMemcpy(d_offsets, offsets, sizeof(uint32_t) * (n_lists + 1),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_prims, prims.data(), sizeof(AggPrim) * n_prims,
                        cudaMemcpyHostToDevice));

  size_t shared_mem = ComputeSharedMemSize(max_n);
  CUDA_CHECK(cudaFuncSetAttribute(kernel::ComputeBatchMultiPrimKernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(shared_mem)));

  uint32_t chunk_size = (n_lists + n_streams_ - 1) / n_streams_;
  for (uint32_t s = 0; s < n_streams_; ++s) {
    uint32_t begin = s * chunk_size;
    if (begin >= n_lists) break;
    uint32_t end = std::min(begin + chunk_size, n_lists);
    uint32_t count = end - begin;

    uint32_t vb = offsets[begin];
    uint32_t vc = offsets[end] - vb;
    if (vc > 0) {
      CUDA_CHECK(cudaMemcpyAsync(d_values + vb, h_flat + vb,
                                 sizeof(FeatureValue) * vc,
                                 cudaMemcpyHostToDevice, streams_[s]));
    }
    kernel::ComputeBatchMultiPrimKernel<<<count, common::kBlockDim,
                                          shared_mem, streams_[s]>>>(
        d_values, d_offsets, d_prims, n_prims, begin, n_lists, d_outputs);
    CUDA_CHECK(cudaMemcpyAsync(h_results + begin * n_prims,
                               d_outputs + begin * n_prims,
                               sizeof(FeatureValue) * count * n_prims,
                               cudaMemcpyDeviceToHost, streams_[s]));
  }
  CUDA_CHECK(cudaGetLastError());

  for (cudaStream_t stream : streams_) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  results.resize(n_lists, std::vector<FeatureValue>(n_prims));
  for (uint32_t i = 0; i < n_lists; ++i) {
    for (uint32_t j = 0; j < n_prims; ++j) {
      results[i][j] = h_results[i * n_prims + j];
    }
  }

  cudaFreeHost(h_results);
  if (d_values) cudaFree(d_values);
  cudaFree(d_offsets);
  cudaFree(d_prims);
  cudaFree(d_outputs);

  return results;
}

// -----------------------------------------------------------------------------
// Per-list single-primitive dispatch (no shared-memory list staging).
// -----------------------------------------------------------------------------
namespace {
enum class PrimClass { kStreaming, kHash, kSortFallback };

__host__ PrimClass ClassifyPrim(AggPrim p) {
  switch (p) {
    case AggPrim::kNumUnique:
      return PrimClass::kHash;
    case AggPrim::kMedian:
    case AggPrim::kMode:
    case AggPrim::kEntropy:
    case AggPrim::kQuarter:
    case AggPrim::kQuartile3:
      return PrimClass::kSortFallback;
    default:
      // count, sum, mean, min, max, variance, std, skew, percent_true,
      // count_greater_than_mean, dfeat
      return PrimClass::kStreaming;
  }
}

// Grow a persistent device buffer on demand (capacity in elements). Only
// reallocates when the requested size exceeds the current capacity.
template <typename T>
__host__ void EnsureDevice(T*& ptr, size_t& cap, size_t need) {
  if (need <= cap) return;
  if (ptr) cudaFree(ptr);
  CUDA_CHECK(cudaMalloc(&ptr, sizeof(T) * need));
  cap = need;
}
}  // namespace

__host__ std::vector<FeatureValue> ExecuteAggPrim::ComputeByPrim(
    const FeatureValue* h_flat, const uint32_t* offsets, uint32_t n_lists,
    uint32_t total_values, const uint8_t* list_prims) {
  std::vector<FeatureValue> results(n_lists);
  if (n_lists == 0) return results;

  EnsureStreams();

  // Persistent device scratch, grown on demand (no per-call cudaMalloc/Free).
  if (total_values > 0) EnsureDevice(d_values_, d_values_cap_, total_values);
  EnsureDevice(d_offsets_, d_offsets_cap_, static_cast<size_t>(n_lists) + 1);
  EnsureDevice(d_outputs_, d_outputs_cap_, n_lists);

  // Default every output to invalid (type kInvalid == 0).
  CUDA_CHECK(cudaMemset(d_outputs_, 0, sizeof(FeatureValue) * n_lists));
  CUDA_CHECK(cudaMemcpy(d_offsets_, offsets, sizeof(uint32_t) * (n_lists + 1),
                        cudaMemcpyHostToDevice));

  // Upload values: pinned, chunked async across streams, then sync. Group lists
  // are scattered, so values must be fully resident before any kernel runs.
  uint32_t chunk = (n_lists + n_streams_ - 1) / n_streams_;
  for (uint32_t s = 0; s < n_streams_; ++s) {
    uint32_t b = s * chunk;
    if (b >= n_lists) break;
    uint32_t e = std::min(b + chunk, n_lists);
    uint32_t vb = offsets[b];
    uint32_t vc = offsets[e] - vb;
    if (vc > 0) {
      CUDA_CHECK(cudaMemcpyAsync(d_values_ + vb, h_flat + vb,
                                 sizeof(FeatureValue) * vc,
                                 cudaMemcpyHostToDevice, streams_[s]));
    }
  }
  for (cudaStream_t stream : streams_) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // Partition list indices by primitive class.
  std::vector<uint32_t> stream_ids;
  std::vector<AggPrim> stream_prims;
  std::vector<uint32_t> hash_ids;
  std::vector<uint32_t> fb_ids;
  std::vector<AggPrim> fb_prims;
  for (uint32_t i = 0; i < n_lists; ++i) {
    AggPrim p = static_cast<AggPrim>(list_prims[i]);
    switch (ClassifyPrim(p)) {
      case PrimClass::kStreaming:
        stream_ids.push_back(i);
        stream_prims.push_back(p);
        break;
      case PrimClass::kHash:
        hash_ids.push_back(i);
        break;
      case PrimClass::kSortFallback:
        fb_ids.push_back(i);
        fb_prims.push_back(p);
        break;
    }
  }

  // Streaming group: one launch, one block per list, no list staging.
  // (d_ids_ is reused by the hash group below; the groups run sequentially.)
  if (!stream_ids.empty()) {
    uint32_t ng = static_cast<uint32_t>(stream_ids.size());
    EnsureDevice(d_ids_, d_ids_cap_, ng);
    EnsureDevice(d_prims_, d_prims_cap_, ng);
    CUDA_CHECK(cudaMemcpy(d_ids_, stream_ids.data(), sizeof(uint32_t) * ng,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prims_, stream_prims.data(), sizeof(AggPrim) * ng,
                          cudaMemcpyHostToDevice));
    kernel::ComputeAggStreamingKernel<<<ng, common::kStreamingAggBlockDim>>>(
        d_values_, d_offsets_, d_ids_, d_prims_, ng, d_outputs_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // NumUnique group: per-list open-addressing hash table in global scratch.
  if (!hash_ids.empty()) {
    uint32_t ng = static_cast<uint32_t>(hash_ids.size());
    std::vector<uint32_t> hoff(ng + 1);
    hoff[0] = 0;
    for (uint32_t k = 0; k < ng; ++k) {
      uint32_t list = hash_ids[k];
      uint32_t n = offsets[list + 1] - offsets[list];
      uint32_t cap = (n == 0) ? 1u : NextPowerOfTwo(n) * 2u;
      hoff[k + 1] = hoff[k] + cap;
    }
    uint32_t total_cap = hoff[ng];
    EnsureDevice(d_ids_, d_ids_cap_, ng);
    EnsureDevice(d_hoff_, d_hoff_cap_, static_cast<size_t>(ng) + 1);
    EnsureDevice(d_hash_, d_hash_cap_, total_cap);
    CUDA_CHECK(cudaMemcpy(d_ids_, hash_ids.data(), sizeof(uint32_t) * ng,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hoff_, hoff.data(), sizeof(uint32_t) * (ng + 1),
                          cudaMemcpyHostToDevice));
    kernel::ComputeNumUniqueHashKernel<<<ng, common::kNumUniqueHashBlockDim>>>(
        d_values_, d_offsets_, d_ids_, ng, d_hash_, d_hoff_, d_outputs_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Sort-only fallback (median/mode/entropy/quarter/quartile3): legacy shared
  // kernel, one list at a time. Unused by the current feature set.
  for (size_t k = 0; k < fb_ids.size(); ++k) {
    uint32_t list = fb_ids[k];
    uint32_t n = offsets[list + 1] - offsets[list];
    size_t shared_mem = ComputeSharedMemSize(n);
    CUDA_CHECK(cudaFuncSetAttribute(kernel::ComputeAggPrimKernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(shared_mem)));
    kernel::ComputeAggPrimKernel<<<1, common::kBlockDim, shared_mem>>>(
        fb_prims[k], d_values_, d_offsets_, list, n_lists, d_outputs_);
    CUDA_CHECK(cudaGetLastError());
  }
  if (!fb_ids.empty()) CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(results.data(), d_outputs_,
                        sizeof(FeatureValue) * n_lists,
                        cudaMemcpyDeviceToHost));
  return results;
}

__host__ void ExecuteAggPrim::Run() {
  std::cout << "[ExecuteAggPrim] Run()" << std::endl;
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
