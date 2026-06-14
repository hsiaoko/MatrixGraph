#include "core/task/gpu_task/execute_agg_prim.cuh"

#include <cmath>
#include <cfloat>
#include <cstring>
#include <iostream>

#include "core/util/cuda_check.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

constexpr uint32_t kExecuteAggPrimBlockSize = 256;

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

namespace kernel {

// -----------------------------------------------------------------------------
// Block reductions
// -----------------------------------------------------------------------------
__device__ inline double BlockSum(double val) {
  __shared__ double sdata[kExecuteAggPrimBlockSize];
  sdata[threadIdx.x] = val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  return sdata[0];
}

__device__ inline FeatureValue BlockMin(FeatureValue val) {
  __shared__ FeatureValue sdata[kExecuteAggPrimBlockSize];
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
  __shared__ FeatureValue sdata[kExecuteAggPrimBlockSize];
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

__device__ inline FeatureValue BlockAggSum(const FeatureValue* values,
                                           uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  double local = 0.0;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    local += values[i].ToDouble();
  }
  return MakeFloat64Value(BlockSum(local));
}

__device__ inline FeatureValue BlockAggMean(const FeatureValue* values,
                                            uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  FeatureValue sum = BlockAggSum(values, n);
  return MakeFloat64Value(sum.ToDouble() / static_cast<double>(n));
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
  return MakeFloat64Value(BlockSum(local) / static_cast<double>(n));
}

__device__ inline FeatureValue BlockAggStd(const FeatureValue* values,
                                           uint32_t n) {
  FeatureValue var = BlockAggVariance(values, n);
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

__device__ inline FeatureValue BlockAggPercentTrue(const FeatureValue* values,
                                                   uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  uint32_t local = 0;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    if (values[i].type == ValueType::kBool && values[i].b) ++local;
  }
  __shared__ uint32_t scratch[kExecuteAggPrimBlockSize];
  uint32_t total = 0;
  BlockExclusiveScan(local, scratch, &total);
  return MakeFloat64Value(static_cast<double>(total) / static_cast<double>(n));
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
  return MakeFloat64Value(BlockSum(local) / static_cast<double>(n));
}

__device__ inline FeatureValue BlockAggCountGreaterThanMean(
    const FeatureValue* values, uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  double mean = BlockAggMean(values, n).ToDouble();
  uint32_t local = 0;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    if (values[i].ToDouble() > mean) ++local;
  }
  __shared__ uint32_t scratch[kExecuteAggPrimBlockSize];
  uint32_t total = 0;
  BlockExclusiveScan(local, scratch, &total);
  return MakeIntValue(static_cast<int64_t>(total));
}

__device__ inline FeatureValue BlockAggDFeat(const FeatureValue* values,
                                             uint32_t n) {
  if (n != 1) return MakeInvalidValue();
  return values[0];
}

__device__ inline FeatureValue BlockApplyAggPrim(AggPrim prim,
                                                 FeatureValue* values,
                                                 uint32_t n) {
  switch (prim) {
    case AggPrim::kCount:
      return BlockAggCount(n);
    case AggPrim::kSum:
      return BlockAggSum(values, n);
    case AggPrim::kMean:
      return BlockAggMean(values, n);
    case AggPrim::kMedian:
      return BlockAggMedian(values, n);
    case AggPrim::kMode:
      return BlockAggMode(values, n);
    case AggPrim::kMax:
      return BlockAggMax(values, n);
    case AggPrim::kMin:
      return BlockAggMin(values, n);
    case AggPrim::kVariance:
      return BlockAggVariance(values, n);
    case AggPrim::kStd:
      return BlockAggStd(values, n);
    case AggPrim::kSkew:
      return BlockAggSkew(values, n);
    case AggPrim::kEntropy:
      return BlockAggEntropy(values, n);
    case AggPrim::kNumUnique:
      return BlockAggNumUnique(values, n);
    case AggPrim::kPercentTrue:
      return BlockAggPercentTrue(values, n);
    case AggPrim::kQuarter:
      return BlockAggQuarter(values, n);
    case AggPrim::kQuartile3:
      return BlockAggQuartile3(values, n);
    case AggPrim::kCountGreaterThanMean:
      return BlockAggCountGreaterThanMean(values, n);
    case AggPrim::kDFeat:
      return BlockAggDFeat(values, n);
    default:
      return MakeInvalidValue();
  }
}

// -----------------------------------------------------------------------------
// Fused compute-all
// -----------------------------------------------------------------------------
__device__ inline AllFeatures BlockComputeAllFeaturesFromValues(
    FeatureValue* values, uint32_t n) {
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
  double sum = BlockSum(local_sum);
  FeatureValue minv = BlockMin(local_min);
  FeatureValue maxv = BlockMax(local_max);

  __shared__ uint32_t scratch[kExecuteAggPrimBlockSize];
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

  // Load values into shared memory cooperatively.
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    shared_buf[i] = d_values[begin + i];
  }
  __syncthreads();

  FeatureValue result = BlockApplyAggPrim(prim, shared_buf, n);

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

  // Load values into shared memory once for all primitives.
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    shared_buf[i] = d_values[begin + i];
  }
  __syncthreads();

  FeatureValue* out_row = d_outputs + list_idx * n_prims;
  for (uint32_t p = 0; p < n_prims; ++p) {
    // Synchronize before reusing shared_buf for order-dependent primitives.
    __syncthreads();
    FeatureValue result = BlockApplyAggPrim(d_prims[p], shared_buf, n);
    if (threadIdx.x == 0) {
      out_row[p] = result;
    }
  }
}

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

  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    shared_buf[i] = d_values[begin + i];
  }
  __syncthreads();

  AllFeatures result = BlockComputeAllFeaturesFromValues(shared_buf, n);

  if (threadIdx.x == 0) {
    d_outputs[list_idx] = result;
  }
}

}  // namespace kernel

// -----------------------------------------------------------------------------
// Host API helpers
// -----------------------------------------------------------------------------
namespace {

__host__ uint32_t NextPowerOfTwo(uint32_t v) {
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

__host__ size_t ComputeSharedMemSize(uint32_t max_n) {
  // BlockBitonicSort may round n up to the next power of two for padding.
  uint32_t padded_n = NextPowerOfTwo(max_n);
  return static_cast<size_t>(padded_n) * sizeof(FeatureValue);
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
}

__host__ ExecuteAggPrim::ExecuteAggPrim(ExecuteAggPrim&& other) noexcept
    : n_streams_(other.n_streams_), streams_(std::move(other.streams_)) {
  other.n_streams_ = 1;
}

__host__ ExecuteAggPrim& ExecuteAggPrim::operator=(
    ExecuteAggPrim&& other) noexcept {
  if (this != &other) {
    DestroyStreams();
    n_streams_ = other.n_streams_;
    streams_ = std::move(other.streams_);
    other.n_streams_ = 1;
  }
  return *this;
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
  kernel::ComputeAggPrimKernel<<<1, kExecuteAggPrimBlockSize, shared_mem>>>(
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
  std::vector<FeatureValue> results;
  if (host_value_lists.empty()) return results;

  std::vector<FeatureValue> flat_values;
  std::vector<uint32_t> offsets;
  BuildOffsets(host_value_lists, flat_values, offsets);
  uint32_t n_lists = static_cast<uint32_t>(host_value_lists.size());
  uint32_t max_n = ComputeMaxN(host_value_lists);

  EnsureStreams();

  FeatureValue* d_values = nullptr;
  uint32_t* d_offsets = nullptr;
  FeatureValue* d_outputs = nullptr;

  CUDA_CHECK(cudaMalloc(&d_values, sizeof(FeatureValue) * flat_values.size()));
  CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(uint32_t) * offsets.size()));
  CUDA_CHECK(cudaMalloc(&d_outputs, sizeof(FeatureValue) * n_lists));

  // Synchronous H2D: data must be resident before any stream's kernel reads it.
  CUDA_CHECK(cudaMemcpy(d_values, flat_values.data(),
                        sizeof(FeatureValue) * flat_values.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(),
                        sizeof(uint32_t) * offsets.size(),
                        cudaMemcpyHostToDevice));

  size_t shared_mem = ComputeSharedMemSize(max_n);

  // Partition lists across streams.
  uint32_t chunk_size = (n_lists + n_streams_ - 1) / n_streams_;
  for (uint32_t s = 0; s < n_streams_; ++s) {
    uint32_t begin = s * chunk_size;
    if (begin >= n_lists) break;
    uint32_t end = std::min(begin + chunk_size, n_lists);
    uint32_t count = end - begin;
    kernel::ComputeAggPrimKernel<<<count, kExecuteAggPrimBlockSize,
                                   shared_mem, streams_[s]>>>(
        prim, d_values, d_offsets, begin, n_lists, d_outputs);
  }
  CUDA_CHECK(cudaGetLastError());

  // Asynchronous D2H per stream.
  results.resize(n_lists);
  for (uint32_t s = 0; s < n_streams_; ++s) {
    uint32_t begin = s * chunk_size;
    if (begin >= n_lists) break;
    uint32_t end = std::min(begin + chunk_size, n_lists);
    uint32_t count = end - begin;
    CUDA_CHECK(cudaMemcpyAsync(results.data() + begin, d_outputs + begin,
                               sizeof(FeatureValue) * count,
                               cudaMemcpyDeviceToHost, streams_[s]));
  }

  // Synchronize all streams.
  for (cudaStream_t stream : streams_) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  cudaFree(d_values);
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
  kernel::ComputeAllAggPrimsKernel<<<1, kExecuteAggPrimBlockSize, shared_mem>>>(
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

  uint32_t chunk_size = (n_lists + n_streams_ - 1) / n_streams_;
  for (uint32_t s = 0; s < n_streams_; ++s) {
    uint32_t begin = s * chunk_size;
    if (begin >= n_lists) break;
    uint32_t end = std::min(begin + chunk_size, n_lists);
    uint32_t count = end - begin;
    kernel::ComputeAllAggPrimsKernel<<<count, kExecuteAggPrimBlockSize,
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
  std::vector<std::vector<FeatureValue>> results;
  if (inputs.empty() || prims.empty()) return results;

  std::vector<FeatureValue> flat_values;
  std::vector<uint32_t> offsets;
  BuildOffsets(inputs, flat_values, offsets);
  uint32_t n_lists = static_cast<uint32_t>(inputs.size());
  uint32_t n_prims = static_cast<uint32_t>(prims.size());
  uint32_t max_n = ComputeMaxN(inputs);

  EnsureStreams();

  FeatureValue* d_values = nullptr;
  uint32_t* d_offsets = nullptr;
  AggPrim* d_prims = nullptr;
  FeatureValue* d_outputs = nullptr;

  CUDA_CHECK(cudaMalloc(&d_values, sizeof(FeatureValue) * flat_values.size()));
  CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(uint32_t) * offsets.size()));
  CUDA_CHECK(cudaMalloc(&d_prims, sizeof(AggPrim) * n_prims));
  CUDA_CHECK(cudaMalloc(&d_outputs, sizeof(FeatureValue) * n_lists * n_prims));

  CUDA_CHECK(cudaMemcpy(d_values, flat_values.data(),
                        sizeof(FeatureValue) * flat_values.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(),
                        sizeof(uint32_t) * offsets.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_prims, prims.data(),
                        sizeof(AggPrim) * n_prims,
                        cudaMemcpyHostToDevice));

  size_t shared_mem = ComputeSharedMemSize(max_n);

  uint32_t chunk_size = (n_lists + n_streams_ - 1) / n_streams_;
  for (uint32_t s = 0; s < n_streams_; ++s) {
    uint32_t begin = s * chunk_size;
    if (begin >= n_lists) break;
    uint32_t end = std::min(begin + chunk_size, n_lists);
    uint32_t count = end - begin;
    kernel::ComputeBatchMultiPrimKernel<<<count, kExecuteAggPrimBlockSize,
                                          shared_mem, streams_[s]>>>(
        d_values, d_offsets, d_prims, n_prims, begin, n_lists, d_outputs);
  }
  CUDA_CHECK(cudaGetLastError());

  // Flat result buffer: row-major [n_lists][n_prims].
  std::vector<FeatureValue> flat_results(n_lists * n_prims);
  for (uint32_t s = 0; s < n_streams_; ++s) {
    uint32_t begin = s * chunk_size;
    if (begin >= n_lists) break;
    uint32_t end = std::min(begin + chunk_size, n_lists);
    uint32_t count = end - begin;
    CUDA_CHECK(cudaMemcpyAsync(
        flat_results.data() + begin * n_prims,
        d_outputs + begin * n_prims,
        sizeof(FeatureValue) * count * n_prims,
        cudaMemcpyDeviceToHost, streams_[s]));
  }

  for (cudaStream_t stream : streams_) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  cudaFree(d_values);
  cudaFree(d_offsets);
  cudaFree(d_prims);
  cudaFree(d_outputs);

  results.resize(n_lists, std::vector<FeatureValue>(n_prims));
  for (uint32_t i = 0; i < n_lists; ++i) {
    for (uint32_t j = 0; j < n_prims; ++j) {
      results[i][j] = flat_results[i * n_prims + j];
    }
  }
  return results;
}

__host__ void ExecuteAggPrim::Run() {
  std::cout << "[ExecuteAggPrim] Run()" << std::endl;
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
