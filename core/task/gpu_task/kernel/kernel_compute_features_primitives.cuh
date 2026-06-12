#ifndef MATRIXGRAPH_CORE_TASK_GPU_TASK_KERNEL_COMPUTE_FEATURES_PRIMITIVES_CUH_
#define MATRIXGRAPH_CORE_TASK_GPU_TASK_KERNEL_COMPUTE_FEATURES_PRIMITIVES_CUH_

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdint>

#include "core/task/gpu_task/compute_features_types.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

/**
 * @file kernel_compute_features_primitives.cuh
 * @brief Block-level aggregation primitives and MGFeatureValue helpers.
 *
 * All aggregation functions assume that `values` has at least blockDim.x
 * entries and that the valid prefix has length `n`.  Order-statistic
 * primitives (median, mode, quartiles, entropy, num_unique) require the caller
 * to pad unused slots with a value larger than any real value (DBL_MAX).
 */

using MGFeatureValue = MatrixGraphFeatureValue;

// Operator codes (match GraphAggregate AggPrim ordering).
constexpr int32_t kAggCount = 0;
constexpr int32_t kAggCountGreaterThanMean = 1;
constexpr int32_t kAggNumUnique = 2;
constexpr int32_t kAggSum = 3;
constexpr int32_t kAggMean = 4;
constexpr int32_t kAggVariance = 5;
constexpr int32_t kAggStd = 6;
constexpr int32_t kAggMode = 7;
constexpr int32_t kAggMin = 8;
constexpr int32_t kAggMax = 9;
constexpr int32_t kAggMedian = 10;
constexpr int32_t kAggQuarter = 11;
constexpr int32_t kAggQuartile3 = 12;
constexpr int32_t kAggEntropy = 13;
constexpr int32_t kAggPercentTrue = 14;
constexpr int32_t kAggSkew = 15;

/** @brief Construct an invalid MGFeatureValue. */
__device__ __host__ inline MGFeatureValue MakeInvalidValue() {
  MGFeatureValue r;
  r.type = MG_VALUE_INVALID;
  r.i64 = 0;
  return r;
}

/** @brief Construct an integer MGFeatureValue. */
__device__ __host__ inline MGFeatureValue MakeIntValue(int64_t v) {
  MGFeatureValue r;
  r.type = MG_VALUE_INT;
  r.i64 = v;
  return r;
}

/** @brief Construct a 64-bit floating point MGFeatureValue. */
__device__ __host__ inline MGFeatureValue MakeFloatValue(double v) {
  MGFeatureValue r;
  r.type = MG_VALUE_FLOAT64;
  r.f64 = v;
  return r;
}

/** @brief Construct a boolean MGFeatureValue. */
__device__ __host__ inline MGFeatureValue MakeBoolValue(bool v) {
  MGFeatureValue r;
  r.type = MG_VALUE_BOOL;
  r.b = v ? 1 : 0;
  return r;
}

/**
 * @brief Convert a typed MGFeatureValue to double for comparison/math.
 *
 * Invalid values are converted to 0.0.  Bool is 1.0 for true and 0.0 for false.
 */
__device__ __host__ inline double ToDouble(const MGFeatureValue& v) {
  switch (static_cast<MatrixGraphValueType>(v.type)) {
    case MG_VALUE_INT:
    case MG_VALUE_TIME:
      return static_cast<double>(v.i64);
    case MG_VALUE_FLOAT64:
      return v.f64;
    case MG_VALUE_FLOAT32:
      return static_cast<double>(v.f64);
    case MG_VALUE_BOOL:
      return v.b ? 1.0 : 0.0;
    default:
      return 0.0;
  }
}

/** @brief Return true if the value is not MG_VALUE_INVALID. */
__device__ __host__ inline bool IsValid(const MGFeatureValue& v) {
  return static_cast<MatrixGraphValueType>(v.type) != MG_VALUE_INVALID;
}

/** @brief Three-way comparison using ToDouble(). */
__device__ __host__ inline int Compare(const MGFeatureValue& a,
                                       const MGFeatureValue& b) {
  double da = ToDouble(a);
  double db = ToDouble(b);
  if (da < db) return -1;
  if (da > db) return 1;
  return 0;
}

// ---------------------------------------------------------------------------
// Block reductions (all threads participate, result broadcast to all threads).
// ---------------------------------------------------------------------------

/** @brief Sum reduction across the block. */
__device__ inline double BlockSum(double local) {
  __shared__ double s_partial[32];
  double x = local;
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffff, x, offset);
  }
  if (threadIdx.x % 32 == 0) s_partial[threadIdx.x / 32] = x;
  __syncthreads();
  if (threadIdx.x == 0) {
    int n_warps = (blockDim.x + 31) / 32;
    double sum = 0.0;
    for (int i = 0; i < n_warps; ++i) sum += s_partial[i];
    s_partial[0] = sum;
  }
  __syncthreads();
  return s_partial[0];
}

/** @brief Minimum reduction across the block. */
__device__ inline double BlockMinVal(double local) {
  __shared__ double s_partial[32];
  double x = local;
  for (int offset = 16; offset > 0; offset >>= 1) {
    x = fmin(x, __shfl_down_sync(0xffffffff, x, offset));
  }
  if (threadIdx.x % 32 == 0) s_partial[threadIdx.x / 32] = x;
  __syncthreads();
  if (threadIdx.x == 0) {
    int n_warps = (blockDim.x + 31) / 32;
    double m = s_partial[0];
    for (int i = 1; i < n_warps; ++i) m = fmin(m, s_partial[i]);
    s_partial[0] = m;
  }
  __syncthreads();
  return s_partial[0];
}

/** @brief Maximum reduction across the block. */
__device__ inline double BlockMaxVal(double local) {
  __shared__ double s_partial[32];
  double x = local;
  for (int offset = 16; offset > 0; offset >>= 1) {
    x = fmax(x, __shfl_down_sync(0xffffffff, x, offset));
  }
  if (threadIdx.x % 32 == 0) s_partial[threadIdx.x / 32] = x;
  __syncthreads();
  if (threadIdx.x == 0) {
    int n_warps = (blockDim.x + 31) / 32;
    double m = s_partial[0];
    for (int i = 1; i < n_warps; ++i) m = fmax(m, s_partial[i]);
    s_partial[0] = m;
  }
  __syncthreads();
  return s_partial[0];
}

// ---------------------------------------------------------------------------
// Block-level bitonic sort on shared/global array of size blockDim.x.
// The caller must pad unused entries with a value larger than any real value
// (we use DBL_MAX sentinel) before calling.
// ---------------------------------------------------------------------------

/** @brief Swap two MGFeatureValue values. */
__device__ inline void SwapValues(MGFeatureValue& a, MGFeatureValue& b) {
  MGFeatureValue tmp = a;
  a = b;
  b = tmp;
}

/**
 * @brief In-place bitonic sort of the first blockDim.x entries of @p arr.
 *
 * All threads participate.  The array must contain at least blockDim.x entries.
 */
__device__ inline void BlockBitonicSort(MGFeatureValue* arr) {
  for (uint32_t k = 2; k <= blockDim.x; k <<= 1) {
    for (uint32_t j = k >> 1; j > 0; j >>= 1) {
      uint32_t ixj = threadIdx.x ^ j;
      if (ixj > threadIdx.x) {
        if ((threadIdx.x & k) == 0) {
          if (Compare(arr[threadIdx.x], arr[ixj]) > 0) {
            SwapValues(arr[threadIdx.x], arr[ixj]);
          }
        } else {
          if (Compare(arr[threadIdx.x], arr[ixj]) < 0) {
            SwapValues(arr[threadIdx.x], arr[ixj]);
          }
        }
      }
      __syncthreads();
    }
  }
}

// ---------------------------------------------------------------------------
// Aggregation primitives.
// All functions assume `values` has at least blockDim.x entries and that the
// valid prefix has length n; the rest must be filled with a sentinel larger
// than any real value.
// ---------------------------------------------------------------------------

/** @brief Number of bindings. */
__device__ inline MGFeatureValue AggCount(const MGFeatureValue*, uint32_t n) {
  return MakeIntValue(static_cast<int64_t>(n));
}

/** @brief Sum of the values. */
__device__ inline MGFeatureValue AggSum(const MGFeatureValue* values,
                                        uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  double local = 0.0;
  if (threadIdx.x < n) local = ToDouble(values[threadIdx.x]);
  double sum = BlockSum(local);
  return MakeFloatValue(sum);
}

/** @brief Arithmetic mean. */
__device__ inline MGFeatureValue AggMean(const MGFeatureValue* values,
                                         uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  double local = 0.0;
  if (threadIdx.x < n) local = ToDouble(values[threadIdx.x]);
  double sum = BlockSum(local);
  return MakeFloatValue(sum / static_cast<double>(n));
}

/** @brief Minimum value. */
__device__ inline MGFeatureValue AggMin(const MGFeatureValue* values,
                                        uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  double local = DBL_MAX;
  if (threadIdx.x < n) local = ToDouble(values[threadIdx.x]);
  double m = BlockMinVal(local);
  return MakeFloatValue(m);
}

/** @brief Maximum value. */
__device__ inline MGFeatureValue AggMax(const MGFeatureValue* values,
                                        uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  double local = -DBL_MAX;
  if (threadIdx.x < n) local = ToDouble(values[threadIdx.x]);
  double m = BlockMaxVal(local);
  return MakeFloatValue(m);
}

/** @brief Population variance. */
__device__ inline MGFeatureValue AggVariance(const MGFeatureValue* values,
                                             uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  double local = 0.0;
  if (threadIdx.x < n) local = ToDouble(values[threadIdx.x]);
  double mean = BlockSum(local) / static_cast<double>(n);
  double diff = 0.0;
  if (threadIdx.x < n) {
    double d = ToDouble(values[threadIdx.x]) - mean;
    diff = d * d;
  }
  double sum = BlockSum(diff);
  return MakeFloatValue(sum / static_cast<double>(n));
}

/** @brief Population standard deviation. */
__device__ inline MGFeatureValue AggStd(const MGFeatureValue* values,
                                        uint32_t n) {
  MGFeatureValue var = AggVariance(values, n);
  if (!IsValid(var)) return var;
  return MakeFloatValue(sqrt(var.f64));
}

/** @brief Fraction of bindings whose boolean value is true. */
__device__ inline MGFeatureValue AggPercentTrue(const MGFeatureValue* values,
                                                uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  double local = 0.0;
  if (threadIdx.x < n) {
    local = (values[threadIdx.x].type == MG_VALUE_BOOL &&
             values[threadIdx.x].b)
                ? 1.0
                : 0.0;
  }
  double cnt = BlockSum(local);
  return MakeFloatValue(cnt / static_cast<double>(n));
}

/** @brief Count of values strictly greater than the mean. */
__device__ inline MGFeatureValue AggCountGreaterThanMean(
    const MGFeatureValue* values, uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  double local = 0.0;
  if (threadIdx.x < n) local = ToDouble(values[threadIdx.x]);
  double mean = BlockSum(local) / static_cast<double>(n);
  double cnt = 0.0;
  if (threadIdx.x < n && ToDouble(values[threadIdx.x]) > mean) cnt = 1.0;
  cnt = BlockSum(cnt);
  return MakeIntValue(static_cast<int64_t>(cnt));
}

/**
 * @brief Median of the values.
 *
 * The caller must sort @p values before calling this function.
 */
__device__ inline MGFeatureValue AggMedian(const MGFeatureValue* values,
                                           uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  if (threadIdx.x == 0) {
    return values[n / 2];
  }
  return values[n / 2];
}

/**
 * @brief First quartile (25th percentile) by linear interpolation.
 *
 * The caller must sort @p values before calling this function.
 */
__device__ inline MGFeatureValue AggQuarter(const MGFeatureValue* values,
                                            uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  if (n == 1) return values[0];
  double pos = 0.25 * static_cast<double>(n - 1);
  uint32_t lo = static_cast<uint32_t>(floor(pos));
  uint32_t hi = static_cast<uint32_t>(ceil(pos));
  double w = pos - static_cast<double>(lo);
  double v = ToDouble(values[lo]) * (1.0 - w) + ToDouble(values[hi]) * w;
  return MakeFloatValue(v);
}

/**
 * @brief Third quartile (75th percentile) by linear interpolation.
 *
 * The caller must sort @p values before calling this function.
 */
__device__ inline MGFeatureValue AggQuartile3(const MGFeatureValue* values,
                                              uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  if (n == 1) return values[0];
  double pos = 0.75 * static_cast<double>(n - 1);
  uint32_t lo = static_cast<uint32_t>(floor(pos));
  uint32_t hi = static_cast<uint32_t>(ceil(pos));
  double w = pos - static_cast<double>(lo);
  double v = ToDouble(values[lo]) * (1.0 - w) + ToDouble(values[hi]) * w;
  return MakeFloatValue(v);
}

/**
 * @brief Most frequent value.
 *
 * The caller must sort @p values before calling this function.  Only thread 0
 * performs the linear scan; other threads return the same value.
 */
__device__ inline MGFeatureValue AggMode(const MGFeatureValue* values,
                                         uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  if (threadIdx.x != 0) return values[0];
  uint32_t max_count = 1;
  uint32_t curr_count = 1;
  uint32_t mode_idx = 0;
  for (uint32_t i = 1; i < n; ++i) {
    if (Compare(values[i], values[i - 1]) == 0) {
      ++curr_count;
    } else {
      curr_count = 1;
    }
    if (curr_count > max_count) {
      max_count = curr_count;
      mode_idx = i;
    }
  }
  return values[mode_idx];
}

/**
 * @brief Number of distinct values.
 *
 * The caller must sort @p values before calling this function.
 */
__device__ inline MGFeatureValue AggNumUnique(const MGFeatureValue* values,
                                              uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  if (threadIdx.x != 0) return MakeIntValue(0);
  uint32_t uniq = 1;
  for (uint32_t i = 1; i < n; ++i) {
    if (Compare(values[i], values[i - 1]) != 0) ++uniq;
  }
  return MakeIntValue(static_cast<int64_t>(uniq));
}

/**
 * @brief Shannon entropy (base 2) of the value distribution.
 *
 * The caller must sort @p values before calling this function.
 */
__device__ inline MGFeatureValue AggEntropy(const MGFeatureValue* values,
                                            uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  if (threadIdx.x != 0) return MakeFloatValue(0.0);
  double entropy = 0.0;
  uint32_t i = 0;
  while (i < n) {
    uint32_t j = i + 1;
    while (j < n && Compare(values[j], values[i]) == 0) ++j;
    double p = static_cast<double>(j - i) / static_cast<double>(n);
    entropy -= p * log2(p);
    i = j;
  }
  return MakeFloatValue(entropy);
}

/**
 * @brief Skewness of the values.
 *
 * Computed as the average standardized third moment.  Returns invalid if the
 * standard deviation is zero.
 */
__device__ inline MGFeatureValue AggSkew(const MGFeatureValue* values,
                                         uint32_t n) {
  if (n == 0) return MakeInvalidValue();
  double local = 0.0;
  if (threadIdx.x < n) local = ToDouble(values[threadIdx.x]);
  double mean = BlockSum(local) / static_cast<double>(n);

  double sq = 0.0;
  if (threadIdx.x < n) {
    double d = ToDouble(values[threadIdx.x]) - mean;
    sq = d * d;
  }
  double variance = BlockSum(sq) / static_cast<double>(n);
  double stdv = sqrt(variance);
  if (stdv == 0.0) return MakeInvalidValue();

  double cube = 0.0;
  if (threadIdx.x < n) {
    double z = (ToDouble(values[threadIdx.x]) - mean) / stdv;
    cube = z * z * z;
  }
  double sum = BlockSum(cube);
  return MakeFloatValue(sum / static_cast<double>(n));
}

// ---------------------------------------------------------------------------
// Top-level dispatcher.  The caller must ensure `values` has
// blockDim.x entries and unused slots are padded with a DBL_MAX sentinel.
// ---------------------------------------------------------------------------

/**
 * @brief Dispatch to the requested aggregation primitive.
 *
 * Order-statistic primitives trigger an in-place bitonic sort before the
 * primitive is evaluated.
 */
__device__ inline MGFeatureValue ApplyAggPrim(int32_t prim,
                                              MGFeatureValue* values,
                                              uint32_t n) {
  // For order-statistic primitives we need a sorted array.
  const bool needs_sort =
      (prim == kAggMedian || prim == kAggMode || prim == kAggNumUnique ||
       prim == kAggEntropy || prim == kAggQuarter || prim == kAggQuartile3);

  if (needs_sort) {
    // Pad unused entries with a value larger than any real value.
    if (threadIdx.x >= n && threadIdx.x < blockDim.x) {
      values[threadIdx.x] = MakeFloatValue(DBL_MAX);
    }
    __syncthreads();
    BlockBitonicSort(values);
    __syncthreads();
  }

  switch (prim) {
    case kAggCount:                 return AggCount(values, n);
    case kAggSum:                   return AggSum(values, n);
    case kAggMean:                  return AggMean(values, n);
    case kAggMin:                   return AggMin(values, n);
    case kAggMax:                   return AggMax(values, n);
    case kAggVariance:              return AggVariance(values, n);
    case kAggStd:                   return AggStd(values, n);
    case kAggMedian:                return AggMedian(values, n);
    case kAggMode:                  return AggMode(values, n);
    case kAggNumUnique:             return AggNumUnique(values, n);
    case kAggEntropy:               return AggEntropy(values, n);
    case kAggQuarter:               return AggQuarter(values, n);
    case kAggQuartile3:             return AggQuartile3(values, n);
    case kAggPercentTrue:           return AggPercentTrue(values, n);
    case kAggSkew:                  return AggSkew(values, n);
    case kAggCountGreaterThanMean:  return AggCountGreaterThanMean(values, n);
    default:                        return MakeInvalidValue();
  }
}

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_GPU_TASK_KERNEL_COMPUTE_FEATURES_PRIMITIVES_CUH_
