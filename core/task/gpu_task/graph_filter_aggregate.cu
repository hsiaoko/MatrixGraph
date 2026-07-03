#include "core/task/gpu_task/graph_filter_aggregate.cuh"

#include <cfloat>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "core/common/consts.h"
#include "core/common/types.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/task/gpu_task/execute_agg_prim.cuh"
#include "core/util/cuda_check.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

namespace {

using FeatureValue = sics::matrixgraph::core::task::FeatureValue;
using ValueType = sics::matrixgraph::core::data_structures::ValueType;
using Attribute = sics::matrixgraph::core::data_structures::Attribute;
using Attributes = sics::matrixgraph::core::data_structures::Attributes;
using AttributeName = sics::matrixgraph::core::data_structures::AttributeName;

using VertexID = sics::matrixgraph::core::common::VertexID;
using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;

__device__ __forceinline__ double AttrToDouble(const Attribute* attr, uint32_t row) {
  if (!attr) return 0.0;
  switch (attr->type) {
    case ValueType::kInt:
    case ValueType::kTime:
      return static_cast<double>(sics::matrixgraph::core::data_structures::GetInt(*attr, row));
    case ValueType::kFloat64:
      return sics::matrixgraph::core::data_structures::GetFloat64(*attr, row);
    case ValueType::kFloat32:
      return static_cast<double>(sics::matrixgraph::core::data_structures::GetFloat32(*attr, row));
    case ValueType::kBool:
      return sics::matrixgraph::core::data_structures::GetBool(*attr, row) ? 1.0 : 0.0;
    default:
      return 0.0;
  }
}

__device__ const Attribute* FindAttr(const Attributes* attrs,
                                     const AttributeName& name) {
  // A single shared Attributes table holds one descriptor per attribute column;
  // the per-vertex value is read by row (= vertex id) via AttrToDouble.
  return attrs->attr_map.find(name);
}

// Fixed-length string support: each string attribute column is stored as an
// array of StringView descriptors pointing at 64-byte fixed-length char arrays.
// Constants carry their 64-byte string inline in FilterOperand::string_val.

__device__ __forceinline__ bool IsStringOperand(const FilterOperand& op,
                                                const Attributes* attrs) {
  if (op.kind == FilterOperand::Kind::kStringConst) return true;
  if (op.kind == FilterOperand::Kind::kAttr ||
      op.kind == FilterOperand::Kind::kPatternAttr) {
    const Attribute* attr = FindAttr(attrs, op.attr_name);
    return attr != nullptr && attr->type == ValueType::kString;
  }
  if (op.kind == FilterOperand::Kind::kSubtract ||
      op.kind == FilterOperand::Kind::kAdd ||
      op.kind == FilterOperand::Kind::kMultiply ||
      op.kind == FilterOperand::Kind::kDivide) {
    const Attribute* attr = FindAttr(attrs, op.attr_name);
    return attr != nullptr && attr->type == ValueType::kString;
  }
  return false;
}

__device__ __forceinline__ void ReadOperandString(const FilterOperand& op,
                                                  const Attributes* attrs,
                                                  uint32_t pivot_id,
                                                  uint32_t neighbor_id,
                                                  char out[64]) {
  for (int i = 0; i < 64; ++i) out[i] = '\0';
  if (op.kind == FilterOperand::Kind::kStringConst) {
    for (int i = 0; i < 64; ++i) out[i] = op.string_val[i];
    return;
  }
  uint32_t row = pivot_id;
  if (op.kind == FilterOperand::Kind::kPatternAttr) {
    row = neighbor_id;
  } else if (op.kind == FilterOperand::Kind::kSubtract ||
             op.kind == FilterOperand::Kind::kAdd ||
             op.kind == FilterOperand::Kind::kMultiply ||
             op.kind == FilterOperand::Kind::kDivide) {
    row = op.pattern_position >= 0 ? neighbor_id : pivot_id;
  }
  const Attribute* attr = FindAttr(attrs, op.attr_name);
  if (attr && attr->type == ValueType::kString) {
    StringView sv = sics::matrixgraph::core::data_structures::GetString(*attr, row);
    uint32_t n = sv.len < 64 ? sv.len : 63;
    for (uint32_t i = 0; i < n; ++i) out[i] = sv.data[i];
    out[n] = '\0';
  }
}

__device__ __forceinline__ bool StringEqual(const char* a, const char* b) {
  for (int i = 0; i < 64; ++i) {
    if (a[i] != b[i]) return false;
    if (a[i] == '\0') return true;
  }
  return true;
}

__device__ __forceinline__ size_t StringLen(const char* s) {
  size_t len = 0;
  for (int i = 0; i < 64 && s[i] != '\0'; ++i) ++len;
  return len;
}

__device__ float lev_jaro_ratio(const char* term_l, const char* term_r) {
  size_t i, j, halflen, trans, match, to;
  size_t len_l = StringLen(term_l);
  size_t len_r = StringLen(term_r);

  float result = 0;
  float md;
  if (len_r == 0 || len_l == 0) {
    if (len_l == 0 && len_r == 0) return 1.0f;
    return 0.0f;
  }

  const char* a = term_l;
  const char* b = term_r;
  if (len_l > len_r) {
    const char* t = a; a = b; b = t;
    i = len_l; len_l = len_r; len_r = i;
  }

  halflen = (len_l + 1) / 2;
  uint16_t idx[64] = {0};
  match = 0;

  for (i = 0; i < halflen; ++i) {
    for (j = 0; j <= i + halflen && j < len_r; ++j) {
      if (j >= 64) break;
      if (a[j] == b[i] && !idx[j]) {
        ++match;
        idx[j] = match;
        break;
      }
    }
  }
  to = len_l + halflen < len_r ? len_l + halflen : len_r;
  for (i = halflen; i < to; ++i) {
    size_t start = (i > halflen) ? (i - halflen) : 0;
    for (j = start; j < len_l; ++j) {
      if (j >= 64) break;
      if (a[j] == b[i] && !idx[j]) {
        ++match;
        idx[j] = match;
        break;
      }
    }
  }

  if (!match) {
    result = 0.0f;
  } else {
    i = 0;
    trans = 0;
    for (j = 0; j < len_l; ++j) {
      if (idx[j]) {
        ++i;
        if (idx[j] != i) ++trans;
      }
    }
    md = static_cast<float>(match);
    result = (md / len_l + md / len_r + (md - trans / 2.0f) / md) / 3.0f;
  }
  return result;
}

__device__ double lev_jaro_winkler_ratio(const char* string1,
                                         const char* string2,
                                         double pfweight = 0.1) {
  double j = lev_jaro_ratio(string1, string2);
  size_t p = 0;
  size_t len1 = StringLen(string1);
  size_t len2 = StringLen(string2);
  size_t m = len1 < len2 ? len1 : len2;
  for (p = 0; p < m; ++p) {
    if (string1[p] != string2[p]) break;
  }
  j += (1.0 - j) * p * pfweight;
  return j > 1.0 ? 1.0 : j;
}

__device__ float jaccard_kernel(const char* term_l, const char* term_r) {
  char c_diff = 'a' - 'A';
  bool bm_l[256] = {false};
  bool bm_r[256] = {false};
  for (int i = 0; term_l[i] && i < 64; ++i) {
    int val = term_l[i];
    if (term_l[i] - 'Z' <= 0) val += c_diff;
    if (val >= 0 && val < 256) bm_l[val] = true;
  }
  for (int i = 0; term_r[i] && i < 64; ++i) {
    int val = term_r[i];
    if (term_r[i] - 'Z' <= 0) val += c_diff;
    if (val >= 0 && val < 256) bm_r[val] = true;
  }
  int count_l = 0, count_r = 0;
  for (int i = 0; i < 256; ++i) {
    count_l += bm_l[i];
    count_r += bm_r[i];
  }
  if (count_r == 0) return 0.0f;
  return static_cast<float>(count_l) / static_cast<float>(count_r);
}

__device__ __forceinline__ double EvalOperand(const FilterOperand& op,
                                              const Attributes* attrs,
                                              uint32_t pivot_id,
                                              uint32_t neighbor_id) {
  switch (op.kind) {
    case FilterOperand::Kind::kConst: {
      if (op.const_type == ValueType::kFloat64) return op.const_f64;
      return static_cast<double>(op.const_i64);
    }
    case FilterOperand::Kind::kStringConst:
      return 0.0;
    case FilterOperand::Kind::kAttr: {
      const Attribute* attr = FindAttr(attrs, op.attr_name);
      return AttrToDouble(attr, pivot_id);
    }
    case FilterOperand::Kind::kPatternAttr: {
      const Attribute* attr = FindAttr(attrs, op.attr_name);
      return AttrToDouble(attr, neighbor_id);
    }
    case FilterOperand::Kind::kSubtract:
    case FilterOperand::Kind::kAdd:
    case FilterOperand::Kind::kMultiply:
    case FilterOperand::Kind::kDivide: {
      uint32_t left_row = op.pattern_position >= 0 ? neighbor_id : pivot_id;
      uint32_t right_row = op.sub_pattern_position >= 0 ? neighbor_id : pivot_id;
      const Attribute* la = FindAttr(attrs, op.attr_name);
      double left = AttrToDouble(la, left_row);
      const Attribute* ra = FindAttr(attrs, op.sub_attr_name);
      double right = AttrToDouble(ra, right_row);
      switch (op.kind) {
        case FilterOperand::Kind::kAdd:      return left + right;
        case FilterOperand::Kind::kMultiply: return left * right;
        case FilterOperand::Kind::kDivide:   return left / right;
        default:                             return left - right;  // kSubtract
      }
    }
  }
  return 0.0;
}

__device__ __forceinline__ bool ApplyOp(FilterCondition::Op op, double left,
                                        double right) {
  switch (op) {
    case FilterCondition::Op::kEq:  return left == right;
    case FilterCondition::Op::kNeq: return left != right;
    case FilterCondition::Op::kGt:  return left > right;
    case FilterCondition::Op::kGte: return left >= right;
    case FilterCondition::Op::kLt:  return left < right;
    case FilterCondition::Op::kLte: return left <= right;
  }
  return false;
}

__device__ __forceinline__ bool AttrPresentAt(const Attributes* attrs,
                                              const AttributeName& name,
                                              uint32_t row) {
  const Attribute* a = FindAttr(attrs, name);
  return a != nullptr &&
         sics::matrixgraph::core::data_structures::IsValidAt(*a, row);
}

// An operand is "present" when the attributes it reads all hold valid values.
// Constants are always present. The CPU treats an invalid/missing operand as
// making the whole comparison false, so callers reject conditions with an
// absent operand.
__device__ __forceinline__ bool OperandPresent(const FilterOperand& op,
                                               const Attributes* attrs,
                                               uint32_t pivot_id,
                                               uint32_t neighbor_id) {
  switch (op.kind) {
    case FilterOperand::Kind::kConst:
      return true;
    case FilterOperand::Kind::kAttr:
      return AttrPresentAt(attrs, op.attr_name, pivot_id);
    case FilterOperand::Kind::kPatternAttr:
      return AttrPresentAt(attrs, op.attr_name, neighbor_id);
    case FilterOperand::Kind::kSubtract:
    case FilterOperand::Kind::kAdd:
    case FilterOperand::Kind::kMultiply: {
      uint32_t lrow = op.pattern_position >= 0 ? neighbor_id : pivot_id;
      uint32_t rrow = op.sub_pattern_position >= 0 ? neighbor_id : pivot_id;
      return AttrPresentAt(attrs, op.attr_name, lrow) &&
             AttrPresentAt(attrs, op.sub_attr_name, rrow);
    }
    case FilterOperand::Kind::kDivide: {
      // Matches the CPU Divide transform: a division by (near) zero yields an
      // invalid value, so treat the operand as absent when |right| < 1e-6.
      uint32_t lrow = op.pattern_position >= 0 ? neighbor_id : pivot_id;
      uint32_t rrow = op.sub_pattern_position >= 0 ? neighbor_id : pivot_id;
      if (!AttrPresentAt(attrs, op.attr_name, lrow) ||
          !AttrPresentAt(attrs, op.sub_attr_name, rrow))
        return false;
      const Attribute* ra = FindAttr(attrs, op.sub_attr_name);
      return fabs(AttrToDouble(ra, rrow)) >= 1e-6;
    }
  }
  return true;
}

__device__ __forceinline__ bool ApplyStringOp(FilterCondition::Op op,
                                               const char left[64],
                                               const char right[64]) {
  switch (op) {
    case FilterCondition::Op::kEq:  return StringEqual(left, right);
    case FilterCondition::Op::kNeq: return !StringEqual(left, right);
    // Similarity predicates are not yet emitted by the host path; they default
    // to a 0.5 threshold so the helper functions are exercised once wired.
    case FilterCondition::Op::kJaro:        return lev_jaro_ratio(left, right) > 0.5f;
    case FilterCondition::Op::kJaroWinkler: return lev_jaro_winkler_ratio(left, right) > 0.5;
    case FilterCondition::Op::kJaccard:     return jaccard_kernel(left, right) > 0.5f;
    default: return false;
  }
}

__device__ __forceinline__ bool EvaluateConditions(
    const FilterAggRequest& req,
    const Attributes* attrs,
    uint32_t pivot_id,
    uint32_t neighbor_id) {
  for (uint32_t i = 0; i < req.n_conditions; ++i) {
    const FilterCondition& c = req.conditions[i];
    // A missing operand makes the comparison false (matches the CPU path).
    if (!OperandPresent(c.left, attrs, pivot_id, neighbor_id) ||
        !OperandPresent(c.right, attrs, pivot_id, neighbor_id)) {
      return false;
    }
    // String operands bypass numeric evaluation and are compared byte-by-byte.
    if (IsStringOperand(c.left, attrs) || IsStringOperand(c.right, attrs)) {
      char left_str[64];
      char right_str[64];
      ReadOperandString(c.left, attrs, pivot_id, neighbor_id, left_str);
      ReadOperandString(c.right, attrs, pivot_id, neighbor_id, right_str);
      if (!ApplyStringOp(c.op, left_str, right_str)) return false;
      continue;
    }
    double l = EvalOperand(c.left, attrs, pivot_id, neighbor_id);
    double r = EvalOperand(c.right, attrs, pivot_id, neighbor_id);
    if (!ApplyOp(c.op, l, r)) return false;
  }
  return true;
}

__device__ __forceinline__ FeatureValue AttrToFeatureValue(const Attribute* attr,
                                                           uint32_t row) {
  if (!attr) return MakeInvalidValue();
  FeatureValue v;
  v.type = attr->type;
  switch (attr->type) {
    case ValueType::kInt:
    case ValueType::kTime:
      v.i64 = sics::matrixgraph::core::data_structures::GetInt(*attr, row);
      break;
    case ValueType::kFloat64:
      v.f64 = sics::matrixgraph::core::data_structures::GetFloat64(*attr, row);
      break;
    case ValueType::kFloat32:
      v.f32 = sics::matrixgraph::core::data_structures::GetFloat32(*attr, row);
      break;
    case ValueType::kBool:
      v.b = sics::matrixgraph::core::data_structures::GetBool(*attr, row);
      break;
    default:
      v.type = ValueType::kInvalid;
      break;
  }
  return v;
}

// Warp reduce helpers.
__device__ __forceinline__ double WarpSum(double val) {
  for (int o = 16; o > 0; o >>= 1) val += __shfl_down_sync(0xffffffffu, val, o);
  return val;
}

__device__ __forceinline__ double BlockSum(double val, double* sdata) {
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  val = WarpSum(val);
  if (lane == 0) sdata[wid] = val;
  __syncthreads();
  if (threadIdx.x == 0) {
    int nwarps = (blockDim.x + 31) >> 5;
    double r = 0.0;
    for (int i = 0; i < nwarps; ++i) r += sdata[i];
    sdata[0] = r;
  }
  __syncthreads();
  double r = sdata[0];
  __syncthreads();
  return r;
}

// NumUnique via per-block hash table in global memory.
static const unsigned long long kHashEmpty = 0xFFFFFFFFFFFFFFFFull;

__device__ __forceinline__ unsigned long long FVHashKey(double v) {
  return static_cast<unsigned long long>(__double_as_longlong(v));
}
__device__ __forceinline__ unsigned long long HashMix64(unsigned long long x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// 64-bit FNV-1a hash over a raw byte span.  Never returns kHashEmpty so the
// per-block hash table can use that value as its "empty" sentinel.
__device__ __forceinline__ unsigned long long StringHash64(const char* data,
                                                           uint32_t len) {
  const unsigned long long kFNVOffset = 14695981039346656037ull;
  const unsigned long long kFNVPrime = 1099511628211ull;
  unsigned long long h = kFNVOffset;
  for (uint32_t i = 0; i < len && i < 64; ++i) {
    h ^= static_cast<unsigned char>(data[i]);
    h *= kFNVPrime;
  }
  return h == kHashEmpty ? 0ull : h;
}

}  // namespace

// -----------------------------------------------------------------------------
// Device kernel: one block per request.
// -----------------------------------------------------------------------------
__launch_bounds__(common::kFilterAggBlockDim)
__global__ void FilterAggKernel(const EdgeIndex* csr_offsets,
                                const VertexID* csr_edges,
                                const uint32_t* edge_labels,
                                const uint32_t* vertex_labels,
                                const Attributes* vertex_attrs,
                                const FilterAggRequest* requests,
                                uint32_t n_requests,
                                unsigned long long* hash_scratch,
                                const uint32_t* hash_offsets,
                                FeatureValue* outputs) {
  uint32_t req_idx = blockIdx.x;
  if (req_idx >= n_requests) return;

  const FilterAggRequest& req = requests[req_idx];
  uint32_t pivot = req.pivot_vertex_id;
  bool outgoing = req.use_outgoing;

  // Incoming edges require a separate in-offsets array; for this version we only
  // support outgoing edges, which the host path validates before dispatch.
  (void)outgoing;
  EdgeIndex begin = csr_offsets[pivot];
  EdgeIndex end = csr_offsets[pivot + 1];

  __shared__ double sd[32];
  __shared__ int si[32];
  __shared__ unsigned int s_count;
  __shared__ int s_has_empty;

  // Per-block hash table for NumUnique.
  unsigned long long* table = nullptr;
  uint32_t cap = 0;
  uint32_t mask = 0;
  if (hash_scratch && hash_offsets) {
    cap = hash_offsets[req_idx + 1] - hash_offsets[req_idx];
    table = hash_scratch + hash_offsets[req_idx];
    mask = cap - 1;
  }
  if (threadIdx.x == 0) { s_count = 0; s_has_empty = 0; }
  for (uint32_t i = threadIdx.x; i < cap; i += blockDim.x) table[i] = kHashEmpty;
  __syncthreads();

  AggPrim prim = static_cast<AggPrim>(req.agg_prim);
  const AttributeName& agg_name = req.agg_attr_name;

  double sum = 0.0;
  double minv = DBL_MAX;
  double maxv = -DBL_MAX;
  int min_idx = -1;
  int max_idx = -1;
  uint32_t count = 0;
  uint32_t true_count = 0;

  for (EdgeIndex eidx = begin + threadIdx.x; eidx < end; eidx += blockDim.x) {
    VertexID nb = csr_edges[eidx];
    if (req.edge_label != 0 && edge_labels && edge_labels[eidx] != req.edge_label)
      continue;
    if (req.target_vertex_label != 0 && vertex_labels &&
        vertex_labels[nb] != req.target_vertex_label)
      continue;

    if (!EvaluateConditions(req, vertex_attrs, pivot, nb)) continue;

    const Attribute* attr = FindAttr(vertex_attrs, agg_name);
    // The CPU drops neighbors whose source value is missing/invalid; do the
    // same so count/sum/min/max/num-unique match.
    if (!attr ||
        !sics::matrixgraph::core::data_structures::IsValidAt(*attr, nb))
      continue;
    double v = AttrToDouble(attr, nb);
    ValueType vtype = attr->type;

    // Count before potential invalid.
    ++count;
    if (vtype == ValueType::kBool && v != 0.0) ++true_count;

    sum += v;
    if (v < minv) { minv = v; min_idx = static_cast<int>(eidx); }
    if (v > maxv) { maxv = v; max_idx = static_cast<int>(eidx); }

    // Insert into per-block hash table for NumUnique.
    if (cap > 0) {
      unsigned long long key;
      if (vtype == ValueType::kString) {
        StringView sv = sics::matrixgraph::core::data_structures::GetString(*attr, nb);
        key = StringHash64(sv.data, sv.len);
      } else {
        key = FVHashKey(v);
      }
      if (key == kHashEmpty) {
        atomicExch(&s_has_empty, 1);
      } else {
        uint32_t slot = static_cast<uint32_t>(HashMix64(key) & mask);
        while (true) {
          unsigned long long old = atomicCAS(&table[slot], kHashEmpty, key);
          if (old == kHashEmpty) { atomicAdd(&s_count, 1u); break; }
          if (old == key) break;
          slot = (slot + 1u) & mask;
        }
      }
    }
  }

  // Block reductions.
  double total_sum = BlockSum(sum, sd);
  uint32_t total_count = static_cast<uint32_t>(BlockSum(static_cast<double>(count), sd) + 0.5);
  uint32_t total_true = static_cast<uint32_t>(BlockSum(static_cast<double>(true_count), sd) + 0.5);

  // Min / max: argmin/argmax across the whole block (every neighbor), not just
  // one warp. First reduce within each warp, then reduce the per-warp winners
  // in shared memory.
  __shared__ double s_minv[32];
  __shared__ int s_mini[32];
  __shared__ double s_maxv[32];
  __shared__ int s_maxi[32];
  {
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    double wminv = minv;
    int wmini = min_idx;
    double wmaxv = maxv;
    int wmaxi = max_idx;
    for (int o = 16; o > 0; o >>= 1) {
      double ov = __shfl_down_sync(0xffffffffu, wminv, o);
      int oi = __shfl_down_sync(0xffffffffu, wmini, o);
      if (ov < wminv || (ov == wminv && oi < wmini)) { wminv = ov; wmini = oi; }
      double xv = __shfl_down_sync(0xffffffffu, wmaxv, o);
      int xi = __shfl_down_sync(0xffffffffu, wmaxi, o);
      if (xv > wmaxv || (xv == wmaxv && xi < wmaxi)) { wmaxv = xv; wmaxi = xi; }
    }
    if (lane == 0) {
      s_minv[wid] = wminv; s_mini[wid] = wmini;
      s_maxv[wid] = wmaxv; s_maxi[wid] = wmaxi;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int nwarps = (blockDim.x + 31) >> 5;
    double mn = s_minv[0]; int mni = s_mini[0];
    double mx = s_maxv[0]; int mxi = s_maxi[0];
    for (int w = 1; w < nwarps; ++w) {
      if (s_minv[w] < mn || (s_minv[w] == mn && s_mini[w] < mni)) { mn = s_minv[w]; mni = s_mini[w]; }
      if (s_maxv[w] > mx || (s_maxv[w] == mx && s_maxi[w] < mxi)) { mx = s_maxv[w]; mxi = s_maxi[w]; }
    }
    si[0] = mni;
    si[1] = mxi;
  }
  __syncthreads();
  int final_min_idx = si[0];
  int final_max_idx = si[1];

  // Second pass over the filtered neighbors: CountGreaterThanMean plus the
  // Variance/Std/Skew moment sums about the mean.
  double mean = (total_count > 0) ? total_sum / static_cast<double>(total_count) : 0.0;
  uint32_t gtm = 0;
  double sum_sq = 0.0;    // Sum (v-mean)^2
  double sum_cube = 0.0;  // Sum (v-mean)^3
  for (EdgeIndex eidx = begin + threadIdx.x; eidx < end; eidx += blockDim.x) {
    VertexID nb = csr_edges[eidx];
    if (req.edge_label != 0 && edge_labels && edge_labels[eidx] != req.edge_label)
      continue;
    if (req.target_vertex_label != 0 && vertex_labels &&
        vertex_labels[nb] != req.target_vertex_label)
      continue;
    if (!EvaluateConditions(req, vertex_attrs, pivot, nb)) continue;
    const Attribute* attr = FindAttr(vertex_attrs, agg_name);
    if (!attr ||
        !sics::matrixgraph::core::data_structures::IsValidAt(*attr, nb))
      continue;
    double v = AttrToDouble(attr, nb);
    if (v > mean) ++gtm;
    double d = v - mean;
    sum_sq += d * d;
    sum_cube += d * d * d;
  }
  uint32_t total_gtm = static_cast<uint32_t>(BlockSum(static_cast<double>(gtm), sd) + 0.5);
  double total_sq = BlockSum(sum_sq, sd);
  double total_cube = BlockSum(sum_cube, sd);

  if (threadIdx.x == 0) {
    FeatureValue r = MakeInvalidValue();
    // Empty result set: the CPU ExecuteAggPrim returns Invalid for every
    // primitive when no neighbor passes the filter (including Count), so leave
    // r invalid and skip the switch.
    if (total_count > 0)
    switch (prim) {
      case AggPrim::kCount:
        r = MakeIntValue(static_cast<int64_t>(total_count));
        break;
      case AggPrim::kSum:
        r = MakeFloat64Value(total_sum);
        break;
      case AggPrim::kMean:
        r = (total_count > 0) ? MakeFloat64Value(total_sum / total_count) : MakeInvalidValue();
        break;
      case AggPrim::kMin:
        if (final_min_idx >= 0 &&
            static_cast<EdgeIndex>(final_min_idx) >= begin &&
            static_cast<EdgeIndex>(final_min_idx) < end) {
          VertexID nb = csr_edges[final_min_idx];
          const Attribute* attr = FindAttr(vertex_attrs, agg_name);
          r = AttrToFeatureValue(attr, nb);
        }
        break;
      case AggPrim::kMax:
        if (final_max_idx >= 0 &&
            static_cast<EdgeIndex>(final_max_idx) >= begin &&
            static_cast<EdgeIndex>(final_max_idx) < end) {
          VertexID nb = csr_edges[final_max_idx];
          const Attribute* attr = FindAttr(vertex_attrs, agg_name);
          r = AttrToFeatureValue(attr, nb);
        }
        break;
      case AggPrim::kNumUnique:
        r = MakeIntValue(static_cast<int64_t>(s_count + (s_has_empty ? 1u : 0u)));
        break;
      case AggPrim::kPercentTrue: {
        // Only meaningful for boolean attributes; match the CPU path which
        // yields invalid for non-bool aggregands.
        const Attribute* aattr = FindAttr(vertex_attrs, agg_name);
        bool is_bool = aattr && aattr->type == ValueType::kBool;
        r = (is_bool && total_count > 0)
                ? MakeFloat64Value(static_cast<double>(total_true) / total_count)
                : MakeInvalidValue();
        break;
      }
      case AggPrim::kCountGreaterThanMean:
        r = MakeIntValue(static_cast<int64_t>(total_gtm));
        break;
      case AggPrim::kVariance:
        // Population variance: Sum (v-mean)^2 / n.
        r = MakeFloat64Value(total_sq / total_count);
        break;
      case AggPrim::kStd:
        r = MakeFloat64Value(sqrt(total_sq / total_count));
        break;
      case AggPrim::kSkew: {
        // Population skew: Sum ((v-mean)/std)^3 / n.  The CPU divides each term
        // by std, so a (near) zero std makes the whole result invalid.
        double variance = total_sq / total_count;
        double std = sqrt(variance);
        r = (std < 1e-6)
                ? MakeInvalidValue()
                : MakeFloat64Value((total_cube / total_count) / (std * std * std));
        break;
      }
      default:
        r = MakeInvalidValue();
        break;
    }
    outputs[req_idx] = r;
  }
}

// -----------------------------------------------------------------------------
// Host implementation
// -----------------------------------------------------------------------------
GraphFilterAggregate::GraphFilterAggregate() = default;

GraphFilterAggregate::~GraphFilterAggregate() {
  DestroyStreams();
  FreeBuffers();
}

GraphFilterAggregate::GraphFilterAggregate(GraphFilterAggregate&& other) noexcept {
  *this = std::move(other);
}

GraphFilterAggregate& GraphFilterAggregate::operator=(
    GraphFilterAggregate&& other) noexcept {
    n_streams_ = other.n_streams_;
    streams_ = std::move(other.streams_);
    d_requests_ = std::move(other.d_requests_);
    requests_cap_ = std::move(other.requests_cap_);
    d_outputs_ = std::move(other.d_outputs_);
    outputs_cap_ = std::move(other.outputs_cap_);
    d_hash_scratch_ = std::move(other.d_hash_scratch_);
    d_hash_offsets_ = std::move(other.d_hash_offsets_);
    hash_scratch_cap_ = std::move(other.hash_scratch_cap_);
    hash_offsets_cap_ = std::move(other.hash_offsets_cap_);
    n_vertices_ = other.n_vertices_;
    n_edges_ = other.n_edges_;
    h_csr_offsets_ = std::move(other.h_csr_offsets_);
    h_csr_edges_ = std::move(other.h_csr_edges_);
    h_edge_labels_ = std::move(other.h_edge_labels_);
    h_vertex_labels_ = std::move(other.h_vertex_labels_);
    d_csr_offsets_ = other.d_csr_offsets_;
    d_csr_edges_ = other.d_csr_edges_;
    d_edge_labels_ = other.d_edge_labels_;
    d_vertex_labels_ = other.d_vertex_labels_;
    per_vertex_attrs_ = std::move(other.per_vertex_attrs_);
    d_vertex_attrs_ = other.d_vertex_attrs_;
    column_buffers_ = std::move(other.column_buffers_);

    other.d_csr_offsets_ = nullptr;
    other.d_csr_edges_ = nullptr;
    other.d_edge_labels_ = nullptr;
    other.d_vertex_labels_ = nullptr;
    other.d_vertex_attrs_ = nullptr;
    other.d_requests_.clear();
    other.requests_cap_.clear();
    other.d_outputs_.clear();
    other.outputs_cap_.clear();
    other.d_hash_scratch_.clear();
    other.d_hash_offsets_.clear();
    other.hash_scratch_cap_.clear();
    other.hash_offsets_cap_.clear();
}

__host__ void GraphFilterAggregate::SetNumStreams(uint32_t n_streams) {
  if (n_streams == 0) n_streams = 1;
  if (n_streams == n_streams_ && !streams_.empty()) return;
  DestroyStreams();
  // Clear per-stream buffer state so a stream-count change does not leave
  // stale pointers/caps behind.
  d_requests_.clear();
  d_outputs_.clear();
  d_hash_offsets_.clear();
  d_hash_scratch_.clear();
  requests_cap_.clear();
  outputs_cap_.clear();
  hash_offsets_cap_.clear();
  hash_scratch_cap_.clear();
  n_streams_ = n_streams;
}

__host__ void GraphFilterAggregate::EnsureStreams() {
  if (streams_.size() == n_streams_ && d_requests_.size() == n_streams_) return;
  DestroyStreams();
  streams_.resize(n_streams_);
  d_requests_.resize(n_streams_, nullptr);
  d_outputs_.resize(n_streams_, nullptr);
  d_hash_offsets_.resize(n_streams_, nullptr);
  d_hash_scratch_.resize(n_streams_, nullptr);
  requests_cap_.resize(n_streams_, 0);
  outputs_cap_.resize(n_streams_, 0);
  hash_offsets_cap_.resize(n_streams_, 0);
  hash_scratch_cap_.resize(n_streams_, 0);
  for (uint32_t i = 0; i < n_streams_; ++i) {
    CUDA_CHECK(cudaStreamCreate(&streams_[i]));
  }
}

__host__ void GraphFilterAggregate::DestroyStreams() {
  for (cudaStream_t s : streams_) {
    if (s) cudaStreamDestroy(s);
  }
  streams_.clear();
}

__host__ void GraphFilterAggregate::FreeBuffers() {
  if (d_csr_offsets_) cudaFree(d_csr_offsets_);
  if (d_csr_edges_) cudaFree(d_csr_edges_);
  if (d_edge_labels_) cudaFree(d_edge_labels_);
  if (d_vertex_labels_) cudaFree(d_vertex_labels_);
  if (d_vertex_attrs_) cudaFree(d_vertex_attrs_);
  if (d_all_conditions_) cudaFree(d_all_conditions_);
  for (FilterAggRequest* p : d_requests_) {
    if (p) cudaFree(p);
  }
  for (FeatureValue* p : d_outputs_) {
    if (p) cudaFree(p);
  }
  for (unsigned long long* p : d_hash_scratch_) {
    if (p) cudaFree(p);
  }
  for (uint32_t* p : d_hash_offsets_) {
    if (p) cudaFree(p);
  }
  for (uint8_t* p : column_buffers_) cudaFree(p);
  d_csr_offsets_ = nullptr;
  d_csr_edges_ = nullptr;
  d_edge_labels_ = nullptr;
  d_vertex_labels_ = nullptr;
  d_vertex_attrs_ = nullptr;
  d_all_conditions_ = nullptr;
  all_conditions_cap_ = 0;
  d_requests_.clear();
  d_outputs_.clear();
  d_hash_scratch_.clear();
  d_hash_offsets_.clear();
  requests_cap_.clear();
  outputs_cap_.clear();
  hash_offsets_cap_.clear();
  hash_scratch_cap_.clear();
  column_buffers_.clear();
  per_vertex_attrs_.clear();
}

__host__ void GraphFilterAggregate::LoadGraphCSR(uint32_t n_vertices,
                                                 uint32_t n_edges,
                                                 const uint32_t* csr_offsets,
                                                 const uint32_t* csr_edges,
                                                 const uint32_t* edge_labels,
                                                 const uint32_t* vertex_labels) {
  FreeBuffers();
  n_vertices_ = n_vertices;
  n_edges_ = n_edges;

  h_csr_offsets_.resize(n_vertices + 1);
  h_csr_edges_.resize(n_edges);
  h_edge_labels_.resize(edge_labels ? n_edges : 0);
  h_vertex_labels_.resize(vertex_labels ? n_vertices : 0);

  std::memcpy(h_csr_offsets_.data(), csr_offsets, sizeof(uint32_t) * (n_vertices + 1));
  std::memcpy(h_csr_edges_.data(), csr_edges, sizeof(uint32_t) * n_edges);
  if (edge_labels) {
    std::memcpy(h_edge_labels_.data(), edge_labels, sizeof(uint32_t) * n_edges);
  }
  if (vertex_labels) {
    std::memcpy(h_vertex_labels_.data(), vertex_labels, sizeof(uint32_t) * n_vertices);
  }

  CUDA_CHECK(cudaMalloc(&d_csr_offsets_, sizeof(uint32_t) * (n_vertices + 1)));
  CUDA_CHECK(cudaMalloc(&d_csr_edges_, sizeof(uint32_t) * n_edges));
  CUDA_CHECK(cudaMemcpy(d_csr_offsets_, h_csr_offsets_.data(),
                        sizeof(uint32_t) * (n_vertices + 1),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_csr_edges_, h_csr_edges_.data(),
                        sizeof(uint32_t) * n_edges,
                        cudaMemcpyHostToDevice));

  if (edge_labels) {
    CUDA_CHECK(cudaMalloc(&d_edge_labels_, sizeof(uint32_t) * n_edges));
    CUDA_CHECK(cudaMemcpy(d_edge_labels_, h_edge_labels_.data(),
                          sizeof(uint32_t) * n_edges,
                          cudaMemcpyHostToDevice));
  }
  if (vertex_labels) {
    CUDA_CHECK(cudaMalloc(&d_vertex_labels_, sizeof(uint32_t) * n_vertices));
    CUDA_CHECK(cudaMemcpy(d_vertex_labels_, h_vertex_labels_.data(),
                          sizeof(uint32_t) * n_vertices,
                          cudaMemcpyHostToDevice));
  }
}

__host__ void GraphFilterAggregate::BuildAttributesFromColumns(
    uint32_t n_columns,
    const GraphAggregateAttributeColumn* columns) {
  // Build a single shared Attributes table: one descriptor per attribute
  // column, each pointing at a full per-vertex column buffer.
  // columns[i].values is a host pointer to n_vertices entries.
  if (n_columns == 0 || n_vertices_ == 0) return;

  std::vector<AttributeName> names(n_columns);
  std::vector<Attribute> attrs(n_columns);
  for (uint32_t c = 0; c < n_columns; ++c) {
    names[c] = AttributeName(reinterpret_cast<const char*>(&columns[c].key));  // 64-bit key as name
    attrs[c].type = static_cast<ValueType>(columns[c].value_type);
    attrs[c].n_rows = n_vertices_;
    attrs[c].n_elements = n_vertices_;
    attrs[c].offsets = nullptr;

    size_t elem_size = 0;
    switch (attrs[c].type) {
      case ValueType::kInt:
      case ValueType::kTime:
        elem_size = sizeof(int64_t);
        break;
      case ValueType::kFloat64:
        elem_size = sizeof(double);
        break;
      case ValueType::kFloat32:
        elem_size = sizeof(float);
        break;
      case ValueType::kBool:
        elem_size = sizeof(uint8_t);
        break;
      case ValueType::kString:
        elem_size = sizeof(StringView);
        break;
      default:
        elem_size = 0;
        break;
    }

    if (elem_size > 0) {
      if (attrs[c].type == ValueType::kString) {
        // String columns from the host are fixed 64-byte char arrays; build a
        // parallel StringView descriptor array that points into a device char
        // buffer so the existing GetString accessor works unchanged. Lengths are
        // computed on the host before the H2D copy.
        const char* h_chars = reinterpret_cast<const char*>(columns[c].values);
        uint8_t* d_chars = nullptr;
        CUDA_CHECK(cudaMalloc(&d_chars, 64 * n_vertices_));
        CUDA_CHECK(cudaMemcpy(d_chars, h_chars, 64 * n_vertices_,
                              cudaMemcpyHostToDevice));

        std::vector<StringView> h_views(n_vertices_);
        const char* d_char_ptr = reinterpret_cast<const char*>(d_chars);
        for (uint32_t r = 0; r < n_vertices_; ++r) {
          uint32_t len = 0;
          for (; len < 64 && h_chars[r * 64 + len] != '\0'; ++len) {}
          h_views[r].data = d_char_ptr + r * 64;
          h_views[r].len = len;
        }
        StringView* d_views = nullptr;
        CUDA_CHECK(cudaMalloc(&d_views, sizeof(StringView) * n_vertices_));
        CUDA_CHECK(cudaMemcpy(d_views, h_views.data(),
                              sizeof(StringView) * n_vertices_,
                              cudaMemcpyHostToDevice));

        attrs[c].data = d_views;
        column_buffers_.push_back(reinterpret_cast<uint8_t*>(d_views));
        column_buffers_.push_back(d_chars);
      } else {
        uint8_t* d_buf = nullptr;
        CUDA_CHECK(cudaMalloc(&d_buf, elem_size * n_vertices_));
        CUDA_CHECK(cudaMemcpy(d_buf, columns[c].values, elem_size * n_vertices_,
                              cudaMemcpyHostToDevice));
        attrs[c].data = d_buf;
        column_buffers_.push_back(d_buf);
      }

      if (columns[c].valid) {
        uint8_t* d_valid = nullptr;
        CUDA_CHECK(cudaMalloc(&d_valid, sizeof(uint8_t) * n_vertices_));
        CUDA_CHECK(cudaMemcpy(d_valid, columns[c].valid,
                              sizeof(uint8_t) * n_vertices_,
                              cudaMemcpyHostToDevice));
        attrs[c].valid = d_valid;
        column_buffers_.push_back(d_valid);
      }
    } else {
      attrs[c].data = nullptr;
    }
  }

  // Build a single shared DeviceAttributes: one hash map of attribute name ->
  // full column descriptor. Per-vertex values are read by row (= vertex id), so
  // we do not need (and cannot afford) one Attributes per vertex.
  per_vertex_attrs_.emplace_back(0u, names.data(), attrs.data(), n_columns);

  Attributes h_view = per_vertex_attrs_[0].View();
  CUDA_CHECK(cudaMalloc(&d_vertex_attrs_, sizeof(Attributes)));
  CUDA_CHECK(cudaMemcpy(d_vertex_attrs_, &h_view, sizeof(Attributes),
                        cudaMemcpyHostToDevice));
}

__host__ void GraphFilterAggregate::LoadVertexAttributes(
    uint32_t n_columns,
    const GraphAggregateAttributeColumn* columns) {
  BuildAttributesFromColumns(n_columns, columns);
}

__host__ void GraphFilterAggregate::EnsureRequestBuffers(uint32_t stream_idx,
                                                          uint32_t n_requests) {
  if (stream_idx >= n_streams_) EnsureStreams();
  if (n_requests <= requests_cap_[stream_idx]) return;
  if (d_requests_[stream_idx]) cudaFree(d_requests_[stream_idx]);
  if (d_outputs_[stream_idx]) cudaFree(d_outputs_[stream_idx]);
  CUDA_CHECK(
      cudaMalloc(&d_requests_[stream_idx], sizeof(FilterAggRequest) * n_requests));
  CUDA_CHECK(
      cudaMalloc(&d_outputs_[stream_idx], sizeof(FeatureValue) * n_requests));
  requests_cap_[stream_idx] = n_requests;
  outputs_cap_[stream_idx] = n_requests;
}

__host__ void GraphFilterAggregate::EnsureScratch(uint32_t stream_idx,
                                                  size_t hash_total) {
  if (stream_idx >= n_streams_) EnsureStreams();
  // hash_offsets is sized per request; we grow it lazily per stream.
  if (hash_offsets_cap_[stream_idx] < requests_cap_[stream_idx] + 1) {
    if (d_hash_offsets_[stream_idx]) cudaFree(d_hash_offsets_[stream_idx]);
    CUDA_CHECK(cudaMalloc(
        &d_hash_offsets_[stream_idx],
        sizeof(uint32_t) * (requests_cap_[stream_idx] + 1)));
    hash_offsets_cap_[stream_idx] = requests_cap_[stream_idx] + 1;
  }
  // hash_total is the exact number of hash-table slots needed for this chunk.
  if (hash_total == 0) hash_total = 1;
  if (hash_total > hash_scratch_cap_[stream_idx]) {
    if (d_hash_scratch_[stream_idx]) cudaFree(d_hash_scratch_[stream_idx]);
    CUDA_CHECK(cudaMalloc(
        &d_hash_scratch_[stream_idx],
        sizeof(unsigned long long) * hash_total));
    hash_scratch_cap_[stream_idx] = hash_total;
  }
}

__host__ std::vector<FeatureValue> GraphFilterAggregate::Compute(
    const std::vector<FilterAggRequest>& requests,
    const FilterCondition* all_conditions,
    size_t n_all_conditions) {
  std::vector<FeatureValue> results;
  if (requests.empty() || n_vertices_ == 0) return results;

  EnsureStreams();

  // Upload the shared conditions once. Each request's host pointer is replaced
  // below by a device pointer at the matching offset in d_all_conditions_.
  if (d_all_conditions_ == nullptr || all_conditions_cap_ < n_all_conditions) {
    if (d_all_conditions_) cudaFree(d_all_conditions_);
    all_conditions_cap_ = std::max(size_t(1), n_all_conditions);
    CUDA_CHECK(cudaMalloc(&d_all_conditions_,
                         sizeof(FilterCondition) * all_conditions_cap_));
  }
  if (n_all_conditions > 0) {
    CUDA_CHECK(cudaMemcpyAsync(
        d_all_conditions_, all_conditions,
        sizeof(FilterCondition) * n_all_conditions,
        cudaMemcpyHostToDevice, streams_[0]));
  }
  CUDA_CHECK(cudaStreamSynchronize(streams_[0]));

  // Precompute per-request hash-slot counts. Only NumUnique needs a hash table.
  std::vector<uint32_t> per_request_hash_cap(requests.size(), 0);
  for (size_t i = 0; i < requests.size(); ++i) {
    if (static_cast<AggPrim>(requests[i].agg_prim) == AggPrim::kNumUnique) {
      uint32_t pid = requests[i].pivot_vertex_id;
      uint32_t deg = 0;
      if (pid < n_vertices_) {
        deg = h_csr_offsets_[pid + 1] - h_csr_offsets_[pid];
      }
      per_request_hash_cap[i] = (deg == 0) ? 1u : NextPowerOfTwo(deg) * 2u;
    }
  }

  const uint32_t total_requests = static_cast<uint32_t>(requests.size());
  results.resize(requests.size());

  // Split requests into memory-bounded chunks. Peak memory is bounded by
  // n_streams * per-chunk budget; the shared conditions buffer is uploaded once
  // above and referenced by offset, so it no longer grows with chunk count.
  std::vector<std::pair<uint32_t, uint32_t>> chunks;
  {
    uint32_t begin = 0;
    while (begin < total_requests) {
      uint32_t end = begin;
      size_t slots = 0;
      while (end < total_requests) {
        uint32_t cap = per_request_hash_cap[end];
        if (end > begin &&
            ((end - begin + 1) > common::kFilterMaxChunkRequests ||
             slots + cap > common::kFilterMaxChunkHashSlots)) {
          break;
        }
        slots += cap;
        ++end;
      }
      chunks.emplace_back(begin, end);
      begin = end;
    }
  }

  for (size_t ci = 0; ci < chunks.size(); ++ci) {
    const uint32_t begin = chunks[ci].first;
    const uint32_t end = chunks[ci].second;
    const uint32_t count = end - begin;
    const uint32_t s = static_cast<uint32_t>(ci % n_streams_);

    if (ci >= n_streams_) {
      CUDA_CHECK(cudaStreamSynchronize(streams_[s]));
    }

    // Hash offset prefix sum for this chunk.
    std::vector<uint32_t> h_hash_offsets(count + 1, 0);
    size_t chunk_hash_total = 0;
    for (uint32_t i = 0; i < count; ++i) {
      uint32_t cap = per_request_hash_cap[begin + i];
      h_hash_offsets[i + 1] = h_hash_offsets[i] + cap;
      chunk_hash_total += cap;
    }

    EnsureRequestBuffers(s, count);
    EnsureScratch(s, chunk_hash_total);

    CUDA_CHECK(cudaMemcpyAsync(
        d_hash_offsets_[s], h_hash_offsets.data(),
        sizeof(uint32_t) * h_hash_offsets.size(), cudaMemcpyHostToDevice,
        streams_[s]));

    // Build device requests: translate host condition pointers into offsets into
    // the single shared device condition buffer.
    std::vector<FilterAggRequest> h_device_requests;
    h_device_requests.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      FilterAggRequest r = requests[begin + i];
      if (r.n_conditions > 0 && all_conditions && r.conditions) {
        size_t offset = r.conditions - all_conditions;
        r.conditions = d_all_conditions_ + offset;
      } else {
        r.conditions = nullptr;
      }
      h_device_requests.push_back(r);
    }
    CUDA_CHECK(cudaMemcpyAsync(
        d_requests_[s], h_device_requests.data(),
        sizeof(FilterAggRequest) * count, cudaMemcpyHostToDevice, streams_[s]));

    FilterAggKernel<<<count, common::kFilterAggBlockDim, 0, streams_[s]>>>(
        d_csr_offsets_, d_csr_edges_, d_edge_labels_, d_vertex_labels_,
        d_vertex_attrs_, d_requests_[s], count, d_hash_scratch_[s],
        d_hash_offsets_[s], d_outputs_[s]);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(
        results.data() + begin, d_outputs_[s],
        sizeof(FeatureValue) * count, cudaMemcpyDeviceToHost, streams_[s]));
  }

  for (uint32_t s = 0; s < n_streams_; ++s) {
    if (streams_[s]) {
      CUDA_CHECK(cudaStreamSynchronize(streams_[s]));
    }
  }

  return results;
}

__host__ void GraphFilterAggregate::Run() {
  std::cout << "[GraphFilterAggregate] Run()" << std::endl;
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
