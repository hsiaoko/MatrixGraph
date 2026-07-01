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

__device__ __forceinline__ double EvalOperand(const FilterOperand& op,
                                              const Attributes* attrs,
                                              uint32_t pivot_id,
                                              uint32_t neighbor_id) {
  switch (op.kind) {
    case FilterOperand::Kind::kConst: {
      if (op.const_type == ValueType::kFloat64) return op.const_f64;
      return static_cast<double>(op.const_i64);
    }
    case FilterOperand::Kind::kAttr: {
      const Attribute* attr = FindAttr(attrs, op.attr_name);
      return AttrToDouble(attr, pivot_id);
    }
    case FilterOperand::Kind::kPatternAttr: {
      const Attribute* attr = FindAttr(attrs, op.attr_name);
      return AttrToDouble(attr, neighbor_id);
    }
    case FilterOperand::Kind::kSubtract: {
      double left = 0.0, right = 0.0;
      uint32_t left_row = op.pattern_position >= 0 ? neighbor_id : pivot_id;
      uint32_t right_row = op.sub_pattern_position >= 0 ? neighbor_id : pivot_id;
      const Attribute* la = FindAttr(attrs, op.attr_name);
      left = AttrToDouble(la, left_row);
      const Attribute* ra = FindAttr(attrs, op.sub_attr_name);
      right = AttrToDouble(ra, right_row);
      return left - right;
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
    case FilterOperand::Kind::kSubtract: {
      uint32_t lrow = op.pattern_position >= 0 ? neighbor_id : pivot_id;
      uint32_t rrow = op.sub_pattern_position >= 0 ? neighbor_id : pivot_id;
      return AttrPresentAt(attrs, op.attr_name, lrow) &&
             AttrPresentAt(attrs, op.sub_attr_name, rrow);
    }
  }
  return true;
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

}  // namespace

// -----------------------------------------------------------------------------
// Device kernel: one block per request.
// -----------------------------------------------------------------------------
__launch_bounds__(256)
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
      unsigned long long key = FVHashKey(v);
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

  // CountGreaterThanMean requires a second pass.
  double mean = (total_count > 0) ? total_sum / static_cast<double>(total_count) : 0.0;
  uint32_t gtm = 0;
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
  }
  uint32_t total_gtm = static_cast<uint32_t>(BlockSum(static_cast<double>(gtm), sd) + 0.5);

  // Variance/Std/Skew need more passes; for the first cut keep only the cheap
  // primitives + NumUnique.  Others return invalid.
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
  if (this != &other) {
    DestroyStreams();
    FreeBuffers();
    n_streams_ = other.n_streams_;
    streams_ = std::move(other.streams_);
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
    d_requests_ = other.d_requests_;
    requests_cap_ = other.requests_cap_;
    d_outputs_ = other.d_outputs_;
    outputs_cap_ = other.outputs_cap_;
    d_hash_scratch_ = other.d_hash_scratch_;
    d_hash_offsets_ = other.d_hash_offsets_;
    hash_scratch_cap_ = other.hash_scratch_cap_;
    hash_offsets_cap_ = other.hash_offsets_cap_;

    other.d_csr_offsets_ = nullptr;
    other.d_csr_edges_ = nullptr;
    other.d_edge_labels_ = nullptr;
    other.d_vertex_labels_ = nullptr;
    other.d_vertex_attrs_ = nullptr;
    other.d_requests_ = nullptr;
    other.d_outputs_ = nullptr;
    other.d_hash_scratch_ = nullptr;
    other.d_hash_offsets_ = nullptr;
  }
  return *this;
}

__host__ void GraphFilterAggregate::SetNumStreams(uint32_t n_streams) {
  if (n_streams == 0) n_streams = 1;
  if (n_streams == n_streams_ && !streams_.empty()) return;
  DestroyStreams();
  n_streams_ = n_streams;
}

__host__ void GraphFilterAggregate::EnsureStreams() {
  if (streams_.size() == n_streams_) return;
  DestroyStreams();
  streams_.resize(n_streams_);
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
  if (d_requests_) cudaFree(d_requests_);
  if (d_outputs_) cudaFree(d_outputs_);
  if (d_hash_scratch_) cudaFree(d_hash_scratch_);
  if (d_hash_offsets_) cudaFree(d_hash_offsets_);
  for (uint8_t* p : column_buffers_) cudaFree(p);
  d_csr_offsets_ = nullptr;
  d_csr_edges_ = nullptr;
  d_edge_labels_ = nullptr;
  d_vertex_labels_ = nullptr;
  d_vertex_attrs_ = nullptr;
  d_requests_ = nullptr;
  d_outputs_ = nullptr;
  d_hash_scratch_ = nullptr;
  d_hash_offsets_ = nullptr;
  // Reset capacities so the Ensure* guards reallocate after a reload; otherwise
  // a second LoadGraphCSR would leave these pointers null while the stale caps
  // make EnsureRequestBuffers/EnsureScratch skip allocation.
  requests_cap_ = 0;
  outputs_cap_ = 0;
  hash_offsets_cap_ = 0;
  hash_scratch_cap_ = 0;
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
      default:
        elem_size = 0;
        break;
    }

    if (elem_size > 0) {
      uint8_t* d_buf = nullptr;
      CUDA_CHECK(cudaMalloc(&d_buf, elem_size * n_vertices_));
      CUDA_CHECK(cudaMemcpy(d_buf, columns[c].values, elem_size * n_vertices_,
                            cudaMemcpyHostToDevice));
      attrs[c].data = d_buf;
      column_buffers_.push_back(d_buf);

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

__host__ void GraphFilterAggregate::EnsureRequestBuffers(uint32_t n_requests) {
  if (n_requests <= requests_cap_) return;
  if (d_requests_) cudaFree(d_requests_);
  if (d_outputs_) cudaFree(d_outputs_);
  CUDA_CHECK(cudaMalloc(&d_requests_, sizeof(FilterAggRequest) * n_requests));
  CUDA_CHECK(cudaMalloc(&d_outputs_, sizeof(FeatureValue) * n_requests));
  requests_cap_ = n_requests;
  outputs_cap_ = n_requests;
}

__host__ void GraphFilterAggregate::EnsureScratch(uint32_t max_degree) {
  // hash_offsets is sized per request; we grow it lazily with requests_cap_.
  if (hash_offsets_cap_ < requests_cap_ + 1) {
    if (d_hash_offsets_) cudaFree(d_hash_offsets_);
    CUDA_CHECK(cudaMalloc(&d_hash_offsets_, sizeof(uint32_t) * (requests_cap_ + 1)));
    hash_offsets_cap_ = requests_cap_ + 1;
  }
  // Per-request hash table size = next_pow2(max_degree) * 2. The scratch holds
  // one such table per request, so the total it must cover is cap *
  // requests_cap_. Compare against that total (hash_scratch_cap_ stores the
  // total element count), not the per-request cap.
  uint32_t cap = (max_degree == 0) ? 1u : NextPowerOfTwo(max_degree) * 2u;
  size_t needed = static_cast<size_t>(cap) * requests_cap_;
  if (needed > hash_scratch_cap_) {
    if (d_hash_scratch_) cudaFree(d_hash_scratch_);
    CUDA_CHECK(cudaMalloc(&d_hash_scratch_, sizeof(unsigned long long) * needed));
    hash_scratch_cap_ = needed;
  }
}

__host__ std::vector<FeatureValue> GraphFilterAggregate::Compute(
    const std::vector<FilterAggRequest>& requests) {
  std::vector<FeatureValue> results;
  if (requests.empty() || n_vertices_ == 0) return results;

  EnsureStreams();
  EnsureRequestBuffers(static_cast<uint32_t>(requests.size()));

  // Compute per-request hash table offsets based on pivot degree.
  std::vector<uint32_t> h_hash_offsets(requests.size() + 1);
  h_hash_offsets[0] = 0;
  uint32_t max_degree = 0;
  for (size_t i = 0; i < requests.size(); ++i) {
    uint32_t pid = requests[i].pivot_vertex_id;
    uint32_t deg = 0;
    if (pid < n_vertices_) {
      deg = h_csr_offsets_[pid + 1] - h_csr_offsets_[pid];
    }
    max_degree = std::max(max_degree, deg);
    uint32_t cap = (deg == 0) ? 1u : NextPowerOfTwo(deg) * 2u;
    h_hash_offsets[i + 1] = h_hash_offsets[i] + cap;
  }
  EnsureScratch(max_degree);

  CUDA_CHECK(cudaMemcpy(d_hash_offsets_, h_hash_offsets.data(),
                        sizeof(uint32_t) * h_hash_offsets.size(),
                        cudaMemcpyHostToDevice));

  // Flatten all conditions and upload them in a single allocation + copy
  // (instead of one cudaMalloc/cudaMemcpy per request).
  std::vector<FilterCondition> flat_conditions;
  std::vector<uint32_t> cond_offsets(requests.size(), 0);
  for (size_t i = 0; i < requests.size(); ++i) {
    cond_offsets[i] = static_cast<uint32_t>(flat_conditions.size());
    for (uint32_t j = 0; j < requests[i].n_conditions; ++j) {
      flat_conditions.push_back(requests[i].conditions[j]);
    }
  }

  FilterCondition* d_all_conditions = nullptr;
  if (!flat_conditions.empty()) {
    CUDA_CHECK(cudaMalloc(&d_all_conditions,
                          sizeof(FilterCondition) * flat_conditions.size()));
    CUDA_CHECK(cudaMemcpy(d_all_conditions, flat_conditions.data(),
                          sizeof(FilterCondition) * flat_conditions.size(),
                          cudaMemcpyHostToDevice));
  }

  // Build the device request array with conditions pointing into the single
  // flattened buffer, then upload it in one copy.
  std::vector<FilterAggRequest> h_device_requests = requests;
  for (size_t i = 0; i < h_device_requests.size(); ++i) {
    h_device_requests[i].conditions =
        (requests[i].n_conditions > 0) ? (d_all_conditions + cond_offsets[i])
                                       : nullptr;
  }
  CUDA_CHECK(cudaMemcpy(d_requests_, h_device_requests.data(),
                        sizeof(FilterAggRequest) * requests.size(),
                        cudaMemcpyHostToDevice));

  FilterAggKernel<<<requests.size(), 256, 0, streams_[0]>>>(
      d_csr_offsets_, d_csr_edges_, d_edge_labels_, d_vertex_labels_,
      d_vertex_attrs_, d_requests_, static_cast<uint32_t>(requests.size()),
      d_hash_scratch_, d_hash_offsets_, d_outputs_);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(streams_[0]));

  results.resize(requests.size());
  CUDA_CHECK(cudaMemcpy(results.data(), d_outputs_,
                        sizeof(FeatureValue) * requests.size(),
                        cudaMemcpyDeviceToHost));

  if (d_all_conditions) cudaFree(d_all_conditions);
  return results;
}

__host__ void GraphFilterAggregate::Run() {
  std::cout << "[GraphFilterAggregate] Run()" << std::endl;
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
