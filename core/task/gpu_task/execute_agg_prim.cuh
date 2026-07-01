#ifndef MATRIXGRAPH_CORE_TASK_GPU_TASK_EXECUTE_AGG_PRIM_CUH_
#define MATRIXGRAPH_CORE_TASK_GPU_TASK_EXECUTE_AGG_PRIM_CUH_

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

#include "core/data_structures/attributes.h"
#include "core/task/gpu_task/task_base.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using ValueType = sics::matrixgraph::core::data_structures::ValueType;
using StringView = sics::matrixgraph::core::data_structures::StringView;

// -----------------------------------------------------------------------------
// Unified value container aligned with featurelib's attribute value types.
// -----------------------------------------------------------------------------
struct FeatureValue {
  ValueType type;
  union {
    int64_t i64;    // kInt, kTime
    double f64;     // kFloat64
    float f32;      // kFloat32
    bool b;         // kBool
    StringView str; // kString
  };

  __host__ __device__ bool IsValid() const {
    return type != ValueType::kInvalid;
  }

  __host__ __device__ double ToDouble() const;

  // Compare two FeatureValues. Returns -1, 0, 1.
  // For numeric/bool/time: compare via ToDouble.
  // For strings: lexicographic compare of StringView content.
  __host__ __device__ int Compare(const FeatureValue& other) const;
};

__host__ __device__ inline FeatureValue MakeIntValue(int64_t v) {
  FeatureValue r;
  r.type = ValueType::kInt;
  r.i64 = v;
  return r;
}

__host__ __device__ inline FeatureValue MakeFloat64Value(double v) {
  FeatureValue r;
  r.type = ValueType::kFloat64;
  r.f64 = v;
  return r;
}

__host__ __device__ inline FeatureValue MakeFloat32Value(float v) {
  FeatureValue r;
  r.type = ValueType::kFloat32;
  r.f32 = v;
  return r;
}

__host__ __device__ inline FeatureValue MakeBoolValue(bool v) {
  FeatureValue r;
  r.type = ValueType::kBool;
  r.b = v;
  return r;
}

__host__ __device__ inline FeatureValue MakeTimeValue(int64_t ms) {
  FeatureValue r;
  r.type = ValueType::kTime;
  r.i64 = ms;
  return r;
}

__host__ __device__ inline FeatureValue MakeStringValue(const char* data,
                                                        uint32_t len) {
  FeatureValue r;
  r.type = ValueType::kString;
  r.str.data = data;
  r.str.len = len;
  return r;
}

__host__ __device__ inline FeatureValue MakeInvalidValue() {
  FeatureValue r;
  r.type = ValueType::kInvalid;
  return r;
}

// Smallest power of two that is >= v (returns 1 for v == 0). Defined in
// execute_agg_prim.cu.
__host__ __device__ uint32_t NextPowerOfTwo(uint32_t v);

// -----------------------------------------------------------------------------
// Aggregation primitives enum.
// Aligned with featurelib/primitive/prim_mapping.go names.
// -----------------------------------------------------------------------------
enum class AggPrim : uint8_t {
  kCount = 0,
  kSum,
  kMean,
  kMedian,
  kMode,
  kMax,
  kMin,
  kVariance,
  kStd,
  kSkew,
  kEntropy,
  kNumUnique,
  kPercentTrue,
  kQuarter,
  kQuartile3,
  kCountGreaterThanMean,
  kDFeat
};

// -----------------------------------------------------------------------------
// Fused output container for ComputeAll.
// -----------------------------------------------------------------------------
struct AllFeatures {
  FeatureValue count;
  FeatureValue sum;
  FeatureValue mean;
  FeatureValue median;
  FeatureValue mode;
  FeatureValue max;
  FeatureValue min;
  FeatureValue variance;
  FeatureValue std;
  FeatureValue skew;
  FeatureValue entropy;
  FeatureValue num_unique;
  FeatureValue percent_true;
  FeatureValue quarter;
  FeatureValue quartile3;
  FeatureValue count_greater_than_mean;
  FeatureValue dfeat;
};

namespace kernel {

// -----------------------------------------------------------------------------
// Block-level aggregation primitives.
// Each function assumes the caller has provided valid shared memory.
// -----------------------------------------------------------------------------
__device__ inline FeatureValue BlockAggCount(uint32_t n);
__device__ inline FeatureValue BlockAggSum(const FeatureValue* values,
                                           uint32_t n);
__device__ inline FeatureValue BlockAggMean(const FeatureValue* values,
                                            uint32_t n);
__device__ inline FeatureValue BlockAggMedian(FeatureValue* values, uint32_t n);
__device__ inline FeatureValue BlockAggMode(FeatureValue* values, uint32_t n);
__device__ inline FeatureValue BlockAggMax(const FeatureValue* values,
                                           uint32_t n);
__device__ inline FeatureValue BlockAggMin(const FeatureValue* values,
                                           uint32_t n);
__device__ inline FeatureValue BlockAggVariance(const FeatureValue* values,
                                                uint32_t n);
__device__ inline FeatureValue BlockAggStd(const FeatureValue* values,
                                           uint32_t n);
__device__ inline FeatureValue BlockAggSkew(const FeatureValue* values,
                                            uint32_t n);
__device__ inline FeatureValue BlockAggEntropy(FeatureValue* values, uint32_t n);
__device__ inline FeatureValue BlockAggNumUnique(FeatureValue* values,
                                                 uint32_t n);
__device__ inline FeatureValue BlockAggPercentTrue(const FeatureValue* values,
                                                   uint32_t n);
__device__ inline FeatureValue BlockAggQuarter(FeatureValue* values, uint32_t n);
__device__ inline FeatureValue BlockAggQuartile3(FeatureValue* values,
                                                 uint32_t n);
__device__ inline FeatureValue BlockAggCountGreaterThanMean(
    const FeatureValue* values, uint32_t n);
__device__ inline FeatureValue BlockAggDFeat(const FeatureValue* values,
                                             uint32_t n);

__device__ inline FeatureValue BlockApplyAggPrim(AggPrim prim,
                                                 FeatureValue* values,
                                                 uint32_t n);

__device__ inline AllFeatures BlockComputeAllFeaturesFromValues(
    FeatureValue* values, uint32_t n);

// -----------------------------------------------------------------------------
// Global kernels
// -----------------------------------------------------------------------------
// list_offset: global list index for the first block in this launch.
// total_n_lists: total number of lists across all streams (for bounds check).
__global__ void ComputeAggPrimKernel(AggPrim prim,
                                     const FeatureValue* d_values,
                                     const uint32_t* d_offsets,
                                     uint32_t list_offset,
                                     uint32_t total_n_lists,
                                     FeatureValue* d_outputs);

// Multi-primitive batch: each input list is processed by one block, which
// computes every requested primitive sequentially. Output layout is row-major:
// d_outputs[list_idx * n_prims + prim_idx].
__global__ void ComputeBatchMultiPrimKernel(const FeatureValue* d_values,
                                            const uint32_t* d_offsets,
                                            const AggPrim* d_prims,
                                            uint32_t n_prims,
                                            uint32_t list_offset,
                                            uint32_t total_n_lists,
                                            FeatureValue* d_outputs);

__global__ void ComputeAllAggPrimsKernel(const FeatureValue* d_values,
                                         const uint32_t* d_offsets,
                                         uint32_t list_offset,
                                         uint32_t total_n_lists,
                                         AllFeatures* d_outputs);

// Streaming reduce kernel (no shared-memory list staging). One block per list;
// the list handled by block b is d_list_ids[b], and its primitive is
// d_list_prims[b]. Values are read directly from global memory and reduced with
// warp-shuffle primitives. Only a few hundred bytes of static shared memory are
// used (one slot per warp), so occupancy is not shared-limited and there is no
// list-length ceiling. Handles the reduce-class primitives only.
__global__ void ComputeAggStreamingKernel(const FeatureValue* d_values,
                                          const uint32_t* d_offsets,
                                          const uint32_t* d_list_ids,
                                          const AggPrim* d_list_prims,
                                          uint32_t n_group,
                                          FeatureValue* d_outputs);

// NumUnique via a per-list open-addressing hash table held in global scratch.
// Block b handles list d_list_ids[b]; its table occupies
// d_hash[d_hash_offsets[b] .. d_hash_offsets[b+1]). No sort, no list staging.
__global__ void ComputeNumUniqueHashKernel(const FeatureValue* d_values,
                                           const uint32_t* d_offsets,
                                           const uint32_t* d_list_ids,
                                           uint32_t n_group,
                                           unsigned long long* d_hash,
                                           const uint32_t* d_hash_offsets,
                                           FeatureValue* d_outputs);

}  // namespace kernel

// -----------------------------------------------------------------------------
// Host-facing task class.
// -----------------------------------------------------------------------------
class ExecuteAggPrim : public TaskBase {
 public:
  ExecuteAggPrim();
  ~ExecuteAggPrim();

  // Disable copy; streams are not copyable.
  ExecuteAggPrim(const ExecuteAggPrim&) = delete;
  ExecuteAggPrim& operator=(const ExecuteAggPrim&) = delete;

  ExecuteAggPrim(ExecuteAggPrim&& other) noexcept;
  ExecuteAggPrim& operator=(ExecuteAggPrim&& other) noexcept;

  // Set the number of CUDA streams for parallel batch processing.
  // Must be called before any batch computation. Default is 1.
  __host__ void SetNumStreams(uint32_t n_streams);

  // Compute a single aggregation primitive on a host-side value list.
  __host__ FeatureValue Compute(AggPrim prim,
                                const FeatureValue* host_values,
                                uint32_t n);

  // Compute the same primitive for many host-side value lists.
  // Uses multiple CUDA streams when n_streams_ > 1.
  __host__ std::vector<FeatureValue> ComputeBatch(
      AggPrim prim,
      const std::vector<std::vector<FeatureValue>>& host_value_lists);

  // Compute all primitives for a single host-side value list.
  __host__ AllFeatures ComputeAll(const FeatureValue* host_values, uint32_t n);

  // Compute all primitives for many host-side value lists.
  // Uses multiple CUDA streams when n_streams_ > 1.
  __host__ std::vector<AllFeatures> ComputeAllBatch(
      const std::vector<std::vector<FeatureValue>>& host_value_lists);

  // Multi-primitive batch: compute a set of primitives for every input list.
  // Input: a 2D matrix where inputs[i] is the i-th value list (row).
  // Output: results[i][j] = prims[j] applied to inputs[i].
  // Uses multiple CUDA streams when n_streams_ > 1.
  __host__ std::vector<std::vector<FeatureValue>> ComputeBatchMultiPrim(
      const std::vector<std::vector<FeatureValue>>& inputs,
      const std::vector<AggPrim>& prims);

  // Flat-input variants. The caller supplies an already-flattened value buffer
  // plus a CSR-style offsets array (length n_lists+1, offsets[0]==0,
  // offsets[n_lists]==total_values). No vector<vector> rebuild, no re-flatten.
  // When h_flat points to pinned (page-locked) host memory, per-stream H2D
  // copies overlap with other streams' kernels.
  __host__ std::vector<FeatureValue> ComputeBatchFlat(
      AggPrim prim, const FeatureValue* h_flat, const uint32_t* offsets,
      uint32_t n_lists, uint32_t total_values, uint32_t max_n);

  __host__ std::vector<std::vector<FeatureValue>> ComputeBatchMultiPrimFlat(
      const FeatureValue* h_flat, const uint32_t* offsets, uint32_t n_lists,
      uint32_t total_values, uint32_t max_n,
      const std::vector<AggPrim>& prims);

  // Per-list single-primitive dispatch with no shared-memory list staging.
  // list_prims[i] is the AggPrim (as uint8) for list i. Reduce-class prims go
  // to ComputeAggStreamingKernel, NumUnique to ComputeNumUniqueHashKernel, and
  // the rare sort-only prims (median/mode/entropy/quarter/quartile3) fall back
  // to the legacy shared-staging single-prim kernel. results[i] is the value
  // for list i. h_flat should be pinned for fast H2D.
  __host__ std::vector<FeatureValue> ComputeByPrim(
      const FeatureValue* h_flat, const uint32_t* offsets, uint32_t n_lists,
      uint32_t total_values, const uint8_t* list_prims);

  // Returns a persistent, pinned host buffer with capacity >= n FeatureValues,
  // grown on demand. The caller writes its converted values here, then passes
  // the pointer to ComputeByPrim. Reused across calls; freed in the destructor.
  __host__ FeatureValue* EnsurePinnedValues(uint32_t n);

  __host__ void Run();

 private:
  __host__ void EnsureStreams();
  __host__ void DestroyStreams();
  // Release all persistent device/pinned scratch buffers.
  __host__ void FreeBuffers();

  uint32_t n_streams_ = 1;
  std::vector<cudaStream_t> streams_;

  // Persistent scratch reused across ComputeByPrim calls (capacities in
  // elements; grown on demand, never shrunk). Removes per-call cudaMalloc/Free.
  FeatureValue* h_values_ = nullptr;       size_t h_values_cap_ = 0;  // pinned
  FeatureValue* d_values_ = nullptr;       size_t d_values_cap_ = 0;
  uint32_t* d_offsets_ = nullptr;          size_t d_offsets_cap_ = 0;
  FeatureValue* d_outputs_ = nullptr;      size_t d_outputs_cap_ = 0;
  uint32_t* d_ids_ = nullptr;              size_t d_ids_cap_ = 0;
  AggPrim* d_prims_ = nullptr;             size_t d_prims_cap_ = 0;
  uint32_t* d_hoff_ = nullptr;             size_t d_hoff_cap_ = 0;
  unsigned long long* d_hash_ = nullptr;   size_t d_hash_cap_ = 0;
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_GPU_TASK_EXECUTE_AGG_PRIM_CUH_
