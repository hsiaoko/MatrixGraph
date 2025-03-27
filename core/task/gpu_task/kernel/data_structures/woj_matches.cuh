#ifndef MATRIXGRAPH_CORE_TASK_KERNEL_DATA_STRUCTURES_WON_MATCHES_CUH_
#define MATRIXGRAPH_CORE_TASK_KERNEL_DATA_STRUCTURES_WON_MATCHES_CUH_

#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/host_buffer.cuh"
#include "core/data_structures/metadata.h"
#include "core/data_structures/unified_buffer.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

class WOJMatches {
private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using BufferVertexID =
      sics::matrixgraph::core::data_structures::Buffer<VertexID>;
  using UnifiedOwnedBufferVertexID =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexID>;

public:
  __host__ WOJMatches(VertexID x, VertexID y);

  __host__ WOJMatches() = default;

  void __host__ Init(VertexID x, VertexID y);

  void __host__ Free();

  __host__ __device__ VertexID BinarySearch(VertexID col, VertexID val) const;

  __inline__ __host__ void SetYOffset(VertexID val) { *y_offset_ = val; }

  __inline__ __host__ void SetXOffset(VertexID val) { *x_offset_ = val; }

  __inline__ __host__ void SetHeader(VertexID idx, VertexID val) {
    header_ptr_[idx] = val;
  }

  __host__ std::pair<VertexID, VertexID> GetJoinKey(const WOJMatches &other);

  __host__ void SetHeader(const VertexID *left_header, VertexID left_x_offset,
                          const VertexID *right_header, VertexID right_x_offset,
                          const std::pair<VertexID, VertexID> &hash_keys);

  __host__ std::vector<WOJMatches *> SplitAndCopy(VertexID n_partitions);

  __host__ void CopyData(const WOJMatches &other);

  __host__ void CopyDataAsync(const WOJMatches &other,
                              const cudaStream_t &stream);

  __inline__ __host__ __device__ VertexID get_y_offset() const {
    return *y_offset_;
  }

  __inline__ __host__ __device__ VertexID get_x_offset() const {
    return *x_offset_;
  }

  __inline__ __host__ __device__ VertexID get_x() const { return x_; }

  __inline__ __host__ __device__ VertexID get_y() const { return y_; }

  __inline__ __host__ __device__ VertexID *get_y_offset_ptr() const {
    return y_offset_;
  }

  __inline__ __host__ __device__ VertexID *get_x_offset_ptr() const {
    return x_offset_;
  }

  __inline__ __host__ __device__ VertexID *get_data_ptr() const {
    return data_;
  }

  __inline__ __host__ __device__ VertexID *get_header_ptr() const {
    return header_ptr_;
  }

  __inline __host__ void Clear() {
    *y_offset_ = 0;
    *x_offset_ = 0;
    cudaMemset(data_, 0, sizeof(VertexID) * x_ * y_);
  }

  void Print(VertexID offset = 3) const;

  VertexID *header_ptr_ = nullptr;
  VertexID *data_ = nullptr;
  VertexID *y_offset_ = nullptr;
  VertexID *x_offset_ = nullptr;

  VertexID y_ = 0;
  VertexID x_ = 0;
};

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_TASK_KERNEL_DATA_STRUCTURES_HASH_BUCKETS_CUH_