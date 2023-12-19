#ifndef HYPERBLOCKER_CORE_GPU_KERNEL_DATA_STRUCTURES_KERNEL_TABLE_H_
#define HYPERBLOCKER_CORE_GPU_KERNEL_DATA_STRUCTURES_KERNEL_TABLE_H_

namespace sics {
namespace hyperblocker {
namespace core {
namespace gpu {
__device__ class SerializedTable {
public:
  __forceinline__ __device__ char *get_data_base_ptr() const { return data_; }
  __forceinline__ __device__ size_t *get_col_size_base_ptr() const {
    return col_size_;
  }
  __forceinline__ __device__ size_t *get_col_offset_base_ptr() const {
    return col_offset_;
  }

  __forceinline__ _device__ char *get_term_ptr(size_t col, size_t row) const {
    return data_ + aligned_tuple_size_ * row + col_offset_[col];
  }

  __forceinline__ __device__ size_t get_aligned_tuple_size() const {
    return aligned_tuple_size_;
  }
  __forceinline__ __device__ size_t get_n_rows() const { return n_rows_; }
  __forceinline__ __device__ size_t get_n_cols() const { return n_cols_; }

private:
  __device__ size_t n_rows_ = 0;
  __device__ size_t n_cols_ = 0;
  __device__ size_t aligned_tuple_size_ = 0;
  __device__ size_t *col_size_;
  __device__ size_t *col_offset_;
  __device__ char *data_;
};

} // namespace gpu
} // namespace core
} // namespace hyperblocker
} // namespace sics
#endif // HYPERBLOCKER_CORE_GPU_KERNEL_DATA_STRUCTURES_KERNEL_TABLE_H_
