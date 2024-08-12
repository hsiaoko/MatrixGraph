#ifndef MATRIXGRAPH_CORE_TASK_KERNEL_MATRIX_OPERATIONS_CUH_
#define MATRIXGRAPH_CORE_TASK_KERNEL_MATRIX_OPERATIONS_CUH_

#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/unified_buffer.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

class MatrixOperationsKernelWrapper {
public:
  // deleting copy constructor
  MatrixOperationsKernelWrapper(const MatrixOperationsKernelWrapper &obj) =
      delete;

  void operator=(const MatrixOperationsKernelWrapper &) = delete;

  // @Description: GetInstance() is a method that returns an instance
  // when it is invoked. It returns the same instance if it is invoked more
  // than once as an instance of Singleton class is already created.
  static MatrixOperationsKernelWrapper *GetInstance();

  static void
  MatrixBitAnd(const cudaStream_t &stream,
               const data_structures::DeviceOwnedBuffer<uint64_t> &matrix_a_buf,
               const data_structures::DeviceOwnedBuffer<uint64_t> &matrix_b_buf,
               data_structures::DeviceOwnedBuffer<uint64_t> *matrix_c_buf,
               uint32_t m, uint32_t k, uint32_t n);

  static void
  MatrixBitCount(const cudaStream_t &stream,
                 const data_structures::DeviceOwnedBuffer<uint64_t> &matrix_buf,
                 data_structures::DeviceOwnedBuffer<uint64_t> *count_buf,
                 uint64_t size);

  static void InitBitTiledMatrixMetadataByLayoutMatrix(
      const cudaStream_t &stream,
      const data_structures::DeviceOwnedBuffer<uint64_t> &layout_matrix,
      data_structures::DeviceOwnedBuffer<uint32_t> *tile_offset_row,
      data_structures::DeviceOwnedBuffer<uint32_t> *tile_row_idx,
      data_structures::DeviceOwnedBuffer<uint32_t> *tile_col_idx,
      uint32_t tile_size);

  static void FillTiles(
      const cudaStream_t &stream, size_t tile_size, size_t n_strips,
      size_t n_nz_tile_a, size_t n_nz_tile_b, size_t n_nz_tile_c,
      const data_structures::UnifiedOwnedBuffer<uint64_t> &layout_matrix_c,
      const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_offset_row_a,
      const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_offset_row_b,
      const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_offset_row_c,
      const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_row_idx_a,
      const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_row_idx_b,
      const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_row_idx_c,
      const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_col_idx_a,
      const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_col_idx_b,
      const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_col_idx_c,
      const data_structures::UnifiedOwnedBuffer<uint64_t> &data_a,
      const data_structures::UnifiedOwnedBuffer<uint64_t> &data_b,
      data_structures::UnifiedOwnedBuffer<uint64_t> *data_c);

private:
  MatrixOperationsKernelWrapper() = default;

  inline static MatrixOperationsKernelWrapper *ptr_ = nullptr;
};

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_TASK_KERNEL_MATRIX_OPERATIONS_CUH_