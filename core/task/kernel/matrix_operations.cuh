#ifndef MATRIXGRAPH_CORE_TASK_KERNEL_MATRIX_OPERATIONS_CUH_
#define MATRIXGRAPH_CORE_TASK_KERNEL_MATRIX_OPERATIONS_CUH_

#include "core/data_structures/device_buffer.cuh"

#include "core/data_structures/device_buffer.cuh"
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

  static void MatrixBitCount(
      const cudaStream_t &stream,
      const data_structures::DeviceOwnedBuffer<uint64_t> &matrix_buf,
      data_structures::DeviceOwnedBuffer<uint64_t> *count_buf,
      uint64_t size
      );

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