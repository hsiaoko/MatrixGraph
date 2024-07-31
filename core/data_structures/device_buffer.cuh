#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_DEVICE_BUFFER_CUH_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_DEVICE_BUFFER_CUH_

#include <cuda_runtime.h>

#include "core/data_structures/host_buffer.cuh"
#include "core/util/cuda_check.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

// Class to manage buffers allocated on the device
template <typename T> class DeviceOwnedBuffer {
public:
  // Default constructor
  DeviceOwnedBuffer() = default;

  // @Brif: Deleted copy constructor and copy assignment operator to prevent
  // copying
  DeviceOwnedBuffer(const DeviceOwnedBuffer<T> &) = delete;
  DeviceOwnedBuffer &operator=(const DeviceOwnedBuffer<T> &) = delete;

  // @Brif: Move constructor and move assignment operator
  DeviceOwnedBuffer(DeviceOwnedBuffer<T> &&r) noexcept {
    if (this != &r) {
      cudaFree(ptr_);
      ptr_ = r.GetPtr();
      s_ = r.GetSize();
      r.SetPtr(nullptr);
      r.SetSize(0);
    }
  }

  DeviceOwnedBuffer &operator=(DeviceOwnedBuffer<T> &&r) noexcept {
    if (this != &r) {
      cudaFree(ptr_);
      ptr_ = r.GetPtr();
      s_ = r.GetSize();
      r.SetPtr(nullptr);
      r.SetSize(0);
    }
    return *this;
  };

  // @Brif:  Deleted copy constructor and copy assignment operator to prevent
  // DeviceOwnedBuffer<T> object get the ownership of the host buffer.
  DeviceOwnedBuffer(Buffer<T> &&h_buff) = delete;

  // @Brif: Constructor with buffer and optional stream
  DeviceOwnedBuffer(const Buffer<T> &h_buf, const cudaStream_t &stream) {
    Init(h_buf, stream);
  }

  // Constructor with buffer size
  DeviceOwnedBuffer(size_t s) { Init(s); }

  // Destructor to free device memory
  ~DeviceOwnedBuffer() {
    cudaFree(ptr_);
    s_ = 0;
  }

  // Initialize the DeviceOwnedBuffer<T> with a buffer
  // Parameters:
  //   h_buf: Reference to the host buffer
  //   stream: CUDA stream for asynchronous memory operations.
  void Init(const Buffer<T> &h_buf, const cudaStream_t &stream) {
    if (ptr_ != nullptr)
      cudaFree(ptr_);
    s_ = h_buf.size;
    CUDA_CHECK(cudaMalloc((void **)&ptr_, s_));
    CUDA_CHECK(
        cudaMemcpyAsync(ptr_, h_buf.data, s_, cudaMemcpyHostToDevice, stream));
  }

  // Initialize the DeviceOwnedBuffer<T> with a buffer
  // Parameters:
  void Init(const Buffer<T> &h_buf) {
    if (ptr_ != nullptr)
      cudaFree(ptr_);
    s_ = h_buf.size;
    CUDA_CHECK(cudaMalloc(&ptr_, s_));
    CUDA_CHECK(cudaMemcpy(ptr_, h_buf.data, s_, cudaMemcpyHostToDevice));
  }

  // Initialize the DeviceOwnedBuffer<T> with a buffer size
  // Parameters:
  //   s: Size of the buffer to be allocated on the device
  void Init(size_t s) {
    if (ptr_ != nullptr)
      cudaFree(ptr_);
    s_ = s;
    CUDA_CHECK(cudaMalloc(&ptr_, s_));
  }

  // Brif: Copy data from device to host asynchronously with the specified
  // stream
  // @Parameters:
  //   stream: CUDA stream for asynchronous memory operations
  //   h_buf: Pointer to the host buffer where the data will be copied
  // @Description:
  //   This function asynchronously copies data from the device to the host
  //   using the specified CUDA stream.
  // @Warning: The size of h_buf must be the same as the size of this->GetSize()
  void Device2Host(const cudaStream_t &stream, Buffer<T> *h_buf) {
    h_buf->size = s_;
    CUDA_CHECK(
        cudaMemcpyAsync(h_buf->data, ptr_, s_, cudaMemcpyDeviceToHost, stream));
  }

  // @Brif: Copy data from device to host synchronously
  // @Parameters:
  //   h_buf: Pointer to the host buffer where the data will be copied
  // @Warning: The size of h_buf must be the same as the size of this->GetSize()
  void Device2Host(Buffer<T> *h_buf) {
    h_buf->size = s_;
    CUDA_CHECK(cudaMemcpy(h_buf->data, ptr_, s_, cudaMemcpyDeviceToHost));
  }

  // @Brif: Get the device buffer pointer
  // @Returns:
  //   Pointer to the device buffer
  // @Warning: The size of h_buf must be the same as the size of this->GetSize()
  T *GetPtr() const { return (ptr_); };

  // @Brif: Get the size of the device buffer
  // @Returns:
  //   Size of the device buffer
  size_t GetSize() const { return s_; };

  size_t GetElementSize() const { return sizeof(T); }

private:
  void SetPtr(T *val) { ptr_ = val; }
  void SetSize(size_t s) { s_ = s; }

  T *ptr_ = nullptr; // Pointer to device memory
  size_t s_ = 0;     // Size of the device memory allocation
};

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_DATA_STRUCTURES_DEVICE_BUFFER_CUH_