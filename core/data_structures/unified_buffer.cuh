#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_UNIFIED_BUFFER_CUH_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_UNIFIED_BUFFER_CUH_

#include <cuda_runtime.h>

#include "core/data_structures/host_buffer.cuh"
#include "core/util/cuda_check.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

// Class to manage buffers allocated on the device
template <typename T> class UnifiedOwnedBuffer {
public:
  // Default constructor
  UnifiedOwnedBuffer() = default;

  // @Brif: Deleted copy constructor and copy assignment operator to prevent
  // copying
  UnifiedOwnedBuffer(const UnifiedOwnedBuffer<T> &) = delete;

  UnifiedOwnedBuffer &operator=(const UnifiedOwnedBuffer<T> &) = delete;

  // @Brif: Move constructor and move assignment operator
  UnifiedOwnedBuffer(UnifiedOwnedBuffer<T> &&r) noexcept {
    if (this != &r) {
      cudaFree(ptr_);
      ptr_ = r.GetPtr();
      s_ = r.GetSize();
      r.SetPtr(nullptr);
      r.SetSize(0);
    }
  }

  UnifiedOwnedBuffer &operator=(UnifiedOwnedBuffer<T> &&r) noexcept {
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
  // UnifiedOwnedBuffer<T> object get the ownership of the host buffer.
  UnifiedOwnedBuffer(Buffer<T> &&h_buff) = delete;

  // Constructor with buffer size
  UnifiedOwnedBuffer(size_t s) { Init(s); }

  // Destructor to free device memory
  ~UnifiedOwnedBuffer() {
    cudaFree(ptr_);
    s_ = 0;
  }

  // Initialize the UnifiedOwnedBuffer<T> with a buffer
  // Parameters:
  void Init(const Buffer<T> &h_buf) {
    if (ptr_ != nullptr)
      cudaFree(ptr_);
    s_ = h_buf.size;
    CUDA_CHECK(cudaMallocManaged(&ptr_, s_));
    memset(ptr_, 0, s_);
  }

  // Initialize the UnifiedOwnedBuffer<T> with a buffer size
  // Parameters:
  //   s: Size of the buffer to be allocated on the device
  void Init(size_t s) {
    if (ptr_ != nullptr)
      cudaFree(ptr_);
    s_ = s;
    CUDA_CHECK(cudaMallocManaged(&ptr_, s_));
    memset(ptr_, 0, s_);
    // CUDA_CHECK(cudaMemset(ptr_, 0, s_));
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