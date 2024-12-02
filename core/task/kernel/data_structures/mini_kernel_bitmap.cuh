#ifndef MATRIXGRAPH_CORE_TASK_KERNEL_MINI_KERNEL_BITMAP_CUH_
#define MATRIXGRAPH_CORE_TASK_KERNEL_MINI_KERNEL_BITMAP_CUH_

#include <cuda_runtime.h>

#include <cstdint>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

class MiniKernelBitmap {
public:
  __device__ MiniKernelBitmap(unsigned size) { Init(size); }

  __device__ MiniKernelBitmap(const MiniKernelBitmap &other) {
    Init(other.GetSize());
    data_ = other.GetData();
  }

  __device__ ~MiniKernelBitmap() = default;

  __device__ void Init(unsigned size) {
    size_ = size;
    data_ = 0;
  }

  __device__ void Clear() { data_ = 0; }

  __device__ void Fill() { data_ = 0xffffffff; }

  __device__ bool GetBit(unsigned i) const {
    if (i > size_)
      return 0;
    return data_ & (1u << i);
  }

  __device__ void SetBit(unsigned i) {
    if (i > size_)
      return;
    data_ |= (unsigned)(1u << i);
  }

  __device__ void ClearBit(const unsigned i) {
    if (i > size_)
      return;
    data_ &= ~(unsigned)(1u << i);
  }

  __device__ unsigned Count() const {
    unsigned count = 0;
    unsigned x = data_;
    x = (x & (0x55555555)) + ((x >> 1) & (0x5555555));
    x = (x & (0x33333333)) + ((x >> 2) & (0x3333333));
    x = (x & (0x0f0f0f0f)) + ((x >> 4) & (0xf0f0f0f));
    x = (x & (0x00ff00ff)) + ((x >> 8) & (0x0ff00ff));
    x = (x & (0x0000ffff)) + ((x >> 16) & (0x0000ffff));
    count += x;
    return count;
  }

  __device__ unsigned GetSize() const { return size_; }

  __device__ unsigned GetData() const { return data_; }

  unsigned size_ = 0;
  unsigned data_ = 0;
};

class HostMiniKernelBitmap {
public:
  HostMiniKernelBitmap(unsigned size) { Init(size); }

  HostMiniKernelBitmap(const HostMiniKernelBitmap &other) {
    Init(other.GetSize());
    data_ = other.GetData();
  }

  ~HostMiniKernelBitmap() = default;

  void Init(unsigned size) {
    size_ = size;
    data_ = 0;
  }

  void Clear() { data_ = 0; }

  void Fill() { data_ = 0xffffffff; }

  bool GetBit(unsigned i) const {
    if (i > size_ || i > 31)
      return 0;
    return data_ & (1u << i);
  }

  void SetBit(unsigned i) {
    if (i > size_ || i > 31)
      return;
    data_ |= (unsigned)(1u << i);
  }

  void ClearBit(const unsigned i) {
    if (i > size_ || i > 31)
      return;
    data_ &= ~(unsigned)(1u << (i));
  }

  unsigned Count() const {

    unsigned x = data_;
    x = (x & (0x55555555)) + ((x >> 1) & (0x5555555));
    x = (x & (0x33333333)) + ((x >> 2) & (0x3333333));
    x = (x & (0x0f0f0f0f)) + ((x >> 4) & (0xf0f0f0f));
    x = (x & (0x00ff00ff)) + ((x >> 8) & (0x0ff00ff));
    x = (x & (0x0000ffff)) + ((x >> 16) & (0x0000ffff));
    return x;
  }

  unsigned GetSize() const { return size_; }

  unsigned GetData() const { return data_; }

  unsigned size_ = 0;
  unsigned data_ = 0;
};

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_TASK_KERNEL_KERNEL_BITMAP_CUH_