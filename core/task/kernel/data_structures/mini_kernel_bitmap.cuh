#ifndef MATRIXGRAPH_CORE_TASK_KERNEL_MINI_KERNEL_BITMAP_CUH_
#define MATRIXGRAPH_CORE_TASK_KERNEL_MINI_KERNEL_BITMAP_CUH_

#include <cuda_runtime.h>

#include <cstdint>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

#define KERNEL_WORD_OFFSET(i) (i >> 6)
#define BIT_OFFSET(i) (i & 0x3f)

class MiniKernelBitmap {
public:
  __device__ MiniKernelBitmap(size_t size) { Init(size); }

  __device__ MiniKernelBitmap(const MiniKernelBitmap &other) {
    Init(other.GetSize());
    data_ = other.GetData();
  }

  __device__ ~MiniKernelBitmap() = default;

  __device__ void Init(size_t size) {
    size_ = size;
    data_ = 0;
  }

  __device__ void Clear() { data_ = 0; }

  __device__ void Fill() {
    data_ = 0xffffffffffffffff;
    for (size_t i = (size_ << 6); i < size_; i++) {
      data_ |= 1ul << BIT_OFFSET(i);
    }
  }

  __device__ bool GetBit(size_t i) const {
    if (i > size_)
      return 0;
    return data_ & (1ul << BIT_OFFSET(i));
  }

  __device__ void SetBit(size_t i) {
    if (i > size_)
      return;
    data_ |= (unsigned long long int)(1ul << BIT_OFFSET(i));
  }

  __device__ void ClearBit(const size_t i) {
    if (i > size_)
      return;
    data_ &= ~(unsigned long long int)(1ul << BIT_OFFSET(i));
  }

  __device__ size_t Count() const {
    size_t count = 0;
    auto x = data_;
    x = (x & (0x5555555555555555)) + ((x >> 1) & (0x5555555555555555));
    x = (x & (0x3333333333333333)) + ((x >> 2) & (0x3333333333333333));
    x = (x & (0x0f0f0f0f0f0f0f0f)) + ((x >> 4) & (0x0f0f0f0f0f0f0f0f));
    x = (x & (0x00ff00ff00ff00ff)) + ((x >> 8) & (0x00ff00ff00ff00ff));
    x = (x & (0x0000ffff0000ffff)) + ((x >> 16) & (0x0000ffff0000ffff));
    x = (x & (0x00000000ffffffff)) + ((x >> 32) & (0x00000000ffffffff));
    count += (size_t)x;
    return count;
  }

  __device__ size_t GetSize() const { return size_; }

  __device__ uint64_t GetData() const { return data_; }

  size_t size_ = 0;
  uint64_t data_ = 0;
};

class HostMiniKernelBitmap {
public:
  HostMiniKernelBitmap(size_t size) { Init(size); }

  HostMiniKernelBitmap(const HostMiniKernelBitmap &other) {
    Init(other.GetSize());
    data_ = other.GetData();
  }

  ~HostMiniKernelBitmap() = default;

  void Init(size_t size) {
    size_ = size;
    data_ = 0;
  }

  void Clear() { data_ = 0; }

  void Fill() {
    data_ = 0xffffffffffffffff;
    for (size_t i = (size_ << 6); i < size_; i++) {
      data_ |= 1ul << BIT_OFFSET(i);
    }
  }

  bool GetBit(size_t i) const {
    if (i > size_)
      return 0;
    return data_ & (1ul << BIT_OFFSET(i));
  }

  void SetBit(size_t i) {
    if (i > size_)
      return;
    data_ |= (unsigned long long int)(1ul << BIT_OFFSET(i));
  }

  void ClearBit(const size_t i) {
    if (i > size_)
      return;
    data_ &= ~(unsigned long long int)(1ul << BIT_OFFSET(i));
  }

  size_t Count() const {
    size_t count = 0;
    auto x = data_;
    x = (x & (0x5555555555555555)) + ((x >> 1) & (0x5555555555555555));
    x = (x & (0x3333333333333333)) + ((x >> 2) & (0x3333333333333333));
    x = (x & (0x0f0f0f0f0f0f0f0f)) + ((x >> 4) & (0x0f0f0f0f0f0f0f0f));
    x = (x & (0x00ff00ff00ff00ff)) + ((x >> 8) & (0x00ff00ff00ff00ff));
    x = (x & (0x0000ffff0000ffff)) + ((x >> 16) & (0x0000ffff0000ffff));
    x = (x & (0x00000000ffffffff)) + ((x >> 32) & (0x00000000ffffffff));
    count += (size_t)x;
    return count;
  }

  size_t GetSize() const { return size_; }

  uint64_t GetData() const { return data_; }

  size_t size_ = 0;
  uint64_t data_ = 0;
};

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // INC_51_11_GRAPH_COMPUTING_MATRIXGRAPH_CORE_TASK_KERNEL_KERNEL_BITMAP_CUH_