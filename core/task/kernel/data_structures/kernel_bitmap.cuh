#ifndef MATRIXGRAPH_CORE_TASK_KERNEL_KERNEL_BITMAP_CUH_
#define MATRIXGRAPH_CORE_TASK_KERNEL_KERNEL_BITMAP_CUH_

#include <cuda_runtime.h>

#include <cstdint>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

#define KERNEL_WORD_OFFSET(i) (i >> 6)
#define BIT_OFFSET(i) (i & 0x3f)

class KernelBitmap {
public:
  __device__ KernelBitmap(size_t size) { Init(size); }

  __device__ KernelBitmap(const KernelBitmap &other) {
    if (this != &other)
      free(data_);

    Init(other.GetSize());
    memcpy(data_, other.GetPtr(),
           (KERNEL_WORD_OFFSET(size_) + 1) * sizeof(uint64_t));
  }

  //__device__ ~KernelBitmap() {
  //  free(data_);
  //  data_ = nullptr;
  //}

  __device__ void Init(size_t size) {
    if (data_ != nullptr) {
      free(data_);
    }
    size_ = size;
    data_ =
        (uint64_t *)malloc(sizeof(uint64_t) * (KERNEL_WORD_OFFSET(size) + 1));
  }

  __device__ void Clear() {
    size_t bm_size = KERNEL_WORD_OFFSET(size_);
    for (size_t i = 0; i <= bm_size; i++)
      data_[i] = 0;
  }

  __device__ void Fill() {
    size_t bm_size = KERNEL_WORD_OFFSET(size_);
    for (size_t i = 0; i < bm_size; i++) {
      data_[i] = 0xffffffffffffffff;
    }
    data_[bm_size] = 0;
    for (size_t i = (bm_size << 6); i < size_; i++) {
      data_[bm_size] |= 1ul << BIT_OFFSET(i);
    }
  }

  __device__ bool GetBit(size_t i) const {
    if (i > size_)
      return 0;
    return data_[KERNEL_WORD_OFFSET(i)] & (1ul << BIT_OFFSET(i));
  }

  __device__ void SetBit(size_t i) {
    if (i > size_)
      return;
    atomicOr((unsigned long long int *)(data_ + KERNEL_WORD_OFFSET(i)),
             (unsigned long long int)(1ul << BIT_OFFSET(i)));
  }

  __device__ void ClearBit(const size_t i) {
    if (i > size_)
      return;
    atomicAnd((unsigned long long int *)(data_ + KERNEL_WORD_OFFSET(i)),
              ~(unsigned long long int)(1ul << BIT_OFFSET(i)));
  }

  __device__ size_t Count() const {
    size_t count = 0;
    for (size_t i = 0; i <= KERNEL_WORD_OFFSET(size_); i++) {
      auto x = data_[i];
      x = (x & (0x5555555555555555)) + ((x >> 1) & (0x5555555555555555));
      x = (x & (0x3333333333333333)) + ((x >> 2) & (0x3333333333333333));
      x = (x & (0x0f0f0f0f0f0f0f0f)) + ((x >> 4) & (0x0f0f0f0f0f0f0f0f));
      x = (x & (0x00ff00ff00ff00ff)) + ((x >> 8) & (0x00ff00ff00ff00ff));
      x = (x & (0x0000ffff0000ffff)) + ((x >> 16) & (0x0000ffff0000ffff));
      x = (x & (0x00000000ffffffff)) + ((x >> 32) & (0x00000000ffffffff));
      count += (size_t)x;
    }
    return count;
  }

  __device__ size_t GetSize() const { return size_; }

  __device__ uint64_t *GetPtr() const { return data_; }

  __device__ bool IsBitmapFull() {
    for (size_t i = 0; i <= KERNEL_WORD_OFFSET(size_); i++) {
      // if()
    }
  }

private:
  size_t size_ = 0;
  uint64_t *data_ = nullptr;
};

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // INC_51_11_GRAPH_COMPUTING_MATRIXGRAPH_CORE_TASK_KERNEL_KERNEL_BITMAP_CUH_