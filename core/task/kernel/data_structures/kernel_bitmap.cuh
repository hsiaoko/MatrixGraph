#ifndef MATRIXGRAPH_CORE_TASK_KERNEL_KERNEL_BITMAP_CUH_
#define MATRIXGRAPH_CORE_TASK_KERNEL_KERNEL_BITMAP_CUH_

#include <cstdint>
#include <cuda_runtime.h>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

#define KERNEL_WORD_OFFSET(i) (i >> 6)
#define BIT_OFFSET(i) (i & 0x3f)

class KernelBitmap {
public:
  __device__ KernelBitmap() = default;

  __device__ KernelBitmap(uint64_t size) { Init(size); }

  __device__ KernelBitmap(uint64_t size, uint64_t *data) {
    size = size;
    data_ = data;
  }

  __device__ KernelBitmap(const KernelBitmap &other) {
    if (this != &other)
      free(data_);

    Init(other.GetSize());
    memcpy(data_, other.GetPtr(),
           (KERNEL_WORD_OFFSET(size_) + 1) * sizeof(uint64_t));
  }

  __device__ ~KernelBitmap() {
    free(data_);
    data_ = nullptr;
  }

  __device__ void Init(uint64_t size, uint64_t *data) {
    if (data_ != nullptr)
      free(data_);
    data_ = data;
    size_ = size;
  }

  __device__ void Init(uint64_t size) {
    if (data_ != nullptr)
      free(data_);

    size_ = size;
    data_ =
        (uint64_t *)malloc(sizeof(uint64_t) * (KERNEL_WORD_OFFSET(size) + 1));
  }

  __device__ void Clear() {
    uint64_t bm_size = KERNEL_WORD_OFFSET(size_);
    for (uint64_t i = 0; i <= bm_size; i++)
      data_[i] = 0;
  }

  __device__ void Fill() {
    uint64_t bm_size = KERNEL_WORD_OFFSET(size_);
    for (uint64_t i = 0; i < bm_size; i++) {
      data_[i] = 0xffffffffffffffff;
    }
    data_[bm_size] = 0;
    for (uint64_t i = (bm_size << 6); i < size_; i++) {
      data_[bm_size] |= 1ul << BIT_OFFSET(i);
    }
  }

  __device__ bool GetBit(uint64_t i) const {
    if (i > size_)
      return 0;
    return data_[KERNEL_WORD_OFFSET(i)] & (1ul << BIT_OFFSET(i));
  }

  __device__ void SetBit(uint64_t i) {
    if (i > size_)
      return;
    atomicOr((unsigned long long int *)(data_ + KERNEL_WORD_OFFSET(i)),
             (unsigned long long int)(1ul << BIT_OFFSET(i)));
  }

  __device__ void ClearBit(const uint64_t i) {
    if (i > size_)
      return;
    atomicAnd((unsigned long long int *)(data_ + KERNEL_WORD_OFFSET(i)),
              ~(unsigned long long int)(1ul << BIT_OFFSET(i)));
  }

  __device__ uint64_t Count() const {
    uint64_t count = 0;
    for (uint64_t i = 0; i <= KERNEL_WORD_OFFSET(size_); i++) {
      auto x = data_[i];
      x = (x & (0x5555555555555555)) + ((x >> 1) & (0x5555555555555555));
      x = (x & (0x3333333333333333)) + ((x >> 2) & (0x3333333333333333));
      x = (x & (0x0f0f0f0f0f0f0f0f)) + ((x >> 4) & (0x0f0f0f0f0f0f0f0f));
      x = (x & (0x00ff00ff00ff00ff)) + ((x >> 8) & (0x00ff00ff00ff00ff));
      x = (x & (0x0000ffff0000ffff)) + ((x >> 16) & (0x0000ffff0000ffff));
      x = (x & (0x00000000ffffffff)) + ((x >> 32) & (0x00000000ffffffff));
      count += (uint64_t)x;
    }
    return count;
  }

  __device__ uint64_t GetSize() const { return size_; }

  __device__ uint64_t *GetPtr() const { return data_; }

private:
  uint64_t size_ = 0;
  uint64_t *data_ = nullptr;
};

class HostKernelBitmap {
public:
  HostKernelBitmap() = default;

  HostKernelBitmap(uint64_t size) { Init(size); }

  ~HostKernelBitmap() {
    cudaFree(data_);
    data_ = nullptr;
  }

  void Init(uint64_t size) {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
    size_ = size;

    cudaMallocManaged(&data_,
                      sizeof(uint64_t) * (KERNEL_WORD_OFFSET(size) + 1));
  }

  void Clear() {
    cudaMemset(data_, 0, sizeof(uint64_t) * (KERNEL_WORD_OFFSET(size_) + 1));
  }

  void ClearAsync(const cudaStream_t &stream) {
    cudaMemsetAsync(data_, 0,
                    sizeof(uint64_t) * (KERNEL_WORD_OFFSET(size_) + 1), stream);
  }

  void Fill() {
    uint64_t bm_size = KERNEL_WORD_OFFSET(size_);
    for (uint64_t i = 0; i < bm_size; i++) {
      data_[i] = 0xffffffffffffffff;
    }
    data_[bm_size] = 0;
    for (uint64_t i = (bm_size << 6); i < size_; i++) {
      data_[bm_size] |= 1ul << BIT_OFFSET(i);
    }
  }

  bool GetBit(uint64_t i) const {
    if (i > size_)
      return 0;
    return data_[KERNEL_WORD_OFFSET(i)] & (1ul << BIT_OFFSET(i));
  }

  void SetBit(uint64_t i) {
    if (i > size_)
      return;
    *(data_ + KERNEL_WORD_OFFSET(i)) |= (1ul << BIT_OFFSET(i));
  }

  void ClearBit(const uint64_t i) {
    if (i > size_)
      return;
    *(data_ + KERNEL_WORD_OFFSET(i)) &= ~(1ul << BIT_OFFSET(i));
  }

  uint64_t Count() const {
    uint64_t count = 0;
    for (uint64_t i = 0; i <= KERNEL_WORD_OFFSET(size_); i++) {
      auto x = data_[i];
      x = (x & (0x5555555555555555)) + ((x >> 1) & (0x5555555555555555));
      x = (x & (0x3333333333333333)) + ((x >> 2) & (0x3333333333333333));
      x = (x & (0x0f0f0f0f0f0f0f0f)) + ((x >> 4) & (0x0f0f0f0f0f0f0f0f));
      x = (x & (0x00ff00ff00ff00ff)) + ((x >> 8) & (0x00ff00ff00ff00ff));
      x = (x & (0x0000ffff0000ffff)) + ((x >> 16) & (0x0000ffff0000ffff));
      x = (x & (0x00000000ffffffff)) + ((x >> 32) & (0x00000000ffffffff));
      count += (uint64_t)x;
    }
    return count;
  }

  uint64_t GetSize() const { return size_; }

  uint64_t *GetPtr() const { return data_; }

private:
  uint64_t size_ = 0;
  uint64_t *data_ = nullptr;
};

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // INC_51_11_GRAPH_COMPUTING_MATRIXGRAPH_CORE_TASK_KERNEL_KERNEL_BITMAP_CUH_