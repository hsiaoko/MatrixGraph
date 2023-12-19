#ifndef HYPERBLOCKER_CORE_GPU_KERNEL_DATA_STRUCTURES_KERNEL_BITMAP_CUH_
#define HYPERBLOCKER_CORE_GPU_KERNEL_DATA_STRUCTURES_KERNEL_BITMAP_CUH_

namespace sics {
namespace hyperblocker {
namespace core {
namespace gpu {

#define GPU_WORD_OFFSET(i) (i >> 3)
#define GPU_BIT_OFFSET(i) (i & 0x1f)

// @DESCRIPTION
//
// Bitmap is a mapping from integers~(indexes) to bits. If the unit is
// occupied, the bit is a nonzero integer constant and if it is empty, the bit
// is zero.
__global__ class KernelBitmap {
public:
  __global__ KernelBitmap() = default;

  KernelBitmap(size_t size, unsigned int *data) {
    size_ = size;
    data_ = data;
  }

  // __global__ ~KernelBitmap() { delete[] data_; }

  //__device__ void Init(size_t size) {
  //  free(data_);
  //  size_ = size;
  //  data_ = new unsigned long long[WORD_OFFSET(size_) + 1];
  //}

  __device__ void Clear() {
    size_t bm_size = GPU_WORD_OFFSET(size_);
    for (size_t i = 0; i <= bm_size; i++)
      data_[i] = 0;
  }

  __device__ void Fill() {
    size_t bm_size = GPU_WORD_OFFSET(size_);
    for (size_t i = 0; i < bm_size; i++) {
      data_[i] = 0xffff;
    }
    data_[bm_size] = 0;
    for (size_t i = (bm_size << 6); i < size_; i++) {
      data_[bm_size] |= 1u << GPU_BIT_OFFSET(i);
    }
  }

  __device__ bool GetBit(size_t i) const {
    if (i > size_)
      return 0;
    return data_[GPU_WORD_OFFSET(i)] & (1u << GPU_BIT_OFFSET(i));
  }

  __device__ void SetBit(size_t i) {
    if (i > size_)
      return;
    //*(data_ + WORD_OFFSET(i)) =
    //    *((data_ + WORD_OFFSET(i))) | (1ul << BIT_OFFSET(i));

    //atomicOr(data_ + WORD_OFFSET(i), 1ull << BIT_OFFSET(i));
    atomicOr(data2_ + GPU_WORD_OFFSET(i), 1 << GPU_BIT_OFFSET(i));
  }

  __device__ size_t Count() const {
    size_t count = 0;
    for (size_t i = 0; i <= size_; i++) {
      if (GetBit(i))
        count++;
    }
    // for (size_t i = 0; i <= WORD_OFFSET(size_); i++) {
    //   auto x = data_[i];
    //   x = (x & (0x5555555555555555)) + ((x >> 1) & (0x5555555555555555));
    //   x = (x & (0x3333333333333333)) + ((x >> 2) & (0x3333333333333333));
    //   x = (x & (0x0f0f0f0f0f0f0f0f)) + ((x >> 4) & (0x0f0f0f0f0f0f0f0f));
    //   x = (x & (0x00ff00ff00ff00ff)) + ((x >> 8) & (0x00ff00ff00ff00ff));
    //   x = (x & (0x0000ffff0000ffff)) + ((x >> 16) & (0x0000ffff0000ffff));
    //   x = (x & (0x00000000ffffffff)) + ((x >> 32) & (0x00000000ffffffff));
    //   count += (size_t)x;
    // }
    return count;
  };

  __device__ size_t get_size() const { return size_; }

  __device__ void SetSize(size_t size) { size_ = size; }

private:
  size_t size_;
  unsigned int *data_ = nullptr;
  unsigned int *data2_ =  nullptr;
};

} // namespace gpu
} // namespace core
} // namespace hyperblocker
} // namespace sics
#endif // INC_51_11_ER_HYPERBLOCKER_CORE_GPU_KERNEL_DATA_STRUCTURES_KERNEL_BITMAP_CUH_
