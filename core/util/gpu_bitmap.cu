#include "core/util/cuda_check.cuh"
#include "core/util/gpu_bitmap.cuh"
#include <cuda_runtime.h>

namespace sics {
namespace matrixgraph {
namespace core {
namespace util {

#define GPU_WORD_OFFSET(i) (i >> 6)
#define GPU_BIT_OFFSET(i) (i & 0x3f)

static __global__ void matrix_count_kernel(size_t size, uint64_t* data,
                                           unsigned long long* count) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  unsigned long long bm_size = GPU_WORD_OFFSET(size);
  unsigned long long idx_offset = GPU_WORD_OFFSET(size);
  unsigned long long idx_bit_offset = GPU_BIT_OFFSET(size);

  for (unsigned int i = tid; i <= bm_size; i += step) {
    unsigned long long x = data[i];

    if (i == idx_offset) {
      unsigned long long mask = (1ul << idx_bit_offset) - 1;
      x = data[i] & mask;
    } else {
      x = data[i];
    }
    x = (x & (0x5555555555555555)) + ((x >> 1) & (0x5555555555555555));
    x = (x & (0x3333333333333333)) + ((x >> 2) & (0x3333333333333333));
    x = (x & (0x0f0f0f0f0f0f0f0f)) + ((x >> 4) & (0x0f0f0f0f0f0f0f0f));
    x = (x & (0x00ff00ff00ff00ff)) + ((x >> 8) & (0x00ff00ff00ff00ff));
    x = (x & (0x0000ffff0000ffff)) + ((x >> 16) & (0x0000ffff0000ffff));
    x = (x & (0x00000000ffffffff)) + ((x >> 32) & (0x00000000ffffffff));
    atomicAdd(count, x);
  }
}

GPUBitmap::GPUBitmap(size_t size) { Init(size); }

GPUBitmap::GPUBitmap(size_t size, uint64_t* data) { Init(size, data); }

GPUBitmap::~GPUBitmap() {
  cudaFree(data_);
  size_ = 0;
}

void GPUBitmap::Init(size_t size) {
  cudaFree(data_);

  size_ = size;
  cudaMallocManaged(&data_,
                    sizeof(unsigned long long) * (GPU_WORD_OFFSET(size_) + 1));
  cudaMemset(data_, 0,
             sizeof(unsigned long long) * (GPU_WORD_OFFSET(size_) + 1));
}

void GPUBitmap::Init(size_t size, uint64_t* data) {
  cudaFree(data_);
  size_ = size;
  data_ = data;
}

size_t GPUBitmap::GPUCount() const {
  dim3 dimBlock(64);
  dim3 dimGrid(64);

  unsigned long long* count_ptr;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaMallocManaged(&count_ptr, sizeof(unsigned long long));
  cudaMemset(count_ptr, 0, sizeof(unsigned long long));

  matrix_count_kernel<<<dimGrid, dimBlock, 0, stream>>>(size_, data_,
                                                        count_ptr);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) CUDA_CHECK(err);

  size_t count = *count_ptr;
  cudaFree(count_ptr);

  return count;
}

size_t GPUBitmap::GPUPreElementCount(size_t idx) const {
  dim3 dimBlock(128);
  dim3 dimGrid(128);

  unsigned long long* count_ptr;
  cudaSetDevice(idx % 4);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaMallocManaged(&count_ptr, sizeof(unsigned long long));
  cudaMemset(count_ptr, 0, sizeof(unsigned long long));

  matrix_count_kernel<<<dimGrid, dimBlock, 0, stream>>>(idx, data_, count_ptr);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) CUDA_CHECK(err);

  size_t count = *count_ptr;
  cudaFree(count_ptr);

  return count;
}

}  // namespace util
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics