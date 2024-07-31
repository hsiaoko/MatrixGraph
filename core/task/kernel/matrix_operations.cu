#include "core/task/kernel/matrix_operations.cuh"

#include <cuda_runtime.h>
#include <iostream>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

#define WORD_OFFSET(i) (i >> 6)
#define BIT_OFFSET(i) (i & 0x3f)

struct Parameters {
  uint64_t *matrix_a;
  uint64_t *matrix_b;
  uint64_t *matrix_c;
  uint64_t m;
  uint64_t k;
  uint64_t n;
};

__device__ static inline uint64_t get_k_bits(uint64_t *data, uint64_t start,
                                             uint64_t end) {
  uint64_t start_word = WORD_OFFSET(start);
  uint64_t end_word = WORD_OFFSET(end);
  uint64_t start_bit = BIT_OFFSET(start);
  uint64_t end_bit = BIT_OFFSET(end);

  uint64_t result = 0;

  if (start_word == end_word) {
    uint64_t mask = (1ull << (end_bit - start_bit + 1)) - 1 << start_bit;
    result = (data[start_word] & mask) >> start_bit;
  } else {
    uint64_t mask = (1ull << (64 - start_bit)) - 1;
    result = (data[start_word] & mask) << (end_bit - start_bit);

    for (uint64_t i = start_word + 1; i < end_word; ++i) {
      result |= data[i] << (64 - start_bit);
      start_bit = 0;
    }

    mask = (1ull) << (end_bit + 1) - 1;
    result |= (data[end_word] & mask) << (64 - start_bit);
  }
  return result;
}

__device__ static inline uint64_t set_bit(uint64_t *data, uint64_t i) {
  *(data + WORD_OFFSET(i)) |= (1ull << BIT_OFFSET(i));
}

__device__ static inline bool get_bit(uint64_t *data, uint64_t i) {
  return data[WORD_OFFSET(i)] & (1ull << BIT_OFFSET(i));
}

__global__ void matrix_and_kernel(Parameters params) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  for (unsigned int row_a = tid; row_a < params.m; row_a += step) {
    uint64_t processor_word_a =
        get_k_bits(params.matrix_a, row_a * params.k, (row_a + 1) * params.k);
    for (unsigned int row_b = 0; row_b < params.n; ++row_b) {
      uint64_t processor_word_b =
          get_k_bits(params.matrix_b, row_b * params.k, (row_b + 1) * params.k);

      if ((processor_word_a & processor_word_b) == 0) {
        continue;
      } else {
      // printf("set: %d, %d\n", row_a, row_b);
        set_bit(params.matrix_c, row_a * params.n + row_b);
      }
    }
  }
  //*(params.matrix_c) = 1;
}

void MatrixOperationsKernelWrapper::MatrixBitAnd(
    const cudaStream_t &stream,
    const data_structures::DeviceOwnedBuffer<uint64_t> &matrix_a_buf,
    const data_structures::DeviceOwnedBuffer<uint64_t> &matrix_b_buf,
    data_structures::DeviceOwnedBuffer<uint64_t> *matrix_c_buf, uint32_t m,
    uint32_t k, uint32_t n) {
  dim3 dimBlock(1);
  dim3 dimGrid(1);

  Parameters params{.matrix_a = matrix_a_buf.GetPtr(),
                    .matrix_b = matrix_b_buf.GetPtr(),
                    .matrix_c = matrix_c_buf->GetPtr(),
                    .m = m,
                    .k = k,
                    .n = n};

  matrix_and_kernel<<<1, 1, 0, stream>>>(params);
}

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics