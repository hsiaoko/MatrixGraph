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
#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

static const uint32_t kProcessorWordSize = 64;

struct ParametersForMatrixBitAnd {
  uint64_t *matrix_a;
  uint64_t *matrix_b;
  uint64_t *matrix_c;
  uint64_t m;
  uint64_t k;
  uint64_t n;
};

struct ParametersForMatrixBitCount {
  unsigned long long *data;
  unsigned long long *count;
  unsigned long long size;
};

struct ParametersInitBitTiledMatrix {
  unsigned long long *layout_matrix;
  unsigned *tile_offset_row;
  unsigned *tile_row_idx;
  unsigned *tile_col_idx;
  unsigned long tile_size;
};

__device__ static inline uint64_t get_bit(uint64_t *data, size_t i,
                                          size_t size) {
  if (i > size)
    return 0;
  return data[WORD_OFFSET(i)] & (1ull << BIT_OFFSET(i));
}

__device__ static inline bool drop() { return false; }

__device__ static inline uint64_t
get_aligned_k_bits(uint64_t *data, uint64_t start, uint64_t end) {
  uint64_t start_word = WORD_OFFSET(start);
  uint64_t end_word = WORD_OFFSET(end);

  uint64_t start_bit = BIT_OFFSET(start);
  uint64_t end_bit = BIT_OFFSET(end);

  uint64_t result = 0;
  uint64_t mask = ((1ull << (end_bit - start_bit)) - 1) << start_bit;

  result = (data[start_word] & mask) >> start_bit;
  return result;
}

__device__ static inline uint64_t set_bit(unsigned long long *data,
                                          unsigned long long i) {
  atomicOr(data + WORD_OFFSET(i), (1ull << BIT_OFFSET(i)));
  // *(data + WORD_OFFSET(i)) |= (1ull << BIT_OFFSET(i));
}

__device__ static inline bool get_bit(uint64_t *data, uint64_t i) {
  return data[WORD_OFFSET(i)] & (1ull << BIT_OFFSET(i));
}

__device__ static inline size_t pre_element_count(const uint64_t *data,
                                                  uint64_t idx) {

  size_t count = 0;
  size_t bm_size = WORD_OFFSET(idx);
  size_t idx_offset = WORD_OFFSET(idx);
  size_t idx_bit_offset = BIT_OFFSET(idx);

  for (size_t i = 0; i <= bm_size; i++) {
    uint64_t x = 0;
    if (i == idx_offset) {
      uint64_t mask = (1ul << idx_bit_offset) - 1;
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
    count += (size_t)x;
  }

  return count;
};

static __global__ void matrix_and_kernel(ParametersForMatrixBitAnd params) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  for (unsigned int row_a = tid; row_a < params.m; row_a += step) {
    for (unsigned int row_b = 0; row_b < params.n; ++row_b) {
      for (unsigned int k = 0; k < params.k; k += kProcessorWordSize) {
        unsigned int upper_bound = min(k + kProcessorWordSize, params.k);

        uint64_t processor_word_a = get_aligned_k_bits(
            params.matrix_a, row_a * params.k, (row_a + 1) * params.k);
        uint64_t processor_word_b = get_aligned_k_bits(
            params.matrix_b, row_b * params.k, (row_b + 1) * params.k);

        if ((processor_word_a & processor_word_b) == 0) {
          continue;
        } else {
          set_bit((unsigned long long *)params.matrix_c,
                  (unsigned long long)row_a * params.n + row_b);
          break;
        }
      }
    }
  }
}

static __global__ void matrix_count_kernel(ParametersForMatrixBitCount params) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  for (unsigned int i = tid; i <= WORD_OFFSET(params.size); i += step) {
    unsigned long long x = params.data[i];
    x = (x & (0x5555555555555555)) + ((x >> 1) & (0x5555555555555555));
    x = (x & (0x3333333333333333)) + ((x >> 2) & (0x3333333333333333));
    x = (x & (0x0f0f0f0f0f0f0f0f)) + ((x >> 4) & (0x0f0f0f0f0f0f0f0f));
    x = (x & (0x00ff00ff00ff00ff)) + ((x >> 8) & (0x00ff00ff00ff00ff));
    x = (x & (0x0000ffff0000ffff)) + ((x >> 16) & (0x0000ffff0000ffff));
    x = (x & (0x00000000ffffffff)) + ((x >> 32) & (0x00000000ffffffff));
    atomicAdd(params.count, x);
  }
}

static __global__ void
init_bit_tiled_matrix_metadata(ParametersInitBitTiledMatrix params) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  for (unsigned int i = tid; i < params.tile_size; i += step) {
    for (unsigned int j = 0; j < params.tile_size; j++) {
      if (get_bit((uint64_t *)params.layout_matrix,
                  (uint64_t)(i * params.tile_size + j))) {
        unsigned long long pre_element_count_val =
            pre_element_count((uint64_t *)params.layout_matrix,
                              (uint64_t)i * params.tile_size + j);
        params.tile_col_idx[pre_element_count_val] = i;
        params.tile_row_idx[pre_element_count_val] = j;
        atomicAdd(params.tile_offset_row + i, (unsigned)1);
      }
    }
  }
}

void MatrixOperationsKernelWrapper::MatrixBitAnd(
    const cudaStream_t &stream,
    const data_structures::DeviceOwnedBuffer<uint64_t> &matrix_a_buf,
    const data_structures::DeviceOwnedBuffer<uint64_t> &matrix_b_buf,
    data_structures::DeviceOwnedBuffer<uint64_t> *matrix_c_buf, uint32_t m,
    uint32_t k, uint32_t n) {
  dim3 dimBlock(64);
  dim3 dimGrid(64);

  ParametersForMatrixBitAnd params{.matrix_a = matrix_a_buf.GetPtr(),
                                   .matrix_b = matrix_b_buf.GetPtr(),
                                   .matrix_c = matrix_c_buf->GetPtr(),
                                   .m = m,
                                   .k = k,
                                   .n = n};

  matrix_and_kernel<<<dimBlock, dimGrid, 0, stream>>>(params);
}

void MatrixOperationsKernelWrapper::MatrixBitCount(
    const cudaStream_t &stream,
    const data_structures::DeviceOwnedBuffer<uint64_t> &matrix_buf,
    data_structures::DeviceOwnedBuffer<uint64_t> *count_buf, uint64_t size) {

  dim3 dimBlock(64);
  dim3 dimGrid(64);
  ParametersForMatrixBitCount params{
      .data = reinterpret_cast<unsigned long long *>(matrix_buf.GetPtr()),
      .count = reinterpret_cast<unsigned long long *>(count_buf->GetPtr()),
      .size = size};
  matrix_count_kernel<<<dimBlock, dimGrid, 0, stream>>>(params);
}

void MatrixOperationsKernelWrapper::InitBitTiledMatrixMetadataByLayoutMatrix(
    const cudaStream_t &stream,
    const data_structures::DeviceOwnedBuffer<uint64_t> &layout_matrix,
    data_structures::DeviceOwnedBuffer<uint32_t> *tile_offset_row,
    data_structures::DeviceOwnedBuffer<uint32_t> *tile_row_idx,
    data_structures::DeviceOwnedBuffer<uint32_t> *tile_col_idx,
    uint32_t tile_size) {

  std::cout << "InitBitTiledMatrixedByLayoutMatrix" << std::endl;
  dim3 dimBlock(1);
  dim3 dimGrid(1);
  ParametersInitBitTiledMatrix params{
      .layout_matrix =
          reinterpret_cast<unsigned long long *>(layout_matrix.GetPtr()),
      .tile_offset_row =
          reinterpret_cast<unsigned *>(tile_offset_row->GetPtr()),
      .tile_row_idx = reinterpret_cast<unsigned *>(tile_row_idx->GetPtr()),
      .tile_col_idx = reinterpret_cast<unsigned *>(tile_col_idx->GetPtr()),
      .tile_size = tile_size};

  init_bit_tiled_matrix_metadata<<<dimBlock, dimGrid, 0, stream>>>(params);
}

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics