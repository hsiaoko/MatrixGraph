#include "core/task/kernel/matrix_operations.cuh"

#include <cuda_runtime.h>
#include <iostream>

#include "core/util/cuda_check.cuh"

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

struct ParametersFillTiles {
  unsigned long tile_size;
  unsigned long n_strips;
  unsigned long n_nz_tile_a;
  unsigned long n_nz_tile_b;
  unsigned long n_nz_tile_c;
  unsigned long tile_unit;
  unsigned long tile_buffer_size;
  unsigned long long *layout_matrix_c;
  unsigned *tile_offset_row_a;
  unsigned *tile_offset_row_b;
  unsigned *tile_offset_row_c;
  unsigned *tile_row_idx_a;
  unsigned *tile_row_idx_b;
  unsigned *tile_row_idx_c;
  unsigned *tile_col_idx_a;
  unsigned *tile_col_idx_b;
  unsigned *tile_col_idx_c;
  uint8_t *data_a;
  uint8_t *data_b;
  uint8_t *data_c;
  uint64_t *csr_offset_a = nullptr;
  uint64_t *csr_offset_b = nullptr;
  uint64_t *csr_offset_c = nullptr;
};

struct ParametersCount {
  unsigned n_nz_tile;
  unsigned n_uint64_t;
  unsigned long long *data;
  unsigned long long *count;
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
}

__device__ static inline bool get_bit(uint64_t *data, uint64_t i) {
  return data[WORD_OFFSET(i)] & (1ull << BIT_OFFSET(i));
}

__device__ static inline unsigned long long
count_per_uint64_t(unsigned long long x) {

  x = (x & (0x5555555555555555)) + ((x >> 1) & (0x5555555555555555));
  x = (x & (0x3333333333333333)) + ((x >> 2) & (0x3333333333333333));
  x = (x & (0x0f0f0f0f0f0f0f0f)) + ((x >> 4) & (0x0f0f0f0f0f0f0f0f));
  x = (x & (0x00ff00ff00ff00ff)) + ((x >> 8) & (0x00ff00ff00ff00ff));
  x = (x & (0x0000ffff0000ffff)) + ((x >> 16) & (0x0000ffff0000ffff));
  x = (x & (0x00000000ffffffff)) + ((x >> 32) & (0x00000000ffffffff));
  return x;
}

__device__ static inline bool single_thread_matrix_bit_and(
    unsigned long tile_size, unsigned long long *matrix_a,
    unsigned long long *matrix_b, unsigned long long *matrix_c) {
  for (unsigned int row_a = 0; row_a < tile_size; ++row_a) {
    for (unsigned int row_b = 0; row_b < tile_size; ++row_b) {
      for (unsigned int k = 0; k < tile_size; k += kProcessorWordSize) {
        uint64_t processor_word_a = get_aligned_k_bits(
            (uint64_t *)matrix_a, row_a * tile_size, (row_a + 1) * tile_size);
        uint64_t processor_word_b = get_aligned_k_bits(
            (uint64_t *)matrix_b, row_b * tile_size, (row_b + 1) * tile_size);

        if ((processor_word_a & processor_word_b) == 0) {
          continue;
        } else {
          set_bit((unsigned long long *)matrix_c,
                  (unsigned long long)row_a * tile_size + row_b);
          break;
        }
      }
    }
  }
}

__device__ static inline void
find_intersection(unsigned size_l, unsigned size_r, unsigned *data_l,
                  unsigned *data_r, unsigned *data_out_l, unsigned *data_out_r,
                  unsigned *n_intersections) {
  unsigned i = 0;
  unsigned j = 0;

  while (i < size_l && j < size_r) {
    if (data_l[i] < data_r[j]) {
      i++;
    } else if (data_l[i] > data_r[j]) {
      j++;
    } else {
      data_out_l[*n_intersections] = i;
      data_out_r[*n_intersections] = j;
      (*n_intersections)++;
      i++;
      j++;
    }
  }
}

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
init_bit_tiled_matrix_metadata_kernel(ParametersInitBitTiledMatrix params) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  for (unsigned int i = tid; i < params.tile_size; i += step) {
    for (unsigned int j = 0; j < params.tile_size; j++) {
      if (get_bit((uint64_t *)params.layout_matrix,
                  (uint64_t)(i * params.tile_size + j))) {
        unsigned long long pre_element_count_val =
            pre_element_count((uint64_t *)params.layout_matrix,
                              (uint64_t)i * params.tile_size + j);
        params.tile_row_idx[pre_element_count_val] = i;
        params.tile_col_idx[pre_element_count_val] = j;
        atomicAdd(params.tile_offset_row + i, (unsigned)1);
      }
    }
  }
}

static __global__ void fill_tiles_kernel(ParametersFillTiles params) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  for (unsigned int i = tid; i < params.n_nz_tile_c; i += step) {
    unsigned int x = params.tile_row_idx_c[i];
    unsigned int y = params.tile_col_idx_c[i];

    unsigned long long *matrix_c =
        ((unsigned long long *)params.data_c) + params.tile_unit * i;

    unsigned int nz_tile_line_x =
        params.tile_offset_row_a[x + 1] - params.tile_offset_row_a[x];

    unsigned int nz_tile_line_y =
        params.tile_offset_row_b[y + 1] - params.tile_offset_row_b[y];

    unsigned nz_idx_x = 0;
    unsigned nz_idx_y = 0;

    while (nz_idx_x < nz_tile_line_x && nz_idx_y < nz_tile_line_y) {
      if (*(params.tile_col_idx_a + params.tile_offset_row_a[x] + nz_idx_x) <
          *(params.tile_col_idx_b + params.tile_offset_row_a[y] + nz_idx_y)) {
        nz_idx_x++;
      } else if (*(params.tile_col_idx_a + params.tile_offset_row_a[x] +
                   nz_idx_x) > *(params.tile_col_idx_b +
                                 params.tile_offset_row_a[y] + nz_idx_y)) {
        nz_idx_y++;
      } else {
        // perform bit and between (params.tile_offset_row_a[x] +
        // idx_intersection_l[l])-th  tile and (params.tile_offset_row_b[y] +
        // idx_intersection_r[r])-th tile.

        unsigned long long *matrix_a =
            ((unsigned long long *)params.data_a) +
            params.tile_unit * (params.tile_offset_row_a[x] + nz_idx_x);
        unsigned long long *matrix_b =
            ((unsigned long long *)params.data_b) +
            params.tile_unit * (params.tile_offset_row_b[y] + nz_idx_y);

        single_thread_matrix_bit_and(params.tile_size, matrix_a, matrix_b,
                                     matrix_c);

        nz_idx_x++;
        nz_idx_y++;
      }
    }
  }
}

static __global__ void fill_csr_tiles_kernel(ParametersFillTiles params) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  printf("%d\n", tid);
  for (unsigned int i = tid; i < params.n_nz_tile_c; i += step) {
    unsigned int x = params.tile_row_idx_c[i];
    unsigned int y = params.tile_col_idx_c[i];

    unsigned long long *matrix_c =
        ((unsigned long long *)params.data_c) + params.tile_unit * i;

    unsigned int nz_tile_line_x =
        params.tile_offset_row_a[x + 1] - params.tile_offset_row_a[x];

    unsigned int nz_tile_line_y =
        params.tile_offset_row_b[y + 1] - params.tile_offset_row_b[y];

    unsigned nz_idx_x = 0;
    unsigned nz_idx_y = 0;
    while (nz_idx_x < nz_tile_line_x && nz_idx_y < nz_tile_line_y) {
      if (*(params.tile_col_idx_a + params.tile_offset_row_a[x] + nz_idx_x) <
          *(params.tile_col_idx_b + params.tile_offset_row_a[y] + nz_idx_y)) {
        nz_idx_x++;
      } else if (*(params.tile_col_idx_a + params.tile_offset_row_a[x] +
                   nz_idx_x) > *(params.tile_col_idx_b +
                                 params.tile_offset_row_a[y] + nz_idx_y)) {
        nz_idx_y++;
      } else {

        size_t csr_offset_a =
            params.csr_offset_a[params.tile_offset_row_a[x] + nz_idx_x];
        size_t csr_offset_b =
            params.csr_offset_b[params.tile_offset_row_b[y] + nz_idx_y];

        printf("offset: %ld, offset: %ld\n", csr_offset_a, csr_offset_b);

        nz_idx_x++;
        nz_idx_y++;
      }
    }
  }
}

static __global__ void count_kernel(ParametersCount params) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  unsigned long long local_count = 0;
  for (unsigned i = tid; i < params.n_uint64_t; i += step) {
    local_count += count_per_uint64_t(params.data[i]);
  }
  atomicAdd(params.count, local_count);
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
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    CUDA_CHECK(err);
  }
}

void MatrixOperationsKernelWrapper::MatrixBitCount(
    const cudaStream_t &stream,
    const data_structures::DeviceOwnedBuffer<uint64_t> &matrix_buf,
    data_structures::DeviceOwnedBuffer<uint64_t> *count_buf, uint64_t size) {

  dim3 dimBlock(256);
  dim3 dimGrid(256);
  ParametersForMatrixBitCount params{
      .data = reinterpret_cast<unsigned long long *>(matrix_buf.GetPtr()),
      .count = reinterpret_cast<unsigned long long *>(count_buf->GetPtr()),
      .size = size};
  matrix_count_kernel<<<dimBlock, dimGrid, 0, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    CUDA_CHECK(err);
  }
}

void MatrixOperationsKernelWrapper::InitBitTiledMatrixMetadataByLayoutMatrix(
    const cudaStream_t &stream,
    const data_structures::DeviceOwnedBuffer<uint64_t> &layout_matrix,
    data_structures::DeviceOwnedBuffer<uint32_t> *tile_offset_row,
    data_structures::DeviceOwnedBuffer<uint32_t> *tile_row_idx,
    data_structures::DeviceOwnedBuffer<uint32_t> *tile_col_idx,
    uint32_t tile_size) {

  dim3 dimBlock(64);
  dim3 dimGrid(64);
  ParametersInitBitTiledMatrix params{
      .layout_matrix =
          reinterpret_cast<unsigned long long *>(layout_matrix.GetPtr()),
      .tile_offset_row =
          reinterpret_cast<unsigned *>(tile_offset_row->GetPtr()),
      .tile_row_idx = reinterpret_cast<unsigned *>(tile_row_idx->GetPtr()),
      .tile_col_idx = reinterpret_cast<unsigned *>(tile_col_idx->GetPtr()),
      .tile_size = tile_size};

  init_bit_tiled_matrix_metadata_kernel<<<dimBlock, dimGrid, 0, stream>>>(
      params);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    CUDA_CHECK(err);
  }
}

void MatrixOperationsKernelWrapper::FillTiles(
    const cudaStream_t &stream, size_t tile_size, size_t n_strips,
    size_t n_nz_tile_a, size_t n_nz_tile_b, size_t n_nz_tile_c,
    const data_structures::UnifiedOwnedBuffer<uint64_t> &layout_matrix_c,
    const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_offset_row_a,
    const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_offset_row_b,
    const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_offset_row_c,
    const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_row_idx_a,
    const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_row_idx_b,
    const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_row_idx_c,
    const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_col_idx_a,
    const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_col_idx_b,
    const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_col_idx_c,
    const data_structures::UnifiedOwnedBuffer<uint64_t> &data_a,
    const data_structures::UnifiedOwnedBuffer<uint64_t> &data_b,
    data_structures::UnifiedOwnedBuffer<uint64_t> *data_c) {

  dim3 dimBlock(256);
  dim3 dimGrid(128);

  auto tile_unit = max(1u, (WORD_OFFSET(tile_size * tile_size)));
  auto tile_buffer_size =
      sizeof(uint64_t) * max(1u, WORD_OFFSET(tile_size * tile_size));

  ParametersFillTiles params{.tile_size = tile_size,
                             .n_strips = n_strips,
                             .n_nz_tile_a = n_nz_tile_a,
                             .n_nz_tile_b = n_nz_tile_b,
                             .n_nz_tile_c = n_nz_tile_c,
                             .tile_unit = tile_unit,
                             .tile_buffer_size = tile_buffer_size,
                             .layout_matrix_c =
                                 (unsigned long long *)layout_matrix_c.GetPtr(),
                             .tile_offset_row_a = tile_offset_row_a.GetPtr(),
                             .tile_offset_row_b = tile_offset_row_b.GetPtr(),
                             .tile_offset_row_c = tile_offset_row_c.GetPtr(),
                             .tile_row_idx_a = tile_row_idx_a.GetPtr(),
                             .tile_row_idx_b = tile_row_idx_b.GetPtr(),
                             .tile_row_idx_c = tile_row_idx_c.GetPtr(),
                             .tile_col_idx_a = tile_col_idx_a.GetPtr(),
                             .tile_col_idx_b = tile_col_idx_b.GetPtr(),
                             .tile_col_idx_c = tile_col_idx_c.GetPtr(),
                             .data_a = (uint8_t *)data_a.GetPtr(),
                             .data_b = (uint8_t *)data_b.GetPtr(),
                             .data_c = (uint8_t *)(data_c->GetPtr())};

  fill_tiles_kernel<<<dimBlock, dimGrid, 0, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    CUDA_CHECK(err);
  }
}

void MatrixOperationsKernelWrapper::FillCSRTiles(
    const cudaStream_t &stream, size_t tile_size, size_t n_strips,
    size_t n_nz_tile_a, size_t n_nz_tile_b, size_t n_nz_tile_c,
    const data_structures::UnifiedOwnedBuffer<uint64_t> &layout_matrix_c,
    const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_offset_row_a,
    const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_offset_row_b,
    const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_offset_row_c,
    const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_row_idx_a,
    const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_row_idx_b,
    const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_row_idx_c,
    const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_col_idx_a,
    const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_col_idx_b,
    const data_structures::UnifiedOwnedBuffer<uint32_t> &tile_col_idx_c,
    const data_structures::UnifiedOwnedBuffer<uint64_t> &csr_offset_a,
    const data_structures::UnifiedOwnedBuffer<uint64_t> &csr_offset_b,
    const data_structures::UnifiedOwnedBuffer<uint64_t> &csr_offset_c,
    const data_structures::UnifiedOwnedBuffer<uint8_t> &data_a,
    const data_structures::UnifiedOwnedBuffer<uint8_t> &data_b,
    data_structures::UnifiedOwnedBuffer<uint8_t> *data_c) {

  dim3 dimBlock(1);
  dim3 dimGrid(1);

  auto tile_unit = max(1u, (WORD_OFFSET(tile_size * tile_size)));
  auto tile_buffer_size =
      sizeof(uint64_t) * max(1u, WORD_OFFSET(tile_size * tile_size));

  ParametersFillTiles params{
      .tile_size = tile_size,
      .n_strips = n_strips,
      .n_nz_tile_a = n_nz_tile_a,
      .n_nz_tile_b = n_nz_tile_b,
      .n_nz_tile_c = n_nz_tile_c,
      .tile_unit = tile_unit,
      .tile_buffer_size = tile_buffer_size,
      .layout_matrix_c = (unsigned long long *)layout_matrix_c.GetPtr(),
      .tile_offset_row_a = tile_offset_row_a.GetPtr(),
      .tile_offset_row_b = tile_offset_row_b.GetPtr(),
      .tile_offset_row_c = tile_offset_row_c.GetPtr(),
      .tile_row_idx_a = tile_row_idx_a.GetPtr(),
      .tile_row_idx_b = tile_row_idx_b.GetPtr(),
      .tile_row_idx_c = tile_row_idx_c.GetPtr(),
      .tile_col_idx_a = tile_col_idx_a.GetPtr(),
      .tile_col_idx_b = tile_col_idx_b.GetPtr(),
      .tile_col_idx_c = tile_col_idx_c.GetPtr(),
      .data_a = data_a.GetPtr(),
      .data_b = data_b.GetPtr(),
      .data_c = data_c->GetPtr(),
      .csr_offset_a = csr_offset_a.GetPtr(),
      .csr_offset_b = csr_offset_b.GetPtr(),
      .csr_offset_c = csr_offset_c.GetPtr(),
  };

  fill_csr_tiles_kernel<<<dimBlock, dimGrid, 0, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    CUDA_CHECK(err);
  }
}

void MatrixOperationsKernelWrapper::Count(
    const cudaStream_t &stream, size_t tile_size, size_t n_strips,
    size_t n_nz_tile, const data_structures::UnifiedOwnedBuffer<uint64_t> &data,
    data_structures::UnifiedOwnedBuffer<uint64_t> *count) {

  dim3 dimBlock(256);
  dim3 dimGrid(256);

  unsigned n_uint64_t = data.GetSize() / sizeof(uint64_t);
  ParametersCount params{.n_uint64_t = n_uint64_t,
                         .data = (unsigned long long *)(data.GetPtr()),
                         .count = (unsigned long long *)(count->GetPtr())};

  count_kernel<<<dimGrid, dimBlock, 0, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    CUDA_CHECK(err);
  }
}

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics