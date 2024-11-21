#include "core/task/ppr_query.cuh"

#include <ctime>
#include <cuda_runtime.h>
#include <execution>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "core/common/host_algorithms.cuh"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/grid_bit_tiled_matrix.cuh"
#include "core/data_structures/host_buffer.cuh"
#include "core/data_structures/metadata.h"
#include "core/data_structures/unified_buffer.cuh"
#include "core/io/grid_bit_tiled_matrix_io.cuh"
#include "core/task/kernel/matrix_operations.cuh"
#include "core/util/atomic.h"
#include "core/util/bitmap.h"
#include "core/util/bitmap_no_ownership.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using sics::matrixgraph::core::data_structures::GridBitTiledMatrix;
using sics::matrixgraph::core::io::GridBitTiledMatrixIO;
using GirdTiledMatrix =
    sics::matrixgraph::core::data_structures::GridBitTiledMatrix;
using DeviceOwnedBufferUint64 =
    sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<uint64_t>;
using DeviceOwnedBufferUint32 =
    sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<uint32_t>;
using UnifiedOwnedBufferUint32 =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint32_t>;
using UnifiedOwnedBufferUint64 =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint64_t>;
using BufferUint64 = sics::matrixgraph::core::data_structures::Buffer<uint64_t>;
using BufferUint32 = sics::matrixgraph::core::data_structures::Buffer<uint32_t>;
using DeviceOwnedBufferUint64 =
    sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<uint64_t>;
using MatrixOperationsKernelWrapper =
    sics::matrixgraph::core::task::kernel::MatrixOperationsKernelWrapper;
using Bitmap = sics::matrixgraph::core::util::Bitmap;
using GPUBitmap = sics::matrixgraph::core::util::GPUBitmap;
using BitmapNoOwnerShip = sics::matrixgraph::core::util::BitmapNoOwnerShip;
using sics::matrixgraph::core::util::atomic::WriteAdd;
using TiledMatrixMetadata =
    sics::matrixgraph::core::data_structures::TiledMatrixMetadata;

// CUDA kernel to add elements of two arrays
__global__ void add(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

__host__ void PPRQuery::LoadData() {
  GridBitTiledMatrixIO grid_tiled_matrix_io;

  grid_tiled_matrix_io.Read(input_path_, &A_);
  grid_tiled_matrix_io.Read(input_path_transposed_, &B_);
  // A_->Print();
  // B_->Print();
  C_ = new GridBitTiledMatrix(A_->get_metadata());
}

__host__ void PPRQuery::InitResultMatrix() {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  auto block_a = A_->GetBitTileMatrixPtrByIdx(0);
  VertexID n_strips = block_a->GetMetadata().n_strips;
  VertexID tile_size = block_a->GetMetadata().tile_size;

  std::cout << "[InitResultMatrix]"
            << " Start - n_strips: " << n_strips << ", tile_size: " << tile_size
            << std::endl;

  VertexID M = A_->get_metadata().n_chunks;
  VertexID K = A_->get_metadata().n_chunks;
  VertexID N = B_->get_metadata().n_chunks;

  VertexID n_chunks = A_->get_metadata().n_chunks;

  std::vector<BufferUint64> buffers_matrix_a;
  std::vector<BufferUint64> buffers_matrix_b;
  std::vector<BufferUint64> buffers_matrix_c;
  buffers_matrix_a.resize(M * K);
  buffers_matrix_b.resize(N * K);
  buffers_matrix_c.resize(M * N);

  std::vector<BufferUint32> tile_offset_row;
  std::vector<BufferUint32> tile_count_row;
  std::vector<BufferUint32> tile_row_idx;
  std::vector<BufferUint32> tile_col_idx;
  tile_offset_row.resize(M * N);
  tile_count_row.resize(M * N);
  tile_row_idx.resize(M * N);
  tile_col_idx.resize(M * N);

  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [this, M, N, K, step, n_strips, &tile_count_row](auto w) {
                  for (VertexID i = w; i < M; i += step) {
                    for (VertexID j = 0; j < N; j++) {
                      tile_count_row[i * N + j].data =
                          new VertexID[n_strips + 1]();
                      tile_count_row[i * N + j].size =
                          sizeof(VertexID) * (n_strips + 1);
                    }
                  }
                });

  std::vector<DeviceOwnedBufferUint64> device_owned_buffers_matrix_a;
  std::vector<DeviceOwnedBufferUint64> device_owned_buffers_matrix_b;
  std::vector<DeviceOwnedBufferUint64> device_owned_buffers_matrix_c;
  device_owned_buffers_matrix_a.resize(M * K);
  device_owned_buffers_matrix_b.resize(K * N);
  device_owned_buffers_matrix_c.resize(M * N);

  std::vector<DeviceOwnedBufferUint32> device_owned_tile_offset_row;
  std::vector<DeviceOwnedBufferUint32> device_owned_tile_count_row;
  std::vector<DeviceOwnedBufferUint32> device_owned_tile_row_idx;
  std::vector<DeviceOwnedBufferUint32> device_owned_tile_col_idx;
  device_owned_tile_offset_row.resize(M * N);
  device_owned_tile_count_row.resize(M * N);
  device_owned_tile_row_idx.resize(M * N);
  device_owned_tile_col_idx.resize(M * N);

  std::vector<cudaStream_t> p_streams_vec;
  p_streams_vec.resize(M * N);
  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [this, M, N, K, step, &p_streams_vec, &mtx](auto w) {
                  for (VertexID i = w; i < N; i += step) {
                    for (VertexID j = 0; j < M; j++) {
                      cudaSetDevice(common::hash_function(i * N + j) % 4);
                      cudaStreamCreate(&p_streams_vec[i * N + j]);
                    }
                  }
                });

  // Step 1 compute layout_matrix for each block of matrix C.
  std::cout << "[InitResultMatrix] Computing layout matrix for each block ..."
            << std::endl;
  auto start_time_0 = std::chrono::system_clock::now();

  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, M, N, K, step, &device_owned_buffers_matrix_a,
       &device_owned_buffers_matrix_b, &device_owned_buffers_matrix_c,
       &p_streams_vec, &buffers_matrix_a, &buffers_matrix_b, &buffers_matrix_c,
       n_strips, n_chunks, tile_size, &mtx](auto w) {
        for (VertexID k = w; k < K; k += step) {
          for (VertexID i = 0; i < M; i++) {
            for (VertexID j = 0; j < N; j++) {

              auto block_a = A_->GetBitTileMatrixPtrByIdx(i * K + k);
              auto block_b = B_->GetBitTileMatrixPtrByIdx(j * K + k);

              if (block_a->GetMetadata().n_nz_tile == 0)
                continue;
              if (block_b->GetMetadata().n_nz_tile == 0)
                continue;

              cudaSetDevice(common::hash_function(i * N + j) % 4);
              cudaStream_t &p_stream = p_streams_vec[i * N + j];

              {
                std::lock_guard<std::mutex> lock(mtx);
                if (device_owned_buffers_matrix_c[i * N + j].GetPtr() ==
                    nullptr) {
                  buffers_matrix_c[i * N + j].size =
                      sizeof(uint64_t) * (WORD_OFFSET(n_strips * n_strips) + 1);
                  cudaHostAlloc(&buffers_matrix_c[i * N + j].data,
                                buffers_matrix_c[i * N + j].size,
                                cudaHostAllocDefault);

                  device_owned_buffers_matrix_c[i * N + j].Init(
                      buffers_matrix_c[i * N + j].GetSize(), p_stream);
                }
              }

              auto &matrix_a_buf = buffers_matrix_a[i * K + k];
              matrix_a_buf.data =
                  (uint64_t *)block_a->GetNzTileBitmapPtr()->data();
              matrix_a_buf.size =
                  block_a->GetNzTileBitmapPtr()->GetBufferSize();

              auto &matrix_b_buf = buffers_matrix_b[j * K + k];
              matrix_b_buf.data =
                  (uint64_t *)block_b->GetNzTileBitmapPtr()->data();
              matrix_b_buf.size =
                  block_b->GetNzTileBitmapPtr()->GetBufferSize();

              device_owned_buffers_matrix_a[i * K + k].Init(matrix_a_buf,
                                                            p_stream);
              device_owned_buffers_matrix_b[j * K + k].Init(matrix_b_buf,
                                                            p_stream);

              MatrixOperationsKernelWrapper::MatrixBitAnd(
                  p_stream, device_owned_buffers_matrix_a[i * K + k],
                  device_owned_buffers_matrix_b[j * K + k],
                  &device_owned_buffers_matrix_c[i * N + j], n_strips, n_strips,
                  n_strips);
            }
          }
        }
      });
  auto start_time_1 = std::chrono::system_clock::now();
  std::cout << "[InitResultMatrix] Tims - MatrixBitAnd elapsed:"
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_1 - start_time_0)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;

  // Step 3 compute Nonzero tile
  std::vector<BufferUint64> buffers_matrix_c_count;
  buffers_matrix_c_count.resize(M * N);
  std::vector<DeviceOwnedBufferUint64> device_owned_buffers_matrix_c_count;
  device_owned_buffers_matrix_c_count.resize(M * N);

  cudaDeviceSynchronize();
  std::cout << "[InitResultMatrix] Counting nz tile ..." << std::endl;
  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [this, M, N, K, step, &buffers_matrix_c, &mtx,
                 &device_owned_buffers_matrix_c_count, &p_streams_vec,
                 &device_owned_buffers_matrix_c, n_strips](auto w) {
                  for (VertexID i = w; i < M; i += step) {
                    for (VertexID j = 0; j < N; j++) {
                      if (device_owned_buffers_matrix_c[i * N + j].GetPtr() ==
                          nullptr) {
                        continue;
                      }
                      cudaSetDevice(common::hash_function(i * N + j) % 4);
                      cudaStream_t &p_stream = p_streams_vec[i * N + j];
                      device_owned_buffers_matrix_c_count[i * N + j].Init(
                          sizeof(uint64_t), p_stream);

                      MatrixOperationsKernelWrapper::MatrixBitCount(
                          p_stream, device_owned_buffers_matrix_c[i * N + j],
                          &device_owned_buffers_matrix_c_count[i * N + j],
                          n_strips * n_strips);
                    }
                  }
                });

  // Copy data back to the host.
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, M, N, K, step, &device_owned_buffers_matrix_a,
       &device_owned_buffers_matrix_b, &device_owned_buffers_matrix_c_count,
       &device_owned_buffers_matrix_c, &p_streams_vec, &buffers_matrix_c,
       &buffers_matrix_c_count, n_strips, n_chunks, tile_size, &mtx](auto w) {
        for (VertexID i = w; i < M; i += step) {
          for (VertexID j = 0; j < N; j++) {
            if (device_owned_buffers_matrix_c[i * N + j].GetPtr() == nullptr) {
              continue;
            }
            cudaSetDevice(common::hash_function(i * N + j) % 4);
            cudaStream_t &p_stream = p_streams_vec[i * N + j];
            buffers_matrix_c_count[i * N + j].size = sizeof(uint64_t);
            cudaHostAlloc(&buffers_matrix_c_count[i * N + j].data,
                          buffers_matrix_c_count[i * N + j].size,
                          cudaHostAllocDefault);
            cudaMemcpyAsync(
                buffers_matrix_c_count[i * N + j].GetPtr(),
                device_owned_buffers_matrix_c_count[i * N + j].GetPtr(),
                device_owned_buffers_matrix_c_count[i * N + j].GetSize(),
                cudaMemcpyDeviceToHost, p_stream);
            cudaMemcpyAsync(buffers_matrix_c[i * N + j].GetPtr(),
                            device_owned_buffers_matrix_c[i * N + j].GetPtr(),
                            device_owned_buffers_matrix_c[i * N + j].GetSize(),
                            cudaMemcpyDeviceToHost, p_stream);
          }
        }
      });
  auto start_time_2 = std::chrono::system_clock::now();
  std::cout << "[InitResultMatrix] Time - Counting nz tiles elapsed:"
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_2 - start_time_1)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;

  std::cout << "[InitResultMatrix] Allocating space for Matrix C ..."
            << std::endl;
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, M, N, K, step, &device_owned_buffers_matrix_a,
       &device_owned_buffers_matrix_b, &device_owned_buffers_matrix_c_count,
       &device_owned_buffers_matrix_c, &p_streams_vec, &buffers_matrix_c_count,
       &buffers_matrix_c, n_strips, n_chunks, tile_size, &mtx](auto w) {
        for (VertexID i = w; i < M; i += step) {
          for (VertexID j = 0; j < N; j++) {
            if (device_owned_buffers_matrix_c[i * N + j].GetPtr() == nullptr) {
              continue;
            }
            VertexID n_nz_tile = *buffers_matrix_c_count[i * N + j].GetPtr();
            if (n_nz_tile == 0)
              continue;
            cudaSetDevice(common::hash_function(i * N + j) % 4);
            cudaStream_t &p_stream = p_streams_vec[i * N + j];

            auto *bit_tiled_matrix_ptr =
                C_->GetBitTileMatrixPtrByIdx(i * N + j);

            TiledMatrixMetadata metadata{.n_strips = n_strips,
                                         .n_nz_tile = n_nz_tile,
                                         .tile_size = tile_size};
            bit_tiled_matrix_ptr->Init(
                metadata, new GPUBitmap(n_strips * tile_size,
                                        buffers_matrix_c[i * N + j].GetPtr()));
          }
        }
      });

  cudaDeviceSynchronize();
  std::cout << "[InitResultMatrix] Initialize BitTiledMatrix metadata."
            << std::endl;
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, M, N, K, step, &device_owned_buffers_matrix_c_count,
       &device_owned_buffers_matrix_c, &tile_count_row, &tile_row_idx,
       &tile_col_idx, &device_owned_tile_offset_row,
       &device_owned_tile_count_row, &device_owned_tile_row_idx,
       &device_owned_tile_col_idx,

       &p_streams_vec, &buffers_matrix_c_count, &buffers_matrix_c, n_strips,
       n_chunks, tile_size, &mtx](auto w) {
        for (VertexID i = w; i < M; i += step) {
          for (VertexID j = 0; j < N; j++) {
            if (device_owned_buffers_matrix_c[i * N + j].GetPtr() == nullptr) {
              continue;
            }
            VertexID n_nz_tile = *buffers_matrix_c_count[i * N + j].GetPtr();
            if (n_nz_tile == 0)
              continue;

            cudaSetDevice(common::hash_function(i * N + j) % 4);
            cudaStream_t &p_stream = p_streams_vec[i * N + j];
            auto *bit_tiled_matrix_ptr =
                C_->GetBitTileMatrixPtrByIdx(i * N + j);

            tile_col_idx[i * N + j].data =
                bit_tiled_matrix_ptr->GetTileColIdxPtr();
            tile_col_idx[i * N + j].size =
                bit_tiled_matrix_ptr->GetMetadata().n_nz_tile *
                sizeof(VertexID);

            tile_row_idx[i * N + j].data =
                bit_tiled_matrix_ptr->GetTileRowIdxPtr();
            tile_row_idx[i * N + j].size =
                bit_tiled_matrix_ptr->GetMetadata().n_nz_tile *
                sizeof(VertexID);

            device_owned_tile_row_idx[i * N + j].Init(tile_row_idx[i * N + j],
                                                      p_stream);
            device_owned_tile_col_idx[i * N + j].Init(tile_col_idx[i * N + j],
                                                      p_stream);
            device_owned_tile_count_row[i * N + j].Init(
                sizeof(VertexID) * (n_strips + 1), p_stream);

            MatrixOperationsKernelWrapper::
                InitBitTiledMatrixMetadataByLayoutMatrix(
                    p_stream, device_owned_buffers_matrix_c[i * N + j],
                    &device_owned_tile_count_row[i * N + j],
                    &device_owned_tile_row_idx[i * N + j],
                    &device_owned_tile_col_idx[i * N + j],
                    bit_tiled_matrix_ptr->GetMetadata().tile_size);
          }
        }
      });

  cudaDeviceSynchronize();
  auto start_time_3 = std::chrono::system_clock::now();
  std::cout << "[InitResultMatrix] Time - Computing meta elapsed:"
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_3 - start_time_2)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;

  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, M, N, K, step, &device_owned_buffers_matrix_c_count,
       &device_owned_buffers_matrix_c, &tile_count_row, &tile_row_idx,
       &tile_col_idx, &device_owned_tile_count_row, &device_owned_tile_row_idx,
       &device_owned_tile_col_idx,

       &p_streams_vec, &buffers_matrix_c_count, &buffers_matrix_c, n_strips,
       n_chunks, tile_size, &mtx](auto w) {
        for (VertexID i = w; i < M; i += step) {
          for (VertexID j = 0; j < N; j++) {
            if (device_owned_buffers_matrix_c[i * N + j].GetPtr() == nullptr) {
              continue;
            }
            VertexID n_nz_tile = *buffers_matrix_c_count[i * N + j].GetPtr();
            if (n_nz_tile == 0)
              continue;

            cudaSetDevice(common::hash_function(i * N + j) % 4);
            cudaStream_t &p_stream = p_streams_vec[i * N + j];

            cudaMemcpyAsync(tile_count_row[i * N + j].GetPtr(),
                            device_owned_tile_count_row[i * N + j].GetPtr(),
                            device_owned_tile_count_row[i * N + j].GetSize(),
                            cudaMemcpyDeviceToHost, p_stream);

            cudaMemcpyAsync(tile_row_idx[i * N + j].GetPtr(),
                            device_owned_tile_row_idx[i * N + j].GetPtr(),
                            device_owned_tile_row_idx[i * N + j].GetSize(),
                            cudaMemcpyDeviceToHost, p_stream);

            cudaMemcpyAsync(tile_col_idx[i * N + j].GetPtr(),
                            device_owned_tile_col_idx[i * N + j].GetPtr(),
                            device_owned_tile_col_idx[i * N + j].GetSize(),
                            cudaMemcpyDeviceToHost, p_stream);
          }
        }
      });
  cudaDeviceSynchronize();

  std::cout << "[InitResultMatrix] Computing tile_offset_row for each "
               "TiledMatrix ..."
            << std::endl;
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, M, N, K, step, &device_owned_buffers_matrix_c_count,
       &device_owned_buffers_matrix_c, &tile_offset_row, &tile_count_row,
       &tile_row_idx, &tile_col_idx, &device_owned_tile_offset_row,
       &device_owned_tile_row_idx, &device_owned_tile_col_idx,

       &p_streams_vec, &buffers_matrix_c_count, &buffers_matrix_c, n_strips,
       n_chunks, tile_size, &mtx](auto w) {
        for (VertexID i = w; i < M; i += step) {
          for (VertexID j = 0; j < N; j++) {
            if (device_owned_buffers_matrix_c[i * N + j].GetPtr() == nullptr) {
              continue;
            }
            VertexID n_nz_tile = *buffers_matrix_c_count[i * N + j].GetPtr();
            if (n_nz_tile == 0)
              continue;

            auto *bit_tiled_matrix_ptr =
                C_->GetBitTileMatrixPtrByIdx(i * N + j);
            auto tile_offset_row = bit_tiled_matrix_ptr->GetTileOffsetRowPtr();

            for (int t = 0; t < bit_tiled_matrix_ptr->GetMetadata().n_strips;
                 t++) {
              tile_offset_row[t + 1] =
                  tile_offset_row[t] + tile_count_row[i * N + j].GetPtr()[t];
            }
          }
        }
      });

  std::cout << "[InitResultMatrix] Done!" << std::endl;
  std::for_each(tile_count_row.begin(), tile_count_row.end(),
                [](auto &d) { delete[] d.data; });
  std::for_each(p_streams_vec.begin(), p_streams_vec.end(),
                [](auto &s) { cudaStreamDestroy(s); });
  std::for_each(buffers_matrix_c_count.begin(), buffers_matrix_c_count.end(),
                [](auto &d) { cudaFreeHost(d.data); });
}

__host__ void PPRQuery::FillTilesUnifiedMemory() {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  auto block_a = A_->GetBitTileMatrixPtrByIdx(0);
  VertexID n_strips = block_a->GetMetadata().n_strips;
  VertexID tile_size = block_a->GetMetadata().tile_size;
  auto tile_buffer_size =
      sizeof(uint64_t) * std::max(1u, WORD_OFFSET(tile_size * tile_size));

  VertexID M = A_->get_metadata().n_chunks;
  VertexID K = A_->get_metadata().n_chunks;
  VertexID N = B_->get_metadata().n_chunks;
  std::vector<cudaStream_t> p_streams_vec;
  p_streams_vec.resize(M * N);
  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [this, M, N, K, step, &p_streams_vec, &mtx](auto w) {
                  for (VertexID i = w; i < N; i += step) {
                    for (VertexID j = 0; j < M; j++) {
                      cudaSetDevice(common::hash_function(i * N + j) % 4);
                      cudaStreamCreate(&p_streams_vec[i * N + j]);
                    }
                  }
                });

  std::cout << "[FillTiles]"
            << " Start - n_strips: " << n_strips << ", tile_size: " << tile_size
            << std::endl;

  std::vector<BufferUint64> layout_matrix_c;
  std::vector<BufferUint32> tile_offset_row_a;
  std::vector<BufferUint32> tile_offset_row_b;
  std::vector<BufferUint32> tile_offset_row_c;
  std::vector<BufferUint32> tile_row_idx_a;
  std::vector<BufferUint32> tile_row_idx_b;
  std::vector<BufferUint32> tile_row_idx_c;
  std::vector<BufferUint32> tile_col_idx_a;
  std::vector<BufferUint32> tile_col_idx_b;
  std::vector<BufferUint32> tile_col_idx_c;
  std::vector<BufferUint64> data_a;
  std::vector<BufferUint64> data_b;
  std::vector<BufferUint64> data_c;

  layout_matrix_c.resize(M * N);
  tile_offset_row_a.resize(M * K);
  tile_offset_row_b.resize(N * K);
  tile_offset_row_c.resize(M * N);
  tile_row_idx_a.resize(M * K);
  tile_row_idx_b.resize(N * K);
  tile_row_idx_c.resize(M * N);
  tile_col_idx_a.resize(M * K);
  tile_col_idx_b.resize(N * K);
  tile_col_idx_c.resize(M * N);
  data_a.resize(M * K);
  data_b.resize(N * K);
  data_c.resize(M * N);

  std::vector<UnifiedOwnedBufferUint64> unified_layout_matrix_c;
  std::vector<UnifiedOwnedBufferUint32> unified_tile_offset_row_a;
  std::vector<UnifiedOwnedBufferUint32> unified_tile_offset_row_b;
  std::vector<UnifiedOwnedBufferUint32> unified_tile_offset_row_c;
  std::vector<UnifiedOwnedBufferUint32> unified_tile_row_idx_a;
  std::vector<UnifiedOwnedBufferUint32> unified_tile_row_idx_b;
  std::vector<UnifiedOwnedBufferUint32> unified_tile_row_idx_c;
  std::vector<UnifiedOwnedBufferUint32> unified_tile_col_idx_a;
  std::vector<UnifiedOwnedBufferUint32> unified_tile_col_idx_b;
  std::vector<UnifiedOwnedBufferUint32> unified_tile_col_idx_c;
  std::vector<UnifiedOwnedBufferUint64> unified_data_a;
  std::vector<UnifiedOwnedBufferUint64> unified_data_b;
  std::vector<UnifiedOwnedBufferUint64> unified_data_c;

  unified_layout_matrix_c.resize(M * N);
  unified_tile_offset_row_a.resize(M * K);
  unified_tile_offset_row_b.resize(N * K);
  unified_tile_offset_row_c.resize(M * N);
  unified_tile_row_idx_a.resize(M * K);
  unified_tile_row_idx_b.resize(N * K);
  unified_tile_row_idx_c.resize(M * N);
  unified_tile_col_idx_a.resize(M * K);
  unified_tile_col_idx_b.resize(N * K);
  unified_tile_col_idx_c.resize(M * N);
  unified_data_a.resize(M * K);
  unified_data_b.resize(N * K);
  unified_data_c.resize(M * N);

  std::cout << "[FillTiles] Initializing buffers for results ..." << std::endl;
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, M, N, K, step, &p_streams_vec, tile_size, tile_buffer_size, &mtx,
       &layout_matrix_c, &tile_offset_row_c, &tile_row_idx_c, &tile_col_idx_c,
       &data_c, &unified_layout_matrix_c, &unified_tile_offset_row_c,
       &unified_tile_row_idx_c, &unified_tile_col_idx_c,
       &unified_data_c](auto w) {
        for (VertexID i = w; i < M; i += step) {
          for (VertexID j = 0; j < N; j++) {
            auto block_c = C_->GetBitTileMatrixPtrByIdx(i * N + j);
            if (block_c == nullptr)
              continue;
            if (block_c->GetMetadata().n_nz_tile == 0)
              continue;

            layout_matrix_c[i * N + j].data =
                (uint64_t *)block_c->GetNzTileBitmapPtr()->data();

            layout_matrix_c[i * N + j].size =
                block_c->GetNzTileBitmapPtr()->size();

            tile_offset_row_c[i * N + j].data = block_c->GetTileOffsetRowPtr();
            tile_offset_row_c[i * N + j].size =
                sizeof(VertexID) * (block_c->GetMetadata().n_strips + 1);

            tile_row_idx_c[i * N + j].data = block_c->GetTileRowIdxPtr();
            tile_row_idx_c[i * N + j].size =
                block_c->GetMetadata().n_nz_tile * sizeof(VertexID);

            tile_col_idx_c[i * N + j].data = block_c->GetTileColIdxPtr();
            tile_col_idx_c[i * N + j].size =
                block_c->GetMetadata().n_nz_tile * sizeof(VertexID);

            data_c[i * N + j].data = block_c->GetDataPtr();
            data_c[i * N + j].size =
                tile_buffer_size * block_c->GetMetadata().n_nz_tile;

            {
              std::lock_guard<std::mutex> lock(mtx);
              unified_layout_matrix_c[i * N + j].Init(
                  layout_matrix_c[i * N + j]);

              unified_tile_offset_row_c[i * N + j].Init(
                  tile_offset_row_c[i * N + j]);

              unified_tile_row_idx_c[i * N + j].Init(tile_row_idx_c[i * N + j]);
              unified_tile_col_idx_c[i * N + j].Init(tile_col_idx_c[i * N + j]);
              unified_data_c[i * N + j].Init(data_c[i * N + j]);
            }
          }
        }
      });

  // Init input Buffer for A_ and B, respectively.
  std::cout << "[FillTiles] Initializing input buffers for A and B."
            << std::endl;
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, M, N, K, step, &p_streams_vec, tile_size, tile_buffer_size, &mtx,
       &tile_offset_row_a, &tile_row_idx_a, &tile_col_idx_a, &data_a,
       &unified_tile_offset_row_a, &unified_tile_row_idx_a,
       &unified_tile_col_idx_a, &unified_data_a](auto w) {
        for (VertexID i = w; i < M; i += step) {
          for (VertexID k = 0; k < K; k++) {
            auto block_a = A_->GetBitTileMatrixPtrByIdx(i * K + k);
            if (block_a->GetMetadata().n_nz_tile == 0)
              continue;

            tile_offset_row_a[i * K + k].data = block_a->GetTileOffsetRowPtr();
            tile_offset_row_a[i * K + k].size =
                sizeof(VertexID) * (block_a->GetMetadata().n_strips + 1);

            tile_row_idx_a[i * K + k].data = block_a->GetTileRowIdxPtr();
            tile_row_idx_a[i * K + k].size =
                block_a->GetMetadata().n_nz_tile * sizeof(VertexID);

            tile_col_idx_a[i * K + k].data = block_a->GetTileColIdxPtr();
            tile_col_idx_a[i * K + k].size =
                block_a->GetMetadata().n_nz_tile * sizeof(VertexID);

            data_a[i * K + k].data = block_a->GetDataPtr();
            data_a[i * K + k].size =
                tile_buffer_size * block_a->GetMetadata().n_nz_tile;

            {
              std::lock_guard<std::mutex> lock(mtx);
              unified_tile_offset_row_a[i * K + k].Init(
                  tile_offset_row_a[i * K + k]);
              unified_tile_row_idx_a[i * K + k].Init(tile_row_idx_a[i * K + k]);
              unified_tile_col_idx_a[i * K + k].Init(tile_col_idx_a[i * K + k]);
              unified_data_a[i * K + k].Init(data_a[i * K + k]);
            }
          }
        }
      });

  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, M, N, K, step, &p_streams_vec, tile_size, tile_buffer_size, &mtx,
       &tile_offset_row_b, &tile_row_idx_b, &tile_col_idx_b, &data_b,
       &unified_tile_offset_row_b, &unified_tile_row_idx_b,
       &unified_tile_col_idx_b, &unified_data_b](auto w) {
        for (VertexID i = w; i < N; i += step) {
          for (VertexID k = 0; k < K; k++) {
            auto block_b = B_->GetBitTileMatrixPtrByIdx(i * K + k);
            if (block_b->GetMetadata().n_nz_tile == 0)
              continue;

            tile_offset_row_b[i * K + k].data = block_b->GetTileOffsetRowPtr();
            tile_offset_row_b[i * K + k].size =
                sizeof(VertexID) * (block_b->GetMetadata().n_strips + 1);

            tile_row_idx_b[i * K + k].data = block_b->GetTileRowIdxPtr();
            tile_row_idx_b[i * K + k].size =
                block_b->GetMetadata().n_nz_tile * sizeof(VertexID);

            tile_col_idx_b[i * K + k].data = block_b->GetTileColIdxPtr();
            tile_col_idx_b[i * K + k].size =
                block_b->GetMetadata().n_nz_tile * sizeof(VertexID);

            data_b[i * K + k].data = block_b->GetDataPtr();
            data_b[i * K + k].size =
                tile_buffer_size * block_b->GetMetadata().n_nz_tile;

            {
              std::lock_guard<std::mutex> lock(mtx);
              unified_tile_offset_row_b[i * K + k].Init(
                  tile_offset_row_b[i * K + k]);
              unified_tile_row_idx_b[i * K + k].Init(tile_row_idx_b[i * K + k]);
              unified_tile_col_idx_b[i * K + k].Init(tile_col_idx_b[i * K + k]);
              unified_data_b[i * K + k].Init(data_b[i * K + k]);
            }
          }
        }
      });

  cudaDeviceSynchronize();

  // Submit Kernel to fill edges into tiles.
  std::vector<int> work_load;
  work_load.resize(4);
  std::cout << "[FillTiles] Filling tiles ..." << std::endl;
  auto start_time_1 = std::chrono::system_clock::now();
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, M, N, K, step, &p_streams_vec, tile_size, tile_buffer_size,
       n_strips, &mtx, &unified_layout_matrix_c, &unified_tile_offset_row_a,
       &unified_tile_row_idx_a, &unified_tile_col_idx_a, &unified_data_a,
       &unified_tile_offset_row_b, &unified_tile_row_idx_b,
       &unified_tile_col_idx_b, &unified_data_b, &unified_tile_offset_row_c,
       &unified_tile_row_idx_c, &unified_tile_col_idx_c, &unified_data_c,
       &work_load](auto w) {
        for (VertexID k = w; k < K; k += step) {
          for (VertexID i = 0; i < M; i++) {
            for (VertexID j = 0; j < N; j++) {
              auto block_a = A_->GetBitTileMatrixPtrByIdx(i * K + k);
              auto block_b = B_->GetBitTileMatrixPtrByIdx(j * K + k);
              auto block_c = C_->GetBitTileMatrixPtrByIdx(i * N + j);

              if (block_c == nullptr)
                continue;
              if (block_b == nullptr)
                continue;
              if (block_a == nullptr)
                continue;

              if (block_a->GetMetadata().n_nz_tile == 0)
                continue;
              if (block_b->GetMetadata().n_nz_tile == 0)
                continue;
              if (block_c->GetMetadata().n_nz_tile == 0)
                continue;

              WriteAdd(&work_load[common::hash_function(i * N + j) % 4], 1);
              {
                std::lock_guard<std::mutex> lock(mtx);
                cudaSetDevice(common::hash_function(i * N + j) % 4);
                cudaStream_t &p_stream = p_streams_vec[i * N + j];
                MatrixOperationsKernelWrapper::FillTiles(
                    p_stream, tile_size, n_strips,
                    block_a->GetMetadata().n_nz_tile,
                    block_b->GetMetadata().n_nz_tile,
                    block_c->GetMetadata().n_nz_tile,
                    unified_layout_matrix_c[i * N + j],
                    unified_tile_offset_row_a[i * K + k],
                    unified_tile_offset_row_b[j * K + k],
                    unified_tile_offset_row_c[i * N + j],
                    unified_tile_row_idx_a[i * K + k],
                    unified_tile_row_idx_b[j * K + k],
                    unified_tile_row_idx_c[i * N + j],
                    unified_tile_col_idx_a[i * K + k],
                    unified_tile_col_idx_b[j * K + k],
                    unified_tile_col_idx_c[i * N + j],
                    unified_data_a[i * K + k], unified_data_b[j * K + k],
                    &unified_data_c[i * N + j]);
              }
            }
          }
        }
      });

  auto start_time_2 = std::chrono::system_clock::now();

  for (int i = 0; i < 4; i++) {
    std::cout << "Device: " << i << " holds " << work_load[i] << " | ";
  }
  std::cout << std::endl;

  std::cout << "[FillTiles] Time - Filling tiles elapse: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_2 - start_time_1)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;

  std::cout << "[FillTiles] Copying result to the result matrix ..."
            << std::endl;
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, M, N, K, step, &p_streams_vec, tile_size, tile_buffer_size,
       n_strips, &mtx, &unified_layout_matrix_c, &unified_tile_offset_row_a,
       &unified_tile_row_idx_a, &unified_tile_col_idx_a, &unified_data_a,
       &unified_tile_offset_row_b, &unified_tile_row_idx_b,
       &unified_tile_col_idx_b, &unified_data_b, &unified_tile_offset_row_c,
       &unified_tile_row_idx_c, &unified_tile_col_idx_c,
       &unified_data_c](auto w) {
        for (VertexID i = w; i < M; i += step) {
          for (VertexID j = 0; j < N; j++) {
            auto block_c = C_->GetBitTileMatrixPtrByIdx(i * N + j);
            if (block_c == nullptr)
              continue;
            if (block_c->GetMetadata().n_nz_tile == 0)
              continue;

            cudaStream_t &p_stream = p_streams_vec[i * N + j];
            {
              std::lock_guard<std::mutex> lock(mtx);
              CUDA_CHECK(cudaMemcpyAsync(block_c->GetDataPtr(),
                                         unified_data_c[i * N + j].GetPtr(),
                                         unified_data_c[i * N + j].GetSize(),
                                         cudaMemcpyDefault, p_stream));
            }
          }
        }
      });

  cudaDeviceSynchronize();
  std::for_each(p_streams_vec.begin(), p_streams_vec.end(),
                [](auto &s) { cudaStreamDestroy(s); });
}

__host__ void PPRQuery::FillTiles() {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  auto block_a = A_->GetBitTileMatrixPtrByIdx(0);
  VertexID n_strips = block_a->GetMetadata().n_strips;
  VertexID tile_size = block_a->GetMetadata().tile_size;
  auto tile_buffer_size =
      sizeof(uint64_t) * std::max(1u, WORD_OFFSET(tile_size * tile_size));

  VertexID M = A_->get_metadata().n_chunks;
  VertexID K = A_->get_metadata().n_chunks;
  VertexID N = B_->get_metadata().n_chunks;
  std::vector<cudaStream_t> p_streams_vec;
  p_streams_vec.resize(M * N);
  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [this, M, N, K, step, &p_streams_vec, &mtx](auto w) {
                  for (VertexID i = w; i < N; i += step) {
                    for (VertexID j = 0; j < M; j++) {
                      cudaSetDevice(common::hash_function(i * N + j) % 4);
                      cudaStreamCreate(&p_streams_vec[i * N + j]);
                    }
                  }
                });

  std::cout << "[FillTiles]"
            << " Start - n_strips: " << n_strips << ", tile_size: " << tile_size
            << std::endl;

  std::vector<BufferUint64> layout_matrix_c;
  std::vector<BufferUint32> tile_offset_row_a;
  std::vector<BufferUint32> tile_offset_row_b;
  std::vector<BufferUint32> tile_offset_row_c;
  std::vector<BufferUint32> tile_row_idx_a;
  std::vector<BufferUint32> tile_row_idx_b;
  std::vector<BufferUint32> tile_row_idx_c;
  std::vector<BufferUint32> tile_col_idx_a;
  std::vector<BufferUint32> tile_col_idx_b;
  std::vector<BufferUint32> tile_col_idx_c;
  std::vector<BufferUint64> data_a;
  std::vector<BufferUint64> data_b;
  std::vector<BufferUint64> data_c;

  layout_matrix_c.resize(M * N);
  tile_offset_row_a.resize(M * K);
  tile_offset_row_b.resize(N * K);
  tile_offset_row_c.resize(M * N);
  tile_row_idx_a.resize(M * K);
  tile_row_idx_b.resize(N * K);
  tile_row_idx_c.resize(M * N);
  tile_col_idx_a.resize(M * K);
  tile_col_idx_b.resize(N * K);
  tile_col_idx_c.resize(M * N);
  data_a.resize(M * K);
  data_b.resize(N * K);
  data_c.resize(M * N);

  std::vector<DeviceOwnedBufferUint64> device_layout_matrix_c;
  std::vector<DeviceOwnedBufferUint32> device_tile_offset_row_a;
  std::vector<DeviceOwnedBufferUint32> device_tile_offset_row_b;
  std::vector<DeviceOwnedBufferUint32> device_tile_offset_row_c;
  std::vector<DeviceOwnedBufferUint32> device_tile_row_idx_a;
  std::vector<DeviceOwnedBufferUint32> device_tile_row_idx_b;
  std::vector<DeviceOwnedBufferUint32> device_tile_row_idx_c;
  std::vector<DeviceOwnedBufferUint32> device_tile_col_idx_a;
  std::vector<DeviceOwnedBufferUint32> device_tile_col_idx_b;
  std::vector<DeviceOwnedBufferUint32> device_tile_col_idx_c;
  std::vector<DeviceOwnedBufferUint64> device_data_a;
  std::vector<DeviceOwnedBufferUint64> device_data_b;
  std::vector<DeviceOwnedBufferUint64> device_data_c;

  device_layout_matrix_c.resize(M * N);
  device_tile_offset_row_a.resize(M * K);
  device_tile_offset_row_b.resize(N * K);
  device_tile_offset_row_c.resize(M * N);
  device_tile_row_idx_a.resize(M * K);
  device_tile_row_idx_b.resize(N * K);
  device_tile_row_idx_c.resize(M * N);
  device_tile_col_idx_a.resize(M * K);
  device_tile_col_idx_b.resize(N * K);
  device_tile_col_idx_c.resize(M * N);
  device_data_a.resize(M * K);
  device_data_b.resize(N * K);
  device_data_c.resize(M * N);

  std::for_each(p_streams_vec.begin(), p_streams_vec.end(),
                [](auto &s) { cudaStreamDestroy(s); });
}

__host__ void PPRQuery::Count(const GridBitTiledMatrix &G) {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  auto block = G.GetBitTileMatrixPtrByIdx(0);

  VertexID n_strips = block->GetMetadata().n_strips;
  VertexID tile_size = block->GetMetadata().tile_size;
  auto tile_buffer_size =
      sizeof(uint64_t) * std::max(1u, WORD_OFFSET(tile_size * tile_size));

  VertexID M = G.get_metadata().n_chunks;
  VertexID K = G.get_metadata().n_chunks;
  VertexID N = G.get_metadata().n_chunks;
  std::vector<cudaStream_t> p_streams_vec;
  p_streams_vec.resize(M * N);
  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [this, M, N, K, step, &p_streams_vec, &mtx](auto w) {
                  for (VertexID i = w; i < N; i += step) {
                    for (VertexID j = 0; j < M; j++) {
                      cudaSetDevice(common::hash_function(i * N + j) % 4);
                      cudaStreamCreate(&p_streams_vec[i * N + j]);
                    }
                  }
                });

  std::vector<BufferUint64> data;
  data.resize(M * N);
  std::vector<UnifiedOwnedBufferUint64> unified_data;
  unified_data.resize(M * N);
  std::vector<UnifiedOwnedBufferUint64> unified_count;
  unified_count.resize(M * N);

  std::cout << "[Count]"
            << " Start - n_strips: " << n_strips << ", tile_size: " << tile_size
            << std::endl;

  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, M, N, K, step, &p_streams_vec, tile_size, &G, tile_buffer_size,
       n_strips, &mtx, &data, &unified_data, &unified_count](auto w) {
        for (VertexID i = w; i < M; i += step) {
          for (VertexID j = 0; j < N; j++) {
            unified_count[i * N + j].Init(sizeof(uint64_t));

            auto block = G.GetBitTileMatrixPtrByIdx(i * N + j);

            if (block == nullptr)
              continue;
            if (block->GetMetadata().n_nz_tile == 0)
              continue;

            data[i * N + j].data = block->GetDataPtr();
            data[i * N + j].size =
                tile_buffer_size * block->GetMetadata().n_nz_tile;

            {
              std::lock_guard<std::mutex> lock(mtx);
              unified_data[i * N + j].Init(data[i * N + j]);
              // std::cout << "(" << i << "," << j << std::endl;
              // block->Print();

              cudaStream_t &p_stream = p_streams_vec[i * N + j];
              cudaSetDevice(common::hash_function(i * N + j) % 4);
              MatrixOperationsKernelWrapper::Count(
                  p_stream, tile_size, n_strips, block->GetMetadata().n_nz_tile,
                  unified_data[i * N + j], &unified_count[i * N + j]);
            }
          }
        }
      });

  cudaDeviceSynchronize();

  uint64_t total_count = 0;
  int i = 0;
  std::for_each(unified_count.begin(), unified_count.end(),
                [this, &total_count, step, &i](auto &w) {
                  WriteAdd(&total_count, *w.GetPtr());
                });

  std::cout << "[Count] done! total_count:" << total_count << std::endl;
}

__host__ void PPRQuery::Run() {

  auto start_time_0 = std::chrono::system_clock::now();
  InitResultMatrix();

  auto start_time_1 = std::chrono::system_clock::now();

  std::cout << "[PPRQuery] Run Step1 InitResultMatrix() elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_1 - start_time_0)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;

  FillTilesUnifiedMemory();
  auto start_time_2 = std::chrono::system_clock::now();
  std::cout << "[PPRQuery] Run Step2 FillTiles() elapsed:"
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_2 - start_time_1)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;

  Count(*C_);

  auto start_time_3 = std::chrono::system_clock::now();
  std::cout << "[PPRQuery] Run Step3 Count() elapsed:"
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_3 - start_time_2)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;
}

} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics