#include "core/task/gemm.cuh"

#include <ctime>
#include <cuda_runtime.h>
#include <execution>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "core/common/consts.h"
#include "core/common/host_algorithms.cuh"
#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/host_buffer.cuh"
#include "core/data_structures/metadata.h"
#include "core/data_structures/unified_buffer.cuh"
#include "core/io/grid_csr_tiled_matrix_io.cuh"
#include "core/task/kernel/matrix_operations.cuh"
#include "core/util/atomic.h"
#include "core/util/bitmap.h"
#include "core/util/bitmap_no_ownership.h"
#include "core/util/format_converter.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using sics::matrixgraph::core::data_structures::GridCSRTiledMatrix;
using sics::matrixgraph::core::io::GridCSRTiledMatrixIO;
using GirdTiledMatrix =
    sics::matrixgraph::core::data_structures::GridCSRTiledMatrix;
using DeviceOwnedBufferUint64 =
    sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<uint64_t>;
using DeviceOwnedBufferUint32 =
    sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<uint32_t>;
using DeviceOwnedBufferUint8 =
    sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<uint8_t>;
using UnifiedOwnedBufferUint32 =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint32_t>;
using UnifiedOwnedBufferUint64 =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint64_t>;
using UnifiedOwnedBufferUint8 =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint8_t>;
using BufferUint64 = sics::matrixgraph::core::data_structures::Buffer<uint64_t>;
using BufferUint8 = sics::matrixgraph::core::data_structures::Buffer<uint8_t>;
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
using Edges = sics::matrixgraph::core::data_structures::Edges;
using sics::matrixgraph::core::util::atomic::WriteAdd;
using sics::matrixgraph::core::util::atomic::WriteMax;
using sics::matrixgraph::core::util::atomic::WriteMin;
using sics::matrixgraph::core::util::format_converter::Edgelist2CSRTiledMatrix;
using GraphMetadata = sics::matrixgraph::core::data_structures::GraphMetadata;
using Edge = sics::matrixgraph::core::data_structures::Edge;
using EdgelistMetadata =
    sics::matrixgraph::core::data_structures::EdgelistMetadata;
using GridGraphMetadata =
    sics::matrixgraph::core::data_structures::GridGraphMetadata;

// CUDA kernel to add elements of two arrays
__host__ void GEMM::LoadData() {
  std::cout << "[GEMM] LoadData()" << std::endl;
  GridCSRTiledMatrixIO grid_csr_tiled_matrix_io;

  grid_csr_tiled_matrix_io.Read(input_path_, &A_);
  grid_csr_tiled_matrix_io.Read(input_path_transposed_, &B_);

  std::cout << "######################## A #######################"
            << std::endl;
  A_->Print();
  // std::cout << "######################## B #######################"
  //           << std::endl;
  // B_->Print();
  // C_ = new GridCSRTiledMatrix(A_->get_metadata());
}

__host__ void GEMM::InitC() {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  auto block_a = A_->GetTiledMatrixPtrByIdx(0);
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
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, M, N, K, step, &device_owned_buffers_matrix_a,
       &device_owned_buffers_matrix_b, &device_owned_buffers_matrix_c,
       &p_streams_vec, &buffers_matrix_a, &buffers_matrix_b, &buffers_matrix_c,
       n_strips, n_chunks, tile_size, &mtx](auto w) {
        for (VertexID k = w; k < K; k += step) {

          for (VertexID i = 0; i < M; i++) {
            for (VertexID j = 0; j < N; j++) {

              auto block_a = A_->GetTiledMatrixPtrByIdx(i * K + k);
              auto block_b = B_->GetTiledMatrixPtrByIdx(j * K + k);

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

  std::cout << "[InitResultMatrix] Allocating space for Matrix C ..."
            << std::endl;
  std::for_each(
      // std::execution::par,
      worker.begin(), worker.end(),
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

            auto *csr_tiled_matrix_ptr = C_->GetTiledMatrixPtrByIdx(i * N + j);

            TiledMatrixMetadata metadata{.n_strips = n_strips,
                                         .n_nz_tile = n_nz_tile,
                                         .tile_size = tile_size};
            csr_tiled_matrix_ptr->Init(
                metadata, new GPUBitmap(n_strips * tile_size,
                                        buffers_matrix_c[i * N + j].GetPtr()));
          }
        }
      });

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
            auto *bit_tiled_matrix_ptr = C_->GetTiledMatrixPtrByIdx(i * N + j);

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

            auto *csr_tiled_matrix_ptr = C_->GetTiledMatrixPtrByIdx(i * N + j);
            auto tile_offset_row = csr_tiled_matrix_ptr->GetTileOffsetRowPtr();

            for (int t = 0; t < csr_tiled_matrix_ptr->GetMetadata().n_strips;
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

__host__ Edges *GEMM::Walks(const GridCSRTiledMatrix &A,
                            const GridCSRTiledMatrix &B, VertexID tile_size,
                            VertexID n_strips) {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  auto tile_buffer_size =
      sizeof(uint64_t) * std::max(1u, WORD_OFFSET(tile_size * tile_size));

  VertexID M = A.get_metadata().n_chunks;
  VertexID K = A.get_metadata().n_chunks;
  VertexID N = B.get_metadata().n_chunks;
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

  std::cout << "[Walks]"
            << " Starting ...  - n_strips: " << n_strips
            << ", tile_size: " << tile_size << std::endl;

  std::vector<BufferUint32> tile_offset_row_a;
  std::vector<BufferUint32> tile_offset_row_b;
  std::vector<BufferUint32> tile_row_idx_a;
  std::vector<BufferUint32> tile_row_idx_b;
  std::vector<BufferUint32> tile_col_idx_a;
  std::vector<BufferUint32> tile_col_idx_b;
  std::vector<BufferUint64> csr_offset_a;
  std::vector<BufferUint64> csr_offset_b;
  std::vector<BufferUint8> data_a;
  std::vector<BufferUint8> data_b;

  tile_offset_row_a.resize(M * K);
  tile_offset_row_b.resize(N * K);
  tile_row_idx_a.resize(M * K);
  tile_row_idx_b.resize(N * K);
  tile_col_idx_a.resize(M * K);
  tile_col_idx_b.resize(N * K);
  csr_offset_a.resize(M * K);
  csr_offset_b.resize(N * K);
  data_a.resize(M * K);
  data_b.resize(N * K);

  std::vector<UnifiedOwnedBufferUint32> unified_csr_n_vertices_a;
  std::vector<UnifiedOwnedBufferUint32> unified_csr_n_vertices_b;

  std::vector<UnifiedOwnedBufferUint64> unified_csr_n_edges_a;
  std::vector<UnifiedOwnedBufferUint64> unified_csr_n_edges_b;
  std::vector<UnifiedOwnedBufferUint64> unified_csr_n_edges_c;

  std::vector<UnifiedOwnedBufferUint32> unified_tile_offset_row_a;
  std::vector<UnifiedOwnedBufferUint32> unified_tile_offset_row_b;
  std::vector<UnifiedOwnedBufferUint32> unified_tile_row_idx_a;
  std::vector<UnifiedOwnedBufferUint32> unified_tile_row_idx_b;
  std::vector<UnifiedOwnedBufferUint32> unified_tile_col_idx_a;
  std::vector<UnifiedOwnedBufferUint32> unified_tile_col_idx_b;
  std::vector<UnifiedOwnedBufferUint64> unified_csr_offset_a;
  std::vector<UnifiedOwnedBufferUint64> unified_csr_offset_b;
  std::vector<UnifiedOwnedBufferUint8> unified_data_a;
  std::vector<UnifiedOwnedBufferUint8> unified_data_b;

  std::vector<UnifiedOwnedBufferUint32> unified_edgelist_c;
  unified_edgelist_c.resize(M * N);

  std::vector<UnifiedOwnedBufferUint32> unified_output_offset;
  unified_output_offset.resize(M * N);

  unified_csr_n_vertices_a.resize(M * K);
  unified_csr_n_vertices_b.resize(N * K);

  unified_csr_n_edges_a.resize(M * K);
  unified_csr_n_edges_b.resize(N * K);

  unified_tile_offset_row_a.resize(M * K);
  unified_tile_offset_row_b.resize(N * K);
  unified_tile_row_idx_a.resize(M * K);
  unified_tile_row_idx_b.resize(N * K);
  unified_tile_col_idx_a.resize(M * K);
  unified_tile_col_idx_b.resize(N * K);
  unified_csr_offset_a.resize(M * K);
  unified_csr_offset_b.resize(N * K);
  unified_data_a.resize(M * K);
  unified_data_b.resize(N * K);

  std::cout << "[Walks] Initializing buffers for results ..." << std::endl;
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, M, N, K, step, &p_streams_vec, tile_size, n_strips,
       tile_buffer_size, &mtx, &unified_edgelist_c,
       &unified_output_offset](auto w) {
        for (VertexID i = w; i < M; i += step) {
          for (VertexID j = 0; j < N; j++) {
            {
              std::lock_guard<std::mutex> lock(mtx);

              unified_edgelist_c[i * N + j].Init(
                  sizeof(VertexID) *
                  sics::matrixgraph::core::common::KDefalutNumEdgesPerTile * 2);

              unified_output_offset[i * N + j].Init(sizeof(EdgeIndex));
            }
          }
        }
      });

  // Init input Buffer for A and B, respectively.
  std::cout << "[Walks] Initializing input buffers for A and B." << std::endl;
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, M, N, K, step, &A, &p_streams_vec, tile_size, tile_buffer_size,
       &mtx, &tile_offset_row_a, &tile_row_idx_a, &tile_col_idx_a,
       &csr_offset_a, &data_a, &unified_tile_offset_row_a,
       &unified_tile_row_idx_a, &unified_tile_col_idx_a, &unified_csr_offset_a,
       &unified_data_a, &unified_csr_n_vertices_a, &unified_csr_n_vertices_b,
       &unified_csr_n_edges_a, &unified_csr_n_edges_b](auto w) {
        for (VertexID i = w; i < M; i += step) {
          for (VertexID k = 0; k < K; k++) {
            auto block_a = A.GetTiledMatrixPtrByIdx(i * K + k);
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
            csr_offset_a[i * K + k].data = block_a->GetCSROffsetPtr();
            csr_offset_a[i * K + k].size =
                block_a->GetMetadata().n_nz_tile * sizeof(uint64_t);

            data_a[i * K + k].data = block_a->GetDataPtr();
            data_a[i * K + k].size = block_a->GetDataBufferSize();

            {
              std::lock_guard<std::mutex> lock(mtx);
              unified_tile_offset_row_a[i * K + k].Init(
                  tile_offset_row_a[i * K + k]);
              unified_tile_row_idx_a[i * K + k].Init(tile_row_idx_a[i * K + k]);
              unified_tile_col_idx_a[i * K + k].Init(tile_col_idx_a[i * K + k]);
              unified_csr_offset_a[i * K + k].Init(csr_offset_a[i * K + k]);
              unified_data_a[i * K + k].Init(data_a[i * K + k]);
              auto subgraph_metadata = block_a->GetCSRMetadata();
              unified_csr_n_vertices_a[i * K + k].Init(
                  sizeof(uint32_t) * block_a->GetMetadata().n_nz_tile);
              unified_csr_n_edges_a[i * K + k].Init(
                  sizeof(uint64_t) * block_a->GetMetadata().n_nz_tile);
              for (VertexID gid = 0; gid < block_a->GetMetadata().n_nz_tile;
                   gid++) {
                unified_csr_n_vertices_a[i * K + k].SetElement(
                    subgraph_metadata[gid].num_vertices, gid);
                unified_csr_n_edges_a[i * K + k].SetElement(
                    subgraph_metadata[gid].num_outgoing_edges, gid);
              }
            }
          }
        }
      });

  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, M, N, K, &B, step, &p_streams_vec, tile_size, tile_buffer_size,
       &mtx, &tile_offset_row_b, &tile_row_idx_b, &tile_col_idx_b,
       &csr_offset_b, &data_b, &unified_tile_offset_row_b,
       &unified_tile_row_idx_b, &unified_tile_col_idx_b, &unified_csr_offset_b,
       &unified_data_b, &unified_csr_n_vertices_b,
       &unified_csr_n_edges_b](auto w) {
        for (VertexID i = w; i < N; i += step) {
          for (VertexID k = 0; k < K; k++) {
            auto block_b = B.GetTiledMatrixPtrByIdx(i * K + k);
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
            csr_offset_b[i * K + k].data = block_b->GetCSROffsetPtr();
            csr_offset_b[i * K + k].size =
                block_b->GetMetadata().n_nz_tile * sizeof(uint64_t);

            data_b[i * K + k].data = block_b->GetDataPtr();
            data_b[i * K + k].size = block_b->GetDataBufferSize();

            {
              std::lock_guard<std::mutex> lock(mtx);
              unified_tile_offset_row_b[i * K + k].Init(
                  tile_offset_row_b[i * K + k]);
              unified_tile_row_idx_b[i * K + k].Init(tile_row_idx_b[i * K + k]);
              unified_tile_col_idx_b[i * K + k].Init(tile_col_idx_b[i * K + k]);
              unified_csr_offset_b[i * K + k].Init(csr_offset_b[i * K + k]);
              unified_data_b[i * K + k].Init(data_b[i * K + k]);

              auto subgraph_metadata = block_b->GetCSRMetadata();
              unified_csr_n_vertices_b[i * K + k].Init(
                  sizeof(uint32_t) * block_b->GetMetadata().n_nz_tile);
              unified_csr_n_edges_b[i * K + k].Init(
                  sizeof(uint64_t) * block_b->GetMetadata().n_nz_tile);
              for (VertexID gid = 0; gid < block_b->GetMetadata().n_nz_tile;
                   gid++) {
                unified_csr_n_vertices_b[i * K + k].SetElement(
                    subgraph_metadata[gid].num_vertices, gid);
                unified_csr_n_edges_b[i * K + k].SetElement(
                    subgraph_metadata[gid].num_outgoing_edges, gid);
              }
            }
          }
        }
      });

  cudaDeviceSynchronize();
  // Submit Kernel to fill edges into tiles.
  std::vector<int> work_load;
  work_load.resize(4);
  std::cout << "[Walks] Walks ..." << std::endl;
  auto start_time_1 = std::chrono::system_clock::now();

  std::for_each(
      // std::execution::par,
      worker.begin(), worker.end(),
      [this, M, N, K, step, &A, &B, &p_streams_vec, tile_size, tile_buffer_size,
       n_strips, &mtx, &unified_csr_n_vertices_a, &unified_csr_n_vertices_b,
       &unified_csr_n_edges_a, &unified_csr_n_edges_b,
       &unified_tile_offset_row_a, &unified_tile_row_idx_a,
       &unified_tile_col_idx_a, &unified_csr_offset_a, &unified_data_a,
       &unified_tile_offset_row_b, &unified_tile_row_idx_b,
       &unified_tile_col_idx_b, &unified_csr_offset_b, &unified_data_b,
       &unified_edgelist_c, &unified_output_offset, &work_load](auto w) {
        for (VertexID i = w; i < M; i += step) {
          for (VertexID j = 0; j < N; j++) {
            for (VertexID k = 0; k < K; k++) {
              auto block_a = A.GetTiledMatrixPtrByIdx(i * K + k);
              auto block_b = B.GetTiledMatrixPtrByIdx(k * K + j);
              if (block_b == nullptr || block_b->GetMetadata().n_nz_tile == 0)
                continue;
              if (block_a == nullptr || block_a->GetMetadata().n_nz_tile == 0)
                continue;

              std::cout << "A: (" << i << ", " << k << ")"
                        << " X "
                        << "B: (" << k << ", " << j << ")"
                        << "C: (" << i << ", " << j << ")"
                        << "nz A: " << block_a->GetMetadata().n_nz_tile
                        << ", nz B: " << block_b->GetMetadata().n_nz_tile
                        << std::endl;

              // std::cout << "- A - " << std::endl;
              // block_a->Print();
              // std::cout << "- B - " << std::endl;
              // block_b->Print();

              cudaSetDevice(common::hash_function(i * N + j) % 4);

              {
                std::lock_guard<std::mutex> lock(mtx);
                cudaStream_t &p_stream = p_streams_vec[i * N + j];
                MatrixOperationsKernelWrapper::Walk(
                    p_stream, tile_size, n_strips,
                    block_a->GetMetadata().n_nz_tile,
                    block_b->GetMetadata().n_nz_tile,
                    unified_csr_n_vertices_a[i * K + k],
                    unified_csr_n_vertices_b[k * K + j],
                    unified_csr_n_edges_a[i * K + k],
                    unified_csr_n_edges_b[k * K + j],
                    unified_tile_offset_row_a[i * K + k],
                    unified_tile_offset_row_b[k * K + j],
                    unified_tile_row_idx_a[i * K + k],
                    unified_tile_row_idx_b[k * K + j],
                    unified_tile_col_idx_a[i * K + k],
                    unified_tile_col_idx_b[k * K + j],
                    unified_csr_offset_a[i * K + k],
                    unified_csr_offset_b[k * K + j], unified_data_a[i * K + k],
                    unified_data_b[k * K + j], &unified_edgelist_c[i * N + j],
                    &unified_output_offset[i * N + j]);
                cudaStreamSynchronize(p_stream);
              }

              // for (auto _ = 0; _ < *(unified_output_offset[i * N +
              // j].GetPtr());
              //      _++) {
              //   std::cout << unified_edgelist_c[i * N + j].GetPtr()[2 * _]
              //             << "=>"
              //             << unified_edgelist_c[i * N + j].GetPtr()[2 * _ +
              //             1]
              //             << std::endl;
              // }
            }
          }
        }
      });
  cudaDeviceSynchronize();

  EdgeIndex output_n_edges = 0;

  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, M, N, K, step, &output_n_edges, &unified_output_offset](auto w) {
        for (auto _ = w; _ < unified_output_offset.size(); _ += step) {
          WriteAdd(&output_n_edges, *(unified_output_offset[_].GetPtr()));
        }
      });

  for (auto _ = 0; _ < unified_output_offset.size(); _++) {
    std::cout << "offset: " << *(unified_output_offset[_].GetPtr())
              << std::endl;
  }

  VertexID *output_edges_buf = new VertexID[output_n_edges * 2]();
  EdgeIndex output_offset = 0;
  for (auto _ = 0; _ < unified_edgelist_c.size(); _++) {
    cudaMemcpy(output_edges_buf + output_offset, unified_edgelist_c[_].GetPtr(),
               sizeof(VertexID) * 2 * *(unified_output_offset[_].GetPtr()),
               cudaMemcpyHostToHost);
    output_offset += *(unified_output_offset[_].GetPtr()) * 2;
  }

  Edges *output_edges = new Edges(output_n_edges, output_edges_buf);

  std::cout << "[Walks] End!" << std::endl;
  return output_edges;
}

__host__ std::vector<Edges> *GEMM::GridPartitioning(const Edges &edges,
                                                    GraphID n_partitions) {
  // std::cout << "[GridCut] Start ..." << std::endl;

  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  // Precompute the size of each edge bucket.
  VertexID scope_per_chunk =
      ceil((edges.get_metadata().max_vid + 1) / (float)n_partitions);

  auto size_per_bucket = new EdgeIndex[n_partitions * n_partitions]();
  auto *n_edges_for_each_block = new EdgeIndex[n_partitions * n_partitions]();
  auto *max_vid_for_each_block = new VertexID[n_partitions * n_partitions]();
  auto *min_vid_for_each_block = new VertexID[n_partitions * n_partitions]();
  Bitmap vertices_bm_for_each_block[n_partitions * n_partitions];

  for (GraphID _ = 0; _ < n_partitions * n_partitions; _++) {
    vertices_bm_for_each_block[_].Init(edges.get_metadata().max_vid);
    min_vid_for_each_block[_] = std::numeric_limits<uint32_t>::max();
  }

  // std::cout << "[GridCut] Computing key parameters under chunk scope of "
  //           << scope_per_chunk << " ...\n"
  //           << std::endl;
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, step, &edges, scope_per_chunk, n_partitions,
       &n_edges_for_each_block, &max_vid_for_each_block,
       &min_vid_for_each_block, &vertices_bm_for_each_block](auto w) {
        for (auto eid = w; eid < edges.get_metadata().num_edges; eid += step) {
          auto *localid_to_globalid = edges.get_localid_to_globalid_ptr();
          auto edge = edges.get_edge_by_index(eid);
          edge.src = localid_to_globalid[edge.src];
          edge.dst = localid_to_globalid[edge.dst];

          auto x = edge.src / scope_per_chunk;
          auto y = edge.dst / scope_per_chunk;
          WriteAdd(&n_edges_for_each_block[x * n_partitions + y], (EdgeIndex)1);

          WriteMax(&max_vid_for_each_block[x * n_partitions + y], edge.src);
          WriteMax(&max_vid_for_each_block[x * n_partitions + y], edge.dst);
          WriteMin(&min_vid_for_each_block[x * n_partitions + y], edge.src);
          WriteMin(&min_vid_for_each_block[x * n_partitions + y], edge.dst);
          if (!vertices_bm_for_each_block[x * n_partitions + y].GetBit(
                  edge.src)) {
            vertices_bm_for_each_block[x * n_partitions + y].SetBit(edge.src);
          }
          if (!vertices_bm_for_each_block[x * n_partitions + y].GetBit(
                  edge.dst)) {
            vertices_bm_for_each_block[x * n_partitions + y].SetBit(edge.dst);
          }
        }
      });

  // std::cout << "[GridCut] Allocating space for each block...\n" << std::endl;
  Edge **edge_blocks_buf = new Edge *[n_partitions * n_partitions]();
  for (GraphID _ = 0; _ < n_partitions * n_partitions; _++) {
    edge_blocks_buf[_] = new Edge[n_edges_for_each_block[_]]();
  }

  auto *offset_for_each_block =
      new std::atomic<EdgeIndex>[n_partitions * n_partitions]();

  // std::cout << "[GridCut] Dropping edges into blocks...\n" << std::endl;
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, step, &edges, scope_per_chunk, n_partitions, &edge_blocks_buf,
       &n_edges_for_each_block, &offset_for_each_block](auto w) {
        for (auto eid = w; eid < edges.get_metadata().num_edges; eid += step) {
          auto *localid_to_globalid = edges.get_localid_to_globalid_ptr();
          auto edge = edges.get_edge_by_index(eid);
          edge.src = localid_to_globalid[edge.src];
          edge.dst = localid_to_globalid[edge.dst];
          auto x = edge.src / scope_per_chunk;
          auto y = edge.dst / scope_per_chunk;
          auto block_id = x * n_partitions + y;
          auto offset = offset_for_each_block[block_id].fetch_add(1);
          edge_blocks_buf[block_id][offset] = edge;
        }
      });

  std::vector<Edges> *edges_blocks = new std::vector<Edges>;
  edges_blocks->reserve(n_partitions * n_partitions);

  // std::cout << "[GridCut] Constructing Edgelist of blocks...\n" << std::endl;
  for (auto _ = 0; _ < n_partitions * n_partitions; _++) {
    EdgelistMetadata meta{.num_vertices = vertices_bm_for_each_block[_].Count(),
                          .num_edges = n_edges_for_each_block[_],
                          .max_vid = max_vid_for_each_block[_],
                          .min_vid = min_vid_for_each_block[_]};
    edges_blocks->emplace_back(Edges(meta, edge_blocks_buf[_]));
  }

  for (auto _ = 0; _ < n_partitions * n_partitions; _++) {
    if (edges_blocks->at(_).get_metadata().num_vertices == 0)
      continue;
    edges_blocks->at(_).GenerateLocalID2GlobalID();
  }

  // std::cout << "[GridCut] End!" << std::endl;
  return edges_blocks;
}

__host__ GridCSRTiledMatrix *GEMM::ConvertGridEdgelist2GridTiledMatrix(
    const std::vector<Edges> &edges_blocks,
    const GridGraphMetadata &grid_graph_metadata, VertexID tile_size) {

  GridCSRTiledMatrix *grid_tiled_matrix =
      new GridCSRTiledMatrix(grid_graph_metadata);

  size_t block_scope = ceil((float)grid_graph_metadata.max_vid /
                            (float)grid_graph_metadata.n_chunks);

  for (GraphID gid = 0; gid < edges_blocks.size(); gid++) {
    auto *csr_tiled_matrix_ptr = grid_tiled_matrix->GetTiledMatrixPtrByIdx(gid);
    csr_tiled_matrix_ptr = Edgelist2CSRTiledMatrix(
        edges_blocks[gid], tile_size, block_scope, csr_tiled_matrix_ptr);
  }

  return grid_tiled_matrix;
}

__host__ void GEMM::FillTiles() {}

__host__ void GEMM::Count(const GridCSRTiledMatrix &G) {}

__host__ void GEMM::Run() {

  auto start_time_0 = std::chrono::system_clock::now();

  auto block_a = A_->GetTiledMatrixPtrByIdx(0);
  VertexID tile_size = block_a->GetMetadata().tile_size;
  VertexID n_strips = block_a->GetMetadata().n_strips;
  VertexID M = A_->get_metadata().n_chunks;
  VertexID K = A_->get_metadata().n_chunks;
  VertexID N = B_->get_metadata().n_chunks;

  auto *edges = Walks(*A_, *B_, tile_size, n_strips);
  auto start_time_1 = std::chrono::system_clock::now();

  std::cout << "[GEMM] Run Step1 Walks() elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_1 - start_time_0)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;

  auto *edges_blocks = GridPartitioning(*edges, M);

  auto start_time_2 = std::chrono::system_clock::now();

  std::cout << "[GEMM] Run Step2 GridPartitioning() elapsed:"
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   start_time_2 - start_time_1)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;

  GridGraphMetadata grid_graph_metadata = {
      .n_chunks = M,
      .n_vertices = edges->get_metadata().num_vertices,
      .n_edges = edges->get_metadata().num_edges,
      .max_vid = A_->get_metadata().max_vid};

  C_ = ConvertGridEdgelist2GridTiledMatrix(*edges_blocks, grid_graph_metadata,
                                           tile_size);
  C_->Print();

  auto start_time_3 = std::chrono::system_clock::now();

  GridCSRTiledMatrixIO graph_csr_tiled_matrix_io;

  graph_csr_tiled_matrix_io.Write(output_path_, *C_);

  std::cout << "[GEMM] Run Step3 ConvertGridEdgelist2GridTiledMatrix() elapsed:"
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