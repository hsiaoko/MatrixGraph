#include "core/task/ppr_query.cuh"

#include <ctime>
#include <cuda_runtime.h>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>

#ifdef TBB_FOUND
#include <execution>
#endif

#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/grid_tiled_matrix.cuh"
#include "core/data_structures/host_buffer.cuh"
#include "core/data_structures/metadata.h"
#include "core/io/grid_tiled_matrix_io.cuh"
#include "core/task/kernel/matrix_operations.cuh"
#include "core/util/bitmap.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using sics::matrixgraph::core::data_structures::GridTiledMatrix;
using sics::matrixgraph::core::io::GridTiledMatrixIO;
using GirdTiledMatrix =
    sics::matrixgraph::core::data_structures::GridTiledMatrix;
using DeviceOwnedBufferUint64 =
    sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<uint64_t>;
using BufferUint64 = sics::matrixgraph::core::data_structures::Buffer<uint64_t>;
using DeviceOwnedBufferUint64 =
    sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<uint64_t>;
using MatrixOperationsKernelWrapper =
    sics::matrixgraph::core::task::kernel::MatrixOperationsKernelWrapper;
using Bitmap = sics::matrixgraph::core::util::Bitmap;
using TiledMatrixMetadata =
    sics::matrixgraph::core::data_structures::TiledMatrixMetadata;

static uint32_t hash_function(uint64_t x) {
  x ^= x >> 16;
  x *= 0x85ebca6b;
  x ^= x >> 13;
  x *= 0xc2b2ae35;
  x ^= x >> 16;
  return x;
}

__host__ void PPRQuery::LoadData() {
  GridTiledMatrixIO grid_tiled_matrix_io;

  grid_tiled_matrix_io.Read(input_path_, &A_);
  grid_tiled_matrix_io.Read(input_path_transposed_, &B_);
  // A_->Print();
  // B_->Print();
  C_ = new GridTiledMatrix(A_->get_metadata());
}

__host__ void PPRQuery::ComputeLayoutMatrix() {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  auto block_a = A_->GetBitTileMatrixPtrByIdx(0);
  VertexID n_strips = block_a->GetMetadata().n_strips;
  VertexID tile_size = block_a->GetMetadata().tile_size;

  std::cout << "[PPRQuery]"
            << " Start - n_strips: " << n_strips << ", tile_size: " << tile_size
            << std::endl;

  VertexID M = A_->get_metadata().n_chunks;
  VertexID K = A_->get_metadata().n_chunks;
  VertexID N = A_->get_metadata().n_chunks;

  VertexID n_chunks = A_->get_metadata().n_chunks;

  std::vector<BufferUint64> buffers_matrix_a;
  std::vector<BufferUint64> buffers_matrix_b;
  std::vector<BufferUint64> buffers_matrix_c;
  buffers_matrix_a.resize(M * K);
  buffers_matrix_b.resize(N * K);
  buffers_matrix_c.resize(M * N);

  std::vector<DeviceOwnedBufferUint64> device_owned_buffers_matrix_a;
  std::vector<DeviceOwnedBufferUint64> device_owned_buffers_matrix_b;
  std::vector<DeviceOwnedBufferUint64> device_owned_buffers_matrix_c;
  device_owned_buffers_matrix_a.resize(M * K);
  device_owned_buffers_matrix_b.resize(K * N);
  device_owned_buffers_matrix_c.resize(M * N);
  std::vector<cudaStream_t> p_streams_vec;
  p_streams_vec.resize(M * N);

  // Step 1 compute layout_matrix for each block of matrix C.
  std::cout << "[PPRQuery] Computing layout matrix for each block ..."
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

              auto block_a = A_->GetBitTileMatrixPtrByIdx(i * K + k);
              auto block_b = B_->GetBitTileMatrixPtrByIdx(j * K + k);

              if (block_a->GetMetadata().n_nz_tile == 0)
                continue;
              if (block_b->GetMetadata().n_nz_tile == 0)
                continue;

              cudaStream_t &p_stream = p_streams_vec[i * N + j];

              {
                std::lock_guard<std::mutex> lock(mtx);
                if (device_owned_buffers_matrix_c[i * N + j].GetPtr() ==
                    nullptr) {
                  cudaSetDevice(hash_function(i * N + j) % 4);
                  buffers_matrix_c[i * N + j].size =
                      sizeof(uint64_t) * (WORD_OFFSET(n_strips * n_strips) + 1);
                  cudaHostAlloc(&buffers_matrix_c[i * N + j].data,
                                buffers_matrix_c[i * N + j].size,
                                cudaHostAllocDefault);

                  cudaStreamCreate(&p_streams_vec[i * N + j]);
                  device_owned_buffers_matrix_c[i * N + j].Init(
                      buffers_matrix_c[i * N + j].GetSize(), p_stream);
                }
              }

              // std::cout << "  Submit (" << i << "," << k << ") X (" << k <<
              // ", "
              //           << j << ") "
              //           << " to " << hash_function(i * N + j) % 4 <<
              //           std::endl;

              auto &matrix_a_buf = buffers_matrix_a[i * K + k];
              matrix_a_buf.data = block_a->GetNzTileBitmapPtr()->data();
              matrix_a_buf.size =
                  block_a->GetNzTileBitmapPtr()->GetBufferSize();

              auto &matrix_b_buf = buffers_matrix_b[j * K + k];
              matrix_b_buf.data = block_b->GetNzTileBitmapPtr()->data();
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

  std::cout << "[PPRQuery] Counting nz tile..." << std::endl;
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
                      cudaSetDevice(hash_function(i * N + j) % 4);
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

  std::cout << "[PPRQuery] Copying data back to host ..." << std::endl;

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
            cudaSetDevice(hash_function(i * N + j) % 4);
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

  cudaDeviceSynchronize();

  std::cout << "[PPRQuery] Allocating space for Matrix C ..." << std::endl;
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
            std::cout << "[PPRQuery] (" << i << "," << j
                      << ") n_nz_tile: " << n_nz_tile << std::endl;

            auto *bit_tiled_matrix_ptr =
                C_->GetBitTileMatrixPtrByIdx(i * N + j);

            TiledMatrixMetadata metadata{.n_strips = n_strips,
                                         .n_nz_tile = n_nz_tile,
                                         .tile_size = tile_size};
            bit_tiled_matrix_ptr->Init(
                metadata, new Bitmap(n_strips * tile_size,
                                     buffers_matrix_c[i * N + j].GetPtr()));
            // bit_tiled_matrix_ptr->Print();
          }
        }
      });

  std::for_each(p_streams_vec.begin(), p_streams_vec.end(),
                [](auto &s) { cudaStreamDestroy(s); });
  std::for_each(buffers_matrix_b.begin(), buffers_matrix_b.end(),
                [](auto &d) { cudaFree(d.data); });
  std::for_each(buffers_matrix_a.begin(), buffers_matrix_a.end(),
                [](auto &d) { cudaFree(d.data); });
  std::for_each(buffers_matrix_c_count.begin(), buffers_matrix_c_count.end(),
                [](auto &d) { cudaFree(d.data); });

  std::cout << "[PPRQuery] Done!" << std::endl;
}

__host__ void PPRQuery::Run() {
  // Step1 Compute Layout Matrix

  auto start_time = std::chrono::system_clock::now();
  ComputeLayoutMatrix();
  auto end_time = std::chrono::system_clock::now();

  std::cout << "[PPRQuery::Run] Step1 elapsed:"
            << std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                     start_time)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;
}

} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics