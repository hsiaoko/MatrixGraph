#include "core/task/ppr_query.cuh"

#include <iostream>

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

__host__ void PPRQuery::LoadData() {
  GridTiledMatrixIO grid_tiled_matrix_io;

  grid_tiled_matrix_io.Read(input_path_, &A_);
  grid_tiled_matrix_io.Read(input_path_transposed_, &B_);
  A_->Print();
  B_->Print();
  C_ = new GridTiledMatrix(A_->get_metadata());
}

__host__ void PPRQuery::ComputeLayoutMatrix() {
  VertexID M = A_->get_metadata().n_chunks;
  VertexID K = A_->get_metadata().n_chunks;
  VertexID N = A_->get_metadata().n_chunks;

  VertexID n_chunks = A_->get_metadata().n_chunks;

  std::vector<BufferUint64> buffers_matrix_c;
  buffers_matrix_c.resize(n_chunks * n_chunks);

  std::vector<DeviceOwnedBufferUint64> device_owned_buffers_matrix_a;
  std::vector<DeviceOwnedBufferUint64> device_owned_buffers_matrix_b;
  std::vector<DeviceOwnedBufferUint64> device_owned_buffers_matrix_c;

  device_owned_buffers_matrix_a.resize(n_chunks * n_chunks);
  device_owned_buffers_matrix_b.resize(n_chunks * n_chunks);
  device_owned_buffers_matrix_c.resize(n_chunks * n_chunks);

  // Step 1 compute layout_matrix for each block of matrix C.
  VertexID n_strips = 0;
  VertexID tile_size = 0;
  std::vector<cudaStream_t *> p_streams_vec;
  p_streams_vec.reserve(M * N);

  for (VertexID i = 0; i < M; i++) {
    for (VertexID j = 0; j < N; j++) {

      cudaStream_t *p_stream = new cudaStream_t;
      p_streams_vec.push_back(p_stream);
      cudaStreamCreate(p_stream);

      buffers_matrix_c[i * n_chunks + j].data =
          new uint64_t[WORD_OFFSET((n_chunks * n_chunks)) + 1]();
      buffers_matrix_c[i * n_chunks + j].size =
          sizeof(uint64_t) * (WORD_OFFSET(n_chunks * n_chunks) + 1);
      device_owned_buffers_matrix_c[i * n_chunks + j].Init(
          buffers_matrix_c[i * n_chunks + j], *p_stream);

      for (VertexID k = 0; k < K; k++) {
        auto block_a = A_->GetBitTileMatrixPtrByIdx(i * K + k);
        auto block_b = B_->GetBitTileMatrixPtrByIdx(j * K + k);

        if (block_a->GetMetadata().n_nz_tile == 0)
          continue;
        if (block_b->GetMetadata().n_nz_tile == 0)
          continue;

        n_strips = block_a->GetMetadata().n_strips;
        tile_size = block_a->GetMetadata().tile_size;

        BufferUint64 matrix_a_buf = {
            .data = block_a->GetNzTileBitmapPtr()->data(),
            .size = block_a->GetNzTileBitmapPtr()->size()};

        BufferUint64 matrix_b_buf = {
            .data = block_b->GetNzTileBitmapPtr()->data(),
            .size = block_b->GetNzTileBitmapPtr()->size()};

        device_owned_buffers_matrix_a[i * n_chunks + k].Init(matrix_a_buf,
                                                             *p_stream);
        device_owned_buffers_matrix_b[j * n_chunks + k].Init(matrix_b_buf,
                                                             *p_stream);

        MatrixOperationsKernelWrapper::MatrixBitAnd(
            *p_stream, device_owned_buffers_matrix_a[i * n_chunks + k],
            device_owned_buffers_matrix_b[j * n_chunks + k],
            &device_owned_buffers_matrix_c[i * n_chunks + j], n_strips,
            n_strips, n_strips);
      }

      cudaMemcpyAsync(buffers_matrix_c[i * n_chunks + j].GetPtr(),
                      device_owned_buffers_matrix_c[i * n_chunks + j].GetPtr(),
                      buffers_matrix_c[i * n_chunks + j].GetSize(),
                      cudaMemcpyDeviceToHost, *p_stream);
    }
  }

  // Step 2 Construct block objects of matrix C.
  cudaDeviceSynchronize();
  for (VertexID i = 0; i < M; i++) {
    for (VertexID j = 0; j < N; j++) {
      Bitmap bm(n_strips * n_strips, buffers_matrix_c[i * n_chunks + j].data);

      VertexID n_nz_tile = bm.Count();
      if (bm.Count() == 0)
        continue;

      TiledMatrixMetadata metadata{
          .n_strips = n_strips, .n_nz_tile = n_nz_tile, .tile_size = tile_size};

      std::cout << "#################################: " << i << "," << j
                << "[PPRQuery] n_nz_tile: " << n_nz_tile << std::endl;

      auto *bit_tiled_matrix_ptr = C_->GetBitTileMatrixPtrByIdx(i * M + j);

      bit_tiled_matrix_ptr->Init(metadata);
      memcpy(bit_tiled_matrix_ptr->GetNzTileBitmapPtr()->data(),
             buffers_matrix_c[i * n_chunks + j].GetPtr(),
             buffers_matrix_c[i * n_chunks + j].GetSize());
      bit_tiled_matrix_ptr->Print();
    }
  }

  std::cout << "[PPRQuery] Done!" << std::endl;
}

__host__ void PPRQuery::Run() {

  // Step1 Compute Layout Matrix
  ComputeLayoutMatrix();

  std::cout << "Running PPRQuery" << std::endl;
}

} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics