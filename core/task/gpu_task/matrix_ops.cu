#include "core/common/consts.h"
#include "core/common/host_algorithms.cuh"
#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/host_buffer.cuh"
#include "core/data_structures/metadata.h"
#include "core/data_structures/unified_buffer.cuh"
#include "core/task/gpu_task/kernel/kernel_matrix_ops.cuh"
#include "core/task/gpu_task/matrix_ops.cuh"
#include "core/util/atomic.h"
#include "core/util/bitmap_no_ownership.h"
#include "core/util/bitmap_ownership.h"
#include <ctime>
#include <cuda_runtime.h>
#include <execution>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
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
using BitmapOwnership = sics::matrixgraph::core::util::BitmapOwnership;
using BitmapNoOwnerShip = sics::matrixgraph::core::util::BitmapNoOwnerShip;
using sics::matrixgraph::core::util::atomic::WriteAdd;
using Edges = sics::matrixgraph::core::data_structures::Edges;
using sics::matrixgraph::core::util::atomic::WriteAdd;
using sics::matrixgraph::core::util::atomic::WriteMax;
using sics::matrixgraph::core::util::atomic::WriteMin;
using GraphMetadata = sics::matrixgraph::core::data_structures::GraphMetadata;
using Edge = sics::matrixgraph::core::data_structures::Edge;
using EdgelistMetadata =
    sics::matrixgraph::core::data_structures::EdgelistMetadata;
using GridGraphMetadata =
    sics::matrixgraph::core::data_structures::GridGraphMetadata;
using sics::matrixgraph::core::common::kDefalutNumEdgesPerBlock;
using sics::matrixgraph::core::common::kDefalutNumEdgesPerTile;
using sics::matrixgraph::core::common::kMaxNumEdges;
using sics::matrixgraph::core::common::kMaxNumEdgesPerBlock;

__host__ void MatrixOps::Run() {}

void MatrixOps::cuBLASMatmult(float* A, float* B, float* C, int m, int k, int n,
                              bool transa_tag, bool transb_tag) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasOperation_t transa = transa_tag ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t transb = transb_tag ? CUBLAS_OP_N : CUBLAS_OP_T;

  float alpha = 1.0f, beta = 0.0f;
  cublasSgeam(handle, transa, transb, m, n, &alpha, A, m, &beta, B, n, C, m);

  cublasDestroy(handle);
}

void MatrixOps::MatMult(float* A, float* B, float* C, int m, int k, int n) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::MatrixOpsKernelWrapper::MatMult(stream, A, B, C, m, k, n);

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
}

void MatrixOps::Activate(float* A, int m, int n) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  kernel::MatrixOpsKernelWrapper::Activate(stream, A, m, n);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
}

void MatrixOps::MatAdd(float* A, float* B, int m, int n) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  kernel::MatrixOpsKernelWrapper::MatAdd(stream, A, B, m, n);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
}

void MatrixOps::Transpose(float* A, float* B, int m, int n) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  kernel::MatrixOpsKernelWrapper::Transpose(stream, A, B, m, n);

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics