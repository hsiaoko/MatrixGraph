#include "core/task/kernel/data_structures/hash_buckets.cuh"
#include "core/util/cuda_check.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

HashBuckets::HashBuckets(VertexID x, VertexID y) : x_(x), y_(y) { Init(x, y); }

void HashBuckets::Init(VertexID x, VertexID y) {
  x_ = x;
  y_ = y;
  CUDA_CHECK(cudaMallocManaged(&data_, sizeof(VertexID *) * x_));
  CUDA_CHECK(cudaMallocManaged(&offset_, sizeof(VertexID) * x_));
  for (VertexID _ = 0; _ < x_; _++) {
    CUDA_CHECK(cudaMallocManaged(&data_[_], sizeof(VertexID) * 2 * y_));
  }
}

void HashBuckets::Free() {
  for (VertexID _ = 0; _ < x_; _++) {
    CUDA_CHECK(cudaFree(data_[_]));
  }
  CUDA_CHECK(cudaFree(data_));
  CUDA_CHECK(cudaFree(offset_));
}

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics