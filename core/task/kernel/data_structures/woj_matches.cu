#include "core/task/kernel/data_structures/woj_matches.cuh"
#include "core/util/cuda_check.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

WOJMatches::WOJMatches(VertexID x, VertexID y) : x_(x), y_(y) { Init(x, y); }

void WOJMatches::Init(VertexID x, VertexID y) {
  x_ = x;
  y_ = y;
  CUDA_CHECK(cudaMallocManaged(&data_, sizeof(VertexID *) * x_));
  CUDA_CHECK(cudaMallocManaged(&offset_, sizeof(VertexID)));
  for (VertexID _ = 0; _ < x_; _++) {
    CUDA_CHECK(cudaMallocManaged(&data_[_], sizeof(VertexID) * 2 * y_));
  }
}

void WOJMatches::Free() {
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