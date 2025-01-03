#include <cassert>

#include "core/common/consts.h"
#include "core/common/types.h"
#include "core/task/kernel/data_structures/woj_matches.cuh"
#include "core/util/bitmap.h"
#include "core/util/cuda_check.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

using sics::matrixgraph::core::common::kMaxVertexID;
using VertexID = sics::matrixgraph::core::common::VertexID;
using Bitmap = sics::matrixgraph::core::util::Bitmap;

WOJMatches::WOJMatches(VertexID x, VertexID y) : x_(x), y_(y) { Init(x, y); }

void WOJMatches::Init(VertexID x, VertexID y) {
  x_ = x;
  y_ = y;
  CUDA_CHECK(cudaMallocManaged(&y_offset_, sizeof(VertexID)));
  CUDA_CHECK(cudaMallocManaged(&x_offset_, sizeof(VertexID)));
  CUDA_CHECK(cudaMallocManaged(&data_, sizeof(VertexID) * x_ * y_));
  CUDA_CHECK(cudaMallocManaged(&header_ptr_, sizeof(VertexID) * x_));
  CUDA_CHECK(cudaMemset(y_offset_, 0, sizeof(VertexID)));
  CUDA_CHECK(cudaMemset(x_offset_, 0, sizeof(VertexID)));
}

VertexID WOJMatches::BinarySearch(VertexID search_col, VertexID target) const {
  int left = 0;
  int right = get_y_offset() - 1;
  int mid = 0;
  VertexID x = get_x_offset();
  while (left <= right) {
    mid = left + (right - left) / 2;
    if (data_[mid * x + search_col] == target)
      return (VertexID)mid;
    else if (data_[mid * x + search_col] < target)
      left = mid + 1;
    else
      right = mid - 1;
  }
  return kMaxVertexID;
}

void WOJMatches::SetHeader(const VertexID *left_header, VertexID left_offset_x,
                           const VertexID *right_header,
                           VertexID right_offset_x,
                           const std::pair<VertexID, VertexID> &hash_keys) {

  *x_offset_ = left_offset_x;
  for (VertexID _ = 0; _ < left_offset_x; _++) {
    header_ptr_[_] = left_header[_];
  }
  for (VertexID _ = 0; _ < right_offset_x; _++) {
    if (_ == hash_keys.second)
      continue;
    header_ptr_[*x_offset_] = right_header[_];
    (*x_offset_)++;
  }
}

std::vector<WOJMatches *> WOJMatches::SplitAndCopy(VertexID n_partitions) {
  std::vector<WOJMatches *> splitted_data;
  splitted_data.resize(n_partitions);

  VertexID base_size = get_y_offset() / n_partitions;
  VertexID remainder = get_y_offset() % n_partitions;

  VertexID count = 0;
  for (VertexID _ = 0; _ < n_partitions; _++) {

    VertexID partition_size =
        base_size + (_ == n_partitions - 1 ? remainder : 0);
    splitted_data[_] = new WOJMatches();
    splitted_data[_]->Init(x_, y_);
    splitted_data[_]->SetYOffset(partition_size);
    splitted_data[_]->SetXOffset(get_x_offset());

    CUDA_CHECK(cudaMemcpy(splitted_data[_]->get_data_ptr(),
                          get_data_ptr() + count * get_x_offset(),
                          sizeof(VertexID) * partition_size * get_x_offset(),
                          cudaMemcpyDefault));

    CUDA_CHECK(cudaMemcpy(splitted_data[_]->get_header_ptr(), get_header_ptr(),
                          sizeof(VertexID) * x_, cudaMemcpyDefault));
    count += partition_size;
  }
  return splitted_data;
}

std::pair<VertexID, VertexID> WOJMatches::GetJoinKey(const WOJMatches &other) {

  Bitmap visited(32);
  VertexID inverted_index[32];
  VertexID left_hash_idx = kMaxVertexID;
  VertexID right_hash_idx = kMaxVertexID;

  for (VertexID _ = 0; _ < get_x_offset(); _++) {
    visited.SetBit(header_ptr_[_]);
    inverted_index[header_ptr_[_]] = _;
  }

  for (VertexID _ = 0; _ < other.get_x_offset(); _++) {
    if (visited.GetBit(other.get_header_ptr()[_])) {
      left_hash_idx = inverted_index[other.get_header_ptr()[_]];
      right_hash_idx = _;
      break;
    }
  }
  return std::make_pair(left_hash_idx, right_hash_idx);
}

void WOJMatches::Free() {
  CUDA_CHECK(cudaFree(data_));
  CUDA_CHECK(cudaFree(x_offset_));
  CUDA_CHECK(cudaFree(y_offset_));
}

void WOJMatches::CopyData(const WOJMatches &other) {
  Init(other.get_x(), other.get_y());
  SetXOffset(other.get_x_offset());
  SetYOffset(other.get_y_offset());
  CUDA_CHECK(cudaMemcpy(get_data_ptr(), other.get_data_ptr(),
                        sizeof(VertexID) * x_ * y_, cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(get_header_ptr(), other.get_header_ptr(),
                        sizeof(VertexID) * x_, cudaMemcpyDefault));
}

void WOJMatches::CopyDataAsync(const WOJMatches &other,
                               const cudaStream_t &stream) {
  Init(other.get_x(), other.get_y());
  SetXOffset(other.get_x_offset());
  SetYOffset(other.get_y_offset());
  CUDA_CHECK(cudaMemcpyAsync(get_data_ptr(), other.get_data_ptr(),
                             sizeof(VertexID) * x_ * y_, cudaMemcpyDefault,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(get_header_ptr(), other.get_header_ptr(),
                             sizeof(VertexID) * x_, cudaMemcpyDefault, stream));
}

void WOJMatches::Print(VertexID offset) const {
  printf("Matches Print(), x_offset: %d, y_offset: %d, x %d, y %d\n",
         get_x_offset(), get_y_offset(), get_x(), get_y());

  std::cout << "\theader: ";
  for (VertexID _ = 0; _ < get_x_offset(); _++) {
    std::cout << header_ptr_[_] << " ";
  }
  std::cout << std::endl;
  VertexID x_offset = get_x_offset();
  VertexID y_offset = get_y_offset();
  y_offset = std::min(offset, y_offset);

  for (VertexID _ = 0; _ < y_offset; _++) {
    for (VertexID __ = 0; __ < x_offset; __++) {
      printf("%d ", data_[_ * x_offset + __]);
    }
    printf("\n");
  }
}

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics