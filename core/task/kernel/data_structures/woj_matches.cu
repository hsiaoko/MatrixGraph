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

// WOJMatches::WOJMatches(const WOJMatches &other) {
//   y_ = other.get_y();
//   x_ = other.get_x();
//   Init(x_, y_);
//   *y_offset_ = other.get_y_offset();
//   *x_offset_ = other.get_x_offset();
//   memcpy(data_, other.get_data_ptr(), sizeof(VertexID) * x_ * y_);
// }
//
// WOJMatches::WOJMatches(WOJMatches &&other) {
//   y_ = other.get_y();
//   x_ = other.get_x();
//   Init(x_, y_);
//   *y_offset_ = other.get_y_offset();
//   *x_offset_ = other.get_x_offset();
//   data_ = other.get_data_ptr();
// }

WOJMatches::WOJMatches(VertexID x, VertexID y) : x_(x), y_(y) { Init(x, y); }

// WOJMatches &WOJMatches::operator=(const WOJMatches &other) {
//   if (this != &other) {
//     delete[] data_;
//
//     y_ = other.get_y();
//     x_ = other.get_x();
//     Init(x_, y_);
//     *y_offset_ = other.get_y_offset();
//     *x_offset_ = other.get_x_offset();
//     memcpy(data_, other.get_data_ptr(), sizeof(VertexID) * x_ * y_);
//   }
//   return *this;
// }
//
// WOJMatches &WOJMatches::operator=(WOJMatches &&other) {
//   if (this != &other) {
//     delete[] data_;
//
//     y_ = other.get_y();
//     x_ = other.get_x();
//     Init(x_, y_);
//     *y_offset_ = other.get_y_offset();
//     *x_offset_ = other.get_x_offset();
//     data_ = other.get_data_ptr();
//   }
//   return *this;
// }

void WOJMatches::Init(VertexID x, VertexID y) {
  x_ = x;
  y_ = y;
  CUDA_CHECK(cudaMallocManaged(&y_offset_, sizeof(VertexID)));
  CUDA_CHECK(cudaMallocManaged(&x_offset_, sizeof(VertexID)));
  CUDA_CHECK(cudaMallocManaged(&data_, sizeof(VertexID) * x_ * y_));
  CUDA_CHECK(cudaMallocManaged(&header_ptr_, sizeof(VertexID) * x_));
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
    std::cout << "set, :_" << _ << " right header: " << right_header[_] << " "
              << hash_keys.second << std::endl;
    header_ptr_[*x_offset_] = right_header[_];
    (*x_offset_)++;
  }
}

std::pair<VertexID, VertexID> WOJMatches::GetJoinKey(const WOJMatches &other) {

  std::cout << "GetJoinKey" << std::endl;
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