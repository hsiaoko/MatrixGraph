#ifndef MATRIXGRAPH_CORE_TASK_KERNEL_ALGORITHMS_HEAP_CUH_
#define MATRIXGRAPH_CORE_TASK_KERNEL_ALGORITHMS_HEAP_CUH_

#include <cstdint>

#include <cuda_runtime.h>

#include "core/common/consts.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

using sics::matrixgraph::core::common::kDefaultHeapCapacity;
class MinHeap {
private:
public:
  __host__ __device__ __forceinline__ void Insert(uint32_t val) {
    if (offset_ < capacity_) {
      data_[offset_++] = val;
      int i = offset_ - 1;
      while (i > 0) {
        int parent = (i - 1) / 2;
        if (data_[i] > data_[parent]) {
          swap(&data_[i], &data_[parent]);
        } else {
          break;
        }
      }
    } else if (val < data_[0]) {
      data_[0] = val;
      heapify(0);
    }
  }

  __host__ __device__ void Clear() {
    memset(data_, 0, sizeof(uint32_t) * capacity_);
    offset_ = 0;
  }

  __host__ __device__ void Print() const {
    printf("MinHeap Print: ");
    for (uint32_t _ = 0; _ < offset_; _++) {
      printf("%u ", data_[_]);
    }
    printf("\n");
  }

  __host__ __device__ __forceinline__ void CopyData(uint32_t *other) {
    memcpy(other, data_, sizeof(uint32_t) * capacity_);
  }

  __host__ __device__ __forceinline__ void MoveData(uint32_t **other) {
    *other = data_;
  }

  __host__ __device__ __forceinline__ uint32_t
  get_element_by_idx(uint32_t idx) const {
    return data_[idx];
  }

  __host__ __device__ __forceinline__ uint32_t get_offset() const {
    return offset_;
  }

  __host__ __device__ __forceinline__ uint32_t get_min() const {
    return data_[0];
  }

private:
  __host__ __device__ __forceinline__ void heapify(int i) {
    int largest = i;
    while (1) {
      int left = 2 * i + 1;
      int right = 2 * i + 2;

      if (left < offset_ && data_[left] > data_[largest]) {
        largest = left;
      }

      if (right < offset_ && data_[right] < data_[largest]) {
        largest = right;
      }

      // The root is already the smallest stop!
      if (largest == i) {
        break;
      }
      swap(&data_[i], &data_[largest]);
      i = largest;
    }
  }

  __host__ __device__ __forceinline__ void swap(uint32_t *a, uint32_t *b) {
    uint32_t tmp = *a;
    *a = *b;
    *b = tmp;
  }

  // current number of elements in the heap.
  uint32_t offset_ = 0;

  // maximum capacity of the heap (k).
  const uint32_t capacity_ = kDefaultHeapCapacity;

  // Array to store heap elements.
public:
  uint32_t data_[kDefaultHeapCapacity] = {0};
};

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_TASK_KERNEL_ALGORITHMS_HEAP_CUH_