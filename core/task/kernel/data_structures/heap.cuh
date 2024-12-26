#ifndef MATRIXGRAPH_CORE_TASK_KERNEL_ALGORITHMS_HEAP_CUH_
#define MATRIXGRAPH_CORE_TASK_KERNEL_ALGORITHMS_HEAP_CUH_

#include <cstdint>

#include <cuda_runtime.h>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

class MinHeap {
public:
  //__host__ __device__ MinHeap(uint32_t capacity) {
  // capacity_ = capacity;
  // offset_ = 0;
  // data_ = (uint32_t *)malloc(capacity_ * sizeof(uint32_t));
  // data_ = (uint32_t *)malloc(capacity_ * sizeof(uint32_t));
  //}

  //__host__ __device__ ~MinHeap() { free(data_);
  //}

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

  __host__ __device__ void Print() const {
    printf("\nMinHeap Print: ");
    for (uint32_t _ = 0; _ < offset_; _++) {
      printf("%u ", data_[_]);
    }
    printf("\n");
  }

  __host__ __device__ __forceinline__ uint32_t SetElementByIdx(uint32_t val,
                                                               uint32_t idx) {
    data_[idx] = val;
  }
  //__forceinline__ uint32_t *get_data_ptr() const { return data_; }

  __host__ __device__ __forceinline__ uint32_t
  get_element_by_idx(uint32_t idx) const {
    return data_[idx];
  }

  __host__ __device__ __forceinline__ uint32_t get_offset() const {
    return offset_;
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

  // Array to store heap elements.
  uint32_t data_[3] = {0};

  // current number of elements in the heap.
  uint32_t offset_ = 0;

  // maximum capacity of the heap (k).
  uint32_t capacity_ = 3;
};

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_TASK_KERNEL_ALGORITHMS_HEAP_CUH_