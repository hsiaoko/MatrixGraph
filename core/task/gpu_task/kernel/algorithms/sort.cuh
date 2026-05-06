#ifndef MATRIXGRAPH_CORE_TASK_KERNEL_ALGORITHMS_SORT_CUH_
#define MATRIXGRAPH_CORE_TASK_KERNEL_ALGORITHMS_SORT_CUH_

#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>

#include "core/common/consts.h"
#include "core/common/types.h"
#include "core/util/cuda_check.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

using VertexID = sics::matrixgraph::core::common::VertexID;
using sics::matrixgraph::core::common::kBlockDim;

template <class T> static __host__ __device__ void swap(T *a, T *b) {
  T temp = *a;
  *a = *b;
  *b = temp;
}

// Merge two sorted subarrays into a single sorted array
static __device__ void Merge(VertexID *input, VertexID *output, VertexID key,
                             VertexID x, int left, int mid, int right) {
  int i = left, j = mid + 1, k = left;

  while (i <= mid && j <= right) {
    if (input[i * x + key] <= input[j * x + key]) {
      memcpy(output + k * x, input + i * x, sizeof(VertexID) * x);
      k++;
      i++;
      // output[k++] = input[i++];
    } else {
      memcpy(output + k * x, input + j * x, sizeof(VertexID) * x);
      k++;
      j++;
      // output[k++] = input[j++];
    }
  }
  while (i <= mid) {
    memcpy(output + k * x, input + i * x, sizeof(VertexID) * x);
    k++;
    i++;
    // output[k++] = input[i++];
  }
  while (j <= right) {
    memcpy(output + k * x, input + j * x, sizeof(VertexID) * x);
    k++;
    j++;
    // output[k++] = input[j++];
  }
}

// Kernel for parallel merge sort
static __global__ void MergeSortKernel(VertexID *input, VertexID *output,
                                       VertexID key, VertexID x, VertexID y,
                                       int step) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int left = idx * step;
  int mid = min(left + step / 2 - 1, y - 1);
  int right = min(left + step - 1, y - 1);

  if (left < y && mid < y) {
    Merge(input, output, key, x, left, mid, right);
  }
}

// Host function for merge sort on managed row buffers. `data_size_bytes` is the
// full WOJ row storage (x * y * sizeof(VertexID)). If `scratch_pool` is
// non-null, it must point to at least `data_size_bytes` bytes; no extra
// allocation is performed (reused across sorts). If null, a temp buffer is
// allocated and freed per call.
static void MergeSort(const cudaStream_t& stream, VertexID* data, VertexID key,
                      VertexID x, VertexID y, size_t data_size_bytes,
                      VertexID* scratch_pool = nullptr) {
  VertexID* input_data = data;
  VertexID* tmp_data = nullptr;
  bool owns_tmp = false;
  if (scratch_pool != nullptr) {
    tmp_data = scratch_pool;
  } else {
    CUDA_CHECK(cudaMallocManaged(&tmp_data, data_size_bytes));
    owns_tmp = true;
  }

  VertexID* in_buf = input_data;
  VertexID* out_buf = tmp_data;

  int step = 2;
  while (step / 2 < static_cast<int>(y)) {
    const int threadsPerBlock = static_cast<int>(kBlockDim);
    int blocks = (static_cast<int>(y) + step - 1) / step;

    MergeSortKernel<<<blocks, threadsPerBlock, 0, stream>>>(in_buf, out_buf,
                                                              key, x, y, step);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUDA_CHECK(err);
    }
    std::swap(in_buf, out_buf);
    step *= 2;
  }

  // Sorted result is in `in_buf` after the last swap (same end state as legacy
  // implementation using `data` / `tmp_data`).
  if (in_buf == input_data) {
    if (owns_tmp) {
      CUDA_CHECK(cudaFree(out_buf));
    }
  } else {
    CUDA_CHECK(cudaMemcpyAsync(input_data, in_buf, data_size_bytes,
                               cudaMemcpyDefault, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (owns_tmp) {
      CUDA_CHECK(cudaFree(in_buf));
    }
  }
}

static __host__ __device__ int Partition(int arr[], int low, int high) {

  // Initialize pivot to be the first element
  int p = arr[low];
  int i = low;
  int j = high;

  while (i < j) {

    // Find the first element greater than
    // the pivot (from starting)
    while (arr[i] <= p && i <= high - 1) {
      i++;
    }

    // Find the first element smaller than
    // the pivot (from last)
    while (arr[j] > p && j >= low + 1) {
      j--;
    }
    if (i < j) {
      swap(&arr[i], &arr[j]);
    }
  }
  swap(&arr[low], &arr[j]);
  return j;
}

static __host__ __device__ void QuickSort(int arr[], int low, int high) {
  if (low < high) {

    // call partition function to find Partition Index
    int pi = Partition(arr, low, high);

    // Recursively call quickSort() for left and right
    // half based on Partition Index
    QuickSort(arr, low, pi - 1);
    QuickSort(arr, pi + 1, high);
  }
}

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_TASK_KERNEL_ALGORITHMS_SORT_CUH_