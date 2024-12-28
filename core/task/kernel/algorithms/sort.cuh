#ifndef MATRIXGRAPH_CORE_TASK_KERNEL_ALGORITHMS_SORT_CUH_
#define MATRIXGRAPH_CORE_TASK_KERNEL_ALGORITHMS_SORT_CUH_

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

static __forceinline__ __device__ void
BitonicCompare(VertexID key, VertexID *data, VertexID i, VertexID j,
               VertexID dir, VertexID n_cols) {
  VertexID temp;
  if (data[n_cols * i + key] > data[n_cols * j + key] == dir) {
    temp = data[n_cols * i];
    data[n_cols * i + key] = data[n_cols * j + key];
    data[n_cols * j + key] = temp;
  }
}

static __global__ void BitonicSortKernel(VertexID key, VertexID *data,
                                         VertexID n, VertexID stage,
                                         VertexID step, VertexID n_cols) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int ixj = idx ^ step;

  if (ixj > idx && ixj < n) {
    VertexID dir = ((dir & stage) == 0) ? 1 : 0;
    BitonicCompare(key, data, idx, ixj, dir, n_cols);
  }
  __syncthreads();
}

static __host__ void BitonicSort(VertexID key, VertexID *data_ptr, VertexID y,
                                 VertexID n_cols, const cudaStream_t &stream) {

  dim3 dimBlock(512);
  dim3 dimGrid(1024);

  for (int stage = 2; stage < y; stage <<= 1) {
    for (int step = stage >> 1; step > 0; step >>= 1) {
      BitonicSortKernel<<<dimGrid, dimBlock, 0>>>(key, data_ptr, y, stage, step,
                                                  n_cols);
      cudaDeviceSynchronize();
    }
  }
}

// Merge two sorted subarrays into a single sorted array
static __device__ void Merge(VertexID *input, VertexID *output, VertexID x,
                             int left, int mid, int right) {
  int i = left, j = mid + 1, k = left;

  while (i <= mid && j <= right) {
    if (input[i * x] <= input[j * x]) {
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
                                       VertexID x, VertexID y, int step) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int left = idx * step;
  int mid = min(left + step / 2 - 1, y - 1);
  int right = min(left + step - 1, y - 1);

  if (left < y && mid < y) {
    Merge(input, output, x, left, mid, right);
  }
}

// Host function for merge sort
static void MergeSort(VertexID *data, VertexID x, VertexID y,
                      VertexID data_size) {
  VertexID *input_data = data;
  VertexID *tmp_data;
  CUDA_CHECK(cudaMallocManaged(&tmp_data, data_size));

  int step = 2;
  while (step / 2 < y) {
    int threadsPerBlock = kBlockDim;
    int blocks = (y + step - 1) / step;

    MergeSortKernel<<<blocks, threadsPerBlock>>>(data, tmp_data, x, y, step);
    cudaDeviceSynchronize();

    // Swap input and output arrays
    std::swap(data, tmp_data);
    step *= 2;
  }

  if (data == input_data) {
    cudaFree(tmp_data);
  } else {
    CUDA_CHECK(cudaMemcpy(input_data, data, x * y * sizeof(VertexID),
                          cudaMemcpyDeviceToDevice));
    cudaFree(data);
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