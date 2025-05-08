#ifndef MATRIXGRAPH_CORE_TASK_KERNEL_ALGORITHMS_HASH_CUH_
#define MATRIXGRAPH_CORE_TASK_KERNEL_ALGORITHMS_HASH_CUH_

#include <cuda_runtime.h>

#include <iostream>

#include "core/common/types.h"
#include "core/util/cuda_check.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

using VertexID = sics::matrixgraph::core::common::VertexID;

__device__ constexpr uint32_t FNV_OFFSET = 0x811c9dc5;
__device__ constexpr uint32_t FNV_PRIME = 0x01000193;

static __forceinline__ __device__ VertexID Hash(VertexID key) {
  key ^= key >> 16;
  key *= 0x853bca6b;
  key ^= key >> 13;
  key *= 0xc2b2ae35;
  key ^= key >> 16;
  return (VertexID)key;
}

static __forceinline__ __host__ __device__ VertexID HashTable(VertexID key) {
  //  switch key
  switch (key) {
    case 0:
      return 12;
    case 1:
      return 3;
    case 2:
      return 11;
    case 3:
      return 9;
    case 4:
      return 15;
    case 5:
      return 2;
    case 6:
      return 8;
    case 7:
      return 4;
    case 8:
      return 13;
    case 9:
      return 10;
    case 10:
      return 5;
    case 11:
      return 7;
    case 12:
      return 14;
    case 13:
      return 6;
    case 14:
      return 1;
    case 15:
      return 0;
  }
  return 0;
}

static __forceinline__ __device__ VertexID NoHash(VertexID key) { return key; }

static __forceinline__ __device__ VertexID FNV_1a_Hash(VertexID key) {
  uint32_t hash = FNV_OFFSET;

  // Process each byte of the integer key
  for (VertexID i = 0; i < sizeof(VertexID); ++i) {
    uint8_t byte = (key >> (i * 8)) & 0xFF;
    // Extract byte
    hash ^= byte;
    // XOR with hash
    hash *= FNV_PRIME;
    // Multiply by prime
  }
  return hash;
}

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_KERNEL_ALGORITHMS_HASH_CUH_