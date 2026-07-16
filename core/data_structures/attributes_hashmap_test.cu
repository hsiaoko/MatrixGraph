#include "core/data_structures/attributes.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using sics::matrixgraph::core::data_structures::Attribute;
using sics::matrixgraph::core::data_structures::AttributeName;
using sics::matrixgraph::core::data_structures::Attributes;
using sics::matrixgraph::core::data_structures::DeviceAttributes;
using sics::matrixgraph::core::data_structures::GetFloat64;
using sics::matrixgraph::core::data_structures::GetInt;
using sics::matrixgraph::core::data_structures::HostHashMap;
using sics::matrixgraph::core::data_structures::kMaxAttributeNameLength;
using sics::matrixgraph::core::data_structures::ValueType;

// ---------------------------------------------------------------------------
// Kernel demonstrating SIMD-style access:
// All threads in a warp share the same Attributes object and look up the
// SAME attribute name (broadcast).  They then access their own row via tid.
// ---------------------------------------------------------------------------
__global__ void simd_lookup_kernel(const Attributes attrs,
                                   int64_t* out_age,
                                   double* out_score,
                                   uint32_t n) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  // All threads in a warp execute find("age") simultaneously (same key,
  // same hash-map, broadcast / cache-friendly).
  const Attribute* age_attr = attrs.attr_map.find(AttributeName("age"));
  if (age_attr) {
    out_age[tid] = GetInt(*age_attr, tid);
  } else {
    out_age[tid] = -1;
  }

  const Attribute* score_attr = attrs.attr_map.find(AttributeName("score"));
  if (score_attr) {
    out_score[tid] = GetFloat64(*score_attr, tid);
  } else {
    out_score[tid] = NAN;
  }
}

int main() {
  std::cout << "[Attributes HashMap Test] Starting..." << std::endl;

  constexpr uint32_t n_rows = 1024;

  // 1. Prepare column data on host.
  std::vector<int64_t> age_data(n_rows);
  std::vector<double> score_data(n_rows);
  for (uint32_t i = 0; i < n_rows; ++i) {
    age_data[i] = static_cast<int64_t>(i * 2);
    score_data[i] = static_cast<double>(i) * 0.5;
  }

  // 2. Copy column data to device memory (managed for simplicity in test).
  int64_t* d_age_data = nullptr;
  double* d_score_data = nullptr;
  cudaMallocManaged(&d_age_data, sizeof(int64_t) * n_rows);
  cudaMallocManaged(&d_score_data, sizeof(double) * n_rows);
  std::memcpy(d_age_data, age_data.data(), sizeof(int64_t) * n_rows);
  std::memcpy(d_score_data, score_data.data(), sizeof(double) * n_rows);

  // 3. Build Attribute descriptors.
  Attribute age_attr{};
  std::strncpy(age_attr.name, "age", kMaxAttributeNameLength);
  age_attr.type = ValueType::kInt;
  age_attr.n_rows = n_rows;
  age_attr.n_elements = n_rows;
  age_attr.data = d_age_data;
  age_attr.offsets = nullptr;

  Attribute score_attr{};
  std::strncpy(score_attr.name, "score", kMaxAttributeNameLength);
  score_attr.type = ValueType::kFloat64;
  score_attr.n_rows = n_rows;
  score_attr.n_elements = n_rows;
  score_attr.data = d_score_data;
  score_attr.offsets = nullptr;

  AttributeName names[] = {AttributeName("age"), AttributeName("score")};
  Attribute attrs[] = {age_attr, score_attr};

  // 4. Build DeviceAttributes (hash map lives in device memory).
  DeviceAttributes dev_attrs(1 /* entity_id / label_id */, names, attrs, 2);

  // 5. Launch kernel: all threads share the same Attributes view.
  int64_t* d_out_age = nullptr;
  double* d_out_score = nullptr;
  cudaMallocManaged(&d_out_age, sizeof(int64_t) * n_rows);
  cudaMallocManaged(&d_out_score, sizeof(double) * n_rows);

  const uint32_t block_size = 256;
  const uint32_t grid_size = (n_rows + block_size - 1) / block_size;
  simd_lookup_kernel<<<grid_size, block_size>>>(
      dev_attrs.View(), d_out_age, d_out_score, n_rows);
  cudaDeviceSynchronize();

  // 6. Verify.
  for (uint32_t i = 0; i < n_rows; ++i) {
    if (d_out_age[i] != age_data[i]) {
      std::cerr << "Age mismatch at i=" << i << " expected=" << age_data[i]
                << " got=" << d_out_age[i] << std::endl;
      return 1;
    }
    if (std::fabs(d_out_score[i] - score_data[i]) > 1e-9) {
      std::cerr << "Score mismatch at i=" << i << " expected=" << score_data[i]
                << " got=" << d_out_score[i] << std::endl;
      return 1;
    }
  }

  std::cout << "[SIMD lookup kernel] PASSED (n=" << n_rows << ")" << std::endl;

  // 7. Host-side lookup via Attributes view pointing to host memory.
  {
    AttributeName h_names[] = {AttributeName("age"), AttributeName("score")};
    Attribute h_attrs[] = {age_attr, score_attr};
    // Temporarily point data back to host arrays for host lookup.
    h_attrs[0].data = age_data.data();
    h_attrs[1].data = score_data.data();

    HostHashMap<AttributeName, Attribute> host_map(h_names, h_attrs, 2);
    Attributes host_view;
    host_view.entity_id = 1;
    host_view.attr_map = host_map.View();

    const Attribute* p = host_view.attr_map.find(AttributeName("age"));
    assert(p != nullptr);
    assert(GetInt(*p, 100) == age_data[100]);
    std::cout << "[Host lookup] PASSED" << std::endl;
  }

  // 8. Missing key test (use host-side map; device pointers cannot be
  // dereferenced on host).
  {
    AttributeName h_names[] = {AttributeName("age"), AttributeName("score")};
    Attribute h_attrs[] = {age_attr, score_attr};
    h_attrs[0].data = age_data.data();
    h_attrs[1].data = score_data.data();

    HostHashMap<AttributeName, Attribute> host_map(h_names, h_attrs, 2);
    Attributes host_view;
    host_view.entity_id = 1;
    host_view.attr_map = host_map.View();

    const Attribute* p = host_view.attr_map.find(AttributeName("nonexistent"));
    assert(p == nullptr);
    std::cout << "[Missing key lookup] PASSED" << std::endl;
  }

  std::cout << "[Attributes HashMap Test] All tests passed." << std::endl;

  cudaFree(d_out_age);
  cudaFree(d_out_score);
  cudaFree(d_age_data);
  cudaFree(d_score_data);
  return 0;
}
