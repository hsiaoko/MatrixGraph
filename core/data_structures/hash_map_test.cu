#include "core/data_structures/hash_map.cuh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

using sics::matrixgraph::core::data_structures::DefaultHash;
using sics::matrixgraph::core::data_structures::DeviceHashMap;
using sics::matrixgraph::core::data_structures::HashMap;
using sics::matrixgraph::core::data_structures::HostHashMap;

// ---------------------------------------------------------------------------
// Device kernel: every thread looks up its key and writes the value to out.
// If the key is not found, write NaN.
// ---------------------------------------------------------------------------
__global__ void lookup_kernel(const HashMap<uint32_t, float> map,
                              const uint32_t* keys,
                              float* out,
                              uint32_t n) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  const float* val = map.find(keys[idx]);
  if (val) {
    out[idx] = *val;
  } else {
    out[idx] = NAN;
  }
}

// ---------------------------------------------------------------------------
// Host test helpers
// ---------------------------------------------------------------------------
static bool float_eq(float a, float b) {
  if (std::isnan(a) && std::isnan(b)) return true;
  return std::fabs(a - b) < 1e-5f;
}

int main() {
  std::cout << "[HashMap Test] Starting..." << std::endl;

  // 1. Basic smoke test with HostHashMap.
  {
    uint32_t keys[] = {10, 20, 30, 40, 50};
    float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    HostHashMap<uint32_t, float> map(keys, values, 5);
    const auto& view = map.View();

    assert(view.get_size() == 5);
    assert(view.contains(10));
    assert(view.contains(50));
    assert(!view.contains(99));

    for (uint32_t i = 0; i < 5; ++i) {
      const float* p = view.find(keys[i]);
      assert(p != nullptr);
      assert(*p == values[i]);
    }
    std::cout << "[HostHashMap smoke test] PASSED" << std::endl;
  }

  // 2. Larger random test (HostHashMap host lookup + DeviceHashMap device
  // lookup).
  {
    constexpr uint32_t n = 100000;
    std::vector<uint32_t> keys(n);
    std::vector<float> values(n);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> val_dist(-1000.0f, 1000.0f);

    // Generate unique keys to avoid undefined duplicate-key behavior.
    std::vector<uint32_t> pool(n);
    for (uint32_t i = 0; i < n; ++i) pool[i] = i + 1;
    std::shuffle(pool.begin(), pool.end(), rng);
    for (uint32_t i = 0; i < n; ++i) {
      keys[i] = pool[i];
      values[i] = val_dist(rng);
    }

    // Host-side map
    HostHashMap<uint32_t, float> host_map(keys.data(), values.data(), n);
    const auto& host_view = host_map.View();
    for (uint32_t i = 0; i < n; ++i) {
      const float* p = host_view.find(keys[i]);
      if (p == nullptr || !float_eq(*p, values[i])) {
        std::cerr << "Host lookup failed at i=" << i << " key=" << keys[i]
                  << std::endl;
        return 1;
      }
    }
    std::cout << "[HostHashMap random lookup] PASSED (n=" << n << ")"
              << std::endl;

    // Device-side map
    DeviceHashMap<uint32_t, float> dev_map(keys.data(), values.data(), n);

    uint32_t* d_keys = nullptr;
    float* d_out = nullptr;
    cudaMallocManaged(&d_keys, sizeof(uint32_t) * n);
    cudaMallocManaged(&d_out, sizeof(float) * n);
    std::memcpy(d_keys, keys.data(), sizeof(uint32_t) * n);

    const uint32_t block_size = 256;
    const uint32_t grid_size = (n + block_size - 1) / block_size;
    lookup_kernel<<<grid_size, block_size>>>(dev_map.View(), d_keys, d_out, n);
    cudaDeviceSynchronize();

    for (uint32_t i = 0; i < n; ++i) {
      if (!float_eq(d_out[i], values[i])) {
        std::cerr << "Device lookup mismatch at i=" << i << " key=" << keys[i]
                  << " expected=" << values[i] << " got=" << d_out[i]
                  << std::endl;
        return 1;
      }
    }
    std::cout << "[DeviceHashMap random lookup] PASSED (n=" << n << ")"
              << std::endl;

    cudaFree(d_keys);
    cudaFree(d_out);
  }

  // 3. Empty map test
  {
    HostHashMap<uint32_t, float> empty_host(nullptr, nullptr, 0);
    const auto& hview = empty_host.View();
    assert(hview.get_size() == 0);
    assert(hview.get_capacity() == 0);
    assert(!hview.contains(1));
    assert(hview.find(1) == nullptr);

    DeviceHashMap<uint32_t, float> empty_dev(nullptr, nullptr, 0);
    const auto& dview = empty_dev.View();
    assert(dview.get_size() == 0);
    assert(dview.get_capacity() == 0);
    std::cout << "[Empty map test] PASSED" << std::endl;
  }

  // 4. Custom struct key test (HostHashMap)
  {
    struct PairKey {
      uint32_t a;
      uint32_t b;
      __host__ __device__ bool operator==(const PairKey& other) const {
        return a == other.a && b == other.b;
      }
    };

    PairKey keys[] = {{1, 2}, {3, 4}, {5, 6}};
    int values[] = {10, 20, 30};

    HostHashMap<PairKey, int> map(keys, values, 3);
    const auto& view = map.View();

    assert(view.contains({1, 2}));
    assert(view.contains({3, 4}));
    assert(!view.contains({1, 3}));
    assert(*view.find({5, 6}) == 30);
    std::cout << "[Custom struct key test] PASSED" << std::endl;
  }

  std::cout << "[HashMap Test] All tests passed." << std::endl;
  return 0;
}
