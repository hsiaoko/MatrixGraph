#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_HASH_MAP_CUH_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_HASH_MAP_CUH_

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

#include "core/util/cuda_check.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

// ---------------------------------------------------------------------------
// DefaultHash  (device-compatible)
// ---------------------------------------------------------------------------
template <typename T>
struct DefaultHash {
  __host__ __device__ uint32_t operator()(const T& key) const {
    static_assert(std::is_trivially_copyable_v<T>,
                  "DefaultHash only supports trivially copyable types");
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&key);
    uint32_t h = 0x811c9dc5u;
    for (uint32_t i = 0; i < sizeof(T); ++i) {
      h ^= static_cast<uint32_t>(bytes[i]);
      h *= 0x01000193u;
    }
    return h;
  }
};

template <>
struct DefaultHash<uint32_t> {
  __host__ __device__ uint32_t operator()(uint32_t key) const {
    key ^= key >> 16;
    key *= 0x85ebca6bu;
    key ^= key >> 13;
    key *= 0xc2b2ae35u;
    key ^= key >> 16;
    return key;
  }
};

template <>
struct DefaultHash<int> {
  __host__ __device__ uint32_t operator()(int key) const {
    return DefaultHash<uint32_t>{}(static_cast<uint32_t>(key));
  }
};

template <>
struct DefaultHash<uint64_t> {
  __host__ __device__ uint32_t operator()(uint64_t key) const {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdull;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ull;
    key ^= key >> 33;
    return static_cast<uint32_t>(key ^ (key >> 32));
  }
};

template <>
struct DefaultHash<int64_t> {
  __host__ __device__ uint32_t operator()(int64_t key) const {
    return DefaultHash<uint64_t>{}(static_cast<uint64_t>(key));
  }
};

template <>
struct DefaultHash<float> {
  __host__ __device__ uint32_t operator()(float key) const {
    static_assert(sizeof(float) == sizeof(uint32_t), "float size mismatch");
    uint32_t u;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(&key);
    uint8_t* dst = reinterpret_cast<uint8_t*>(&u);
    for (uint32_t i = 0; i < sizeof(uint32_t); ++i) dst[i] = src[i];
    return DefaultHash<uint32_t>{}(u);
  }
};

template <>
struct DefaultHash<double> {
  __host__ __device__ uint32_t operator()(double key) const {
    static_assert(sizeof(double) == sizeof(uint64_t), "double size mismatch");
    uint64_t u;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(&key);
    uint8_t* dst = reinterpret_cast<uint8_t*>(&u);
    for (uint32_t i = 0; i < sizeof(uint64_t); ++i) dst[i] = src[i];
    return DefaultHash<uint64_t>{}(u);
  }
};

// ---------------------------------------------------------------------------
// DefaultEqual
// ---------------------------------------------------------------------------
template <typename T>
struct DefaultEqual {
  __host__ __device__ bool operator()(const T& a, const T& b) const {
    return a == b;
  }
};

// ---------------------------------------------------------------------------
// HashMap  (lightweight non-owning view)
//
// The caller is responsible for ensuring the underlying memory is accessible
// from the execution space where find()/contains() is invoked:
//   - HostHashMap  -> pointers live in host memory
//   - DeviceHashMap -> pointers live in device memory
// ---------------------------------------------------------------------------
template <typename Key,
          typename Value,
          typename Hash = DefaultHash<Key>,
          typename KeyEqual = DefaultEqual<Key>>
struct HashMap {
  static_assert(std::is_trivially_copyable_v<Key>,
                "HashMap Key must be trivially copyable");
  static_assert(std::is_trivially_copyable_v<Value>,
                "HashMap Value must be trivially copyable");

  Key* keys = nullptr;
  Value* values = nullptr;
  uint8_t* occupied = nullptr;  // 0 = empty, 1 = filled
  uint32_t size = 0;
  uint32_t capacity = 0;

  __host__ __device__ const Value* find(const Key& k) const {
    if (capacity == 0) return nullptr;
    const uint32_t h = Hash{}(k);
    uint32_t idx = h & (capacity - 1);
    for (uint32_t i = 0; i < capacity; ++i) {
      const uint32_t probe = (idx + i) & (capacity - 1);
      if (!occupied[probe]) return nullptr;
      if (KeyEqual{}(keys[probe], k)) return &values[probe];
    }
    return nullptr;
  }

  __host__ __device__ Value* find(const Key& k) {
    if (capacity == 0) return nullptr;
    const uint32_t h = Hash{}(k);
    uint32_t idx = h & (capacity - 1);
    for (uint32_t i = 0; i < capacity; ++i) {
      const uint32_t probe = (idx + i) & (capacity - 1);
      if (!occupied[probe]) return nullptr;
      if (KeyEqual{}(keys[probe], k)) return &values[probe];
    }
    return nullptr;
  }

  __host__ __device__ bool contains(const Key& k) const {
    return find(k) != nullptr;
  }

  __host__ __device__ uint32_t get_size() const { return size; }
  __host__ __device__ uint32_t get_capacity() const { return capacity; }
};

// ---------------------------------------------------------------------------
// HostHashMap  (owns host memory)
// ---------------------------------------------------------------------------
template <typename Key,
          typename Value,
          typename Hash = DefaultHash<Key>,
          typename KeyEqual = DefaultEqual<Key>>
class HostHashMap {
 public:
  HostHashMap() = default;

  HostHashMap(const Key* keys, const Value* values, uint32_t n,
              float load_factor = 0.7f) {
    if (n == 0) return;

    uint32_t min_cap =
        static_cast<uint32_t>(static_cast<float>(n) / load_factor) + 1;
    capacity_ = 1;
    while (capacity_ < min_cap) capacity_ <<= 1;

    keys_ = new Key[capacity_];
    values_ = new Value[capacity_];
    occupied_ = new uint8_t[capacity_]();

    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t h = Hash{}(keys[i]);
      uint32_t idx = h & (capacity_ - 1);
      for (uint32_t probe = 0; probe < capacity_; ++probe) {
        const uint32_t slot = (idx + probe) & (capacity_ - 1);
        if (!occupied_[slot]) {
          keys_[slot] = keys[i];
          values_[slot] = values[i];
          occupied_[slot] = 1;
          break;
        }
      }
    }

    view_.keys = keys_;
    view_.values = values_;
    view_.occupied = occupied_;
    view_.size = n;
    view_.capacity = capacity_;
  }

  ~HostHashMap() {
    delete[] keys_;
    delete[] values_;
    delete[] occupied_;
  }

  HostHashMap(const HostHashMap&) = delete;
  HostHashMap& operator=(const HostHashMap&) = delete;

  HostHashMap(HostHashMap&& other) noexcept {
    keys_ = other.keys_;
    values_ = other.values_;
    occupied_ = other.occupied_;
    capacity_ = other.capacity_;
    view_ = other.view_;
    other.keys_ = nullptr;
    other.values_ = nullptr;
    other.occupied_ = nullptr;
    other.capacity_ = 0;
    other.view_ = HashMap<Key, Value, Hash, KeyEqual>{};
  }

  HostHashMap& operator=(HostHashMap&& other) noexcept {
    if (this != &other) {
      delete[] keys_;
      delete[] values_;
      delete[] occupied_;
      keys_ = other.keys_;
      values_ = other.values_;
      occupied_ = other.occupied_;
      capacity_ = other.capacity_;
      view_ = other.view_;
      other.keys_ = nullptr;
      other.values_ = nullptr;
      other.occupied_ = nullptr;
      other.capacity_ = 0;
      other.view_ = HashMap<Key, Value, Hash, KeyEqual>{};
    }
    return *this;
  }

  const HashMap<Key, Value, Hash, KeyEqual>& View() const { return view_; }
  HashMap<Key, Value, Hash, KeyEqual>& View() { return view_; }

 private:
  HashMap<Key, Value, Hash, KeyEqual> view_;
  Key* keys_ = nullptr;
  Value* values_ = nullptr;
  uint8_t* occupied_ = nullptr;
  uint32_t capacity_ = 0;
};

// ---------------------------------------------------------------------------
// DeviceHashMap  (owns device memory; builds on host temp buffer then H2D)
// ---------------------------------------------------------------------------
template <typename Key,
          typename Value,
          typename Hash = DefaultHash<Key>,
          typename KeyEqual = DefaultEqual<Key>>
class DeviceHashMap {
 public:
  DeviceHashMap() = default;

  DeviceHashMap(const Key* keys, const Value* values, uint32_t n,
                float load_factor = 0.7f) {
    if (n == 0) return;

    uint32_t min_cap =
        static_cast<uint32_t>(static_cast<float>(n) / load_factor) + 1;
    capacity_ = 1;
    while (capacity_ < min_cap) capacity_ <<= 1;

    // Build on host first, then copy to device.
    Key* h_keys = new Key[capacity_];
    Value* h_values = new Value[capacity_];
    uint8_t* h_occupied = new uint8_t[capacity_]();

    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t h = Hash{}(keys[i]);
      uint32_t idx = h & (capacity_ - 1);
      for (uint32_t probe = 0; probe < capacity_; ++probe) {
        const uint32_t slot = (idx + probe) & (capacity_ - 1);
        if (!h_occupied[slot]) {
          h_keys[slot] = keys[i];
          h_values[slot] = values[i];
          h_occupied[slot] = 1;
          break;
        }
      }
    }

    CUDA_CHECK(cudaMalloc(&keys_, sizeof(Key) * capacity_));
    CUDA_CHECK(cudaMalloc(&values_, sizeof(Value) * capacity_));
    CUDA_CHECK(cudaMalloc(&occupied_, sizeof(uint8_t) * capacity_));

    CUDA_CHECK(
        cudaMemcpy(keys_, h_keys, sizeof(Key) * capacity_, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(values_, h_values, sizeof(Value) * capacity_,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(occupied_, h_occupied, sizeof(uint8_t) * capacity_,
                          cudaMemcpyHostToDevice));

    delete[] h_keys;
    delete[] h_values;
    delete[] h_occupied;

    view_.keys = keys_;
    view_.values = values_;
    view_.occupied = occupied_;
    view_.size = n;
    view_.capacity = capacity_;
  }

  ~DeviceHashMap() {
    if (keys_) cudaFree(keys_);
    if (values_) cudaFree(values_);
    if (occupied_) cudaFree(occupied_);
  }

  DeviceHashMap(const DeviceHashMap&) = delete;
  DeviceHashMap& operator=(const DeviceHashMap&) = delete;

  DeviceHashMap(DeviceHashMap&& other) noexcept {
    keys_ = other.keys_;
    values_ = other.values_;
    occupied_ = other.occupied_;
    capacity_ = other.capacity_;
    view_ = other.view_;
    other.keys_ = nullptr;
    other.values_ = nullptr;
    other.occupied_ = nullptr;
    other.capacity_ = 0;
    other.view_ = HashMap<Key, Value, Hash, KeyEqual>{};
  }

  DeviceHashMap& operator=(DeviceHashMap&& other) noexcept {
    if (this != &other) {
      if (keys_) cudaFree(keys_);
      if (values_) cudaFree(values_);
      if (occupied_) cudaFree(occupied_);
      keys_ = other.keys_;
      values_ = other.values_;
      occupied_ = other.occupied_;
      capacity_ = other.capacity_;
      view_ = other.view_;
      other.keys_ = nullptr;
      other.values_ = nullptr;
      other.occupied_ = nullptr;
      other.capacity_ = 0;
      other.view_ = HashMap<Key, Value, Hash, KeyEqual>{};
    }
    return *this;
  }

  const HashMap<Key, Value, Hash, KeyEqual>& View() const { return view_; }
  HashMap<Key, Value, Hash, KeyEqual>& View() { return view_; }

 private:
  HashMap<Key, Value, Hash, KeyEqual> view_;
  Key* keys_ = nullptr;
  Value* values_ = nullptr;
  uint8_t* occupied_ = nullptr;
  uint32_t capacity_ = 0;
};

}  // namespace data_structures
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_DATA_STRUCTURES_HASH_MAP_CUH_
