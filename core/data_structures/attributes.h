#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_ATTRIBUTES_H_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_ATTRIBUTES_H_

#include <cuda_runtime.h>
#include <cstdint>

#include "core/data_structures/hash_map.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

// Maximum length of an attribute name (excluding null terminator).
constexpr uint32_t kMaxAttributeNameLength = 63;

// ValueType mirrors the Go enum used by igeenum.
enum class ValueType : uint8_t {
  kInvalid = 0,
  kFloat64 = 1,
  kFloatList = 2,
  kInt = 3,
  kIntList = 4,
  kString = 5,
  kStringList = 6,
  kTime = 7,
  kTimeList = 8,
  kBool = 9,
  kFloat32 = 10,
};

__host__ __device__ inline const char* ValueTypeToString(ValueType type) {
  switch (type) {
    case ValueType::kInvalid:    return "Invalid";
    case ValueType::kFloat64:    return "Float64";
    case ValueType::kFloatList:  return "FloatList";
    case ValueType::kInt:        return "Int";
    case ValueType::kIntList:    return "IntList";
    case ValueType::kString:     return "String";
    case ValueType::kStringList: return "StringList";
    case ValueType::kTime:       return "Time";
    case ValueType::kTimeList:   return "TimeList";
    case ValueType::kBool:       return "Bool";
    case ValueType::kFloat32:    return "Float32";
  }
  return "Unknown";
}

// A lightweight view of a string that only stores a pointer and a length.
struct StringView {
  const char* data;
  uint32_t len;
};

// Column-oriented attribute storage: one name/type plus a value per entity.
struct Attribute {
  char name[kMaxAttributeNameLength + 1];  // null-terminated
  ValueType type = ValueType::kInvalid;
  uint32_t n_rows = 0;       // number of entities (rows)
  uint32_t n_elements = 0;   // total number of primitive elements in data
  const void* data = nullptr;      // pointer to the flat value buffer
  const uint32_t* offsets = nullptr;  // nullptr for scalars; length n_rows + 1 for lists
  const uint8_t* valid = nullptr;  // nullptr = all valid; else n_rows bytes (1 = valid)
};

// Whether row `row` of `attr` holds a valid (present) value. A null valid mask
// means every row is valid.
__host__ __device__ inline bool IsValidAt(const Attribute& attr, uint32_t row) {
  return attr.valid == nullptr || attr.valid[row] != 0;
}

// ---------------------------------------------------------------------------
// Scalar accessors
// ---------------------------------------------------------------------------
__host__ __device__ inline double GetFloat64(const Attribute& attr,
                                             uint32_t row) {
  return static_cast<const double*>(attr.data)[row];
}

__host__ __device__ inline float GetFloat32(const Attribute& attr,
                                            uint32_t row) {
  return static_cast<const float*>(attr.data)[row];
}

__host__ __device__ inline int64_t GetInt(const Attribute& attr, uint32_t row) {
  return static_cast<const int64_t*>(attr.data)[row];
}

__host__ __device__ inline int64_t GetTime(const Attribute& attr,
                                           uint32_t row) {
  return static_cast<const int64_t*>(attr.data)[row];
}

__host__ __device__ inline bool GetBool(const Attribute& attr, uint32_t row) {
  return static_cast<const uint8_t*>(attr.data)[row] != 0;
}

__host__ __device__ inline StringView GetString(const Attribute& attr,
                                                uint32_t row) {
  return static_cast<const StringView*>(attr.data)[row];
}

// ---------------------------------------------------------------------------
// List accessors
// ---------------------------------------------------------------------------
__host__ __device__ inline uint32_t GetListSize(const Attribute& attr,
                                                uint32_t row) {
  return attr.offsets[row + 1] - attr.offsets[row];
}

__host__ __device__ inline const double* GetFloatList(const Attribute& attr,
                                                      uint32_t row,
                                                      uint32_t* size) {
  const uint32_t start = attr.offsets[row];
  *size = attr.offsets[row + 1] - start;
  return static_cast<const double*>(attr.data) + start;
}

__host__ __device__ inline const int64_t* GetIntList(const Attribute& attr,
                                                     uint32_t row,
                                                     uint32_t* size) {
  const uint32_t start = attr.offsets[row];
  *size = attr.offsets[row + 1] - start;
  return static_cast<const int64_t*>(attr.data) + start;
}

__host__ __device__ inline const int64_t* GetTimeList(const Attribute& attr,
                                                      uint32_t row,
                                                      uint32_t* size) {
  const uint32_t start = attr.offsets[row];
  *size = attr.offsets[row + 1] - start;
  return static_cast<const int64_t*>(attr.data) + start;
}

__host__ __device__ inline const StringView* GetStringList(const Attribute& attr,
                                                           uint32_t row,
                                                           uint32_t* size) {
  const uint32_t start = attr.offsets[row];
  *size = attr.offsets[row + 1] - start;
  return static_cast<const StringView*>(attr.data) + start;
}

// ---------------------------------------------------------------------------
// AttributeName  (fixed-length string key for hash maps)
// ---------------------------------------------------------------------------
struct AttributeName {
  char data[kMaxAttributeNameLength + 1];

  __host__ __device__ AttributeName() { data[0] = '\0'; }

  __host__ __device__ AttributeName(const char* src) {
    uint32_t i = 0;
    for (; i < kMaxAttributeNameLength && src[i] != '\0'; ++i) {
      data[i] = src[i];
    }
    data[i] = '\0';
  }

  __host__ __device__ bool operator==(const AttributeName& other) const {
    for (uint32_t i = 0; i <= kMaxAttributeNameLength; ++i) {
      if (data[i] != other.data[i]) return false;
      if (data[i] == '\0') return true;
    }
    return true;
  }
};

// Hash / Equal specializations for AttributeName
template <>
struct DefaultHash<AttributeName> {
  __host__ __device__ uint32_t operator()(const AttributeName& key) const {
    uint32_t h = 0x811c9dc5u;
    for (uint32_t i = 0; i < kMaxAttributeNameLength; ++i) {
      h ^= static_cast<uint32_t>(key.data[i]);
      h *= 0x01000193u;
      if (key.data[i] == '\0') break;
    }
    return h;
  }
};

template <>
struct DefaultEqual<AttributeName> {
  __host__ __device__ bool operator()(const AttributeName& a,
                                      const AttributeName& b) const {
    return a == b;
  }
};

// ---------------------------------------------------------------------------
// Attributes  (per-label / per-graph attribute table)
//
// A lightweight view that can be passed by value to a kernel.  All threads
// typically share the same Attributes object and access their own row via
// their vertex / thread id inside the Attribute columns.
// ---------------------------------------------------------------------------
struct Attributes {
  uint32_t vertex_id = 0;  // label id, graph id, or any entity grouping id
  HashMap<AttributeName, Attribute> attr_map;
};

// ---------------------------------------------------------------------------
// DeviceAttributes  (builds an Attributes table in device memory)
//
// The caller is responsible for keeping the Attribute::data / Attribute::offsets
// buffers alive on the device; DeviceAttributes only manages the hash-map
// buckets (keys / values / occupied) that map attribute names to Attribute
// descriptors.
// ---------------------------------------------------------------------------
class DeviceAttributes {
 public:
  DeviceAttributes() = default;

  // Build from host-side attribute name / Attribute descriptor pairs.
  DeviceAttributes(uint32_t vertex_id,
                   const AttributeName* names,
                   const Attribute* attrs,
                   uint32_t n,
                   float load_factor = 0.7f) {
    if (n == 0) {
      view_.vertex_id = vertex_id;
      return;
    }

    uint32_t min_cap = static_cast<uint32_t>(static_cast<float>(n) / load_factor) + 1;
    capacity_ = 1;
    while (capacity_ < min_cap) capacity_ <<= 1;

    // Build on host first, then copy H2D.
    AttributeName* h_keys = new AttributeName[capacity_];
    Attribute* h_values = new Attribute[capacity_];
    uint8_t* h_occupied = new uint8_t[capacity_]();

    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t h = DefaultHash<AttributeName>{}(names[i]);
      uint32_t idx = h & (capacity_ - 1);
      for (uint32_t probe = 0; probe < capacity_; ++probe) {
        const uint32_t slot = (idx + probe) & (capacity_ - 1);
        if (!h_occupied[slot]) {
          h_keys[slot] = names[i];
          h_values[slot] = attrs[i];
          h_occupied[slot] = 1;
          break;
        }
      }
    }

    CUDA_CHECK(cudaMalloc(&d_keys_, sizeof(AttributeName) * capacity_));
    CUDA_CHECK(cudaMalloc(&d_values_, sizeof(Attribute) * capacity_));
    CUDA_CHECK(cudaMalloc(&d_occupied_, sizeof(uint8_t) * capacity_));

    CUDA_CHECK(cudaMemcpy(d_keys_, h_keys, sizeof(AttributeName) * capacity_,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values_, h_values, sizeof(Attribute) * capacity_,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_occupied_, h_occupied, sizeof(uint8_t) * capacity_,
                          cudaMemcpyHostToDevice));

    delete[] h_keys;
    delete[] h_values;
    delete[] h_occupied;

    view_.vertex_id = vertex_id;
    view_.attr_map.keys = d_keys_;
    view_.attr_map.values = d_values_;
    view_.attr_map.occupied = d_occupied_;
    view_.attr_map.size = n;
    view_.attr_map.capacity = capacity_;
  }

  ~DeviceAttributes() {
    if (d_keys_) cudaFree(d_keys_);
    if (d_values_) cudaFree(d_values_);
    if (d_occupied_) cudaFree(d_occupied_);
  }

  DeviceAttributes(const DeviceAttributes&) = delete;
  DeviceAttributes& operator=(const DeviceAttributes&) = delete;

  DeviceAttributes(DeviceAttributes&& other) noexcept {
    d_keys_ = other.d_keys_;
    d_values_ = other.d_values_;
    d_occupied_ = other.d_occupied_;
    capacity_ = other.capacity_;
    view_ = other.view_;
    other.d_keys_ = nullptr;
    other.d_values_ = nullptr;
    other.d_occupied_ = nullptr;
    other.capacity_ = 0;
    other.view_ = Attributes{};
  }

  DeviceAttributes& operator=(DeviceAttributes&& other) noexcept {
    if (this != &other) {
      if (d_keys_) cudaFree(d_keys_);
      if (d_values_) cudaFree(d_values_);
      if (d_occupied_) cudaFree(d_occupied_);
      d_keys_ = other.d_keys_;
      d_values_ = other.d_values_;
      d_occupied_ = other.d_occupied_;
      capacity_ = other.capacity_;
      view_ = other.view_;
      other.d_keys_ = nullptr;
      other.d_values_ = nullptr;
      other.d_occupied_ = nullptr;
      other.capacity_ = 0;
      other.view_ = Attributes{};
    }
    return *this;
  }

  const Attributes& View() const { return view_; }
  Attributes& View() { return view_; }

 private:
  Attributes view_;
  AttributeName* d_keys_ = nullptr;
  Attribute* d_values_ = nullptr;
  uint8_t* d_occupied_ = nullptr;
  uint32_t capacity_ = 0;
};

}  // namespace data_structures
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_DATA_STRUCTURES_ATTRIBUTES_H_
