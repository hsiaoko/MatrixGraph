#ifndef CORE_UTIL_BITMAP_OWNERSHIP_H_
#define CORE_UTIL_BITMAP_OWNERSHIP_H_

#include <cassert>
#include <cstdint>
#include <cstring>

#include "core/util/bitmap.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace util {

// @DESCRIPTION
//
// BitmapOwnership is a mapping from integers~(indexes) to bits. If the unit is
// occupied, the bit is a nonzero integer constant and if it is empty, the bit
// is zero.
// make sure the pointer is created by new[].
class BitmapOwnership : public Bitmap {
 public:
  BitmapOwnership() = default;

  BitmapOwnership(size_t size) { Init(size); }

  // copy constructor
  BitmapOwnership(const BitmapOwnership& other) {
    size_ = other.size();
    data_ = new uint64_t[WORD_OFFSET(size_) + 1]();
    memcpy(data_, other.GetDataBasePointer(),
           (WORD_OFFSET(size_) + 1) * sizeof(uint64_t));
  };

  // move constructor
  BitmapOwnership(BitmapOwnership&& other) noexcept {
    SetSize(other.size());
    SetDataPtr(other.data());
    other.SetSize(0);
    other.SetDataPtr(nullptr);
  }

  // copy assignment
  BitmapOwnership& operator=(const BitmapOwnership& other) {
    if (this != &other) {
      delete[] data_;
      size_ = other.size();
      data_ = new uint64_t[WORD_OFFSET(size_) + 1]();
      memcpy(data_, other.GetDataBasePointer(),
             (WORD_OFFSET(size_) + 1) * sizeof(uint64_t));
    }
    return *this;
  };

  // move assignment
  BitmapOwnership& operator=(BitmapOwnership&& other) noexcept {
    if (this != &other) {
      delete[] data_;
      size_ = other.size_;
      data_ = other.data_;
      other.size_ = 0;
      other.data_ = nullptr;
    }
    return *this;
  };

  ~BitmapOwnership() {
    delete[] data_;
    size_ = 0;
  }

  void Init(size_t size) {
    delete[] data_;
    size_ = size;
    data_ = new uint64_t[WORD_OFFSET(size) + 1]();
  }
};

}  // namespace util
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
#endif  // CORE_UTIL_BITMAP_H_
