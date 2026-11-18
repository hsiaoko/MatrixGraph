#ifndef MATRIXGRAPH_CORE_UTIL_BITMAP_NO_OWNERSHIP_H_
#define MATRIXGRAPH_CORE_UTIL_BITMAP_NO_OWNERSHIP_H_

// BitmapNoOwnership: a Bitmap view over caller-owned storage; never frees it.
#include "core/util/bitmap.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace util {

class BitmapNoOwnership : public Bitmap {
 public:
  BitmapNoOwnership() = default;

  BitmapNoOwnership(size_t size) { size_ = size; }

  BitmapNoOwnership(size_t size, uint64_t* data) { Init(size, data); }

  ~BitmapNoOwnership() {
    data_ = nullptr;
    size_ = 0;
  };

  void Init(size_t size, uint64_t* data) override {
    this->data_ = data;
    this->size_ = size;
  }
};

}  // namespace util
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_UTIL_BITMAP_NO_OWNERSHIP_H_