#ifndef GRAPH_COMPUTING_MATRIXGRAPH_CORE_UTIL_BITMAP_NO_OWNERSHIP_H_
#define GRAPH_COMPUTING_MATRIXGRAPH_CORE_UTIL_BITMAP_NO_OWNERSHIP_H_

#include "core/util/bitmap.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace util {

class BitmapNoOwnerShip : public Bitmap {
 public:
  BitmapNoOwnerShip() = default;

  BitmapNoOwnerShip(size_t size) { size_ = size; }

  BitmapNoOwnerShip(size_t size, uint64_t* data) { Init(size, data); }

  ~BitmapNoOwnerShip() {
    std::cout << "~BitmapNoOwnerShip" << std::endl;

    data_ = nullptr;
    size_ = 0;
  };

  void Init(size_t size, uint64_t* data) override {
    size_ = size;
    data_ = data;
  }
};

}  // namespace util
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // GRAPH_COMPUTING_MATRIXGRAPH_CORE_UTIL_BITMAP_NO_OWNERSHIP_H_