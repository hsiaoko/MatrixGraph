#ifndef MATRIXGRAPH_CORE_UTIL_GPU_BITMAP_CUH_
#define MATRIXGRAPH_CORE_UTIL_GPU_BITMAP_CUH_

#include "core/util/bitmap.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace util {

// @DESCRIPTION
//
// Bitmap is a mapping from integers~(indexes) to bits. If the unit is
// occupied, the bit is a nonzero integer constant and if it is empty, the bit
// is zero.
__host__ class GPUBitmap : public Bitmap {
 public:
  GPUBitmap(size_t size);

  GPUBitmap(size_t size, uint64_t* data);

  ~GPUBitmap();

  void Init(size_t size);

  void Init(size_t size, uint64_t* data);

  size_t GPUCount() const;

  size_t GPUPreElementCount(size_t idx) const;
};

}  // namespace util
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_UTIL_GPU_BITMAP_CUH_