#ifndef CORE_GPU_KERNEL_DATA_STRUCTURES_KERNEL_BITMAP_CUH_
#define CORE_GPU_KERNEL_DATA_STRUCTURES_KERNEL_BITMAP_CUH_

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

#endif  // CORE_GPU_KERNEL_DATA_STRUCTURES_KERNEL_BITMAP_CUH_