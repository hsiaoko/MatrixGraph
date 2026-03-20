#include "core/common/consts.h"
#include "core/common/host_algorithms.cuh"
#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/heap.cuh"
#include "core/data_structures/host_buffer.cuh"
#include "core/data_structures/kernel_bitmap.cuh"
#include "core/data_structures/kernel_bitmap_no_ownership.cuh"
#include "core/data_structures/mini_kernel_bitmap.cuh"
#include "core/data_structures/unified_buffer.cuh"
#include "core/task/gpu_task/kernel/algorithms/hash.cuh"
#include "core/task/gpu_task/kernel/algorithms/sort.cuh"
#include "core/task/gpu_task/kernel/kernel_gar_match.cuh"
#include "core/task/gpu_task/kernel/kernel_woj_subiso.cuh"
#include "core/util/bitmap_ownership.h"
#include "core/util/execution_policy.h"
#include <cuda_runtime.h>
#include <iostream>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

GARMatchKernelWrapper* GARMatchKernelWrapper::GetInstance() {
  if (ptr_ == nullptr) {
    ptr_ = new GARMatchKernelWrapper();
  }
  return ptr_;
}

int GARMatchKernelWrapper::GARMatch(const GARGraphArrays& g,
                                    const GARPatternArrays& p,
                                    GARMatchArrays* out) {
  std::cout << "GARMatchKernelWrapper::GARMatch" << std::endl;
  dim3 dimBlock(kBlockDim);
  dim3 dimGrid(kGridDim);
  (void)g;
  (void)p;
  if (out == nullptr) {
    return 1;
  }
  if (out->num_conditions) {
    *(out->num_conditions) = 0;
  }
  if (out->row_size) {
    *(out->row_size) = 0;
  }
  if (out->match_size) {
    *(out->match_size) = 0;
  }
  return 0;
}

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
