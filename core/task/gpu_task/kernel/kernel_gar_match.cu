#include "core/task/gpu_task/kernel/kernel_gar_match.cuh"

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

int GARMatchKernelWrapper::GARMatch(const GARGraphParams& g,
                                    const GARPatternParams& p,
                                    GARMatchOutput* out) {
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
