#ifndef MATRIXGRAPH_CORE_UTIL_CPU_PARALLEL_SCOPE_H_
#define MATRIXGRAPH_CORE_UTIL_CPU_PARALLEL_SCOPE_H_

#include <climits>
#include <cstddef>
#include <memory>

#include <oneapi/tbb/global_control.h>

namespace sics {
namespace matrixgraph {
namespace core {
namespace util {

// RAII: caps worker parallelism for std::execution::par when libstdc++ routes via oneTBB.
// max_threads == 0: no control (default hardware/TBB pool behavior).
class CpuParallelScope {
 public:
  explicit CpuParallelScope(std::size_t max_threads) {
    if (max_threads > 0) {
      int lim = max_threads > static_cast<std::size_t>(INT_MAX)
                    ? INT_MAX
                    : static_cast<int>(max_threads);
      gc_ = std::make_unique<oneapi::tbb::global_control>(
          oneapi::tbb::global_control::max_allowed_parallelism, lim);
    }
  }

 private:
  std::unique_ptr<oneapi::tbb::global_control> gc_;
};

}  // namespace util
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_UTIL_CPU_PARALLEL_SCOPE_H_
