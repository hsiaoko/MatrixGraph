#include "core/util/parallel.h"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

namespace sics {
namespace matrixgraph {
namespace core {
namespace util {

void ParallelForRange(size_t begin, size_t end,
                      const std::function<void(size_t, size_t)>& body,
                      bool parallel) {
  if (begin >= end) return;
  // Serial for tiny ranges: thread wake-up dominates and call sites use
  // worker-sized vectors (one element per thread) as their common shape.
  if (!parallel || end - begin < 2) {
    body(begin, end);
    return;
  }
  tbb::parallel_for(tbb::blocked_range<size_t>(begin, end, 1),
                    [&body](const tbb::blocked_range<size_t>& r) {
                      body(r.begin(), r.end());
                    });
}

}  // namespace util
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics