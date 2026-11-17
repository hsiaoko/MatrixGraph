#ifndef MATRIXGRAPH_CORE_UTIL_PARALLEL_H_
#define MATRIXGRAPH_CORE_UTIL_PARALLEL_H_

#include <cstddef>
#include <functional>

namespace sics {
namespace matrixgraph {
namespace core {
namespace util {

/**
 * Single host-parallel entry point for the library.
 *
 * The definition lives in parallel.cpp, which includes oneTBB. This header
 * deliberately includes neither TBB nor <execution>: nvcc translation units
 * cannot parse either (EDG fails on GCC's avx512bf16/amxtileintrin headers;
 * see .audit/engine-grounding.md), so host code inside .cu files reaches the
 * TBB implementation across the link boundary instead of the include
 * boundary. Pure CXX translation units use the same symbol.
 *
 * body(lo, hi) processes the half-open index range [lo, hi). Callers must
 * make disjoint-range writes only; shared state needs its own
 * synchronization, exactly as with any parallel algorithm.
 */
void ParallelForRange(size_t begin, size_t end,
                      const std::function<void(size_t, size_t)>& body,
                      bool parallel = true);

/** Runs f over [first, last); requires random-access iterators. */
template <class It, class F>
void ParallelFor(It first, It last, F f, bool parallel = true) {
  const size_t n = static_cast<size_t>(last - first);
  ParallelForRange(
      0, n,
      [&f, first](size_t lo, size_t hi) {
        for (size_t i = lo; i < hi; ++i) {
          f(*(first + static_cast<std::ptrdiff_t>(i)));
        }
      },
      parallel);
}

}  // namespace util
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_UTIL_PARALLEL_H_