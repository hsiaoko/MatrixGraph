#ifndef MATRIXGRAPH_CORE_UTIL_EXECUTION_POLICY_H_
#define MATRIXGRAPH_CORE_UTIL_EXECUTION_POLICY_H_

#include <algorithm>

/**
 * 执行策略：nvcc 与 GCC 12+ 的 avx512bf16/amxtileintrin 不兼容，
 * <execution> 会间接包含这些头文件导致编译失败。
 * 因此对仅由 CUDA 宿主编译(.cu/.cuh→nvcc，__CUDACC__) 的单元的 ParForEach 只能用串行
 * std::for_each（与历史行为一致）。
 * 纯 CXX 宿主翻译单元（例如 tools/graph_converter/graph_converter.cpp）
 * __CUDACC__ 未定义，可使用 std::execution::par，format_converter 等会并行。
 */
#ifdef __CUDACC__
#define MATRIXGRAPH_EXEC_POLICY /* 串行：无 parallel 执行策略 */
template <typename Iter, typename F>
inline void ParForEach(Iter begin, Iter end, F f) {
  std::for_each(begin, end, f);
}
#else
#include <execution>
#define MATRIXGRAPH_EXEC_POLICY std::execution::par,
template <typename Iter, typename F>
inline void ParForEach(Iter begin, Iter end, F&& f) {
  std::for_each(std::execution::par, begin, end, std::forward<F>(f));
}
#endif

#endif  // MATRIXGRAPH_CORE_UTIL_EXECUTION_POLICY_H_
