#ifndef MATRIXGRAPH_CORE_UTIL_EXECUTION_POLICY_H_
#define MATRIXGRAPH_CORE_UTIL_EXECUTION_POLICY_H_

#include <algorithm>

/**
 * 执行策略：nvcc 与 GCC 12+ 的 avx512bf16/amxtileintrin 不兼容，
 * <execution> 会间接包含这些头文件导致编译失败。
 * 因此 CUDA 编译时使用单线程，C++ 编译时使用并行。
 * ParForEach：共享的并行 for_each，使用模板函数避免 lambda 捕获列表逗号问题。
 */
#ifdef __CUDACC__
#include <numeric>  /* std::iota 用于 worker 初始化 */
#include <thread>   /* std::thread::hardware_concurrency() 用于 worker 并行 */
#define MATRIXGRAPH_EXEC_POLICY /* 单线程：nvcc 不兼容 */
/* 显式模板参数强制选择非并行重载，避免 nvcc 与 execution policy 重载的解析冲突 */
template <typename Iter, typename F>
inline void ParForEach(Iter begin, Iter end, F f) {
  std::for_each<Iter, F>(begin, end, f);
}
#else
#include <execution>
#define MATRIXGRAPH_EXEC_POLICY std::execution::par,
template <typename Iter, typename F>
inline void ParForEach(Iter begin, Iter end, F f) {
  std::for_each(std::execution::par, begin, end, f);
}
#endif

#endif  // MATRIXGRAPH_CORE_UTIL_EXECUTION_POLICY_H_
