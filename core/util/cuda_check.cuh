#ifndef GRAPH_COMPUTING_MATRIXGRAPH_CORE_UTIL_CUDA_CHECK_CUH_
#define GRAPH_COMPUTING_MATRIXGRAPH_CORE_UTIL_CUDA_CHECK_CUH_

#include <iostream>
#include <string>
#include <cuda_runtime.h>

static void HandleError(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    std::cout << "CUDA error: " << cudaGetErrorString(error) << " in " << file
              << " at line " << line << std::endl;
    exit(EXIT_FAILURE);
  }
}

static void LogInfo(size_t val, const char *file, int line) {
  std::cout << val << " in " << file << " at line " << line << std::endl;
}

static void LogInfo(const std::string &str, const char *file, int line) {
  std::cout << str << " in " << file << " at line " << line << std::endl;
}

#define CUDA_CHECK(error) HandleError(error, __FILE__, __LINE__)

#define CUDA_LOG_INFO(info) LogInfo(info, __FILE__, __LINE__)

#endif // MATRIXGRAPH_CORE_UTIL_CUDA_CHECK_CUH_
