#ifndef MATRIXGRAPH_CORE_UTIL_CUDA_CHECK_CUH_
#define MATRIXGRAPH_CORE_UTIL_CUDA_CHECK_CUH_

#include <cuda_runtime.h>

#include <iostream>
#include <string>

#include "core/util/error.h"

// Library code reports failures by throwing; only entry points translate
// them into exit codes (a library that closes the host process cannot be
// reused in-process). Callers that need a specific policy catch util::Error.
inline void HandleError(cudaError_t error, const char* file, int line) {
  if (error != cudaSuccess) {
    throw sics::matrixgraph::core::util::Error(
        sics::matrixgraph::core::util::ErrorCode::kCuda,
        std::string("CUDA error: ") + cudaGetErrorString(error) + " in " +
            file + " at line " + std::to_string(line));
  }
}

inline void LogInfo(size_t val, const char* file, int line) {
  std::cout << val << " in " << file << " at line " << line << std::endl;
}

inline void LogInfo(const std::string& str, const char* file, int line) {
  std::cout << str << " in " << file << " at line " << line << std::endl;
}

#define CUDA_CHECK(error) HandleError(error, __FILE__, __LINE__)

#define CUDA_LOG_INFO(info) LogInfo(info, __FILE__, __LINE__)

#endif  // MATRIXGRAPH_CORE_UTIL_CUDA_CHECK_CUH_
