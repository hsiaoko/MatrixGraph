#ifndef MATRIXGRAPH_CORE_UTIL_ERROR_H_
#define MATRIXGRAPH_CORE_UTIL_ERROR_H_

#include <cstdint>
#include <stdexcept>
#include <string>

namespace sics {
namespace matrixgraph {
namespace core {
namespace util {

/**
 * Error codes are append-only: numeric values back callers' expectations
 * (and, on the preserved go-api branch, the C boundary's contract). Add new
 * values at the end; never renumber.
 */
enum class ErrorCode : int32_t {
  kOk = 0,
  kInvalidArgument = 1,
  kNotFound = 2,
  kUnsupported = 3,
  kCuda = 4,
  kIo = 5,
};

/**
 * The library's single error type. Internal code throws; only entry points
 * (app mains, a future C boundary) translate it into exit codes. Library
 * code never calls exit()/abort(): a library that kills its host process
 * cannot be reused in-process.
 */
class Error : public std::runtime_error {
 public:
  Error(ErrorCode code, const std::string& message)
      : std::runtime_error(message), code_(code) {}

  ErrorCode code() const noexcept { return code_; }

 private:
  ErrorCode code_;
};

}  // namespace util
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_UTIL_ERROR_H_