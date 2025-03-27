#ifndef MATRIXGRAPH_CORE_COMPONENTS_TASK_CPU_TASK_BASE_H_
#define MATRIXGRAPH_CORE_COMPONENTS_TASK_CPU_TASK_BASE_H_

#include <future>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

class CPUTaskBase {
 public:
  ~CPUTaskBase() = default;
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_COMPONENTS_TASK_CPU_TASK_BASE_H_
