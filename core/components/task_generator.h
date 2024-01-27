#ifndef _MATRIXGRAPH_CORE_COMPONENTS_EXECUTIONPLAN_GENERATOR_H_
#define _MATRIXGRAPH_CORE_COMPONENTS_EXECUTIONPLAN_GENERATOR_H_

#include <string>

namespace sics {
namespace matrixgraph {
namespace core {
namespace components {

class ExecutionPlan() {
  // ...
}

class ExecutionGenerator {
public:
  TaskGenerator(const std::string &root_path) { root_path_ = root_path; }

  friend class ExecutionPlan;

private:
  std::string root_path_;
};

} // namespace components
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // MATRIXGRAPH_CORE_COMPONENTS_EXECUTIONPLAN_GENERATOR_H_
