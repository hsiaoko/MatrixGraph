#ifndef _MATRIXGRAPH_CORE_COMPONENTS_TASKGENERATOR_H_
#define _MATRIXGRAPH_CORE_COMPONENTS_TASKGENERATOR_H_

#include <string>

namespace sics {
namespace matrixgraph {
namespace core {
namespace components {

class Task() {
  // ...
}

class TaskGenerator {
public:
  TaskGenerator(const std::string &root_path) { root_path_ = root_path; }

  friend class Task;

private:
  std::string root_path_;
};

} // namespace components
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // MATRIXGRAPH_CORE_COMPONENTS_TASKGENERATOR_H_
