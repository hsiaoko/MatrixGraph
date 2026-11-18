#ifndef MATRIXGRAPH_CORE_TASK_CPU_TASK_CPU_TASK_BASE_H_
#define MATRIXGRAPH_CORE_TASK_CPU_TASK_CPU_TASK_BASE_H_

// CPUTaskBase: the CPU task contract. One virtual entry point so callers
// need no type switch.
namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

// Base for CPU tasks. The contract is deliberately minimal: one virtual
// entry point so the facade can run any task without a type switch. Resource
// ownership for CPU work (TBB arena) moves in with the engine layer.
class CPUTaskBase {
 public:
  virtual ~CPUTaskBase() = default;

  virtual void Run() {}
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_CPU_TASK_CPU_TASK_BASE_H_
