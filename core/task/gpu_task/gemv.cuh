#ifndef MATRIXGRAPH_CORE_TASK_GEMV_CUH_
#define MATRIXGRAPH_CORE_TASK_GEMV_CUH_

#include "core/common/types.h"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/unified_buffer.cuh"
#include "core/task/gpu_task/task_base.cuh"
#include <string>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

class GEMV : public TaskBase {
 private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using GraphID = sics::matrixgraph::core::common::GraphID;

 public:
  GEMV(const std::string& input_path) : input_path_(input_path) {}

  __host__ void Run();

 private:
  const std::string input_path_;

  __host__ void LoadData();
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // USE_MATRIXGRAPH_GEMV_CUH
