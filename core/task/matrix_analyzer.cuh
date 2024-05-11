#ifndef MATRIXGRAPH_CORE_COMPONENTS_MATRIX_ANALYZER_CUH_
#define MATRIXGRAPH_CORE_COMPONENTS_MATRIX_ANALYZER_CUH_

#include <cuda_runtime.h>
#include <string>
#include <unistd.h>

#include <climits>
#include <condition_variable>
#include <mutex>

#include "core/common/types.h"
#include "core/components/scheduler/CHBL_scheduler.h"
#include "core/components/scheduler/even_split_scheduler.h"
#include "core/components/scheduler/round_robin_scheduler.h"
#include "core/components/task"
#include "core/gpu/global_func.cuh"
#include "core/gpu/host_func.cuh"
#include "core/gpu/kernel_data_structures/kernel_bitmap.cuh"
#include "core/gpu/kernel_data_structures/kernel_table.cuh"
#include "core/util/set_operations.h"

#include "core/task/task_base.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

class MatrixAnalyzer : public TaskBase {
private:
  using TiledMatrix = sics::matrixgraph::core::data_structures::TiledMatrix;
  using Tile = sics::matrixgraph::core::data_structures::Tile;
  using Mask = sics::matrixgraph::core::data_structures::Mask;
  using VertexID = sics::matrixgraph::core::common::VertexID;

public:
private:
  const std::string data_path_;
  const std::string output_path_;

  std::vector<TiledMatrix *> vec_tiled_matrix_ptr_;
};

} // namespace components
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // MATRIXGRAPH_CORE_COMPONENTS_MATRIX_ANALYZER_CUH_
