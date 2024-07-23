#include "core/task/ppr_query.cuh"

#include <iostream>

#include "core/io/grid_tiled_matrix_io.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using sics::matrixgraph::core::data_structures::GridTiledMatrix;
using sics::matrixgraph::core::io::GridTiledMatrixIO;

__host__ void PPRQuery::LoadData() {
  GridTiledMatrixIO grid_tiled_matrix_io;

  GridTiledMatrix *grid_tiled_matrix_ptr;

  grid_tiled_matrix_io.Read(input_path_, grid_tiled_matrix_ptr);
}

__host__ void PPRQuery::Run() { std::cout << "Running PPRQuery" << std::endl; }

} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics