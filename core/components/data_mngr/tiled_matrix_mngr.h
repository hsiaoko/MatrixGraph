#ifndef MATRIXGRAPH_CORE_COMPONENTS_DATA_MNGR_TILED_MATRIX_MNGR_H_
#define MATRIXGRAPH_CORE_COMPONENTS_DATA_MNGR_TILED_MATRIX_MNGR_H_

#include <string>

#include "core/components/data_mngr/data_mngr_base.h"
#include "core/data_structures/tiled_matrix.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace components {

class TiledMatrixMngr : public DataMngrBase {
private:
  using TiledMatrix = sics::matrixgraph::core::data_structures::TiledMatrix;

public:
  TiledMatrixMngr(const std::string &root_path) : root_path_(root_path) {
    auto p_tiled_matrix = new TiledMatrix();
    p_tiled_matrix->Read(root_path_);
  }

  void GetData(size_t i, void *data) override{
      // TODO: implement
  };

private:
  std::string root_path_;
  TiledMatrix *p_tiled_matrix_;
};

} // namespace components
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // MATRIXGRAPH_CORE_COMPONENTS_DATA_MNGR_TILED_MATRIX_MNGR_H_
