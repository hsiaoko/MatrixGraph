#ifndef MATRIXGRAPH_CORE_COMPONENTS_DATA_MNGR_TILED_MATRIX_MNGR_H_
#define MATRIXGRAPH_CORE_COMPONENTS_DATA_MNGR_TILED_MATRIX_MNGR_H_

#include <string>

#include "core/components/data_mngr/data_mngr_base.h"
#include "core/data_structures/tiled_matrix.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace components {

class TiledMatrixMngr : public DataMngrBase {
private:
  using TiledMatrix = sics::matrixgraph::core::data_structures::TiledMatrix;

public:
  TiledMatrixMngr(const std::string &root_path) : root_path_(root_path) {
    p_tiled_matrix_ = new TiledMatrix();
    p_tiled_matrix_transposed_ = new TiledMatrix();
    p_tiled_matrix_->Read(root_path_ + "origin/");
    p_tiled_matrix_transposed_->Read(root_path_ + "transposed/");
  }

  void GetData(void *data) override {
    // TODO: implement
  };

  TiledMatrix *GetTiledMatrixPtr() {
    return p_tiled_matrix_;
  }

  TiledMatrix *GetTransposedTiledMatrixPtr() {
    return p_tiled_matrix_transposed_;
  }

private:
  std::string root_path_;
  TiledMatrix *p_tiled_matrix_transposed_;
  TiledMatrix *p_tiled_matrix_;
};

} // namespace components
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // MATRIXGRAPH_CORE_COMPONENTS_DATA_MNGR_TILED_MATRIX_MNGR_H_
