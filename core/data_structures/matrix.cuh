#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_MATRIX_CUH_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_MATRIX_CUH_

#include <string>
#include <stdint.h>


namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

class Matrix {
 public:
  Matrix() = default;

  Matrix(uint32_t x, uint32_t y) : x_(x), y_(y) { data_ = new float[x_ * y_]; }

  void Init(size_t n) {
    if (data_ != nullptr) delete[] data_;
    data_ = new float[n]();
  }
  void Read(const std::string& root_path);

  void Print(uint32_t k = 3) const;

  uint32_t get_x() const { return x_; }

  uint32_t get_y() const { return y_; }

  float* GetPtr() const { return data_; }

 private:
  uint32_t x_ = 0;
  uint32_t y_ = 0;
  float* data_ = nullptr;
};

}  // namespace data_structures
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_DATA_STRUCTURES_MATRIX_CUH_
