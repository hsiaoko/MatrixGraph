#include <fstream>
#include <iostream>

#include "core/data_structures/matrix.cuh"
#include "yaml-cpp/yaml.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

void Matrix::Read(const std::string& root_path) {
  std::string meta_path = root_path + "meta.yaml";
  std::string data_path = root_path + "embedding.bin";
  try {
    YAML::Node config = YAML::LoadFile(meta_path);

    x_ = config["x"].as<uint32_t>();
    y_ = config["y"].as<uint32_t>();

    data_ = new float[x_ * y_]();

  } catch (const YAML::Exception& e) {
    std::cerr << "YAML Error: " << e.what() << std::endl;
  }

  std::ifstream data_file(data_path, std::ios::binary);
  if (!data_file) throw std::runtime_error("Error reading file: " + data_path);
  data_file.seekg(0, std::ios::end);
  size_t file_size = data_file.tellg();
  data_file.seekg(0, std::ios::beg);
  data_file.read(reinterpret_cast<char*>(data_), file_size);
  data_file.close();
}

void Matrix::Print(uint32_t k) const {
  printf("Print Matrix(%d, %d) ...\n", x_, y_);
  k = k < x_ ? k : x_;

  for (uint32_t _ = 0; _ < k; _++) {
    for (uint32_t __ = 0; __ < y_; __++) {
      std::cout << data_[_ * y_ + __] << " ";
    }
    std::cout << std::endl;
  }
}

}  // namespace data_structures
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
