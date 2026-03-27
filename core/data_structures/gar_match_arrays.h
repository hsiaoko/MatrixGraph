#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_GAR_MATCH_ARRAYS_H_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_GAR_MATCH_ARRAYS_H_

#include <algorithm>
#include <cstdint>
#include <iostream>

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

// Flattened match arrays (output).
struct GARMatchArrays {
  int* num_conditions = nullptr;

  uint32_t* row_pivot_id = nullptr;
  int32_t* row_cond_j = nullptr;
  int32_t* row_pos = nullptr;
  int32_t* row_offset = nullptr;
  int32_t* row_count = nullptr;
  int row_capacity = 0;
  int* row_size = nullptr;

  uint32_t* matched_v_ids = nullptr;
  int match_capacity = 0;
  int* match_size = nullptr;

  void Print(int k = 5) const {
    const int nc = (num_conditions ? *num_conditions : -1);
    const int rs = (row_size ? *row_size : 0);
    const int ms = (match_size ? *match_size : 0);
    const int show_rows = std::max(0, std::min(k, rs));
    std::cout << "[GARMatchArrays] num_conditions=" << nc
              << " row_size=" << rs << "/" << row_capacity
              << " match_size=" << ms << "/" << match_capacity << std::endl;
    if (show_rows <= 0) {
      std::cout << "[GARMatchArrays] no rows to print." << std::endl;
      return;
    }
    for (int i = 0; i < show_rows; ++i) {
      const uint32_t pivot = row_pivot_id ? row_pivot_id[i] : 0;
      const int32_t cond_j = row_cond_j ? row_cond_j[i] : -1;
      const int32_t pos = row_pos ? row_pos[i] : -1;
      const int32_t offset = row_offset ? row_offset[i] : -1;
      const int32_t count = row_count ? row_count[i] : 0;
      std::cout << "  [row " << i << "] pivot=" << pivot
                << " cond_j=" << cond_j << " pos=" << pos
                << " offset=" << offset << " count=" << count;
      if (matched_v_ids && offset >= 0 && count > 0) {
        const int begin = offset;
        const int safe_end = std::max(begin, std::min(begin + count, ms));
        const int show_ids = std::min(k, std::max(0, safe_end - begin));
        std::cout << " matched=[";
        for (int j = 0; j < show_ids; ++j) {
          if (j) std::cout << ", ";
          std::cout << matched_v_ids[begin + j];
        }
        if (show_ids < (safe_end - begin)) std::cout << ", ...";
        std::cout << "]";
      }
      std::cout << std::endl;
    }
    if (show_rows < rs) {
      std::cout << "  ... (" << (rs - show_rows) << " more rows)" << std::endl;
    }
  }
};

}  // namespace data_structures
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_DATA_STRUCTURES_GAR_MATCH_ARRAYS_H_
