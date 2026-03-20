#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_GAR_MATCH_ARRAYS_H_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_GAR_MATCH_ARRAYS_H_

#include <cstdint>

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
};

}  // namespace data_structures
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_DATA_STRUCTURES_GAR_MATCH_ARRAYS_H_
