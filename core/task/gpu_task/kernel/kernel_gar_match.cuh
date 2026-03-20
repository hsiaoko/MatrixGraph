#ifndef MATRIXGRAPH_CORE_TASK_KERNEL_KERNEL_GAR_MATCH_CUH_
#define MATRIXGRAPH_CORE_TASK_KERNEL_KERNEL_GAR_MATCH_CUH_

#include <cstdint>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

// Serialized graph arrays (input g).
struct GARGraphParams {
  const uint32_t* v_id = nullptr;
  const int32_t* v_label_idx = nullptr;
  int n_vertices = 0;

  const uint32_t* e_src = nullptr;
  const uint32_t* e_dst = nullptr;
  const uint32_t* e_id = nullptr;
  const int32_t* e_label_idx = nullptr;
  int n_edges = 0;
};

// Serialized pattern arrays (input p).
struct GARPatternParams {
  const int32_t* node_label_idx = nullptr;
  int n_nodes = 0;

  const int32_t* edge_src = nullptr;
  const int32_t* edge_dst = nullptr;
  const int32_t* edge_label_idx = nullptr;
  int n_edges = 0;
};

// Flattened match arrays (output).
struct GARMatchOutput {
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

class GARMatchKernelWrapper {
 public:
  GARMatchKernelWrapper(const GARMatchKernelWrapper& obj) = delete;
  void operator=(const GARMatchKernelWrapper&) = delete;

  static GARMatchKernelWrapper* GetInstance();

  // Placeholder: returns 0 and writes empty output.
  static int GARMatch(const GARGraphParams& g, const GARPatternParams& p,
                      GARMatchOutput* out);

 private:
  GARMatchKernelWrapper() = default;
  inline static GARMatchKernelWrapper* ptr_ = nullptr;
};

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_KERNEL_KERNEL_GAR_MATCH_CUH_
