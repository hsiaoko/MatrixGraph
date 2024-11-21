#include "core/common/types.h"
#include "core/task/kernel/kernel_subiso.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

struct ParametersSubIso {
  const unsigned long tile_size;
  const unsigned long n_strips;
  const unsigned long n_nz_tile_a;
  const unsigned long n_nz_tile_b;
  const unsigned long tile_unit;
  const unsigned long tile_buffer_size;
  unsigned *tile_offset_row_a;
  unsigned *tile_offset_row_b;
  unsigned *tile_row_idx_a;
  unsigned *tile_row_idx_b;
  unsigned *tile_col_idx_a;
  unsigned *tile_col_idx_b;
  uint8_t *data_a;
  uint8_t *data_b;
  uint32_t *csr_n_vertices_a = nullptr;
  uint32_t *csr_n_vertices_b = nullptr;
  uint64_t *csr_n_edges_a = nullptr;
  uint64_t *csr_n_edges_b = nullptr;
  uint64_t *csr_offset_a = nullptr;
  uint64_t *csr_offset_b = nullptr;
  uint32_t *edgelist_c = nullptr;
  uint32_t *output_offset = nullptr;
};

void SubIsoKernelWrapper::SubIso(
    const cudaStream_t &stream, size_t tile_size, size_t n_strips,
    size_t n_nz_tile_g,
    const data_structures::UnifiedOwnedBuffer<uint8_t> &data_p,
    const data_structures::UnifiedOwnedBuffer<VertexLabel> &v_label_p,
    const data_structures::UnifiedOwnedBuffer<VertexID> &csr_n_vertices_g,
    const data_structures::UnifiedOwnedBuffer<VertexID> &csr_n_edges_g,
    const data_structures::UnifiedOwnedBuffer<VertexID> &tile_offset_row_g,
    const data_structures::UnifiedOwnedBuffer<VertexID> &tile_row_idx_g,
    const data_structures::UnifiedOwnedBuffer<VertexID> &tile_col_idx_g,
    const data_structures::UnifiedOwnedBuffer<uint64_t> &csr_offset_g,
    const data_structures::UnifiedOwnedBuffer<uint8_t> &data_g,
    const std::vector<data_structures::UnifiedOwnedBuffer<VertexID>> &m,
    const std::vector<data_structures::UnifiedOwnedBuffer<EdgeIndex>>
        &m_offset) {

  dim3 dimBlock(256);
  dim3 dimGrid(256);

  // count_kernel<<<dimGrid, dimBlock, 0, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    CUDA_CHECK(err);
  }
}

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics