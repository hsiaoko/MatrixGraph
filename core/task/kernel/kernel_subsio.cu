#include <cuda_runtime.h>
#include <iostream>

#include "core/common/types.h"
#include "core/task/kernel/kernel_subiso.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
using VertexID = sics::matrixgraph::core::common::VertexID;

struct ParametersSubIso {
  VertexID n_vertices_p;
  EdgeIndex n_edges_p;
  uint8_t *data_p;
  VertexLabel *v_label_p;
  size_t tile_size;
  size_t n_strips;
  size_t n_nz_tile_g;
  VertexID *n_vertices_for_each_csr_g;
  EdgeIndex *n_edges_for_each_csr_g;
  VertexID *tile_offset_row_g;
  VertexID *tile_row_idx_g;
  VertexID *tile_col_idx_g;
  uint64_t *csr_offset_g;
  uint8_t *data_g;
  VertexID **m;
  EdgeIndex **m_offset;
};

static __device__ void entend(uint32_t level) {}

static __global__ void subiso_kernel(ParametersSubIso params) {

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  // Phrase pattern
  VertexID *globalid_p = (VertexID *)params.data_p;
  VertexID *in_degree_p = globalid_p + params.n_vertices_p;
  VertexID *out_degree_p = in_degree_p + params.n_vertices_p;
  EdgeIndex *in_offset_p = (EdgeIndex *)(out_degree_p + params.n_vertices_p);
  EdgeIndex *out_offset_p =
      (EdgeIndex *)(in_offset_p + params.n_vertices_p + 1);
  EdgeIndex *in_edges_p = (EdgeIndex *)(out_offset_p + params.n_vertices_p + 1);
  VertexID *out_edges_p = in_edges_p + params.n_edges_p;
  VertexID *edges_globalid_by_localid_p = out_edges_p + params.n_edges_p;

  VertexID u = globalid_p[0];
  printf("tid: %d, u %d\n", tid, u);
}

void SubIsoKernelWrapper::SubIso(
    const cudaStream_t &stream, VertexID n_vertices_p, EdgeIndex n_edges_p,
    const data_structures::UnifiedOwnedBuffer<uint8_t> &data_p,
    const data_structures::UnifiedOwnedBuffer<VertexLabel> &v_label_p,
    size_t tile_size, size_t n_strips, size_t n_nz_tile_g,
    const data_structures::UnifiedOwnedBuffer<VertexID>
        &n_vertices_for_each_csr_g,
    const data_structures::UnifiedOwnedBuffer<EdgeIndex>
        &n_edges_for_each_csr_g,
    const data_structures::UnifiedOwnedBuffer<VertexID> &tile_offset_row_g,
    const data_structures::UnifiedOwnedBuffer<VertexID> &tile_row_idx_g,
    const data_structures::UnifiedOwnedBuffer<VertexID> &tile_col_idx_g,
    const data_structures::UnifiedOwnedBuffer<uint64_t> &csr_offset_g,
    const data_structures::UnifiedOwnedBuffer<uint8_t> &data_g,
    const std::vector<data_structures::UnifiedOwnedBuffer<VertexID>> &m,
    const std::vector<data_structures::UnifiedOwnedBuffer<EdgeIndex>>
        &m_offset) {

  dim3 dimBlock(1);
  dim3 dimGrid(1);

  std::vector<VertexID *> m_ptr;
  m_ptr.reserve(n_edges_p);

  std::vector<EdgeIndex *> m_offset_ptr;
  m_offset_ptr.reserve(n_edges_p);

  for (auto _ = 0; _ < m.size(); _++) {
    m_offset_ptr[_] = m_offset[_].GetPtr();
    m_ptr[_] = m[_].GetPtr();
  }

  ParametersSubIso params{
      .n_vertices_p = n_vertices_p,
      .n_edges_p = n_edges_p,
      .data_p = data_p.GetPtr(),
      .v_label_p = v_label_p.GetPtr(),
      .tile_size = tile_size,
      .n_strips = n_strips,
      .n_nz_tile_g = n_nz_tile_g,
      .n_vertices_for_each_csr_g = n_vertices_for_each_csr_g.GetPtr(),
      .n_edges_for_each_csr_g = n_edges_for_each_csr_g.GetPtr(),
      .tile_offset_row_g = tile_offset_row_g.GetPtr(),
      .tile_row_idx_g = tile_row_idx_g.GetPtr(),
      .tile_col_idx_g = tile_col_idx_g.GetPtr(),
      .csr_offset_g = csr_offset_g.GetPtr(),
      .data_g = data_g.GetPtr(),
      .m = m_ptr.data(),
      .m_offset = m_offset_ptr.data()};

  subiso_kernel<<<dimGrid, dimBlock, 0, stream>>>(params);
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