#include "core/data_structures/csr_tiled_matrix.cuh"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <execution>
#include <mutex>
#include <numeric>
#include <thread>

#include "core/common/consts.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/util/bitmap.h"
#include "core/util/bitmap_no_ownership.h"
#include "core/util/cuda_check.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;
using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;

void CSRTiledMatrix::Init(const TiledMatrixMetadata &metadata,
                          GPUBitmap *nz_tile_bm) {

  metadata_.n_strips = metadata.n_strips;
  metadata_.n_nz_tile = metadata.n_nz_tile;
  metadata_.tile_size = metadata.tile_size;

  if (tile_row_idx_ != nullptr)
    delete[] tile_row_idx_;
  if (tile_col_idx_ != nullptr)
    delete[] tile_col_idx_;
  if (tile_offset_row_ != nullptr)
    delete[] tile_offset_row_;
  if (nz_tile_bm_ != nullptr)
    delete nz_tile_bm_;
  if (csr_offset_ != nullptr)
    delete[] csr_offset_;

  tile_row_idx_ = new VertexID[metadata_.n_nz_tile]();
  tile_col_idx_ = new VertexID[metadata_.n_nz_tile]();
  tile_offset_row_ = new VertexID[metadata_.n_strips + 1]();
  csr_offset_ = new uint64_t[metadata_.n_nz_tile + 1]();

  metadata_vec_.resize(metadata_.n_nz_tile);
  nz_tile_bm_ = nz_tile_bm;

  auto default_n_edges =
      sics::matrixgraph::core::common::KDefalutNumEdgesPerTile;
  auto default_n_vertices =
      sics::matrixgraph::core::common::KDefalutNumVerticesPerTile;
  auto tile_buffer_size = sizeof(VertexID) * (default_n_vertices + 1) +
                          sizeof(VertexID) * (default_n_vertices + 1) +
                          sizeof(VertexID) * (default_n_vertices) +
                          sizeof(VertexID) * (default_n_vertices) +
                          sizeof(VertexID) * (default_n_vertices) +
                          sizeof(VertexID) * (default_n_vertices) +
                          sizeof(EdgeIndex) * (default_n_edges)*2;

  CUDA_CHECK(cudaHostAlloc((void **)&data_,
                           tile_buffer_size * metadata_.n_nz_tile,
                           cudaHostAllocDefault));

  if (nz_tile_bm == nullptr) {
    uint64_t *bm_data;
    size_t bm_size = pow(metadata_.n_strips, 2);
    CUDA_CHECK(cudaHostAlloc((void **)&bm_data,
                             sizeof(uint64_t) * (WORD_OFFSET(bm_size) + 1),
                             cudaHostAllocDefault));
    nz_tile_bm_ = new GPUBitmap(bm_size, bm_data);
  } else {
    nz_tile_bm_ = nz_tile_bm;
  }
}

void CSRTiledMatrix::Init(VertexID tile_size, VertexID n_strips, Bitmap *bm) {}

void CSRTiledMatrix::Print() const {
  if (metadata_.n_nz_tile == 0)
    return;
  std::cout << "[CSRTiledMatrix Print], n_nz_tile: " << metadata_.n_nz_tile
            << std::endl;

  std::cout << " * tile_offset_row: ";

  for (VertexID _ = 0; _ < metadata_.n_strips + 1; _++) {
    std::cout << " " << tile_offset_row_[_];
  }
  std::cout << std::endl;

  for (VertexID _ = 0; _ < metadata_.n_nz_tile; _++) {
    std::cout << "offset: " << _ << ", x: " << tile_row_idx_[_]
              << " y: " << tile_col_idx_[_] << std::endl;
  }

  for (VertexID _ = 0; _ < metadata_.n_strips; _++) {
    for (VertexID __ = 0; __ < metadata_.n_strips; __++) {
      std::cout << GetNzTileBitmapPtr()->GetBit(_ * metadata_.n_strips + __)
                << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  for (VertexID _ = 0; _ < metadata_.n_nz_tile; _++) {
    auto *csr_base_ptr = GetCSRBasePtrByIdx(_);
    ImmutableCSR csr(GetCSRMetadataByIdx(_));
    csr.ParseBasePtr(csr_base_ptr);
    csr.PrintGraph(3);
  }
  std::cout << std::endl;
}

uint8_t *CSRTiledMatrix::GetCSRBasePtrByIdx(uint32_t idx) const {
  assert(idx < metadata_.n_nz_tile);
  return data_ + csr_offset_[idx];
}

void CSRTiledMatrix::InitData(uint64_t size) {
  CUDA_CHECK(cudaHostAlloc((void **)&data_, size, cudaHostAllocDefault));
}

void CSRTiledMatrix::SetBit(VertexID row, VertexID col) {
  nz_tile_bm_->SetBit(row * metadata_.n_strips + col);
}

uint64_t CSRTiledMatrix::GetDataBufferSize() const {
  return csr_offset_[metadata_.n_nz_tile];
}

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics