#include "core/data_structures/tiled_matrix.cuh"

#include <algorithm>
#include <atomic>
#include <cuda_runtime.h>
#include <filesystem>
#include <mutex>
#include <thread>
#include <unordered_map>

#ifdef TBB_FOUND
#include <execution>
#endif

#include "core/util/atomic.h"
#include "core/util/bitmap.cuh"
#include "core/util/cuda_check.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

void Mask::Show() const {
  std::cout << "     MASK     " << (int)mask_size_ << "X" << (int)mask_size_
            << std::endl;
  for (size_t i = 0; i < mask_size_; i++) {
    for (size_t j = 0; j < mask_size_; j++) {
      std::cout << GetBit(i, j) << " ";
    }
    std::cout << std::endl;
  }
}

void Tile::InitHost(TileIndex tile_size, VertexID tile_x, VertexID tile_y,
                    VertexID n_nz, TileIndex *bar_offset, TileIndex *row_idx,
                    TileIndex *col_idx, VertexLabel *data, Mask *mask) {
  tile_size_ = tile_size;
  tile_x_ = tile_x;
  tile_y_ = tile_y;
  n_nz_ = n_nz;

  if (bar_offset_ != nullptr)
    delete[] bar_offset_;
  if (mask_ != nullptr)
    delete mask_;
  if (row_idx_ != nullptr)
    delete[] row_idx_;
  if (col_idx_ != nullptr)
    delete[] col_idx_;
  if (data_ != nullptr)
    delete[] data_;

  if (bar_offset == nullptr)
    bar_offset_ = new TileIndex[tile_size_]();
  else
    bar_offset_ = bar_offset;
  if (mask == nullptr)
    mask_ = new Mask(tile_size_);
  else
    mask_ = mask;
  if (row_idx == nullptr)
    row_idx_ = new TileIndex[n_nz_]();
  else
    row_idx_ = row_idx;
  if (col_idx == nullptr)
    col_idx_ = new TileIndex[n_nz_]();
  else
    col_idx_ = col_idx;
  if (data_ == nullptr) {
    data_ = new VertexLabel[n_nz_]();
  } else
    data_ = data;

  mask_ptr_ = new Bitmap(tile_size * tile_size);
}

void Mask::InitDevice(TileIndex mask_size) {
  mask_size_ = mask_size;

  uint64_t *init_val;
  cudaMalloc((void **)&init_val,
             sizeof(uint64_t) * (WORD_OFFSET(mask_size_ * mask_size_) + 1));
  bm_ = new Bitmap(mask_size_ * mask_size_, init_val);
}

void Mask::FreeDevice() { bm_->FreeDevice(); }

void Tile::InitDevice(TileIndex tile_size, VertexID tile_x, VertexID tile_y,
                      VertexID n_nz) {
  tile_size_ = tile_size;
  tile_x_ = tile_x;
  tile_y_ = tile_y;
  n_nz_ = n_nz;

  if (bar_offset_ != nullptr)
    cudaFree(bar_offset_);
  if (mask_ != nullptr)
    cudaFree(mask_);

  if (row_idx_ != nullptr)
    cudaFree(row_idx_);
  if (col_idx_ != nullptr)
    cudaFree(col_idx_);
  if (data_ != nullptr)
    cudaFree(data_);

  cudaMalloc(reinterpret_cast<void **>(&bar_offset_),
             tile_size_ * sizeof(TileIndex));
  cudaMalloc(reinterpret_cast<void **>(&row_idx_), n_nz_ * sizeof(TileIndex));
  cudaMalloc(reinterpret_cast<void **>(&col_idx_), n_nz_ * sizeof(TileIndex));
  cudaMalloc(reinterpret_cast<void **>(&data_), n_nz_ * sizeof(VertexLabel));
  mask_ = new Mask();
  mask_->InitDevice(tile_size_);
}

void Tile::FreeDevice() {
  cudaFree(bar_offset_);
  cudaFree(row_idx_);
  cudaFree(col_idx_);
  cudaFree(data_);
  mask_->FreeDevice();
}

void Tile::MemcpyAsyncHost2Device(const Tile &tile,
                                  const cudaStream_t &stream) {
  assert(tile_x_ == tile.tile_x_);
  assert(tile_y_ == tile.tile_y_);
  assert(n_nz_ == tile.n_nz_);
  assert(tile_size_ == tile.tile_size_);

  cudaMemcpyAsync(bar_offset_, tile.bar_offset_,
                  sizeof(TileIndex) * (tile_size_), cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(row_idx_, tile.row_idx_, sizeof(TileIndex) * n_nz_,
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(col_idx_, tile.col_idx_, sizeof(TileIndex) * n_nz_,
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(data_, tile.data_, sizeof(VertexLabel) * n_nz_,
                  cudaMemcpyHostToDevice, stream);

  cudaMemcpyAsync(mask_->GetDataPtr()->GetDataBasePointer(),
                  tile.mask_->GetDataPtr()->GetDataBasePointer(),
                  sizeof(uint64_t) *
                      (WORD_OFFSET(tile.mask_->GetDataPtr()->size()) + 1),
                  cudaMemcpyHostToDevice, stream);
}

void Tile::MemcpyAsyncDevice2Host(const Tile &tile,
                                  const cudaStream_t &stream) {
  assert(tile_x_ == tile.tile_x_);
  assert(tile_y_ == tile.tile_y_);
  assert(n_nz_ == tile.n_nz_);
  assert(tile_size_ == tile.tile_size_);

  cudaMemcpyAsync(bar_offset_, tile.bar_offset_,
                  sizeof(TileIndex) * (tile_size_), cudaMemcpyDeviceToHost,
                  stream);
  cudaMemcpyAsync(row_idx_, tile.row_idx_, sizeof(TileIndex) * n_nz_,
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(col_idx_, tile.col_idx_, sizeof(TileIndex) * n_nz_,
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(data_, tile.data_, sizeof(VertexLabel) * n_nz_,
                  cudaMemcpyDeviceToHost, stream);

  cudaMemcpyAsync(mask_->GetDataPtr()->GetDataBasePointer(),
                  tile.mask_->GetDataPtr()->GetDataBasePointer(),
                  sizeof(uint64_t) *
                      (WORD_OFFSET(tile.mask_->GetDataPtr()->size()) + 1),
                  cudaMemcpyDeviceToHost, stream);
}

void TiledMatrix::Init(const ImmutableCSR &immutable_csr, size_t tile_size) {
  auto start_time = std::chrono::system_clock::now();
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  auto mask = new Mask(tile_size);

  size_t max_n_nz_tile =
      std::max(size_t(pow((immutable_csr.get_num_vertices() / tile_size), 2)),
               (size_t)1) +
      64;

  CUDA_LOG_INFO(max_n_nz_tile);
  util::Bitmap is_tile_exist(max_n_nz_tile);
  struct TileBaseInfo {
    VertexID n_nz;
    VertexID tile_offset;
  };

  auto tile_id_2_base_info = new std::unordered_map<VertexID, TileBaseInfo>;

  VertexID n_rows = ceil(immutable_csr.get_num_vertices() / (float)tile_size);

  CUDA_LOG_INFO("Step0: Get Non-empty tiles.");
  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [step, &immutable_csr, &is_tile_exist, tile_size, &mtx,
                 &tile_id_2_base_info, n_rows](auto w) {
                  for (auto vid = w; vid < immutable_csr.get_num_vertices();
                       vid += step) {
                    auto v = immutable_csr.GetVertexByLocalID(vid);
                    VertexID tile_x = (vid / (tile_size));
                    for (int nbr = 0; nbr < v.outdegree; nbr++) {
                      VertexID tile_y = ceil(v.outgoing_edges[nbr] / tile_size);
                      VertexID tile_id = XYToTileId(tile_x, tile_y, n_rows);
                      if (is_tile_exist.GetBit(tile_id)) {
                        std::lock_guard<std::mutex> lock(mtx);
                        auto iter = tile_id_2_base_info->find(tile_id);
                        sics::matrixgraph::core::util::atomic::WriteAdd(
                            &(iter->second.n_nz), (VertexID)1);
                      } else {
                        {
                          std::lock_guard<std::mutex> lock(mtx);
                          TileBaseInfo base_info = {1, 0};
                          tile_id_2_base_info->insert(
                              std::make_pair(tile_id, base_info));
                          is_tile_exist.SetBit(tile_id);
                        }
                      }
                    }
                  }
                });

  size_t n_nz_tile = is_tile_exist.Count();
  std::cout << "n_nz_tile: " << n_nz_tile << std::endl;

  Tile **data_ptr = new Tile *[n_nz_tile];
  std::vector<Tile *> data_ptr_vec;
  data_ptr_vec.reserve(n_nz_tile);
  VertexID *tile_col_idx = new VertexID[n_nz_tile]();
  VertexID *tile_row_idx = new VertexID[n_nz_tile]();
  VertexID *tile_n_nz = new VertexID[n_nz_tile]();
  // VertexID *tile_ptr = new VertexID[tile_size + 1]();

  VertexID *n_tile_per_row = new VertexID[n_rows]();

  CUDA_LOG_INFO(max_n_nz_tile);
  CUDA_LOG_INFO(n_rows);
  VertexID tile_ptr_offset = 0;
  CUDA_LOG_INFO("Step1: Construct tiles.");
  for (int tile_id = 0; tile_id < max_n_nz_tile; tile_id++) {
    if (is_tile_exist.GetBit(tile_id)) {
      if (tile_id % 100000 == 0)
        std::cout << tile_id << "/" << n_nz_tile << std::endl;

      VertexID tile_x, tile_y;
      TileIdToXY(tile_id, n_rows, &tile_x, &tile_y);

      n_tile_per_row[tile_x]++;
      tile_col_idx[tile_ptr_offset] = tile_y;
      tile_row_idx[tile_ptr_offset] = tile_x;
      auto iter = tile_id_2_base_info->find(tile_id);
      auto n_nz_element = iter->second.n_nz;
      tile_n_nz[tile_ptr_offset] = n_nz_element;
      auto p_tile = new Tile(tile_size, tile_x, tile_y, n_nz_element);
      data_ptr[tile_ptr_offset] = p_tile;
      iter->second.tile_offset = tile_ptr_offset;
      tile_ptr_offset++;
    }
  }

  CUDA_LOG_INFO("Step2: Drop edge into tile");
  // Fill edge into tile.
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [step, &immutable_csr, tile_size, &data_ptr, &tile_id_2_base_info,
       n_rows](auto w) {
        for (auto vid = w; vid < immutable_csr.get_num_vertices();
             vid += step) {
          auto v = immutable_csr.GetVertexByLocalID(vid);
          VertexID tile_x = (vid / (tile_size));
          for (int nbr = 0; nbr < v.outdegree; nbr++) {
            VertexID tile_y = ceil(v.outgoing_edges[nbr] / tile_size);
            auto tile_id = XYToTileId(tile_x, tile_y, n_rows);
            auto tile_offset =
                tile_id_2_base_info->find(tile_id)->second.tile_offset;
            auto x = vid % tile_size;
            auto y = v.outgoing_edges[nbr] % tile_size;
            auto tile = data_ptr[tile_offset];
            auto mask = tile->GetMaskPtr();
            mask->SetBit(x, y);
          }
        }
      });

  CUDA_LOG_INFO("Step3: Fill data into tile");
  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [step, n_nz_tile, tile_size, &data_ptr](auto w) {
                  for (auto i = w; i < n_nz_tile; i += step) {
                    auto tile = data_ptr[i];
                    auto offset = 0;
                    for (auto x = 0; x < tile_size; x++) {
                      tile->GetBarOffsetPtr()[x] = offset;
                      for (auto y = 0; y < tile_size; y++) {
                        if (tile->GetBit(x, y)) {
                          tile->GetRowIdxPtr()[offset] = x;
                          tile->GetColIdxPtr()[offset] = y;
                          offset++;
                        }
                      }
                    }
                    tile->GetBarOffsetPtr()[tile_size - 1] = offset;
                    // tile->Show();
                  }
                });

  auto end_time = std::chrono::system_clock::now();
  std::cout << "Elapse time "
            << std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                     start_time)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;

  CUDA_LOG_INFO("finished ###");

  VertexID *tile_id_to_tile_ptr_idx = new VertexID[max_n_nz_tile]();

  // VertexID tile_col_idx[is_tile_exist.Count()] = {0};
  // VertexID tile_row_idx[is_tile_exist.Count()] = {0};
  // VertexID tile_n_nz[is_tile_exist.Count() + 1] = {0};
  // VertexID tile_ptr[tile_size + 1] = {0};

  // Assistant data structure in
  VertexID *tile_id_to_tile_ptr_offset = new VertexID[max_n_nz_tile]();

  //  for (int tile_id = 0; tile_id < max_n_nz_tile; tile_id++) {
  //    if (n_nz_element_in_tile[tile_id] > 0) {
  //      VertexID tile_x = (tile_id / (tile_size));
  //      VertexID tile_y = tile_id % (tile_size);
  //      n_tile_per_row[tile_x]++;
  //      tile_id_to_tile_ptr_idx[tile_id] = tile_ptr_offset;
  //      data_ptr[tile_ptr_offset] =
  //          new Tile(tile_size, tile_x, tile_y,
  //          n_nz_element_in_tile[tile_id]);
  //      tile_n_nz[tile_ptr_offset] = n_nz_element_in_tile[tile_id];
  //      tile_col_idx[tile_ptr_offset] = tile_y;
  //      tile_row_idx[tile_ptr_offset] = tile_x;
  //      tile_id_to_tile_ptr_offset[tile_id] = tile_ptr_offset;
  //      tile_ptr_offset++;
  //    }
  //  }
  //  for (int i = 0; i < tile_size; i++) {
  //    tile_ptr[i + 1] = n_tile_per_row[i] + tile_ptr[i];
  //  }
  //  CUDA_LOG_INFO("###");

  // Fill edge into tile.
  // std::for_each(worker.begin(), worker.end(),
  //              [step, &immutable_csr, tile_size, &n_nz_element_in_tile,
  //               &tile_id_to_tile_ptr_offset, &data_ptr](auto w) {
  //                for (auto vid = w; vid < immutable_csr.get_num_vertices();
  //                     vid += step) {
  //                  auto v = immutable_csr.GetVertexByLocalID(vid);
  //                  VertexID tile_x = (vid / (tile_size));
  //                  for (int nbr = 0; nbr < v.outdegree; nbr++) {
  //                    VertexID tile_y = ceil(v.outgoing_edges[nbr] /
  //                    tile_size); auto tile_id = tile_x * tile_size + tile_y;
  //                    auto tile_offset = tile_id_to_tile_ptr_offset[tile_id];
  //                    auto tile = data_ptr[tile_offset];
  //                    auto x = vid % tile_size;
  //                    auto y = v.outgoing_edges[nbr] % tile_size;
  //                    auto mask = tile->GetMaskPtr();
  //                    mask->SetBit(x, y);
  //                  }
  //                }
  //              });

  // for (int i = 0; i < 3; i++) {
  //   //  for (int i = 0; i < is_tile_exist.Count(); i++) {
  //   CUDA_LOG_INFO(i);

  //  data_ptr[i]->Show();
  //}

  // std::for_each(worker.begin(), worker.end(),
  //               [step, &immutable_csr, &is_tile_exist, tile_size,
  //                &n_nz_element_in_tile](auto w) {
  //                 for (auto vid = w; vid < immutable_csr.get_num_vertices();
  //                      vid += step) {
  //                 }
  //               });

  // std::unordered_map<VertexID, Tile *> tile_map;
  // std::for_each(worker.begin(), worker.end(),
  //               [step, tile_size, &immutable_csr, &is_tile_exist, n_nz_tile,
  //                &mtx, &tile_map](auto w) {
  //                 for (auto tile_id = w; tile_id < n_nz_tile; tile_id +=
  //                 step) {
  //                   if (n_nz_element_in_tile[tile_id] > 0) {
  //                     if (auto search = tile_map.find(tile_id);
  //                         search != tile_map.end()) {
  //                       std::lock_guard<std::mutex> lock(mtx);
  //                       // tile_map.insert();
  //                       auto tile_x = (tile_id / tile_size) % tile_size;
  //                       auto tile_y = tile_id % tile_size;
  //                     }
  //                     // auto tile = new Tile(tile_size, tile_x, tile_y,
  //                     // tile_mask->GetDataPtr()->Count());
  //                     // dict.insert(std::make_pair(tile_id, new Tile));
  //                   }
  //                 }
  //               });

  while (1)
    ;
  delete[] tile_col_idx;
  delete[] tile_row_idx;
  delete[] tile_n_nz;
  // delete[] tile_ptr;
}

/*
void TiledMatrix::Init(const ImmutableCSR &immutable_csr, size_t tile_size) {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;
  std::iota(worker.begin(), worker.end(), 0);

  auto mask = new Mask(tile_size);

  // Step 1. Generate the global tile structure of the matrix: tile_ptr,
  // tile_col_idx, tile_n_nz where (1) tile_ptr used to store memory offset of
  // the tiles in tile rows; (2) tile_col_index used to store tile column
  // indices; (3) tile_n_nz used to store the number of non-zeros in each
  // tile.
  VertexID n_rows = ceil((float)immutable_csr.get_num_vertices() /
                         (float)mask->get_mask_size_x());
  VertexID n_cols = ceil((float)immutable_csr.get_num_vertices() /
                         (float)mask->get_mask_size_y());

  VertexID *tile_ptr = new VertexID[n_cols + 1]();

  std::vector<VertexID> tile_n_nz_vec;
  tile_n_nz_vec.reserve(n_cols);
  tile_n_nz_vec.push_back(0);
  std::vector<VertexID> tile_row_idx;
  tile_row_idx.reserve(n_cols);
  std::vector<VertexID> tile_col_idx;
  tile_col_idx.reserve(n_cols);

  VertexID tile_x_scope =
      std::max((VertexID)ceil((float)immutable_csr.get_num_vertices() /
                              (float)mask->get_mask_size_x()),
               (VertexID)1);
  VertexID tile_y_scope =
      std::max((VertexID)ceil((float)immutable_csr.get_num_vertices() /
                              (float)mask->get_mask_size_y()),
               (VertexID)1);

  VertexID tile_count = 0;
  std::vector<Tile *> tile_vec;
  tile_vec.reserve(n_rows * n_cols / 1024);
  auto step = worker.size();

  std::cout << "XXXXXXXXXXXXXX" << std::endl;
  std::cout << "tile_x_scope: " << tile_x_scope << std::endl;
#ifdef TBB_FOUND
  for (VertexID tile_x = 0; tile_x < tile_x_scope; tile_x++) {
    VertexID tile_row_offset = 0;

    std::cout << "x: " << tile_x << std::endl;

    std::for_each(
        std::execution::par, worker.begin(), worker.end(),
        [&mtx, step, tile_x, tile_x_scope, tile_y_scope, tile_size,
         &immutable_csr, &tile_row_offset, &tile_count, &tile_n_nz_vec,
         &tile_row_idx, &tile_col_idx, &tile_vec, &mask](auto &w) {
          for (auto tile_y = w; tile_y < tile_y_scope; tile_y += step) {
            bool is_nz_tile = false;

            TileIndex n_nz = 0;
            auto tile_offset = new TileIndex[tile_size]();
            auto tile_mask = new Mask(tile_size);

            VertexID tile_col_offset = 0;

            std::vector<TileIndex> row_idx;
            std::vector<TileIndex> col_idx;
            row_idx.reserve(tile_size + 64);
            col_idx.reserve(tile_size + 64);

            for (TileIndex x = 0; x < tile_size; x++) {
              auto uid = tile_x * tile_size + x;
              if (uid > immutable_csr.get_max_vid())
                break;
              auto u = immutable_csr.GetVertexByLocalID(uid);
              auto nbr_start = tile_y * tile_size;
              auto nbr_end = (tile_y + 1) * tile_size;

              bool is_nz_x = false;
              uint8_t n_nz_by_x = 0;

              // binary search to get the index of the first element within
              // nbr_start and the index of latest
              //
              uint32_t nbr_idx_start = 0;
              uint32_t nbr_idx_end = u.outdegree;

              uint32_t mid = nbr_idx_start + (nbr_idx_end - nbr_idx_start) / 2;

              if (u.outgoing_edges[mid] < nbr_start)
                nbr_idx_start = mid + 1;
              if (u.outgoing_edges[mid] >= nbr_end)
                nbr_idx_end = mid - 1;

              for (auto nbr = nbr_idx_start; nbr < nbr_idx_end; nbr++) {

                if (u.outgoing_edges[nbr] >= nbr_start &&
                    u.outgoing_edges[nbr] < nbr_end) {
                  is_nz_tile = true;
                  is_nz_x = true;
                  n_nz++;
                  n_nz_by_x++;
                  row_idx.push_back(x);
                  col_idx.push_back(u.outgoing_edges[nbr] - nbr_start);
                  tile_mask->SetBit(x, u.outgoing_edges[nbr] - nbr_start);
                } else if (u.outgoing_edges[nbr] >= nbr_end) {
                  break;
                }
              }
              tile_offset[x + 1] = n_nz_by_x + tile_offset[x];
            }

            // If it is NO Zero tile.
            if (is_nz_tile) {
              std::lock_guard<std::mutex> guard(mtx);
              tile_row_offset++;
              tile_count++;
              tile_n_nz_vec.push_back(tile_n_nz_vec.back() +
                                      tile_mask->GetDataPtr()->Count());
              tile_row_idx.push_back(tile_x);
              tile_col_idx.push_back(tile_y);

              auto tile = new Tile(tile_size, tile_x, tile_y,
                                   tile_mask->GetDataPtr()->Count());
              tile->SetBarOffsetPtr(tile_offset);
              tile->SetMaskPtr(tile_mask);
              memcpy(tile->GetRowIdxPtr(), row_idx.data(),
                     sizeof(TileIndex) * row_idx.size());
              memcpy(tile->GetColIdxPtr(), col_idx.data(),
                     sizeof(TileIndex) * col_idx.size());
              tile_vec.emplace_back(tile);
              mask->SetBit(tile_x, tile_y);
            } else {
              delete[] tile_offset;
              delete tile_mask;
            }
          }
        });

    // Finished A line
    tile_ptr[tile_x + 1] = tile_row_offset + tile_ptr[tile_x];
  }

  Init(n_rows, n_cols, tile_vec.size(), tile_vec.data(), mask, tile_ptr,
       tile_row_idx.data(), tile_col_idx.data(), tile_n_nz_vec.data());

#else
  std::cout << "Init error: no TBB" << std::endl;
#endif
}
*/

void TiledMatrix::InitTranspose(const ImmutableCSR &immutable_csr,
                                size_t tile_size) {
#ifdef TBB_FOUND
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;
  std::iota(worker.begin(), worker.end(), 0);

  auto mask = new Mask(tile_size);

  // Step 1. Generate the global tile structure of the matrix: tile_ptr,
  // tile_col_idx, tile_n_nz where (1) tile_ptr used to store memory offset of
  // the tiles in tile rows; (2) tile_col_index used to store tile column
  // indices; (3) tile_n_nz used to store the number of non-zeros in each
  // tile.
  VertexID n_rows = ceil((float)immutable_csr.get_num_vertices() /
                         (float)mask->get_mask_size_x());
  VertexID n_cols = ceil((float)immutable_csr.get_num_vertices() /
                         (float)mask->get_mask_size_y());

  VertexID *tile_ptr = new VertexID[n_cols + 1]();

  std::vector<VertexID> tile_n_nz_vec;
  tile_n_nz_vec.reserve(n_cols);
  tile_n_nz_vec.push_back(0);
  std::vector<VertexID> tile_row_idx;
  tile_row_idx.reserve(n_cols);
  std::vector<VertexID> tile_col_idx;
  tile_col_idx.reserve(n_cols);

  VertexID tile_x_scope = ceil((float)immutable_csr.get_num_vertices() /
                               (float)mask->get_mask_size_x());
  VertexID tile_y_scope = ceil((float)immutable_csr.get_num_vertices() /
                               (float)mask->get_mask_size_y());

  VertexID tile_count = 0;
  std::vector<Tile *> tile_vec;

  for (VertexID tile_y = 0; tile_y < tile_y_scope; tile_y++) {
    VertexID tile_row_offset = 0;

    auto step = worker.size();

    std::for_each(
        std::execution::par, worker.begin(), worker.end(),
        [&mtx, step, tile_y, tile_y_scope, tile_x_scope, tile_size,
         &immutable_csr, &tile_row_offset, &tile_count, &tile_n_nz_vec,
         &tile_row_idx, &tile_col_idx, &tile_vec, &mask](auto &w) {
          for (auto tile_x = w; tile_x < tile_x_scope; tile_x += step) {
            bool is_nz_tile = false;

            TileIndex n_nz = 0;
            auto tile_offset = new TileIndex[tile_size]();
            auto tile_mask = new Mask(tile_size);

            VertexID tile_col_offset = 0;

            std::vector<TileIndex> row_idx;
            std::vector<TileIndex> col_idx;
            for (TileIndex y = 0; y < tile_size; y++) {
              auto uid = tile_y * tile_size + y;
              if (uid > immutable_csr.get_max_vid())
                break;
              auto u = immutable_csr.GetVertexByLocalID(uid);
              auto nbr_start = tile_x * tile_size;
              auto nbr_end = (tile_x + 1) * tile_size;

              bool is_nz_y = false;
              uint8_t n_nz_by_y = 0;

              for (auto nbr = 0; nbr < u.indegree; nbr++) {
                if (u.incoming_edges[nbr] >= nbr_start &&
                    u.incoming_edges[nbr] < nbr_end) {
                  is_nz_tile = true;
                  is_nz_y = true;
                  n_nz++;
                  n_nz_by_y++;
                  row_idx.push_back(y);
                  col_idx.push_back(u.incoming_edges[nbr] - nbr_start);
                  tile_mask->SetBit(y, u.incoming_edges[nbr] - nbr_start);
                } else if (u.incoming_edges[nbr] >= nbr_end) {
                  break;
                }
              }
              tile_offset[y + 1] = n_nz_by_y + tile_offset[y];
            }

            // If it is NO Zero tile.
            if (is_nz_tile) {
              std::lock_guard<std::mutex> guard(mtx);
              tile_col_offset++;
              tile_count++;
              tile_n_nz_vec.push_back(tile_n_nz_vec.back() +
                                      tile_mask->GetDataPtr()->Count());
              tile_row_idx.push_back(tile_y);
              tile_col_idx.push_back(tile_x);

              auto tile = new Tile(tile_size, tile_y, tile_x,
                                   tile_mask->GetDataPtr()->Count());
              tile->SetBarOffsetPtr(tile_offset);
              tile->SetMaskPtr(tile_mask);
              memcpy(tile->GetRowIdxPtr(), row_idx.data(),
                     sizeof(TileIndex) * row_idx.size());
              memcpy(tile->GetColIdxPtr(), col_idx.data(),
                     sizeof(TileIndex) * col_idx.size());
              tile_vec.emplace_back(tile);
              mask->SetBit(tile_y, tile_x);
            } else {
              delete[] tile_offset;
              delete tile_mask;
            }
          }
        });

    // Finished A line
    tile_ptr[tile_y + 1] = tile_row_offset + tile_ptr[tile_y];
  }

  Init(n_rows, n_cols, tile_vec.size(), tile_vec.data(), mask, tile_ptr,
       tile_row_idx.data(), tile_col_idx.data(), tile_n_nz_vec.data());
#else
  std::cout << "no TBB" << std::endl;
#endif
}

void TiledMatrix::Write(const std::string &root_path) {
  // Init path
  if (!std::filesystem::exists(root_path))
    std::filesystem::create_directory(root_path);

  if (!std::filesystem::exists(root_path + "tiles"))
    std::filesystem::create_directory(root_path + "tiles");

  if (!std::filesystem::exists(root_path + "mask"))
    std::filesystem::create_directory(root_path + "mask");

  // Write tile_ptr, tile_col_idx tile_n_nz
  std::ofstream tile_ptr_file(root_path + "tile_ptr.bin");
  std::ofstream row_idx_file(root_path + "tile_row_idx.bin");
  std::ofstream col_idx_file(root_path + "tile_col_idx.bin");
  std::ofstream tile_n_nz_file(root_path + "tile_n_nz.bin");

  tile_ptr_file.write(reinterpret_cast<char *>(tile_ptr_),
                      sizeof(VertexID) * (metadata_.n_rows + 1));
  row_idx_file.write(reinterpret_cast<char *>(tile_row_idx_),
                     sizeof(VertexID) * metadata_.n_nz_tile);
  col_idx_file.write(reinterpret_cast<char *>(tile_col_idx_),
                     sizeof(VertexID) * metadata_.n_nz_tile);
  tile_n_nz_file.write(reinterpret_cast<char *>(tile_n_nz_),
                       sizeof(VertexID) * metadata_.n_nz_tile);

  // Write tile
  for (auto i = 0; i < metadata_.n_nz_tile; i++) {
    std::ofstream data_file(root_path + "tiles/" + std::to_string(i) + ".bin");

    // Init Data
    auto tile_data = data_ptr_[i]->GetDataPtr();
    for (int j = 0; j < data_ptr_[i]->get_n_nz(); j++) {
      tile_data[j] |= 1;
    }

    VertexID n_nz = data_ptr_[i]->get_n_nz();
    VertexID tile_x = data_ptr_[i]->get_tile_x();
    VertexID tile_y = data_ptr_[i]->get_tile_y();
    data_file.write(reinterpret_cast<char *>(&n_nz), sizeof(VertexID));
    data_file.write(reinterpret_cast<char *>(&tile_x), sizeof(VertexID));
    data_file.write(reinterpret_cast<char *>(&tile_y), sizeof(VertexID));
    TileIndex tile_size = data_ptr_[i]->get_tile_size();
    data_file.write(reinterpret_cast<char *>(&tile_size), sizeof(TileIndex));

    data_file.write(reinterpret_cast<char *>(data_ptr_[i]->GetBarOffsetPtr()),
                    sizeof(TileIndex) * data_ptr_[i]->get_tile_size());
    data_file.write(reinterpret_cast<char *>(data_ptr_[i]->GetRowIdxPtr()),
                    sizeof(TileIndex) * data_ptr_[i]->get_n_nz());
    data_file.write(reinterpret_cast<char *>(data_ptr_[i]->GetColIdxPtr()),
                    sizeof(TileIndex) * data_ptr_[i]->get_n_nz());

    data_file.write(reinterpret_cast<char *>(data_ptr_[i]->GetDataPtr()),
                    sizeof(VertexLabel) * data_ptr_[i]->get_n_nz());

    // Write mask.
    std::ofstream mask_file(root_path + "mask/" + std::to_string(i) + ".bin");
    mask_file.write(
        reinterpret_cast<char *>(
            data_ptr_[i]->GetMaskPtr()->GetDataPtr()->GetDataBasePointer()),
        sizeof(uint64_t) *
            (data_ptr_[i]->GetMaskPtr()->GetDataPtr()->GetMaxWordOffset() + 1));

    data_file.close();
    mask_file.close();
  }

  // Write Metadata
  std::ofstream out_meta_file(root_path + "meta.yaml");
  YAML::Node out_node;
  out_node["TiledMatrix"]["n_rows"] = metadata_.n_rows;
  out_node["TiledMatrix"]["n_cols"] = metadata_.n_cols;
  out_node["TiledMatrix"]["n_nz_tile"] = metadata_.n_nz_tile;

  out_meta_file << out_node << std::endl;
  out_meta_file.close();
}

void Tile::Show(bool is_transposed) const {
  std::cout << "**********  Tile: (" << (int)tile_x_ << ", " << (int)tile_y_
            << ")"
            << " n_nz: " << n_nz_ << " ************" << std::endl;
  if (n_nz_ == 0) {
    std::cout << "   empty" << std::endl
              << "****************************" << std::endl;
    return;
  }
  std::cout << " bar_offset: ";
  for (int i = 0; i < tile_size_; i++) {
    std::cout << (int)bar_offset_[i] << " ";
  }
  std::cout << std::endl << " row_idx: ";
  for (VertexID i = 0; i < n_nz_; i++) {
    std::cout << (int)row_idx_[i] << " ";
  }
  std::cout << std::endl << " col_idx: ";
  for (VertexID i = 0; i < n_nz_; i++) {
    std::cout << (int)col_idx_[i] << " ";
  }
  std::cout << std::endl << " data: ";
  for (VertexID i = 0; i < n_nz_; i++) {
    std::cout << data_[i] << " ";
  }
  std::cout << std::endl << std::endl;
  mask_->Show();
  std::cout << "****************************" << std::endl;
}

Mask *Tile::GetOutputMask(Mask &mask_a, Mask &mask_b) {
  if (mask_a.mask_size_ != mask_b.mask_size_) {
    std::cout << "Error: mask_a.mask_size_ != mask_b.mask_size_" << std::endl;
    return nullptr;
  }

  Mask *output = new Mask(mask_a.mask_size_);

  mask_a.Show();
  mask_b.Show();
  for (TileIndex i = 0; i < mask_a.mask_size_; i++) {

    uint64_t intersection = 0;
    // Get WORD_OFFSET of mask_a & mask_b;

    auto word_offset_a = WORD_OFFSET((i + 1) * mask_a.mask_size_);

    uint64_t scope_A = ~(0xffffffffffffffff << (mask_a.mask_size_ * (i + 1)) |
                         ~(0xffffffffffffffff << (mask_a.mask_size_ * i)));

    uint64_t a_f =
        (mask_a.GetDataPtr()->GetFragment(i * mask_a.mask_size_) & scope_A) >>
        (mask_a.mask_size_ * i);

    for (TileIndex j = 0; j < mask_b.mask_size_; j++) {
      auto word_offset_b = WORD_OFFSET((j + 1) * mask_b.mask_size_);

      uint64_t scope_B = ~(0xffffffffffffffff << (mask_b.mask_size_ * (j + 1)) |
                           ~(0xffffffffffffffff << (mask_b.mask_size_ * j)));

      uint64_t b_f =
          (mask_b.GetDataPtr()->GetFragment(j * mask_b.mask_size_) & scope_B) >>
          (mask_b.mask_size_ * j);
      if ((a_f & b_f) != 0) {
        // Set i, j as 1.
        intersection |= (1 << j);
      }
    }
    *(output->GetDataPtr()->GetPFragment(i * mask_a.mask_size_)) |=
        intersection << (mask_a.mask_size_ * i);
  }
  return output;
}

void TiledMatrix::Show() const {
  std::cout << "TiledMatrix: " << std::endl;
  std::cout << "  n_rows: " << metadata_.n_rows << std::endl;
  std::cout << "  n_cols: " << metadata_.n_cols << std::endl;
  std::cout << "  n_nz_tile: " << metadata_.n_nz_tile << std::endl;

  std::cout << "  tile_ptr: ";
  for (VertexID i = 0; i < metadata_.n_rows + 1; i++) {
    std::cout << tile_ptr_[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "  tile_row_idx: ";
  for (VertexID i = 0; i < metadata_.n_nz_tile; i++) {
    std::cout << tile_row_idx_[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "  tile_col_idx: ";
  for (VertexID i = 0; i < metadata_.n_nz_tile; i++) {
    std::cout << tile_col_idx_[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "  tile_n_nz: ";
  for (VertexID i = 0; i < metadata_.n_nz_tile; i++) {
    std::cout << tile_n_nz_[i] << " ";
  }
  std::cout << std::endl;

  for (size_t i = 0; i < metadata_.n_nz_tile; i++) {
    data_ptr_[i]->Show();
  }
}

void TiledMatrix::ShowAbs() const {
  std::cout << "TiledMatrix: " << std::endl;
  std::cout << "  n_rows: " << metadata_.n_rows << std::endl;
  std::cout << "  n_cols: " << metadata_.n_cols << std::endl;
  std::cout << "  n_nz_tile: " << metadata_.n_nz_tile << std::endl;
}

void TiledMatrix::Read(const std::string &root_path) {
  std::cout << "Read: " << root_path << std::endl;
  YAML::Node metadata_node;
  try {
    metadata_node = YAML::LoadFile(root_path + "meta.yaml");
    metadata_ = metadata_node.as<TiledMatrixMetadata>();
  } catch (YAML::BadFile &e) {
    std::cout << "meta.yaml file read failed! " << e.msg << std::endl;
  }

  tile_ptr_ = new VertexID[metadata_.n_rows + 1]();
  tile_row_idx_ = new VertexID[metadata_.n_nz_tile]();
  tile_col_idx_ = new VertexID[metadata_.n_nz_tile]();
  tile_n_nz_ = new VertexID[metadata_.n_nz_tile]();

  // Read tile_ptr, tile_col_idx tile_n_nz
  std::ifstream tile_ptr_file(root_path + "tile_ptr.bin");
  std::ifstream row_idx_file(root_path + "tile_row_idx.bin");
  std::ifstream col_idx_file(root_path + "tile_col_idx.bin");
  std::ifstream tile_n_nz_file(root_path + "tile_n_nz.bin");

  tile_ptr_file.read(reinterpret_cast<char *>(tile_ptr_),
                     sizeof(VertexID) * (metadata_.n_rows + 1));
  row_idx_file.read(reinterpret_cast<char *>(tile_row_idx_),
                    sizeof(VertexID) * metadata_.n_nz_tile);
  col_idx_file.read(reinterpret_cast<char *>(tile_col_idx_),
                    sizeof(VertexID) * metadata_.n_nz_tile);
  tile_n_nz_file.read(reinterpret_cast<char *>(tile_n_nz_),
                      sizeof(VertexID) * metadata_.n_nz_tile);

  // Read tiles.
  data_ptr_ = new Tile *[metadata_.n_nz_tile]();
  for (auto i = 0; i < metadata_.n_nz_tile; i++) {
    std::ifstream data_file(root_path + "tiles/" + std::to_string(i) + ".bin",
                            std::ios::binary);

    TileIndex tile_size;
    VertexID n_nz, tile_x, tile_y;

    data_file.read(reinterpret_cast<char *>(&n_nz), sizeof(VertexID));
    data_file.read(reinterpret_cast<char *>(&tile_x), sizeof(VertexID));
    data_file.read(reinterpret_cast<char *>(&tile_y), sizeof(VertexID));
    data_file.read(reinterpret_cast<char *>(&tile_size), sizeof(TileIndex));

    data_ptr_[i] = new Tile(tile_size, tile_x, tile_y, n_nz);

    data_file.read(reinterpret_cast<char *>(data_ptr_[i]->GetBarOffsetPtr()),
                   sizeof(TileIndex) * data_ptr_[i]->get_tile_size());
    data_file.read(reinterpret_cast<char *>(data_ptr_[i]->GetRowIdxPtr()),
                   sizeof(TileIndex) * data_ptr_[i]->get_n_nz());
    data_file.read(reinterpret_cast<char *>(data_ptr_[i]->GetColIdxPtr()),
                   sizeof(TileIndex) * data_ptr_[i]->get_n_nz());
    data_file.read(reinterpret_cast<char *>(data_ptr_[i]->GetDataPtr()),
                   sizeof(VertexLabel) * data_ptr_[i]->get_n_nz());
    data_file.close();

    std::ifstream mask_file(root_path + "mask/" + std::to_string(i) + ".bin");

    mask_file.read(
        reinterpret_cast<char *>(
            data_ptr_[i]->GetMaskPtr()->GetDataPtr()->GetDataBasePointer()),
        sizeof(uint64_t) *
            (data_ptr_[i]->GetMaskPtr()->GetDataPtr()->GetMaxWordOffset() + 1));
    mask_file.close();
  }
}

void TiledMatrix::Init(VertexID n_rows, VertexID n_cols, VertexID n_nz_tile,
                       Tile **data_ptr, Mask *mask, VertexID *tile_ptr,
                       VertexID *tile_row_idx, VertexID *tile_col_idx,
                       VertexID *tile_n_nz) {
  metadata_.n_rows = n_rows;
  metadata_.n_cols = n_cols;
  metadata_.n_nz_tile = n_nz_tile;

  data_ptr_ = new Tile *[n_nz_tile];
  memcpy(data_ptr_, data_ptr, sizeof(Tile *) * n_nz_tile);
  mask_ = mask;

  tile_ptr_ = new VertexID[n_rows + 1]();
  memcpy(tile_ptr_, tile_ptr, sizeof(VertexID) * (n_rows + 1));

  tile_row_idx_ = new VertexID[n_nz_tile]();
  memcpy(tile_row_idx_, tile_row_idx, sizeof(VertexID) * (n_nz_tile));

  tile_col_idx_ = new VertexID[n_nz_tile]();
  memcpy(tile_col_idx_, tile_col_idx, sizeof(VertexID) * (n_nz_tile));

  tile_n_nz_ = new VertexID[n_nz_tile]();
  memcpy(tile_n_nz_, tile_n_nz, sizeof(VertexID) * (n_nz_tile));
}

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics