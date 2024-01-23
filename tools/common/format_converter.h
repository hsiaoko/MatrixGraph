#ifndef MATRIXGRAPH_CORE_COMMON_FORMAT_CONVERTER_H_
#define MATRIXGRAPH_CORE_COMMON_FORMAT_CONVERTER_H_

#include <cmath>
#include <list>
#include <numeric>
#include <vector>

#include "core/common/types.h"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/immutable_csr.h"
#include "core/data_structures/tiled_matrix.cuh"
#include "core/util/bitmap.h"

namespace sics {
namespace matrixgraph {
namespace tools {
namespace format_converter {

using sics::matrixgraph::core::common::EdgeIndex;
using sics::matrixgraph::core::common::TileIndex;
using sics::matrixgraph::core::common::VertexID;
using sics::matrixgraph::core::data_structures::Edges;
using sics::matrixgraph::core::data_structures::ImmutableCSR;
using sics::matrixgraph::core::data_structures::Mask;
using sics::matrixgraph::core::data_structures::Tile;
using sics::matrixgraph::core::data_structures::TiledMatrix;
using sics::matrixgraph::core::util::Bitmap;
using sics::matrixgraph::core::util::atomic::WriteAdd;
using sics::matrixgraph::core::util::atomic::WriteMin;
using Vertex = sics::matrixgraph::core::data_structures::ImmutableCSRVertex;

ImmutableCSR *Edgelist2ImmutableCSR(const Edges &edgelist) {
  auto parallelism = std::thread::hardware_concurrency();

  auto immutable_csr = new ImmutableCSR();

  auto edgelist_metadata = edgelist.get_metadata();
  auto aligned_max_vid =
      (((edgelist.get_metadata().max_vid + 1) >> 6) << 6) + 64;
  auto visited = Bitmap(aligned_max_vid);
  auto num_in_edges_by_vid = new VertexID[aligned_max_vid]();
  auto num_out_edges_by_vid = new VertexID[aligned_max_vid]();
  VertexID min_vid = MAX_VERTEX_ID;

  std::vector<EdgeIndex> parallel_scope_edges(edgelist_metadata.num_edges);
  std::vector<EdgeIndex> parallel_scope_vertices(aligned_max_vid);

  std::iota(parallel_scope_edges.begin(), parallel_scope_edges.end(), 0);
  std::iota(parallel_scope_vertices.begin(), parallel_scope_vertices.end(), 0);

  // Compute min_vid and obtain the num of incoming/outgoing edges for each
  // vertex.
  std::for_each(parallel_scope_edges.begin(), parallel_scope_edges.end(),
                [&edgelist, &visited, &num_in_edges_by_vid,
                 &num_out_edges_by_vid, &min_vid](auto i) {
                  auto e = edgelist.get_edge_by_index(i);
                  visited.SetBit(e.src);
                  visited.SetBit(e.dst);
                  WriteAdd(num_in_edges_by_vid + e.dst, (VertexID)1);
                  WriteAdd(num_out_edges_by_vid + e.src, (VertexID)1);
                  WriteMin(&min_vid, e.src);
                  WriteMin(&min_vid, e.dst);
                });

  auto buffer_csr_vertices = new Vertex[aligned_max_vid]();
  EdgeIndex count_in_edges = 0, count_out_edges = 0;

  // Malloc space for each vertex.
  std::for_each(
      parallel_scope_vertices.begin(), parallel_scope_vertices.end(),
      [&visited, &buffer_csr_vertices, &count_in_edges, &count_out_edges,
       &num_in_edges_by_vid, &num_out_edges_by_vid](auto j) {
        if (!visited.GetBit(j))
          return;
        buffer_csr_vertices[j].vid = j;
        buffer_csr_vertices[j].indegree = num_in_edges_by_vid[j];
        buffer_csr_vertices[j].outdegree = num_out_edges_by_vid[j];
        buffer_csr_vertices[j].incoming_edges =
            new VertexID[num_in_edges_by_vid[j]]();
        buffer_csr_vertices[j].outgoing_edges =
            new VertexID[num_out_edges_by_vid[j]]();
        WriteAdd(&count_in_edges, (EdgeIndex)buffer_csr_vertices[j].indegree);
        WriteAdd(&count_out_edges, (EdgeIndex)buffer_csr_vertices[j].outdegree);
      });

  delete[] num_in_edges_by_vid;
  delete[] num_out_edges_by_vid;

  EdgeIndex *offset_in_edges = new EdgeIndex[aligned_max_vid]();
  EdgeIndex *offset_out_edges = new EdgeIndex[aligned_max_vid]();

  // Fill edges in each vertex.
  std::for_each(parallel_scope_edges.begin(), parallel_scope_edges.end(),
                [&edgelist, &buffer_csr_vertices, &offset_in_edges,
                 &offset_out_edges](auto j) {
                  auto e = edgelist.get_edge_by_index(j);
                  EdgeIndex offset_out = 0, offset_in = 0;
                  offset_out =
                      __sync_fetch_and_add(offset_out_edges + e.src, 1);
                  offset_in = __sync_fetch_and_add(offset_in_edges + e.dst, 1);
                  buffer_csr_vertices[e.src].outgoing_edges[offset_out] = e.dst;
                  buffer_csr_vertices[e.dst].incoming_edges[offset_in] = e.src;
                });
  delete[] offset_in_edges;
  delete[] offset_out_edges;

  // Construct CSR graph.
  auto buffer_globalid = new VertexID[edgelist.get_metadata().num_vertices]();
  auto buffer_indegree = new VertexID[edgelist.get_metadata().num_vertices]();
  auto buffer_outdegree = new VertexID[edgelist.get_metadata().num_vertices]();

  VertexID vid = 0;
  auto vid_map = new VertexID[aligned_max_vid]();
  for (VertexID i = 0; i < aligned_max_vid; i++) {
    if (!visited.GetBit(i))
      continue;
    buffer_globalid[vid] = buffer_csr_vertices[i].vid;
    buffer_indegree[vid] = buffer_csr_vertices[i].indegree;
    buffer_outdegree[vid] = buffer_csr_vertices[i].outdegree;
    vid_map[i] = vid;
    vid++;
  }
  auto buffer_in_offset = new EdgeIndex[edgelist.get_metadata().num_vertices]();
  auto buffer_out_offset =
      new EdgeIndex[edgelist.get_metadata().num_vertices]();

  // Compute offset for each vertex.
  for (VertexID i = 1; i < edgelist.get_metadata().num_vertices; i++) {
    buffer_in_offset[i] = buffer_in_offset[i - 1] + buffer_indegree[i - 1];
    buffer_out_offset[i] = buffer_out_offset[i - 1] + buffer_outdegree[i - 1];
  }

  auto buffer_in_edges = new VertexID[count_in_edges]();
  auto buffer_out_edges = new VertexID[count_out_edges]();

  // Fill edges.
  vid = 0;
  std::for_each(parallel_scope_vertices.begin(), parallel_scope_vertices.end(),
                [&visited, &buffer_csr_vertices, &buffer_in_edges,
                 &buffer_out_edges, &buffer_in_offset, &buffer_out_offset,
                 &vid_map](auto j) {
                  if (!visited.GetBit(j))
                    return;
                  if (buffer_csr_vertices[j].indegree != 0) {
                    memcpy(buffer_in_edges + buffer_in_offset[vid_map[j]],
                           buffer_csr_vertices[j].incoming_edges,
                           buffer_csr_vertices[j].indegree * sizeof(VertexID));
                  }
                  if (buffer_csr_vertices[j].outdegree != 0) {
                    memcpy(buffer_out_edges + buffer_out_offset[vid_map[j]],
                           buffer_csr_vertices[j].outgoing_edges,
                           buffer_csr_vertices[j].outdegree * sizeof(VertexID));
                  }
                });

  delete[] buffer_csr_vertices;
  delete[] vid_map;

  immutable_csr->SetGlobalIDBuffer(buffer_globalid);
  immutable_csr->SetInDegreeBuffer(buffer_indegree);
  immutable_csr->SetOutDegreeBuffer(buffer_outdegree);
  immutable_csr->SetInOffsetBuffer(buffer_in_offset);
  immutable_csr->SetOutOffsetBuffer(buffer_out_offset);
  immutable_csr->SetIncomingEdgesBuffer(buffer_in_edges);
  immutable_csr->SetOutgoingEdgesBuffer(buffer_out_edges);

  immutable_csr->SetNumVertices(edgelist.get_metadata().num_vertices);
  immutable_csr->SetNumIncomingEdges(count_in_edges);
  immutable_csr->SetNumOutgoingEdges(count_out_edges);
  immutable_csr->SetMaxVid(edgelist.get_metadata().max_vid);
  immutable_csr->SetMinVid(min_vid);

  return immutable_csr;
}

TiledMatrix *
ImmutableCSR2TransposedTiledMatrix(const ImmutableCSR &immutable_csr) {
  std::cout << "ImmutableCSR2TransposedTiledMatrix" << std::endl;

  TiledMatrix *tiled_matrix = new TiledMatrix();

  auto tile_size = 4;

  auto mask = new Mask(tile_size);

  // Step 1. Generate the global tile structure of the matrix: tile_ptr,
  // tile_col_idx, tile_n_nz where (1) tile_ptr used to store memory offset of
  // the tiles in tile rows; (2) tile_col_index used to store tile column
  // indices; (3) tile_n_nz used to store the number of non-zeros in each tile.
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

  // Generate transposed TileMatrix.
  for (VertexID tile_x = 0; tile_x < tile_x_scope; tile_x++) {
    VertexID tile_col_offset = 0;

    for (VertexID tile_y = 0; tile_y < tile_y_scope; tile_y++) {
      bool is_nz_tile = false;

      TileIndex n_nz = 0;

      auto tile_offset = new TileIndex[tile_size]();
      auto tile_mask = new Mask(tile_size);

      std::vector<TileIndex> row_idx;
      std::vector<TileIndex> col_idx;

      for (TileIndex y = 0; y < tile_size; y++) {
        auto uid = tile_y * tile_size + y;
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

      if (is_nz_tile) {
        tile_col_offset++;
        tile_count++;
        tile_n_nz_vec.push_back(tile_n_nz_vec.back() + n_nz);
        tile_row_idx.push_back(tile_y);
        tile_col_idx.push_back(tile_x);

        auto tile = new Tile(tile_size, tile_y, tile_x, n_nz);
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
    tile_ptr[tile_x + 1] = tile_col_offset + tile_ptr[tile_x];
  }

  // Step 2. Construct tiled_matrix.
  tiled_matrix->Init(n_rows, n_cols, tile_vec.size(), tile_vec.data(), mask,
                     tile_ptr, tile_row_idx.data(), tile_col_idx.data(),
                     tile_n_nz_vec.data());
  tiled_matrix->MarkasTransposed();

  return tiled_matrix;
}

TiledMatrix *ImmutableCSR2TiledMatrix(const ImmutableCSR &immutable_csr) {
  std::cout << "ImmutableCSR2TiledMatrix" << std::endl;

  TiledMatrix *tiled_matrix = new TiledMatrix();

  auto tile_size = 4;

  auto mask = new Mask(tile_size);

  // Step 1. Generate the global tile structure of the matrix: tile_ptr,
  // tile_col_idx, tile_n_nz where (1) tile_ptr used to store memory offset of
  // the tiles in tile rows; (2) tile_col_index used to store tile column
  // indices; (3) tile_n_nz used to store the number of non-zeros in each tile.
  VertexID n_rows = ceil((float)immutable_csr.get_num_vertices() /
                         (float)mask->get_mask_size_x());
  VertexID n_cols = ceil((float)immutable_csr.get_num_vertices() /
                         (float)mask->get_mask_size_y());

  VertexID *tile_ptr = new VertexID[n_rows + 1]();

  std::vector<VertexID> tile_n_nz_vec;
  tile_n_nz_vec.reserve(n_cols);
  tile_n_nz_vec.push_back(0);
  std::vector<VertexID> tile_col_idx, tile_row_idx;
  tile_col_idx.reserve(n_cols);
  tile_row_idx.reserve(n_cols);

  VertexID tile_x_scope = ceil((float)immutable_csr.get_num_vertices() /
                               (float)mask->get_mask_size_x());
  VertexID tile_y_scope = ceil((float)immutable_csr.get_num_vertices() /
                               (float)mask->get_mask_size_y());

  VertexID tile_count = 0;

  std::vector<Tile *> tile_vec;

  // Generate TileMatrix.
  for (VertexID tile_x = 0; tile_x < tile_x_scope; tile_x++) {
    VertexID tile_row_offset = 0;

    for (VertexID tile_y = 0; tile_y < tile_y_scope; tile_y++) {
      bool is_nz_tile = false;

      TileIndex n_nz = 0;
      auto tile_offset = new TileIndex[tile_size]();
      auto tile_mask = new Mask(tile_size);

      VertexID tile_col_offset = 0;

      std::vector<TileIndex> row_idx;
      std::vector<TileIndex> col_idx;

      for (TileIndex x = 0; x < tile_size; x++) {
        auto uid = tile_x * tile_size + x;
        auto u = immutable_csr.GetVertexByLocalID(uid);
        auto nbr_start = tile_y * tile_size;
        auto nbr_end = (tile_y + 1) * tile_size;

        bool is_nz_x = false;
        uint8_t n_nz_by_x = 0;

        for (auto nbr = 0; nbr < u.outdegree; nbr++) {
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

      if (is_nz_tile) {
        tile_row_offset++;
        tile_count++;
        tile_n_nz_vec.push_back(tile_n_nz_vec.back() + n_nz);
        tile_row_idx.push_back(tile_x);
        tile_col_idx.push_back(tile_y);

        auto tile = new Tile(tile_size, tile_x, tile_y, n_nz);
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
    tile_ptr[tile_x + 1] = tile_row_offset + tile_ptr[tile_x];
  }

  // Step 2. Construct tiled_matrix.
  tiled_matrix->Init(n_rows, n_cols, tile_vec.size(), tile_vec.data(), mask,
                     tile_ptr, tile_row_idx.data(), tile_col_idx.data(),
                     tile_n_nz_vec.data());

  return tiled_matrix;
}

} // namespace format_converter
} // namespace tools
} // namespace matrixgraph
} // namespace sics
#endif // MATRIXGRAPH_CORE_COMMON_FORMAT_CONVERTER_H_
