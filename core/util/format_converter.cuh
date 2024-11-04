#ifndef MATRIXGRAPH_CORE_UTIL_FORMAT_CONVERTER_H_
#define MATRIXGRAPH_CORE_UTIL_FORMAT_CONVERTER_H_

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <execution>
#include <iomanip>
#include <list>
#include <numeric>
#include <vector>

#include "core/common/types.h"
#include "core/data_structures/bit_tiled_matrix.cuh"
#include "core/data_structures/csr_tiled_matrix.cuh"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/metadata.h"
#include "core/util/bitmap.h"
#include "core/util/cuda_check.cuh"
#include "core/util/gpu_bitmap.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace util {
namespace format_converter {

using sics::matrixgraph::core::common::EdgeIndex;
using sics::matrixgraph::core::common::TileIndex;
using sics::matrixgraph::core::common::VertexID;
using sics::matrixgraph::core::data_structures::BitTiledMatrix;
using sics::matrixgraph::core::data_structures::CSRTiledMatrix;
using sics::matrixgraph::core::data_structures::Edge;
using sics::matrixgraph::core::data_structures::EdgelistMetadata;
using sics::matrixgraph::core::data_structures::Edges;
using sics::matrixgraph::core::data_structures::ImmutableCSR;
using sics::matrixgraph::core::util::Bitmap;
using sics::matrixgraph::core::util::GPUBitmap;
using sics::matrixgraph::core::util::atomic::WriteAdd;
using sics::matrixgraph::core::util::atomic::WriteMin;
using Vertex = sics::matrixgraph::core::data_structures::ImmutableCSRVertex;
using GraphID = sics::matrixgraph::core::common::GraphID;
using sics::matrixgraph::core::util::atomic::WriteAdd;
using sics::matrixgraph::core::util::atomic::WriteMax;
using TiledMatrixMetadata =
    sics::matrixgraph::core::data_structures::TiledMatrixMetadata;
using SubGraphMetadata =
    sics::matrixgraph::core::data_structures::SubGraphMetadata;

static Edges *ImmutableCSR2Edgelist(const ImmutableCSR &immutable_csr) {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  EdgeIndex n_edges = immutable_csr.get_num_outgoing_edges();

  EdgelistMetadata edgelist_metadata{
      .num_vertices = immutable_csr.get_num_vertices(),
      .num_edges = immutable_csr.get_num_outgoing_edges(),
      .max_vid = immutable_csr.get_max_vid(),
      .min_vid = 0};

  auto *edge_ptr = new Edge[n_edges]();

  std::cout << "[ImmutableCSR2Edgelist] Converting immutable csr to edge buffer"
            << std::endl;
  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [step, &immutable_csr, &edge_ptr](auto w) {
                  for (VertexID vid = w; vid < immutable_csr.get_num_vertices();
                       vid += step) {

                    auto offset = immutable_csr.GetOutOffsetByLocalID(vid);
                    auto degree = immutable_csr.GetOutDegreeByLocalID(vid);
                    auto *out_edges =
                        immutable_csr.GetOutgoingEdgesByLocalID(vid);

                    VertexID global_id =
                        immutable_csr.GetGlobalIDByLocalID(vid);
                    for (EdgeIndex i = 0; i < degree; i++) {
                      Edge e;
                      e.src = global_id;
                      e.dst = out_edges[i];
                      edge_ptr[offset + i] = e;
                    }
                  }
                });
  auto *edges = new Edges(edgelist_metadata, edge_ptr);
  return edges;
}

static ImmutableCSR *Edgelist2ImmutableCSR(const Edges &edgelist) {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  auto immutable_csr = new ImmutableCSR();

  auto edgelist_metadata = edgelist.get_metadata();
  auto aligned_max_vid =
      (((edgelist.get_metadata().max_vid + 1) >> 6) << 6) + 64;
  auto visited = Bitmap(aligned_max_vid);
  auto *num_in_edges_by_vid = new EdgeIndex[aligned_max_vid]();
  auto *num_out_edges_by_vid = new EdgeIndex[aligned_max_vid]();
  VertexID min_vid = MAX_VERTEX_ID;

  std::vector<VertexID> parallel_scope_vertices(aligned_max_vid);
  std::iota(parallel_scope_vertices.begin(), parallel_scope_vertices.end(), 0);

  // Compute min_vid and obtain the num of incoming/outgoing edges for each
  // vertex.
  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [step, &min_vid, &num_out_edges_by_vid, &num_in_edges_by_vid,
                 &visited, &edgelist](auto &w) {
                  for (EdgeIndex i = w; i < edgelist.get_metadata().num_edges;
                       i += step) {
                    auto e = edgelist.get_edge_by_index(i);
                    visited.SetBit(e.src);
                    visited.SetBit(e.dst);
                    WriteAdd(num_in_edges_by_vid + e.dst, (EdgeIndex)1);
                    WriteAdd(num_out_edges_by_vid + e.src, (EdgeIndex)1);
                    WriteMin(&min_vid, e.src);
                    WriteMin(&min_vid, e.dst);
                  }
                });

  auto buffer_csr_vertices = new Vertex[aligned_max_vid]();
  EdgeIndex count_in_edges = 0, count_out_edges = 0;

  // Malloc space for each vertex.
  std::for_each(
      std::execution::par, parallel_scope_vertices.begin(),
      parallel_scope_vertices.end(),
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
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [step, &min_vid, &num_out_edges_by_vid, &num_in_edges_by_vid, &visited,
       &edgelist, &buffer_csr_vertices, &offset_in_edges,
       &offset_out_edges](auto &w) {
        for (EdgeIndex i = w; i < edgelist.get_metadata().num_edges;
             i += step) {
          auto e = edgelist.get_edge_by_index(i);
          EdgeIndex offset_out = 0, offset_in = 0;
          offset_out = __sync_fetch_and_add(offset_out_edges + e.src, 1);
          offset_in = __sync_fetch_and_add(offset_in_edges + e.dst, 1);
          buffer_csr_vertices[e.src].outgoing_edges[offset_out] = e.dst;
          buffer_csr_vertices[e.dst].incoming_edges[offset_in] = e.src;
        }
      });
  delete[] offset_in_edges;
  delete[] offset_out_edges;

  // Construct CSR graph.
  VertexID *buffer_globalid = nullptr;
  if (edgelist.get_localid_to_globalid_ptr() == nullptr) {
    buffer_globalid = new VertexID[edgelist.get_metadata().num_vertices]();
  } else {
    buffer_globalid = edgelist.get_localid_to_globalid_ptr();
  }
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

  VertexID max_degree = 0, mean_degree = 0, mean_in_degree = 0,
           min_degree = MAX_VERTEX_ID;
  for (VertexID i = 0; i < edgelist.get_metadata().num_vertices; i++) {
    WriteMax(&max_degree, buffer_outdegree[i]);
    WriteMin(&min_degree, buffer_outdegree[i]);
    WriteAdd(&mean_degree, buffer_outdegree[i]);
    WriteAdd(&mean_in_degree, buffer_indegree[i]);
  }
  auto *buffer_in_offset =
      new EdgeIndex[edgelist.get_metadata().num_vertices + 1]();
  auto *buffer_out_offset =
      new EdgeIndex[edgelist.get_metadata().num_vertices + 1]();

  // Compute offset for each vertex.
  for (VertexID i = 0; i < edgelist.get_metadata().num_vertices; i++) {
    buffer_in_offset[i + 1] = buffer_in_offset[i] + buffer_indegree[i];
    buffer_out_offset[i + 1] = buffer_out_offset[i] + buffer_outdegree[i];
  }

  auto buffer_in_edges = new VertexID[count_in_edges]();
  auto buffer_out_edges = new VertexID[count_out_edges]();

  // Fill edges.
  vid = 0;
  std::for_each(std::execution::par, parallel_scope_vertices.begin(),
                parallel_scope_vertices.end(),
                [&visited, &buffer_csr_vertices, &buffer_in_edges,
                 &buffer_out_edges, &buffer_in_offset, &buffer_out_offset,
                 &vid_map](auto j) {
                  if (!visited.GetBit(j))
                    return;
                  if (buffer_csr_vertices[j].indegree != 0) {
                    memcpy(buffer_in_edges + buffer_in_offset[vid_map[j]],
                           buffer_csr_vertices[j].incoming_edges,
                           buffer_csr_vertices[j].indegree * sizeof(VertexID));
                    std::sort(buffer_in_edges + buffer_in_offset[vid_map[j]],
                              buffer_in_edges + buffer_in_offset[vid_map[j]] +
                                  buffer_csr_vertices[j].indegree);
                  }
                  if (buffer_csr_vertices[j].outdegree != 0) {
                    memcpy(buffer_out_edges + buffer_out_offset[vid_map[j]],
                           buffer_csr_vertices[j].outgoing_edges,
                           buffer_csr_vertices[j].outdegree * sizeof(VertexID));
                    std::sort(buffer_out_edges + buffer_out_offset[vid_map[j]],
                              buffer_out_edges + buffer_out_offset[vid_map[j]] +
                                  buffer_csr_vertices[j].outdegree);
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

  std::cout << "[Edgelist2ImmutableCSR] done ! - max degree:" << max_degree
            << " mean degree: "
            << (float)mean_degree / (float)edgelist.get_metadata().num_vertices
            << " mean in degree: "
            << (float)mean_in_degree /
                   (float)edgelist.get_metadata().num_vertices
            << " min degree: " << min_degree << "   "
            << edgelist.get_metadata().num_vertices << std::endl;
  return immutable_csr;
}

static BitTiledMatrix *Edgelist2BitTiledMatrix(const Edges &edges,
                                               size_t tile_size,
                                               size_t block_scope) {

  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();
  auto *bit_tiled_matrix = new BitTiledMatrix();

  VertexID n_strips = ceil((float)block_scope / (float)tile_size);

  auto tile_visited = new GPUBitmap(n_strips * n_strips);

  VertexID n_nz_tile_per_row[n_strips] = {0};
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [step, &edges, block_scope, tile_size, &tile_visited, &n_nz_tile_per_row,
       n_strips](auto w) {
        for (auto eid = w; eid < edges.get_metadata().num_edges; eid += step) {

          auto e = edges.get_edge_by_index(eid);
          auto *localid_to_globalid = edges.get_localid_to_globalid_ptr();

          auto tile_x = (localid_to_globalid[e.src] % block_scope) / tile_size;
          auto tile_y = (localid_to_globalid[e.dst] % block_scope) / tile_size;

          if (!tile_visited->GetBit(tile_x * n_strips + tile_y)) {
            tile_visited->SetBit(tile_x * n_strips + tile_y);
            WriteAdd(&n_nz_tile_per_row[tile_x], (VertexID)1);
          }
        }
      });

  size_t n_nz_tile = tile_visited->GPUCount();

  TiledMatrixMetadata metadata = {
      .n_strips = n_strips,
      .n_nz_tile = n_nz_tile,
      .tile_size = tile_size,
  };

  bit_tiled_matrix->Init(metadata, tile_visited);

  auto *tile_offset_row = bit_tiled_matrix->GetTileOffsetRowPtr();

  for (size_t _ = 0; _ < n_strips; _++) {
    tile_offset_row[_ + 1] = tile_offset_row[_] + n_nz_tile_per_row[_];
  }

  std::cout << "[Edgelist2BitTiledMatrix] Creating None Zero Tile: "
            << n_nz_tile << " in total" << std::endl;
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [step, &edges, block_scope, tile_size, &tile_visited, &bit_tiled_matrix,
       &n_nz_tile_per_row, n_strips](auto w) {
        for (auto eid = w; eid < edges.get_metadata().num_edges; eid += step) {

          auto *localid_to_globalid = edges.get_localid_to_globalid_ptr();

          auto e = edges.get_edge_by_index(eid);

          auto x_within_tile =
              (localid_to_globalid[e.src] % block_scope) % tile_size;
          auto y_within_tile =
              (localid_to_globalid[e.dst] % block_scope) % tile_size;

          auto tile_x = (localid_to_globalid[e.src] % block_scope) / tile_size;
          auto tile_y = (localid_to_globalid[e.dst] % block_scope) / tile_size;

          size_t tile_row_offset = 0;
          // if (tile_x * n_strips + tile_y < 33554432) {
          if (1) {
            auto start_time_0 = std::chrono::system_clock::now();
            tile_row_offset =
                bit_tiled_matrix->GetNzTileBitmapPtr()->PreElementCount(
                    tile_x * n_strips + tile_y);
            auto start_time_1 = std::chrono::system_clock::now();
            if (eid % 30000 == 0) {
              std::cout
                  << "w: " << w << "|" << std::setprecision(3)
                  << (float)eid / (float)edges.get_metadata().num_edges << " "
                  << eid << "/ " << edges.get_metadata().num_edges << " "
                  << tile_x * n_strips + tile_y << " cpu time"
                  << std::chrono::duration_cast<std::chrono::microseconds>(
                         start_time_1 - start_time_0)
                             .count() /
                         (double)CLOCKS_PER_SEC
                  << std::endl;
            }
          } else {
            auto start_time_0 = std::chrono::system_clock::now();

            tile_row_offset =
                bit_tiled_matrix->GetNzTileBitmapPtr()->GPUPreElementCount(
                    tile_x * n_strips + tile_y);

            auto start_time_1 = std::chrono::system_clock::now();

            if (eid % 30000 == 0) {
              std::cout
                  << "w: " << w << "|" << std::setprecision(3)
                  << (float)eid / (float)edges.get_metadata().num_edges << " "
                  << eid << "/ " << edges.get_metadata().num_edges << " "
                  << tile_x * n_strips + tile_y << " gpu time"
                  << std::chrono::duration_cast<std::chrono::microseconds>(
                         start_time_1 - start_time_0)
                             .count() /
                         (double)CLOCKS_PER_SEC
                  << std::endl;
            }
          }

          auto *tile_row_idx = bit_tiled_matrix->GetTileRowIdxPtr();
          tile_row_idx[tile_row_offset] = tile_x;

          auto *tile_col_idx = bit_tiled_matrix->GetTileColIdxPtr();
          tile_col_idx[tile_row_offset] = tile_y;

          auto *tile_ptr = bit_tiled_matrix->GetDataPtrByIdx(tile_row_offset);

          __sync_fetch_and_or(
              tile_ptr + WORD_OFFSET(x_within_tile * tile_size + y_within_tile),
              1ull << BIT_OFFSET(x_within_tile * tile_size + y_within_tile));
        }
      });
  std::cout << "[Edgelist2BitTiledMatrix] Done!" << std::endl;
  return bit_tiled_matrix;
}

static CSRTiledMatrix *
Edgelist2CSRTiledMatrix(const Edges &edges, size_t tile_size,
                        size_t block_scope, CSRTiledMatrix *output = nullptr) {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  VertexID n_strips = ceil((float)block_scope / (float)tile_size);
  VertexID n_csr = n_strips * n_strips;

  CSRTiledMatrix *csr_tiled_matrix = nullptr;

  if (output == nullptr) {
    csr_tiled_matrix = new CSRTiledMatrix();
  } else {
    csr_tiled_matrix = output;
  }

  if (edges.get_metadata().num_edges == 0) {
    TiledMatrixMetadata tile_meta_data = {
        .n_strips = n_strips, .n_nz_tile = 0, .tile_size = tile_size};
    csr_tiled_matrix->Init(tile_meta_data);
    return csr_tiled_matrix;
  }

  VertexID n_nz_tile_per_row[n_strips] = {0};

  std::vector<std::unique_ptr<Vertex[]>> buffer_csr_vertices_vec(n_csr);
  std::generate(
      buffer_csr_vertices_vec.begin(), buffer_csr_vertices_vec.end(),
      [tile_size]() { return std::make_unique<Vertex[]>(tile_size); });

  std::vector<std::unique_ptr<VertexID[]>> num_out_edges_by_vid_vec(n_csr);
  std::generate(
      num_out_edges_by_vid_vec.begin(), num_out_edges_by_vid_vec.end(),
      [tile_size]() { return std::make_unique<VertexID[]>(tile_size); });

  std::vector<EdgeIndex> count_out_edges_vec;
  count_out_edges_vec.resize(n_csr);

  std::vector<Bitmap> visited_vec;
  visited_vec.resize(n_csr);
  std::generate(visited_vec.begin(), visited_vec.end(),
                [tile_size]() { return Bitmap(tile_size); });
  Bitmap graph_visited(n_csr);
  std::vector<VertexID> gid_map(n_csr);

  // Step 1. compute tile_id for each edge.
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [&edges, step, block_scope, tile_size, n_strips, &buffer_csr_vertices_vec,
       &num_out_edges_by_vid_vec, &visited_vec, &graph_visited](auto &w) {
        for (EdgeIndex eid = w; eid < edges.get_metadata().num_edges;
             eid += step) {
          auto e = edges.get_edge_by_index(eid);
          auto *localid_to_globalid = edges.get_localid_to_globalid_ptr();

          auto tile_x = (localid_to_globalid[e.src] % block_scope) / tile_size;
          auto tile_y = (localid_to_globalid[e.dst] % block_scope) / tile_size;

          auto x_within_tile =
              (localid_to_globalid[e.src] % block_scope) % tile_size;
          auto y_within_tile =
              (localid_to_globalid[e.dst] % block_scope) % tile_size;

          WriteAdd(&(num_out_edges_by_vid_vec[tile_x * n_strips + tile_y]
                                             [x_within_tile]),
                   (VertexID)1);

          visited_vec[tile_x * n_strips + tile_y].SetBit(x_within_tile);
          graph_visited.SetBit(tile_x * n_strips + tile_y);
        }
      });

  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [step, n_strips, tile_size, n_csr, &buffer_csr_vertices_vec,
       &graph_visited, &num_out_edges_by_vid_vec, &count_out_edges_vec, &edges,
       &visited_vec](auto &w) {
        for (auto tile_id = w; tile_id < n_csr; tile_id += step) {

          if (!graph_visited.GetBit(tile_id))
            continue;

          for (auto i = 0; i < tile_size; i++) {
            if (!visited_vec[tile_id].GetBit(i))
              continue;
            // buffer_csr_vertices_vec[tile_id][i].vid =
            //     (tile_id / n_strips) * tile_size + i;

            buffer_csr_vertices_vec[tile_id][i].outdegree =
                num_out_edges_by_vid_vec[tile_id][i];

            buffer_csr_vertices_vec[tile_id][i].outgoing_edges =
                new VertexID[num_out_edges_by_vid_vec[tile_id][i]]();

            WriteAdd(&count_out_edges_vec[tile_id],
                     (EdgeIndex)buffer_csr_vertices_vec[tile_id][i].outdegree);
          }
        }
      });

  // Step 2. compute global id for src.
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [&edges, step, block_scope, tile_size, n_strips, &buffer_csr_vertices_vec,
       &num_out_edges_by_vid_vec, &visited_vec, &graph_visited](auto &w) {
        for (EdgeIndex eid = w; eid < edges.get_metadata().num_edges;
             eid += step) {
          auto e = edges.get_edge_by_index(eid);
          auto *localid_to_globalid = edges.get_localid_to_globalid_ptr();

          auto tile_x = (localid_to_globalid[e.src] % block_scope) / tile_size;
          auto tile_y = (localid_to_globalid[e.dst] % block_scope) / tile_size;

          auto x_within_tile =
              (localid_to_globalid[e.src] % block_scope) % tile_size;
          buffer_csr_vertices_vec[tile_x * n_strips + tile_y][x_within_tile]
              .vid = localid_to_globalid[e.src];
        }
      });

  size_t n_nz_tile = graph_visited.Count();
  TiledMatrixMetadata tile_meta_data = {
      .n_strips = n_strips, .n_nz_tile = n_nz_tile, .tile_size = tile_size};
  csr_tiled_matrix->Init(tile_meta_data, new GPUBitmap(n_csr * n_csr));

  VertexID gid = 0;
  std::for_each(worker.begin(), worker.end(),
                [step, n_strips, &gid, tile_size, &visited_vec, n_csr,
                 &buffer_csr_vertices_vec, &num_out_edges_by_vid_vec,
                 &count_out_edges_vec, &csr_tiled_matrix, &graph_visited,
                 &gid_map](auto &w) {
                  for (auto tile_id = w; tile_id < n_csr; tile_id += step) {
                    if (!graph_visited.GetBit(tile_id))
                      continue;
                    VertexID n_vertices = visited_vec[tile_id].Count();

                    gid_map[tile_id] = gid;
                    SubGraphMetadata metadata = {
                        .gid = tile_id,
                        .num_vertices = n_vertices,
                        .num_incoming_edges = 0,
                        .num_outgoing_edges = count_out_edges_vec[tile_id],
                        .max_vid = tile_size,
                        .min_vid = 0};

                    csr_tiled_matrix->SetCSRMetadata(metadata, gid);
                    gid++;
                  }
                });

  size_t total_mem = 0;
  size_t n_mem_size[n_nz_tile] = {0};

  auto *csr_offset_ptr = csr_tiled_matrix->GetCSROffsetPtr();

  for (VertexID i = 0; i < n_nz_tile; i++) {
    auto metadata = csr_tiled_matrix->GetCSRMetadataByIdx(i);

    size_t size_globalid_buf = sizeof(VertexID) * metadata.num_vertices;
    size_t size_in_degree_buf = sizeof(VertexID) * metadata.num_vertices;
    size_t size_out_degree_buf = sizeof(VertexID) * metadata.num_vertices;
    size_t size_in_offset_buf = sizeof(EdgeIndex) * (metadata.num_vertices + 1);
    size_t size_out_offset_buf =
        sizeof(EdgeIndex) * (metadata.num_vertices + 1);
    size_t size_in_edges_buf = sizeof(VertexID) * metadata.num_incoming_edges;
    size_t size_out_edges_buf = sizeof(VertexID) * metadata.num_outgoing_edges;
    size_t size_edges_globalid_buf = sizeof(VertexID) * metadata.max_vid;
    n_mem_size[i] = size_globalid_buf + size_edges_globalid_buf +
                    size_in_degree_buf + size_in_offset_buf +
                    size_out_degree_buf + size_out_offset_buf +
                    size_out_edges_buf + size_in_edges_buf;
    total_mem += n_mem_size[i];
    csr_offset_ptr[i + 1] = csr_offset_ptr[i] + n_mem_size[i];
  }

  num_out_edges_by_vid_vec.clear();

  csr_tiled_matrix->InitData(total_mem);

  std::vector<std::unique_ptr<EdgeIndex[]>> offset_out_edges_vec(n_nz_tile);
  std::generate(
      offset_out_edges_vec.begin(), offset_out_edges_vec.end(),
      [tile_size]() { return std::make_unique<EdgeIndex[]>(tile_size); });

  std::vector<VertexID *> buffer_globalid_vec(n_nz_tile);
  std::vector<VertexID *> buffer_outdegree_vec(n_nz_tile);
  std::vector<EdgeIndex *> buffer_out_offset_vec(n_nz_tile);
  std::vector<VertexID *> buffer_out_edges_vec(n_nz_tile);
  std::vector<VertexID *> y_vid_map_vec(n_nz_tile);

  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [step, n_csr, n_nz_tile, &csr_tiled_matrix, &buffer_globalid_vec,
       &buffer_outdegree_vec, &buffer_out_offset_vec, &buffer_out_edges_vec,
       &y_vid_map_vec](auto &w) {
        for (auto i = w; i < n_nz_tile; i += step) {
          auto *csr_base_ptr = csr_tiled_matrix->GetCSRBasePtrByIdx(i);
          ImmutableCSR csr(csr_tiled_matrix->GetCSRMetadataByIdx(i));
          csr.ParseBasePtr(csr_base_ptr);
          buffer_out_offset_vec[i] = csr.GetOutOffsetBasePointer();
          buffer_outdegree_vec[i] = csr.GetOutDegreeBasePointer();
          buffer_globalid_vec[i] = csr.GetGloablIDBasePointer();
          buffer_out_edges_vec[i] = csr.GetOutgoingEdgesBasePointer();
          y_vid_map_vec[i] = csr.GetEdgesGloablIDBasePointer();
        }
      });

  // Fill edges in each vertex.
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [tile_size, &edges, step, block_scope, n_strips, &buffer_csr_vertices_vec,
       &y_vid_map_vec, &offset_out_edges_vec, &csr_tiled_matrix,
       &gid_map](auto &w) {
        for (EdgeIndex eid = w; eid < edges.get_metadata().num_edges;
             eid += step) {

          auto e = edges.get_edge_by_index(eid);

          auto *localid_to_globalid = edges.get_localid_to_globalid_ptr();
          auto tile_x = (localid_to_globalid[e.src] % block_scope) / tile_size;
          auto tile_y = (localid_to_globalid[e.dst] % block_scope) / tile_size;

          auto x_within_tile =
              (localid_to_globalid[e.src] % block_scope) % tile_size;
          auto y_within_tile =
              (localid_to_globalid[e.dst] % block_scope) % tile_size;

          EdgeIndex offset_out = 0;
          auto gid = gid_map[tile_x * n_strips + tile_y];
          offset_out = __sync_fetch_and_add(
              offset_out_edges_vec[gid].get() + x_within_tile, 1);

          buffer_csr_vertices_vec[tile_x * n_strips + tile_y][x_within_tile]
              .outgoing_edges[offset_out] = y_within_tile;

          y_vid_map_vec[gid][y_within_tile] = localid_to_globalid[e.dst];
        }
      });

  offset_out_edges_vec.clear();

  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [tile_size, &edges, step, block_scope, n_strips, n_csr,
                 &offset_out_edges_vec, &buffer_csr_vertices_vec, &visited_vec,
                 &gid_map, &graph_visited, &buffer_outdegree_vec](auto &w) {
                  for (auto i = w; i < n_csr; i += step) {
                    if (!graph_visited.GetBit(i))
                      continue;
                    VertexID vid = 0;
                    for (VertexID j = 0; j < tile_size; j++) {
                      if (!visited_vec[i].GetBit(j))
                        continue;
                      buffer_outdegree_vec[gid_map[i]][vid++] =
                          buffer_csr_vertices_vec[i][j].outdegree;
                    }
                  }
                });

  // Construct offset.
  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [tile_size, &edges, step, block_scope, n_strips, n_csr,
                 n_nz_tile, &offset_out_edges_vec, &buffer_csr_vertices_vec,
                 &visited_vec, &buffer_outdegree_vec, &buffer_out_offset_vec,
                 &csr_tiled_matrix](auto &w) {
                  for (auto i = w; i < n_nz_tile; i += step) {
                    auto metadata = csr_tiled_matrix->GetCSRMetadataByIdx(i);
                    for (VertexID j = 1; j < metadata.num_vertices; j++) {
                      buffer_out_offset_vec[i][j] =
                          buffer_out_offset_vec[i][j - 1] +
                          buffer_outdegree_vec[i][j - 1];
                    }
                  }
                });

  // Fill edges.
  std::for_each(
      worker.begin(), worker.end(),
      [&visited_vec, step, n_csr, tile_size, &buffer_out_edges_vec,
       &buffer_csr_vertices_vec, &buffer_outdegree_vec, &gid_map,
       &graph_visited, &buffer_out_offset_vec](auto &w) {
        for (auto i = w; i < n_csr; i += step) {
          if (!graph_visited.GetBit(i))
            continue;
          VertexID vid = 0;
          auto gid = gid_map[i];
          for (VertexID j = 0; j < tile_size; j++) {
            if (!visited_vec[i].GetBit(j))
              continue;
            memcpy(buffer_out_edges_vec[gid] + buffer_out_offset_vec[gid][vid],
                   buffer_csr_vertices_vec[i][j].outgoing_edges,
                   buffer_csr_vertices_vec[i][j].outdegree * sizeof(VertexID));

            vid++;
          }
        }
      });

  // Generate global id
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [&visited_vec, step, n_csr, n_nz_tile, tile_size, &buffer_out_edges_vec,
       &buffer_csr_vertices_vec, &buffer_globalid_vec, &graph_visited, &gid_map,
       &buffer_out_offset_vec](auto &w) {
        for (auto i = w; i < n_csr; i += step) {
          if (!graph_visited.GetBit(i))
            continue;
          VertexID vid = 0;
          auto gid = gid_map[i];
          for (VertexID j = 0; j < tile_size; j++) {
            if (!visited_vec[i].GetBit(j))
              continue;
            buffer_globalid_vec[gid][vid] = buffer_csr_vertices_vec[i][j].vid;
            vid++;
          }
        }
      });

  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [n_strips, &visited_vec, step, n_csr, n_nz_tile, tile_size,
       &n_nz_tile_per_row, &buffer_out_edges_vec, &buffer_csr_vertices_vec,
       &buffer_globalid_vec, &buffer_outdegree_vec, &buffer_out_offset_vec,
       &num_out_edges_by_vid_vec, &count_out_edges_vec, &y_vid_map_vec,
       &csr_tiled_matrix](auto &w) {
        for (auto i = w; i < n_nz_tile; i += step) {
          auto *csr_base_ptr = csr_tiled_matrix->GetCSRBasePtrByIdx(i);
          auto *tile_row_idx_ptr = csr_tiled_matrix->GetTileRowIdxPtr();
          auto *tile_col_idx_ptr = csr_tiled_matrix->GetTileColIdxPtr();
          tile_row_idx_ptr[i] =
              csr_tiled_matrix->GetCSRMetadataByIdx(i).gid / n_strips;
          tile_col_idx_ptr[i] =
              csr_tiled_matrix->GetCSRMetadataByIdx(i).gid % n_strips;
          WriteAdd(&n_nz_tile_per_row[tile_row_idx_ptr[i]], (VertexID)1);
          csr_tiled_matrix->SetBit(tile_row_idx_ptr[i], tile_col_idx_ptr[i]);
        }
      });

  auto *tile_offset_row_ptr = csr_tiled_matrix->GetTileOffsetRowPtr();
  for (int i = 0; i < n_strips; i++)
    tile_offset_row_ptr[i + 1] = tile_offset_row_ptr[i] + n_nz_tile_per_row[i];

  return csr_tiled_matrix;
}

} // namespace format_converter
} // namespace util
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_COMMON_FORMAT_CONVERTER_H_