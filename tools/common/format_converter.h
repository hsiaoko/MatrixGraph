#ifndef MATRIXGRAPH_CORE_COMMON_FORMAT_CONVERTER_H_
#define MATRIXGRAPH_CORE_COMMON_FORMAT_CONVERTER_H_

#include <cmath>
#include <list>
#include <numeric>
#include <vector>

#include "core/common/types.h"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/tiled_matrix.cuh"
#include "core/util/bitmap.h"

#ifdef TBB_FOUND
#include <execution>
#endif

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
#ifdef TBB_FOUND
  std::for_each(std::execution::par, parallel_scope_edges.begin(),
                parallel_scope_edges.end(),
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
#else
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
#endif

  auto buffer_csr_vertices = new Vertex[aligned_max_vid]();
  EdgeIndex count_in_edges = 0, count_out_edges = 0;

  // Malloc space for each vertex.
#ifdef TBB_FOUND
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
#else
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
#endif

  delete[] num_in_edges_by_vid;
  delete[] num_out_edges_by_vid;

  EdgeIndex *offset_in_edges = new EdgeIndex[aligned_max_vid]();
  EdgeIndex *offset_out_edges = new EdgeIndex[aligned_max_vid]();

  // Fill edges in each vertex.
#ifdef TBB_FOUND
  std::for_each(std::execution::par, parallel_scope_edges.begin(),
                parallel_scope_edges.end(),
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
#else
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
#endif
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
#ifdef TBB_FOUND
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
#else
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
#endif

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

} // namespace format_converter
} // namespace tools
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_COMMON_FORMAT_CONVERTER_H_