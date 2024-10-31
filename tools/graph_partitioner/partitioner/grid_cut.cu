#include "tools/graph_partitioner/partitioner/grid_cut.cuh"

#include <algorithm>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "core/common/consts.h"
#include "core/common/types.h"
#include "core/common/yaml_config.h"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/metadata.h"
#include "core/util/atomic.h"
#include "core/util/bitmap.h"

#include "core/util/format_converter.cuh"
#include "tools/common/edgelist_subgraphs_io.cuh"

namespace sics {
namespace matrixgraph {
namespace tools {
namespace partitioner {

using VertexID = sics::matrixgraph::core::common::VertexID;
using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using EdgelistMetadata =
    sics::matrixgraph::core::data_structures::EdgelistMetadata;
using Bitmap = sics::matrixgraph::core::util::Bitmap;
using Edge = sics::matrixgraph::core::data_structures::Edge;
using Edges = sics::matrixgraph::core::data_structures::Edges;
using sics::matrixgraph::core::util::atomic::WriteAdd;
using sics::matrixgraph::core::util::atomic::WriteMax;
using sics::matrixgraph::core::util::atomic::WriteMin;
using GraphMetadata = sics::matrixgraph::core::data_structures::GraphMetadata;
using EdgelistSubGraphsIO =
    sics::matrixgraph::tools::common::EdgelistSubGraphsIO;

void GridCutPartitioner::RunPartitioner() {
  assert(n_partitions_ <= core::common::kMaxNChunks);

  std::cout << "[GridCutPartitioner] Loading edgelist...\n" << std::endl;
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  // Load Yaml node (Edgelist metadata).
  YAML::Node node = YAML::LoadFile(input_path_ + "meta.yaml");

  sics::matrixgraph::core::data_structures::EdgelistMetadata edgelist_metadata =
      {node["EdgelistBin"]["num_vertices"].as<VertexID>(),
       node["EdgelistBin"]["num_edges"].as<EdgeIndex>(),
       node["EdgelistBin"]["max_vid"].as<VertexID>()};

  auto buffer_edges =
      new sics::matrixgraph::core::data_structures::Edge[edgelist_metadata
                                                             .num_edges]();

  std::ifstream in_file(input_path_ + "edgelist.bin");
  if (!in_file) {
    std::cout << "Open file failed: " + input_path_ + "edgelist.bin"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  in_file.read(reinterpret_cast<char *>(buffer_edges),
               sizeof(sics::matrixgraph::core::data_structures::Edge) *
                   edgelist_metadata.num_edges);
  sics::matrixgraph::core::data_structures::Edges edges(edgelist_metadata,
                                                        buffer_edges);

  edges.ShowGraph(3);

  // Precompute the size of each edge bucket.
  VertexID scope_per_chunk =
      ceil((edges.get_metadata().max_vid + 1) / (float)n_partitions_);

  auto size_per_bucket = new EdgeIndex[n_partitions_ * n_partitions_]();
  auto *n_edges_for_each_block = new EdgeIndex[n_partitions_ * n_partitions_]();
  auto *max_vid_for_each_block = new VertexID[n_partitions_ * n_partitions_]();
  auto *min_vid_for_each_block = new VertexID[n_partitions_ * n_partitions_]();
  Bitmap vertices_bm_for_each_block[n_partitions_ * n_partitions_];

  for (GraphID _ = 0; _ < n_partitions_ * n_partitions_; _++) {
    vertices_bm_for_each_block[_].Init(edges.get_metadata().max_vid);
    min_vid_for_each_block[_] = std::numeric_limits<uint32_t>::max();
  }

  std::cout
      << "[GridCutPartitioner] Computing key parameters under chunk scope of "
      << scope_per_chunk << " ...\n"
      << std::endl;
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, step, &edges, scope_per_chunk, &n_edges_for_each_block,
       &max_vid_for_each_block, &min_vid_for_each_block,
       &vertices_bm_for_each_block](auto w) {
        for (auto eid = w; eid < edges.get_metadata().num_edges; eid += step) {
          auto edge = edges.get_edge_by_index(eid);
          auto x = edge.src / scope_per_chunk;
          auto y = edge.dst / scope_per_chunk;
          WriteAdd(&n_edges_for_each_block[x * n_partitions_ + y],
                   (EdgeIndex)1);

          WriteMax(&max_vid_for_each_block[x * n_partitions_ + y], edge.src);
          WriteMax(&max_vid_for_each_block[x * n_partitions_ + y], edge.dst);
          WriteMin(&min_vid_for_each_block[x * n_partitions_ + y], edge.src);
          WriteMin(&min_vid_for_each_block[x * n_partitions_ + y], edge.dst);
          if (!vertices_bm_for_each_block[x * n_partitions_ + y].GetBit(
                  edge.src)) {
            vertices_bm_for_each_block[x * n_partitions_ + y].SetBit(edge.src);
          }
          if (!vertices_bm_for_each_block[x * n_partitions_ + y].GetBit(
                  edge.dst)) {
            vertices_bm_for_each_block[x * n_partitions_ + y].SetBit(edge.dst);
          }
        }
      });

  std::cout << "[GridCutPartitioner] Allocating space for each block...\n"
            << std::endl;
  Edge **edge_blocks_buf = new Edge *[n_partitions_ * n_partitions_]();
  for (GraphID _ = 0; _ < n_partitions_ * n_partitions_; _++) {
    edge_blocks_buf[_] = new Edge[n_edges_for_each_block[_]]();
  }

  auto *offset_for_each_block =
      new std::atomic<EdgeIndex>[n_partitions_ * n_partitions_]();

  std::cout << "[GridCutPartitioner] Dropping edges into blocks...\n"
            << std::endl;
  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [this, step, &edges, scope_per_chunk, &edge_blocks_buf,
                 &n_edges_for_each_block, &offset_for_each_block](auto w) {
                  for (auto eid = w; eid < edges.get_metadata().num_edges;
                       eid += step) {
                    auto edge = edges.get_edge_by_index(eid);
                    auto x = edge.src / scope_per_chunk;
                    auto y = edge.dst / scope_per_chunk;
                    auto block_id = x * n_partitions_ + y;
                    auto offset = offset_for_each_block[block_id].fetch_add(1);
                    edge_blocks_buf[block_id][offset] = edge;
                  }
                });

  std::vector<Edges> edge_blocks;
  edge_blocks.reserve(n_partitions_ * n_partitions_);

  std::cout << "[GridCutPartitioner] Constructing Edgelist of blocks...\n"
            << std::endl;
  for (auto _ = 0; _ < n_partitions_ * n_partitions_; _++) {
    EdgelistMetadata meta{.num_vertices = vertices_bm_for_each_block[_].Count(),
                          .num_edges = n_edges_for_each_block[_],
                          .max_vid = max_vid_for_each_block[_],
                          .min_vid = min_vid_for_each_block[_]};
    edge_blocks.emplace_back(Edges(meta, edge_blocks_buf[_]));
  }

  for (auto _ = 0; _ < n_partitions_ * n_partitions_; _++) {
    if (edge_blocks[_].get_metadata().num_vertices == 0)
      continue;
    std::cout << "[GridCutPartitioner] Reassigning vertex ids for block: "
              << " x: " << _ / n_partitions_ << " y: " << _ % n_partitions_
              << " ...\n"
              << std::endl;
    edge_blocks[_].ReassignVertexIDs();
    edge_blocks[_].ShowGraph(3);
  }

  std::cout << "[GridCutPartitioner] Writing to disk!" << std::endl;
  GraphMetadata graph_metadata;
  graph_metadata.num_vertices = edges.get_metadata().num_vertices;
  graph_metadata.num_edges = edges.get_metadata().num_edges;
  graph_metadata.num_subgraphs = n_partitions_ * n_partitions_;
  graph_metadata.max_vid = edges.get_metadata().max_vid;
  graph_metadata.min_vid = 0;

  EdgelistSubGraphsIO::Write(output_path_, edge_blocks, graph_metadata);

  std::cout << "[GridCutPartitioner] Done! " << output_path_ << std::endl;

  delete[] offset_for_each_block;
  delete[] edge_blocks_buf;
  delete[] max_vid_for_each_block;
  delete[] min_vid_for_each_block;
}

} // namespace partitioner
} // namespace tools
} // namespace matrixgraph
} // namespace sics