#include "core/data_structures/edgelist.h"

#include <execution>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <thread>

#include "core/common/consts.h"
#include "core/common/types.h"
#include "core/util/atomic.h"
#include "core/util/bitmap.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

using VertexID = sics::matrixgraph::core::common::VertexID;
using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using sics::matrixgraph::core::util::atomic::WriteAdd;
using sics::matrixgraph::core::util::atomic::WriteMax;
using sics::matrixgraph::core::util::atomic::WriteMin;
using std::filesystem::create_directory;
using std::filesystem::exists;

Edges::Edges(const Edges &edges) {
  edgelist_metadata_ = edges.get_metadata();
  edges_ptr_ = new Edge[edgelist_metadata_.num_edges]();
  memcpy(edges_ptr_, edges.get_base_ptr(),
         sizeof(Edge) * edgelist_metadata_.num_edges);
  if (edges.get_localid_to_globalid_ptr() != nullptr) {
    memcpy(localid_to_globalid_, edges.get_localid_to_globalid_ptr(),
           sizeof(Edge) * edgelist_metadata_.num_vertices);
  }
}

Edges::Edges(EdgeIndex n_edges, VertexID *edges_buf,
             VertexID *localid2globalid) {
  Init(n_edges, edges_buf, localid2globalid);
}

void Edges::Init(EdgeIndex n_edges, VertexID *edges_buf,
                 VertexID *localid2globalid) {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  Bitmap bm(n_edges);
  edges_ptr_ = new Edge[n_edges]();
  VertexID max_vid = 0;
  VertexID min_vid = MAX_VERTEX_ID;

  // Get Min Max vertex ID.
  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [this, n_edges, step, &edges_buf, &min_vid, &max_vid](auto w) {
                  for (EdgeIndex _ = w; _ < n_edges; _ += step) {
                    edges_ptr_[_].src = edges_buf[_ * 2];
                    edges_ptr_[_].dst = edges_buf[_ * 2 + 1];
                    WriteMin(&min_vid, edges_ptr_[_].src);
                    WriteMin(&min_vid, edges_ptr_[_].dst);
                    WriteMax(&max_vid, edges_ptr_[_].src);
                    WriteMax(&max_vid, edges_ptr_[_].dst);
                  }
                });

  Bitmap visited(max_vid);

  // Get number of vertices.
  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [this, &visited, step, n_edges](auto w) {
                  for (EdgeIndex _ = w; _ < n_edges; _ += step) {
                    visited.SetBit(edges_ptr_[_].src);
                    visited.SetBit(edges_ptr_[_].dst);
                  }
                });

  edgelist_metadata_.num_edges = n_edges;
  edgelist_metadata_.num_vertices = visited.Count();
  edgelist_metadata_.max_vid = max_vid;
  edgelist_metadata_.min_vid = min_vid;
  if (localid2globalid == nullptr) {
    GenerateLocalID2GlobalID();
  } else {
    localid_to_globalid_ = localid2globalid;
  }
}

void Edges::WriteToBinary(const std::string &output_path) {
  if (!std::filesystem::exists(output_path))
    std::filesystem::create_directory(output_path);

  std::ofstream out_data_file(output_path + "edgelist.bin");
  std::ofstream out_localid2globalid_file(output_path + "localid2globalid.bin");
  std::ofstream out_meta_file(output_path + "meta.yaml");

  out_data_file.write(reinterpret_cast<char *>(edges_ptr_),
                      sizeof(Edge) * edgelist_metadata_.num_edges);

  out_localid2globalid_file.write(
      reinterpret_cast<char *>(localid_to_globalid_),
      sizeof(VertexID) * edgelist_metadata_.num_vertices);

  YAML::Node node;
  node["EdgelistBin"]["num_vertices"] = edgelist_metadata_.num_vertices;
  node["EdgelistBin"]["num_edges"] = edgelist_metadata_.num_edges;
  node["EdgelistBin"]["max_vid"] = edgelist_metadata_.max_vid;
  node["EdgelistBin"]["min_vid"] = edgelist_metadata_.min_vid;
  out_meta_file << node << std::endl;

  out_data_file.close();
  out_localid2globalid_file.close();
  out_meta_file.close();
}

void Edges::ReadFromBin(const std::string &input_path) {
  YAML::Node node = YAML::LoadFile(input_path + "meta.yaml");

  edgelist_metadata_ = {node["EdgelistBin"]["num_vertices"].as<VertexID>(),
                        node["EdgelistBin"]["num_edges"].as<EdgeIndex>(),
                        node["EdgelistBin"]["max_vid"].as<VertexID>()};

  edges_ptr_ =
      new sics::matrixgraph::core::data_structures::Edge[edgelist_metadata_
                                                             .num_edges]();

  std::ifstream in_file(input_path + "edgelist.bin");
  if (!in_file) {
    std::cout << "Open file failed: " + input_path + "edgelist.bin"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  in_file.read(reinterpret_cast<char *>(edges_ptr_),
               sizeof(Edge) * edgelist_metadata_.num_edges);

  std::ifstream in_localid2globalid_file(input_path + "localid2globalid.bin");
  if (!in_localid2globalid_file) {
    std::cout << "Open file failed: " + input_path + "localid2globalid.bin"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  in_file.read(reinterpret_cast<char *>(localid_to_globalid_),
               sizeof(VertexID) * edgelist_metadata_.num_vertices);
}

void Edges::ReadFromCSV(const std::string &filename, const std::string &sep,
                        bool compressed) {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  std::ifstream in_file(filename);

  in_file.seekg(0, std::ios::end);
  size_t length = in_file.tellg();
  in_file.seekg(0, std::ios::beg);

  char *buff = new char[length]();
  in_file.read(buff, length);
  std::string content(buff, length);

  EdgeIndex n_edges = count(content.begin(), content.end(), '\n');
  auto buffer_edges = new VertexID[n_edges * 2]();
  std::stringstream ss(content);
  delete[] buff;

  EdgeIndex index = 0;
  VertexID max_vid = 0, compressed_vid = 0;
  std::string line, vid_str;

  while (getline(ss, line, '\n')) {
    if (*line.c_str() == '\0')
      break;
    std::stringstream ss_line(line);
    while (getline(ss_line, vid_str, *sep.c_str())) {
      VertexID vid = stoll(vid_str);
      sics::matrixgraph::core::util::atomic::WriteMax(&max_vid, vid);
      buffer_edges[index++] = vid;
    }
  }
  content.clear();
  in_file.close();

  auto aligned_max_vid = (((max_vid + 1) >> 6) << 6) + 64;
  edges_ptr_ = new Edge[n_edges]();
  Bitmap bitmap(aligned_max_vid);

  auto vid_map = new VertexID[aligned_max_vid]();
  auto compressed_buffer_edges = new VertexID[n_edges * 2]();

  // Compute the mapping between origin vid to compressed vid.
  for (EdgeIndex index = 0; index < n_edges * 2; index++) {
    if (!bitmap.GetBit(buffer_edges[index])) {
      bitmap.SetBit(buffer_edges[index]);
      vid_map[buffer_edges[index]] = compressed_vid++;
    }
  }

  for (EdgeIndex i = 0; i < n_edges; i++) {
    edges_ptr_[i].src = buffer_edges[2 * i];
    edges_ptr_[i].dst = buffer_edges[2 * i + 1];
  }

  delete[] buffer_edges;
  delete[] vid_map;
  delete[] compressed_buffer_edges;

  // Compute metadata.
  edgelist_metadata_.num_edges = n_edges;
  edgelist_metadata_.num_vertices = bitmap.Count();
  edgelist_metadata_.max_vid = max_vid;
  if (compressed) {
    std::cout << "[Edges] Reading CSV with compressed ..." << std::endl;
    GenerateLocalID2GlobalID();
  } else {
    std::cout << "[Edges] Reading CSV without compressed ..." << std::endl;
    if (localid_to_globalid_ != nullptr)
      delete[] localid_to_globalid_;
    localid_to_globalid_ = new VertexID[edgelist_metadata_.num_vertices]();
    std::for_each(std::execution::par, worker.begin(), worker.end(),
                  [this, step](auto w) {
                    for (auto i = w; i < get_metadata().num_vertices;
                         i += step) {
                      std::cout << "->" << localid_to_globalid_[i] << " " << i
                                << std::endl;
                      localid_to_globalid_[i] = i;
                      std::cout << "->" << localid_to_globalid_[i] << " " << i
                                << std::endl;
                    }
                  });
  }
}

void Edges::GenerateLocalID2GlobalID() {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  VertexID *new_localid_to_globalid =
      new VertexID[edgelist_metadata_.num_vertices]();

  VertexID *vid_map = new VertexID[edgelist_metadata_.max_vid + 1]();

  VertexID compressed_vid = 0;

  Bitmap bitmap(edgelist_metadata_.max_vid);

  for (EdgeIndex index = 0; index < edgelist_metadata_.num_edges; index++) {
    auto e = get_edge_by_index(index);
    if (localid_to_globalid_ != nullptr) {
      e.src = localid_to_globalid_[e.src];
      e.dst = localid_to_globalid_[e.dst];
    }

    if (!bitmap.GetBit(e.src)) {
      bitmap.SetBit(e.src);
      vid_map[e.src] = compressed_vid++;
    }
    if (!bitmap.GetBit(e.dst)) {
      bitmap.SetBit(e.dst);
      vid_map[e.dst] = compressed_vid++;
    }
  }

  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [this, step, &vid_map, &new_localid_to_globalid](auto w) {
                  for (auto i = w; i < get_metadata().num_edges; i += step) {
                    auto e = get_edge_by_index(i);
                    if (localid_to_globalid_ != nullptr) {
                      e.src = localid_to_globalid_[e.src];
                      e.dst = localid_to_globalid_[e.dst];
                    }
                    new_localid_to_globalid[vid_map[e.src]] = e.src;
                    new_localid_to_globalid[vid_map[e.dst]] = e.dst;
                    edges_ptr_[i].src = vid_map[e.src];
                    edges_ptr_[i].dst = vid_map[e.dst];
                  }
                });
  delete[] vid_map;
  delete[] localid_to_globalid_;
  localid_to_globalid_ = new_localid_to_globalid;
}

void Edges::Compacted() {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  if (localid_to_globalid_ == nullptr) {
    localid_to_globalid_ = new VertexID[get_metadata().num_vertices]();
  } else {
    memset(localid_to_globalid_, 0,
           sizeof(VertexID) * get_metadata().num_vertices);
  }

  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [this, step](auto w) {
                  for (auto i = w; i < get_metadata().num_edges; i += step) {
                    auto e = get_edge_by_index(i);
                    localid_to_globalid_[e.src] = e.src;
                    localid_to_globalid_[e.dst] = e.dst;
                  }
                });
}

void Edges::Transpose() {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();
  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [this, step](auto w) {
                  for (auto i = w; i < get_metadata().num_edges; i += step) {
                    VertexID tmp = edges_ptr_[i].src;
                    edges_ptr_[i].src = edges_ptr_[i].dst; // swap src and dst
                    edges_ptr_[i].dst = tmp;
                  }
                });
}

void Edges::SortBySrc() {
  std::sort(std::execution::par, edges_ptr_,
            edges_ptr_ + edgelist_metadata_.num_edges);
}

void Edges::ShowGraph(EdgeIndex n_edges) const {
  std::cout << "[ShowGraph] n_edges:" << edgelist_metadata_.num_edges
            << ", n_vertices:" << edgelist_metadata_.num_vertices << std::endl;
  EdgeIndex min_n_edges = min(edgelist_metadata_.num_edges, n_edges);
  if (localid_to_globalid_ != nullptr) {
    for (EdgeIndex i = 0; i < min_n_edges; i++) {
      std::cout << localid_to_globalid_[edges_ptr_[i].src] << " "
                << localid_to_globalid_[edges_ptr_[i].dst] << std::endl;
    }
  } else {
    for (EdgeIndex i = 0; i < min_n_edges; i++) {
      std::cout << edges_ptr_[i].src << " " << edges_ptr_[i].dst << std::endl;
    }
  }
}

VertexID Edges::get_globalid_by_localid(VertexID localid) const {
  if (localid_to_globalid_ == nullptr)
    return localid;
  return localid_to_globalid_[localid];
}

void Edges::SetLocalIDToGlobalID(VertexID *localid_to_globalid) {
  if (localid_to_globalid_ != nullptr)
    delete[] localid_to_globalid;
  localid_to_globalid_ = localid_to_globalid;
}

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics