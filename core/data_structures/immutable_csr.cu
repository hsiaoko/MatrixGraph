#include "core/data_structures/immutable_csr.cuh"

#include <algorithm>
#include <execution>
#include <fstream>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>

#include "core/common/consts.h"
#include "core/util/atomic.h"
#include "core/util/bitmap.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

using sics::matrixgraph::core::util::atomic::WriteAdd;
using sics::matrixgraph::core::util::atomic::WriteMax;
using sics::matrixgraph::core::util::atomic::WriteMin;

void ImmutableCSR::PrintGraphAbs(VertexID display_num) const {
  std::cout << "### GID: " << metadata_.gid
            << ",  num_vertices: " << metadata_.num_vertices
            << ", num_incoming_edges: " << metadata_.num_incoming_edges
            << ", num_outgoing_edges: " << metadata_.num_outgoing_edges
            << " ###" << std::endl;
  for (VertexID i = 0; i < metadata_.num_vertices; i++) {
    if (i > display_num)
      break;
    auto u = GetVertexByLocalID(i);
    std::stringstream ss;
    ss << "  ===vid: " << u.vid << ", indegree: " << u.indegree
       << ", outdegree: " << u.outdegree << "===" << std::endl;
    std::string s = ss.str();
    std::cout << s << std::endl;
  }
}

void ImmutableCSR::PrintGraph(VertexID display_num) const {
  std::cout << "### GID: " << metadata_.gid
            << ",  num_vertices: " << metadata_.num_vertices
            << ", num_incoming_edges: " << metadata_.num_incoming_edges
            << ", num_outgoing_edges: " << metadata_.num_outgoing_edges
            << " Show top " << display_num << " ###" << std::endl;
  for (VertexID i = 0; i < metadata_.num_vertices; i++) {
    if (i > display_num)
      break;
    auto u = GetVertexByLocalID(i);
    std::stringstream ss;

    ss << "  ===vid: " << u.vid << ", indegree: " << u.indegree
       << ", outdegree: " << u.outdegree << ", label: " << u.vlabel
       << "===" << std::endl;
    if (u.indegree != 0) {
      ss << "    Incoming edges: ";
      for (VertexID i = 0; i < u.indegree; i++)
        ss << u.incoming_edges[i] << ",";
      ss << std::endl << std::endl;
    }
    if (u.outdegree != 0) {
      ss << "    Outgoing edges: ";
      for (VertexID i = 0; i < u.outdegree; i++) {
        if (edges_globalid_by_localid_base_pointer_ != nullptr) {
          ss << edges_globalid_by_localid_base_pointer_[u.outgoing_edges[i]]
             << ",";
        } else {
          ss << u.outgoing_edges[i] << ",";
        }
      }

      ss << std::endl << std::endl;
    }
    ss << "****************************************" << std::endl;
    std::string s = ss.str();
    std::cout << s << std::endl;
  }
}

void ImmutableCSR::ParseBasePtr(uint8_t *graph_base_pointer) {
  SetGlobalIDBuffer(reinterpret_cast<VertexID *>(graph_base_pointer));
  SetInDegreeBuffer(
      reinterpret_cast<VertexID *>(globalid_by_localid_base_pointer_) +
      metadata_.num_vertices);
  SetOutDegreeBuffer(reinterpret_cast<VertexID *>(indegree_base_pointer_) +
                     metadata_.num_vertices);
  SetInOffsetBuffer(reinterpret_cast<EdgeIndex *>(
      reinterpret_cast<VertexID *>(outdegree_base_pointer_) +
      metadata_.num_vertices));
  SetOutOffsetBuffer(reinterpret_cast<EdgeIndex *>(
      reinterpret_cast<VertexID *>(in_offset_base_pointer_) +
      metadata_.num_vertices + 1));
  SetIncomingEdgesBuffer(reinterpret_cast<VertexID *>(
      reinterpret_cast<EdgeIndex *>(out_offset_base_pointer_) +
      metadata_.num_vertices + 1));
  SetOutgoingEdgesBuffer(reinterpret_cast<VertexID *>(
      incoming_edges_base_pointer_ + metadata_.num_incoming_edges));
  SetEdgesGlobalIDBuffer(reinterpret_cast<VertexID *>(
      outgoing_edges_base_pointer_ + metadata_.num_outgoing_edges));
}

void ImmutableCSR::Write(const std::string &root_path, GraphID gid) {
  std::cout << "Write: " << root_path << std::endl;

  if (!std::filesystem::exists(root_path))
    std::filesystem::create_directory(root_path);
  if (!std::filesystem::exists(root_path + "graphs"))
    std::filesystem::create_directory(root_path + "graphs");
  if (!std::filesystem::exists(root_path + "label"))
    std::filesystem::create_directory(root_path + "label");

  std::ofstream out_meta_file(root_path + "meta.yaml");
  std::ofstream out_data_file(root_path + "graphs/" + std::to_string(gid) +
                              ".bin");

  // Write topology of graph.
  out_data_file.write(reinterpret_cast<char *>(GetGloablIDBasePointer()),
                      sizeof(VertexID) * get_num_vertices());
  out_data_file.write(reinterpret_cast<char *>(GetInDegreeBasePointer()),
                      sizeof(VertexID) * get_num_vertices());
  out_data_file.write(reinterpret_cast<char *>(GetOutDegreeBasePointer()),
                      sizeof(VertexID) * get_num_vertices());
  out_data_file.write(reinterpret_cast<char *>(GetInOffsetBasePointer()),
                      sizeof(EdgeIndex) * (get_num_vertices() + 1));
  out_data_file.write(reinterpret_cast<char *>(GetOutOffsetBasePointer()),
                      sizeof(EdgeIndex) * (get_num_vertices() + 1));
  out_data_file.write(reinterpret_cast<char *>(GetIncomingEdgesBasePointer()),
                      sizeof(VertexID) * get_num_incoming_edges());
  out_data_file.write(reinterpret_cast<char *>(GetOutgoingEdgesBasePointer()),
                      sizeof(VertexID) * get_num_outgoing_edges());
  out_data_file.write(reinterpret_cast<char *>(GetEdgesGloablIDBasePointer()),
                      sizeof(VertexID) * (get_max_vid() + 1));

  std::cout << "write size " <<

      sizeof(VertexID) * get_num_vertices() +
          sizeof(VertexID) * get_num_vertices() +
          sizeof(VertexID) * get_num_vertices() +
          sizeof(EdgeIndex) * (get_num_vertices() + 1) +
          sizeof(EdgeIndex) * (get_num_vertices() + 1) +
          sizeof(VertexID) * get_num_incoming_edges() +
          sizeof(VertexID) * get_num_outgoing_edges() +
          sizeof(VertexID) * (get_max_vid() + 1)

            << std::endl;

  std::cout << "global: " << sizeof(VertexID) * get_num_vertices() << std::endl;
  std::cout << "in degree: " << sizeof(VertexID) * get_num_vertices()
            << std::endl;
  std::cout << "out degree: " << sizeof(VertexID) * get_num_vertices()
            << std::endl;
  std::cout << "in offset " << sizeof(EdgeIndex) * (get_num_vertices() + 1)
            << std::endl;
  std::cout << "out offset" << sizeof(EdgeIndex) * (get_num_vertices() + 1)
            << std::endl;
  std::cout << "in edges: " << sizeof(VertexID) * get_num_outgoing_edges()
            << std::endl;
  std::cout << "out edges: " << sizeof(VertexID) * get_num_outgoing_edges()
            << std::endl;
  std::cout << "edges vid: " << sizeof(VertexID) * (get_max_vid() + 1)
            << std::endl;

  // Write label data with all 0.
  std::ofstream out_label_file(root_path + "label/" + std::to_string(gid) +
                               ".bin");
  auto buffer_label = GetVLabelBasePointer();
  out_label_file.write(reinterpret_cast<char *>(buffer_label),
                       sizeof(VertexLabel) * get_num_vertices());

  out_data_file.close();
  out_label_file.close();

  // Write graph metadata.
  std::vector<SubGraphMetadata> subgraph_metadata_vec;
  subgraph_metadata_vec.push_back(
      {gid, get_num_vertices(), get_num_incoming_edges(),
       get_num_outgoing_edges(), get_max_vid(), get_min_vid()});

  YAML::Node out_node;
  out_node["GraphMetadata"]["num_vertices"] = get_num_vertices();
  out_node["GraphMetadata"]["num_edges"] =
      get_num_incoming_edges() + get_num_outgoing_edges();
  out_node["GraphMetadata"]["max_vid"] = get_max_vid();
  out_node["GraphMetadata"]["min_vid"] = get_min_vid();
  out_node["GraphMetadata"]["count_border_vertices"] = 0;
  out_node["GraphMetadata"]["num_subgraphs"] = 1;
  out_node["GraphMetadata"]["subgraphs"] = subgraph_metadata_vec;
  out_meta_file << out_node << std::endl;
  std::cout << "Write Successfully" << std::endl;
  out_meta_file.close();
}

void ImmutableCSR::Read(const std::string &root_path) {
  std::cout << "Read: " << root_path << std::endl;

  std::cout << root_path + "meta.yaml" << std::endl;
  YAML::Node metadata_node;
  GraphMetadata metadata;
  try {
    metadata_node = YAML::LoadFile(root_path + "meta.yaml");
    metadata = metadata_node.as<GraphMetadata>();
  } catch (YAML::BadFile &e) {
    std::cout << "meta.yaml file read failed! " << e.msg << std::endl;
  }

  assert(metadata.num_subgraphs == 1);

  SetMaxVid(metadata.subgraphs[0].max_vid);
  SetMinVid(metadata.subgraphs[0].min_vid);
  SetNumIncomingEdges(metadata.subgraphs[0].num_incoming_edges);
  SetNumOutgoingEdges(metadata.subgraphs[0].num_outgoing_edges);
  SetNumVertices(metadata.subgraphs[0].num_vertices);

  std::ifstream data_file(root_path + "graphs/" + "0.bin", std::ios::binary);
  std::ifstream label_file(root_path + "label/" + "0.bin", std::ios::binary);

  if (!data_file)
    throw std::runtime_error("Error reading file: " + root_path + "graphs/" +
                             "0.bin");

  if (!label_file)
    throw std::runtime_error("Error reading file: " + root_path + "label/" +
                             "0.bin");

  data_file.seekg(0, std::ios::end);
  size_t file_size = data_file.tellg();
  data_file.seekg(0, std::ios::beg);

  graph_base_pointer_ = std::make_unique<uint8_t[]>(file_size);

  // Read the file data.
  data_file.read(reinterpret_cast<char *>(graph_base_pointer_.get()),
                 file_size);

  SetGlobalIDBuffer(reinterpret_cast<VertexID *>(graph_base_pointer_.get()));
  SetInDegreeBuffer(
      reinterpret_cast<VertexID *>(globalid_by_localid_base_pointer_) +
      metadata_.num_vertices);
  SetOutDegreeBuffer(reinterpret_cast<VertexID *>(indegree_base_pointer_) +
                     metadata_.num_vertices);
  SetInOffsetBuffer(reinterpret_cast<EdgeIndex *>(
      reinterpret_cast<VertexID *>(outdegree_base_pointer_) +
      metadata_.num_vertices));
  SetOutOffsetBuffer(reinterpret_cast<EdgeIndex *>(
      reinterpret_cast<EdgeIndex *>(in_offset_base_pointer_) +
      (metadata_.num_vertices + 1)));

  SetIncomingEdgesBuffer(reinterpret_cast<VertexID *>(
      reinterpret_cast<EdgeIndex *>(out_offset_base_pointer_) +
      (metadata_.num_vertices + 1)));
  SetOutgoingEdgesBuffer(reinterpret_cast<VertexID *>(
      incoming_edges_base_pointer_ + metadata_.num_incoming_edges));
  SetEdgesGlobalIDBuffer(reinterpret_cast<VertexID *>(
      outgoing_edges_base_pointer_ + metadata_.num_outgoing_edges));

  auto ptr = GetEdgesGloablIDBasePointer();

  label_file.seekg(0, std::ios::end);
  file_size = label_file.tellg();
  label_file.seekg(0, std::ios::beg);
  vertex_label_base_pointer_ = std::make_unique<VertexLabel[]>(file_size);

  // Read the label.
  label_file.read(reinterpret_cast<char *>(GetVLabelBasePointer()), file_size);
  std::cout << "Read Successfully" << std::endl;

  data_file.close();
}

ImmutableCSRVertex ImmutableCSR::GetVertexByLocalID(VertexID i) const {
  ImmutableCSRVertex v;
  v.vid = globalid_by_localid_base_pointer_[i];
  v.vlabel = vertex_label_base_pointer_.get()[i];
  if (get_num_incoming_edges() > 0) {
    v.indegree = GetInDegreeByLocalID(i);
    v.incoming_edges = incoming_edges_base_pointer_ + GetInOffsetByLocalID(i);
  }
  if (get_num_outgoing_edges() > 0) {
    v.outdegree = GetOutDegreeByLocalID(i);
    v.outgoing_edges = outgoing_edges_base_pointer_ + GetOutOffsetByLocalID(i);
  }
  return v;
}

void ImmutableCSR::GenerateVLabel(VertexID range) {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<> dis(0, range);

  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [this, step, &dis, &gen](auto w) {
                  for (auto vid = w; vid < get_num_vertices(); vid += step) {
                    auto vlabel_ptr = GetVLabelBasePointer();
                    vlabel_ptr[vid] = dis(gen);
                  }
                });
}

void ImmutableCSR::SortByDegree() {
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  auto n_vertices = get_num_vertices();
  VidCountPair *vids_and_degrees = new VidCountPair[n_vertices]();

  std::cout << "[SortByDegree] Computing degree of each vertex" << std::endl;
  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [this, step, &vids_and_degrees, n_vertices](auto w) {
                  for (auto vid = w; vid < n_vertices; vid += step) {
                    vids_and_degrees[vid].vid = vid;
                    vids_and_degrees[vid].count = GetOutDegreeByLocalID(vid);
                  }
                });

  std::cout << "[SortByDegree] Sorting" << std::endl;
  std::sort(std::execution::par, vids_and_degrees,
            vids_and_degrees + n_vertices,
            [](const auto a, const auto b) { return a.count > b.count; });

  auto new_buffer_globalid = new VertexID[get_num_vertices()]();
  auto new_buffer_indegree = new VertexID[get_num_vertices()]();
  auto new_buffer_outdegree = new VertexID[get_num_vertices()]();
  auto new_buffer_in_offset = new EdgeIndex[get_num_vertices() + 1]();
  auto new_buffer_out_offset = new EdgeIndex[get_num_vertices() + 1]();
  auto new_buffer_in_edges = new VertexID[get_num_incoming_edges()]();
  auto new_buffer_out_edges = new VertexID[get_num_outgoing_edges()]();

  std::cout << "[SortByDegree] Computing offset" << std::endl;
  for (VertexID i = 0; i < n_vertices; i++) {
    auto local_vid = vids_and_degrees[i].vid;
    new_buffer_out_offset[i + 1] =
        new_buffer_out_offset[i] + GetOutDegreeByLocalID(local_vid);
    new_buffer_in_offset[i + 1] =
        new_buffer_in_offset[i] + GetInDegreeByLocalID(local_vid);
  }

  auto *new_id_by_old_id = new VertexID[n_vertices]();

  metadata_.max_vid = 0;
  metadata_.min_vid = common::kMaxVertexID;

  std::cout << "[SortByDegree] Replacing old val by new val." << std::endl;
  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [this, step, n_vertices, &new_id_by_old_id, &vids_and_degrees,
                 &new_buffer_globalid, &new_buffer_indegree,
                 &new_buffer_outdegree, &new_buffer_in_offset,
                 &new_buffer_out_offset, &new_buffer_in_edges,
                 &new_buffer_out_edges](auto w) {
                  for (VertexID i = w; i < n_vertices; i += step) {
                    auto local_vid = vids_and_degrees[i].vid;
                    new_id_by_old_id[local_vid] = i;
                    new_buffer_globalid[i] = GetGlobalIDByLocalID(local_vid);
                    new_buffer_indegree[i] = GetInDegreeByLocalID(local_vid);
                    new_buffer_outdegree[i] = GetOutDegreeByLocalID(local_vid);

                    WriteMax(&metadata_.max_vid, new_buffer_globalid[i]);
                    WriteMin(&metadata_.min_vid, new_buffer_globalid[i]);

                    auto in_edge_ptr = GetIncomingEdgesBasePointer();
                    auto out_edge_ptr = GetOutgoingEdgesBasePointer();
                    memcpy(new_buffer_out_edges + new_buffer_out_offset[i],
                           out_edge_ptr + GetOutOffsetByLocalID(local_vid),
                           sizeof(VertexID) * new_buffer_outdegree[i]);
                    memcpy(new_buffer_in_edges + new_buffer_in_offset[i],
                           in_edge_ptr + GetInOffsetByLocalID(local_vid),
                           sizeof(VertexID) * new_buffer_indegree[i]);
                  }
                });

  // re-assign id for each vertex.
  std::cout << "[SortByDegree] Reassigning id for each vertex" << std::endl;
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, step, n_vertices, &new_id_by_old_id, &vids_and_degrees,
       &new_buffer_globalid, &new_buffer_indegree, &new_buffer_outdegree,
       &new_buffer_in_offset, &new_buffer_out_offset, &new_buffer_in_edges,
       &new_buffer_out_edges](auto w) {
        for (auto i = w; i < n_vertices; i += step) {
          new_buffer_out_edges[i] = new_id_by_old_id[new_buffer_out_edges[i]];
          new_buffer_in_edges[i] = new_id_by_old_id[new_buffer_in_edges[i]];
        }
      });

  SetGlobalIDBuffer(new_buffer_globalid);
  SetIncomingEdgesBuffer(new_buffer_in_edges);
  SetOutgoingEdgesBuffer(new_buffer_out_edges);
  SetInDegreeBuffer(new_buffer_indegree);
  SetOutDegreeBuffer(new_buffer_outdegree);
  SetInOffsetBuffer(new_buffer_in_offset);
  SetOutOffsetBuffer(new_buffer_out_offset);
  delete[] new_id_by_old_id;
  std::cout << "[SortByDegree] Done!" << std::endl;
}

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics