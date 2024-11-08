#include "core/data_structures/immutable_csr.cuh"

#include <algorithm>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_map>

#ifdef TBB_FOUND
#include <execution>
#endif

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
       << ", outdegree: " << u.outdegree << "===" << std::endl;
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
                      sizeof(VertexID) * get_num_outgoing_edges());
  out_data_file.write(reinterpret_cast<char *>(GetOutgoingEdgesBasePointer()),
                      sizeof(VertexID) * get_num_outgoing_edges());

  // Write label data with all 0.
  std::ofstream out_label_file(root_path + "label/" + std::to_string(gid) +
                               ".bin");
  auto buffer_label = new VertexLabel[get_num_vertices()]();
  out_label_file.write(reinterpret_cast<char *>(buffer_label),
                       sizeof(VertexLabel) * get_num_vertices());
  delete[] buffer_label;

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

  data_file.seekg(0, std::ios::end);
  size_t file_size = data_file.tellg();
  data_file.seekg(0, std::ios::beg);

  graph_base_pointer_ = std::make_unique<uint8_t[]>(file_size);

  // Read the file data.
  data_file.read(reinterpret_cast<char *>(graph_base_pointer_.get()),
                 file_size);

  if (!data_file)
    throw std::runtime_error("Error reading file: " + root_path + "graphs/" +
                             "0.bin");

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
      reinterpret_cast<VertexID *>(in_offset_base_pointer_) +
      (metadata_.num_vertices + 1)));
  SetIncomingEdgesBuffer(reinterpret_cast<VertexID *>(
      reinterpret_cast<EdgeIndex *>(out_offset_base_pointer_) +
      (metadata_.num_vertices + 1)));
  SetOutgoingEdgesBuffer(reinterpret_cast<VertexID *>(
      incoming_edges_base_pointer_ + metadata_.num_incoming_edges));
  SetEdgesGlobalIDBuffer(reinterpret_cast<VertexID *>(
      outgoing_edges_base_pointer_ + metadata_.num_outgoing_edges));

  label_file.seekg(0, std::ios::end);
  file_size = label_file.tellg();
  label_file.seekg(0, std::ios::beg);
  vertex_label_base_pointer_ = std::make_unique<VertexLabel[]>(file_size);

  // Read the label.
  label_file.read(reinterpret_cast<char *>(vertex_label_base_pointer_.get()),
                  file_size);
  std::cout << "Read Successfully" << std::endl;
  data_file.close();
}

ImmutableCSRVertex ImmutableCSR::GetVertexByLocalID(VertexID i) const {
  ImmutableCSRVertex v;
  v.vid = globalid_by_localid_base_pointer_[i];
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
  auto new_buffer_in_offset = new EdgeIndex[get_num_vertices()]();
  auto new_buffer_out_offset = new EdgeIndex[get_num_vertices()]();
  auto new_buffer_in_edges = new VertexID[get_num_incoming_edges()]();
  auto new_buffer_out_edges = new VertexID[get_num_outgoing_edges()]();

  std::cout << "[SortByDegree] Computing offset" << std::endl;
  for (VertexID i = 0; i < n_vertices - 1; i++) {
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

void ImmutableCSR::SortByDistance(VertexID sim_granularity) {
  std::cout << "SortDistance" << std::endl;
  auto parallelism = std::thread::hardware_concurrency();
  std::vector<size_t> worker(parallelism);
  std::mutex mtx;
  std::iota(worker.begin(), worker.end(), 0);
  auto step = worker.size();

  auto n_vertices = get_num_vertices();

  auto n_block = ceil(get_num_vertices() / (float)sim_granularity);

  VidCountPair *vids_and_out_degrees = new VidCountPair[n_vertices];
  VidCountPair *vids_and_in_degrees = new VidCountPair[n_vertices];
  VidCountPair *vids_and_degrees = new VidCountPair[n_vertices];

  VidCountPair *vids_and_overlapped_count = new VidCountPair[n_vertices];
  VidCountPair *in_vids_and_overlapped_count = new VidCountPair[n_vertices];
  VidCountPair *out_vids_and_overlapped_count = new VidCountPair[n_vertices];

  util::Bitmap k_out_degree(n_vertices);
  util::Bitmap k_in_degree(n_vertices);
  util::Bitmap k_degree(n_vertices);

  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [this, step, &vids_and_out_degrees, &vids_and_in_degrees,
                 &vids_and_degrees, n_vertices, &k_out_degree, &k_in_degree,
                 &k_degree](auto w) {
                  for (auto vid = w; vid < n_vertices; vid += step) {
                    vids_and_out_degrees[vid].vid = vid;
                    vids_and_out_degrees[vid].count =
                        GetOutDegreeByLocalID(vid);
                    vids_and_in_degrees[vid].vid = vid;
                    vids_and_in_degrees[vid].count = GetInDegreeByLocalID(vid);
                    vids_and_degrees[vid].vid = vid;
                    vids_and_degrees[vid].count =
                        GetInDegreeByLocalID(vid) + GetOutDegreeByLocalID(vid);
                    k_out_degree.SetBit(GetOutDegreeByLocalID(vid));
                    k_in_degree.SetBit(GetInDegreeByLocalID(vid));
                    k_degree.SetBit(GetInDegreeByLocalID(vid));
                    k_degree.SetBit(GetOutDegreeByLocalID(vid));
                  }
                });

  // const int n_pivot = k_degree.Count();
  const int n_pivot = k_degree.Count();
  const int n_out_pivot = k_out_degree.Count();
  const int n_in_pivot = k_in_degree.Count();

  std::cout << k_degree.Count() << std::endl;

  VertexID scope_upper_bound[k_degree.Count()];
  VertexID scope_lower_bound[k_degree.Count()];

  VertexID scope_out_upper_bound[k_out_degree.Count()];
  VertexID scope_out_lower_bound[k_out_degree.Count()];

  VertexID scope_in_upper_bound[k_in_degree.Count()];
  VertexID scope_in_lower_bound[k_in_degree.Count()];

  memset(scope_upper_bound, 0, sizeof(VertexID) * n_pivot);
  memset(scope_out_upper_bound, 0, sizeof(VertexID) * n_out_pivot);
  memset(scope_in_upper_bound, 0, sizeof(VertexID) * n_in_pivot);

  // for (auto i = 0; i < k_out_degree.Count(); ++i)
  //   scope_lower_bound[i] = MAX_VERTEX_ID;
  for (auto i = 0; i < n_pivot; ++i)
    scope_lower_bound[i] = MAX_VERTEX_ID;
  for (auto i = 0; i < n_in_pivot; ++i)
    scope_in_lower_bound[i] = MAX_VERTEX_ID;
  for (auto i = 0; i < n_out_pivot; ++i)
    scope_out_lower_bound[i] = MAX_VERTEX_ID;

  std::sort(vids_and_degrees, vids_and_degrees + n_vertices);
  std::sort(vids_and_out_degrees, vids_and_out_degrees + n_vertices);
  std::sort(vids_and_in_degrees, vids_and_in_degrees + n_vertices);

  /// Compute the scope of each pivot
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, step, &vids_and_degrees, &vids_and_out_degrees,
       &vids_and_in_degrees, n_vertices, &k_out_degree, &k_in_degree, &k_degree,
       &scope_lower_bound, &scope_upper_bound, &scope_out_lower_bound,
       &scope_out_upper_bound, &scope_in_lower_bound,
       &scope_in_upper_bound](auto w) {
        for (auto idx = w; idx < n_vertices; idx += step) {
          auto pivot_idx =
              k_degree.PreElementCount(vids_and_degrees[idx].count);
          util::atomic::WriteMax(&scope_upper_bound[pivot_idx], (VertexID)idx);
          util::atomic::WriteMin(&scope_lower_bound[pivot_idx], (VertexID)idx);

          pivot_idx =
              k_out_degree.PreElementCount(vids_and_out_degrees[idx].count);
          util::atomic::WriteMax(&scope_out_upper_bound[pivot_idx],
                                 (VertexID)idx);
          util::atomic::WriteMin(&scope_out_lower_bound[pivot_idx],
                                 (VertexID)idx);
          pivot_idx =
              k_in_degree.PreElementCount(vids_and_in_degrees[idx].count);
          util::atomic::WriteMax(&scope_in_upper_bound[pivot_idx],
                                 (VertexID)idx);
          util::atomic::WriteMin(&scope_in_lower_bound[pivot_idx],
                                 (VertexID)idx);
        }
      });

  // Compute the pivot bitmap.
  std::vector<util::Bitmap> overlap_bm_vec;
  std::vector<util::Bitmap> overlap_out_bm_vec;
  std::vector<util::Bitmap> overlap_in_bm_vec;

  overlap_bm_vec.resize(n_pivot, n_vertices);
  overlap_out_bm_vec.resize(n_out_pivot, n_vertices);
  overlap_in_bm_vec.resize(n_in_pivot, n_vertices);

  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, step, sim_granularity, n_out_pivot, n_in_pivot, n_pivot,
       &vids_and_degrees, &vids_and_out_degrees, &vids_and_in_degrees,
       n_vertices, &k_out_degree, &k_in_degree, &k_degree, &scope_lower_bound,
       &scope_upper_bound, &scope_in_lower_bound, &scope_in_upper_bound,
       &scope_out_lower_bound, &scope_out_upper_bound, &overlap_out_bm_vec,
       &overlap_in_bm_vec, &overlap_bm_vec](auto w) {
        for (auto pivot_idx = w; pivot_idx < n_out_pivot; pivot_idx += step) {
          auto vertex_pivot = (scope_out_upper_bound[pivot_idx] -
                               scope_out_lower_bound[pivot_idx]) /
                                  2 +
                              scope_out_lower_bound[pivot_idx];

          auto v = GetVertexByLocalID(vids_and_out_degrees[vertex_pivot].vid);

          for (auto nbr_i = 0; nbr_i < v.outdegree; nbr_i++) {
            overlap_out_bm_vec[pivot_idx].SetBit(v.outgoing_edges[nbr_i]);
          }
        }
        for (auto pivot_idx = w; pivot_idx < n_in_pivot; pivot_idx += step) {
          auto vertex_pivot = (scope_in_upper_bound[pivot_idx] -
                               scope_in_lower_bound[pivot_idx]) /
                                  2 +
                              scope_in_lower_bound[pivot_idx];

          auto v = GetVertexByLocalID(vids_and_in_degrees[vertex_pivot].vid);

          for (auto nbr_i = 0; nbr_i < v.indegree; nbr_i++) {
            overlap_in_bm_vec[pivot_idx].SetBit(v.incoming_edges[nbr_i]);
          }
        }
        for (auto pivot_idx = w; pivot_idx < n_pivot; pivot_idx += step) {
          auto vertex_pivot =
              (scope_upper_bound[pivot_idx] - scope_lower_bound[pivot_idx]) /
                  2 +
              scope_in_lower_bound[pivot_idx];

          auto v = GetVertexByLocalID(vids_and_degrees[vertex_pivot].vid);

          for (auto nbr_i = 0; nbr_i < v.indegree; nbr_i++) {
            overlap_in_bm_vec[pivot_idx].SetBit(v.incoming_edges[nbr_i]);
          }
          for (auto nbr_i = 0; nbr_i < v.outdegree; nbr_i++) {
            overlap_out_bm_vec[pivot_idx].SetBit(v.outgoing_edges[nbr_i]);
          }
        }
      });

  // Get the overlap between each vertex and pivot.
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, step, sim_granularity, &vids_and_degrees, &vids_and_out_degrees,
       &vids_and_in_degrees, n_vertices, &k_out_degree,
       &vids_and_overlapped_count, &out_vids_and_overlapped_count,
       &in_vids_and_overlapped_count, &scope_out_lower_bound,
       &scope_out_upper_bound, &scope_in_lower_bound, &scope_in_upper_bound,
       &scope_lower_bound, &scope_upper_bound, &overlap_bm_vec,
       &overlap_out_bm_vec, &overlap_in_bm_vec](auto w) {
        for (auto idx = w; idx < n_vertices; idx += step) {
          auto pivot_idx =
              k_out_degree.PreElementCount(vids_and_out_degrees[idx].count);
          util::Bitmap tmp_bm(overlap_out_bm_vec[pivot_idx]);

          auto v = GetVertexByLocalID(vids_and_out_degrees[idx].vid);

          out_vids_and_overlapped_count[idx].vid = v.vid;
          for (auto nbr_i = 0; nbr_i < v.outdegree; nbr_i++) {
            if (tmp_bm.GetBit(v.outgoing_edges[nbr_i])) {
              out_vids_and_overlapped_count[idx].count++;
            }
          }
        }
      });

  for (size_t pivot_idx = 0; pivot_idx < k_out_degree.Count(); pivot_idx++) {
    std::sort(out_vids_and_overlapped_count + scope_out_lower_bound[pivot_idx],
              out_vids_and_overlapped_count + scope_out_upper_bound[pivot_idx]);
  }

  auto new_buffer_globalid = new VertexID[get_num_vertices()]();
  auto new_buffer_indegree = new VertexID[get_num_vertices()]();
  auto new_buffer_outdegree = new VertexID[get_num_vertices()]();
  auto new_buffer_in_offset = new EdgeIndex[get_num_vertices()]();
  auto new_buffer_out_offset = new EdgeIndex[get_num_vertices()]();
  auto new_buffer_in_edges = new VertexID[get_num_incoming_edges()]();
  auto new_buffer_out_edges = new VertexID[get_num_outgoing_edges()]();

  // Construct Graph.
  VertexID *vid_map = new VertexID[get_num_vertices()]();
  std::for_each(
      std::execution::par, worker.begin(), worker.end(),
      [this, step, n_vertices, &vids_and_degrees, &vids_and_out_degrees,
       &vids_and_in_degrees, &vids_and_overlapped_count, &vid_map](auto w) {
        for (auto new_vid = w; new_vid < n_vertices; new_vid += step) {
          auto vid = vids_and_degrees[new_vid].vid;
          vid_map[vid] = new_vid;
        }
      });

  for (VertexID i = 0; i < n_vertices - 1; i++) {
    auto local_vid = vids_and_degrees[i].vid;
    new_buffer_out_offset[i + 1] =
        new_buffer_out_offset[i] + GetOutDegreeByLocalID(local_vid);
    new_buffer_in_offset[i + 1] =
        new_buffer_in_offset[i] + GetInDegreeByLocalID(local_vid);
  }

  metadata_.max_vid = 0;
  metadata_.min_vid = common::kMaxVertexID;

  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [this, step, n_vertices, &vids_and_overlapped_count,
                 &vids_and_degrees, &vids_and_out_degrees, &vids_and_in_degrees,
                 &new_buffer_globalid, &new_buffer_indegree,
                 &new_buffer_outdegree, &new_buffer_in_offset,
                 &new_buffer_out_offset, &new_buffer_in_edges,
                 &new_buffer_out_edges](auto w) {
                  for (VertexID i = w; i < n_vertices; i += step) {
                    auto local_vid = vids_and_degrees[i].vid;
                    new_buffer_globalid[i] = i;
                    new_buffer_indegree[i] = GetInDegreeByLocalID(local_vid);
                    new_buffer_outdegree[i] = GetOutDegreeByLocalID(local_vid);

                    WriteMax(&metadata_.max_vid, i);
                    WriteMin(&metadata_.min_vid, i);

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

  // Reassign vertex id to each vertex.

  auto n_outgoing_edges = get_num_outgoing_edges();
  std::for_each(std::execution::par, worker.begin(), worker.end(),
                [this, step, n_outgoing_edges, &vid_map, &new_buffer_in_edges,
                 &new_buffer_out_edges](auto w) {
                  for (auto i = w; i < n_outgoing_edges; i += step) {
                    new_buffer_out_edges[i] = vid_map[new_buffer_out_edges[i]];
                    new_buffer_in_edges[i] = vid_map[new_buffer_in_edges[i]];
                  }
                });

  delete[] globalid_by_localid_base_pointer_;
  SetGlobalIDBuffer(new_buffer_globalid);
  delete[] incoming_edges_base_pointer_;
  SetIncomingEdgesBuffer(new_buffer_in_edges);
  delete[] outgoing_edges_base_pointer_;
  SetOutgoingEdgesBuffer(new_buffer_out_edges);
  delete[] indegree_base_pointer_;
  SetInDegreeBuffer(new_buffer_indegree);
  delete[] outdegree_base_pointer_;
  SetOutDegreeBuffer(new_buffer_outdegree);
  delete[] in_offset_base_pointer_;
  SetInOffsetBuffer(new_buffer_in_offset);
  delete[] out_offset_base_pointer_;
  SetOutOffsetBuffer(new_buffer_out_offset);

  delete[] vid_map;
  delete[] vids_and_out_degrees;
  delete[] vids_and_in_degrees;
  delete[] vids_and_degrees;
  delete[] vids_and_overlapped_count;
  delete[] in_vids_and_overlapped_count;
  delete[] out_vids_and_overlapped_count;
}

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics