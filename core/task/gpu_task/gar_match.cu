#include "core/task/gpu_task/gar_match.cuh"
#include "core/task/gpu_task/kernel/kernel_gar_match.cuh"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using GARMatchKernelWrapper =
    sics::matrixgraph::core::task::kernel::GARMatchKernelWrapper;

namespace {

std::string EscapeShellSingleQuotes(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out += c;
    }
  }
  return out;
}

std::string ExecCommand(const std::string& command) {
  std::array<char, 256> buffer{};
  std::string output;
  FILE* pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) {
    return output;
  }
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) !=
         nullptr) {
    output += buffer.data();
  }
  pclose(pipe);
  return output;
}

bool ExtractIntField(const std::string& json, const std::string& field_name,
                     int* out) {
  if (out == nullptr) return false;
  const std::string key = "\"" + field_name + "\":";
  auto pos = json.find(key);
  if (pos == std::string::npos) return false;
  pos += key.size();
  while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos])))
    ++pos;
  size_t end = pos;
  while (end < json.size() && std::isdigit(static_cast<unsigned char>(json[end])))
    ++end;
  if (end == pos) return false;
  *out = std::stoi(json.substr(pos, end - pos));
  return true;
}

}  // namespace

__host__ int GARMatch::Run(
    const uint32_t* g_v_id, const int32_t* g_v_label_idx, int g_n_vertices,
    const uint32_t* g_e_src, const uint32_t* g_e_dst, const uint32_t* g_e_id,
    const int32_t* g_e_label_idx, int g_n_edges,
    const int32_t* p_node_label_idx, int p_n_nodes, const int32_t* p_edge_src,
    const int32_t* p_edge_dst, const int32_t* p_edge_label_idx, int p_n_edges,
    int* out_num_conditions, uint32_t* out_row_pivot_id,
    int32_t* out_row_cond_j, int32_t* out_row_pos, int32_t* out_row_offset,
    int32_t* out_row_count, int out_row_capacity, int* out_row_size,
    uint32_t* out_matched_v_ids, int out_match_capacity, int* out_match_size) {
  std::cout << "[GARMatch] Run() ..." << std::endl;
  GARGraphArrays g{
      .v_id = g_v_id,
      .v_label_idx = g_v_label_idx,
      .n_vertices = g_n_vertices,
      .e_src = g_e_src,
      .e_dst = g_e_dst,
      .e_id = g_e_id,
      .e_label_idx = g_e_label_idx,
      .n_edges = g_n_edges,
  };
  GARPatternArrays p{
      .node_label_idx = p_node_label_idx,
      .n_nodes = p_n_nodes,
      .edge_src = p_edge_src,
      .edge_dst = p_edge_dst,
      .edge_label_idx = p_edge_label_idx,
      .n_edges = p_n_edges,
  };
  GARMatchArrays out{
      .num_conditions = out_num_conditions,
      .row_pivot_id = out_row_pivot_id,
      .row_cond_j = out_row_cond_j,
      .row_pos = out_row_pos,
      .row_offset = out_row_offset,
      .row_count = out_row_count,
      .row_capacity = out_row_capacity,
      .row_size = out_row_size,
      .matched_v_ids = out_matched_v_ids,
      .match_capacity = out_match_capacity,
      .match_size = out_match_size,
  };
  GARMatchKernelWrapper::GARMatch(g, p, &out);

  return 0;
}

__host__ void GARMatch::LoadData() {
  std::cout << "[GARMatch] LoadData() ..." << std::endl;
  ResetOwnedBuffers();
  YAML::Node cfg = YAML::LoadFile(config_path_);

  const std::string scheme = cfg["arango"]["scheme"].as<std::string>("http");
  const std::string host = cfg["arango"]["host"].as<std::string>("127.0.0.1");
  const int port = cfg["arango"]["port"].as<int>(8529);
  const std::string db = cfg["arango"]["database"].as<std::string>("_system");
  const std::string user = cfg["arango"]["username"].as<std::string>("root");
  const std::string pass = cfg["arango"]["password"].as<std::string>("");
  const std::string pivot_graphs_collection =
      cfg["collections"]["pivot_graphs"].as<std::string>("pivot_graphs");

  const std::string auth_url = scheme + "://" + user + ":" + pass + "@" + host +
                               ":" + std::to_string(port);
  const std::string base_db_url =
      auth_url + "/_db/" + db + "/_api";

  const std::string ping_cmd = "curl -s '" + EscapeShellSingleQuotes(auth_url) +
                               "/_api/version'";
  const std::string ping_resp = ExecCommand(ping_cmd);
  if (ping_resp.find("\"server\":\"arango\"") == std::string::npos) {
    throw std::runtime_error(
        "Failed to connect to ArangoDB with config: " + config_path_);
  }
  std::cout << "[GARMatch] ArangoDB connected: " << host << ":" << port
            << ", db=" << db << std::endl;

  const std::string count_cmd =
      "curl -s '" + EscapeShellSingleQuotes(base_db_url + "/collection/" +
                                            pivot_graphs_collection + "/count") +
      "'";
  const std::string count_resp = ExecCommand(count_cmd);
  int pivot_graph_count = 0;
  if (!ExtractIntField(count_resp, "count", &pivot_graph_count)) {
    std::cout << "[GARMatch] Warning: cannot parse pivot_graph count, use fallback."
              << std::endl;
    pivot_graph_count = 1;
  }
  std::cout << "[GARMatch] pivot_graphs count: " << pivot_graph_count
            << std::endl;

  // Build a small graph bootstrap from DB metadata/count for now.
  // This keeps the pipeline runnable before full JSON payload parsing.
  owned_g_.n_vertices = std::max(2, std::min(pivot_graph_count + 1, 64));
  owned_g_.n_edges = owned_g_.n_vertices - 1;
  owned_g_.v_id = std::make_unique<uint32_t[]>(owned_g_.n_vertices);
  owned_g_.v_label_idx = std::make_unique<int32_t[]>(owned_g_.n_vertices);
  owned_g_.e_src = std::make_unique<uint32_t[]>(owned_g_.n_edges);
  owned_g_.e_dst = std::make_unique<uint32_t[]>(owned_g_.n_edges);
  owned_g_.e_id = std::make_unique<uint32_t[]>(owned_g_.n_edges);
  owned_g_.e_label_idx = std::make_unique<int32_t[]>(owned_g_.n_edges);
  std::fill_n(owned_g_.v_label_idx.get(), owned_g_.n_vertices, 0);
  std::fill_n(owned_g_.e_label_idx.get(), owned_g_.n_edges, 0);
  for (int i = 0; i < owned_g_.n_vertices; ++i) {
    owned_g_.v_id[i] = static_cast<uint32_t>(i);
  }
  for (int e = 0; e < owned_g_.n_edges; ++e) {
    owned_g_.e_src[e] = static_cast<uint32_t>(e);
    owned_g_.e_dst[e] = static_cast<uint32_t>(e + 1);
    owned_g_.e_id[e] = static_cast<uint32_t>(e);
  }
  g_ = GARGraphArrays{
      .v_id = owned_g_.v_id.get(),
      .v_label_idx = owned_g_.v_label_idx.get(),
      .n_vertices = owned_g_.n_vertices,
      .e_src = owned_g_.e_src.get(),
      .e_dst = owned_g_.e_dst.get(),
      .e_id = owned_g_.e_id.get(),
      .e_label_idx = owned_g_.e_label_idx.get(),
      .n_edges = owned_g_.n_edges,
  };

  // Generate a simple pattern: node(0) -> node(1)
  owned_p_.n_nodes = 2;
  owned_p_.n_edges = 1;
  owned_p_.node_label_idx = std::make_unique<int32_t[]>(owned_p_.n_nodes);
  owned_p_.edge_src = std::make_unique<int32_t[]>(owned_p_.n_edges);
  owned_p_.edge_dst = std::make_unique<int32_t[]>(owned_p_.n_edges);
  owned_p_.edge_label_idx = std::make_unique<int32_t[]>(owned_p_.n_edges);
  owned_p_.node_label_idx[0] = 0;
  owned_p_.node_label_idx[1] = 0;
  owned_p_.edge_src[0] = 0;
  owned_p_.edge_dst[0] = 1;
  owned_p_.edge_label_idx[0] = 0;
  gar_pattern_arrays_ = GARPatternArrays{
      .node_label_idx = owned_p_.node_label_idx.get(),
      .n_nodes = owned_p_.n_nodes,
      .edge_src = owned_p_.edge_src.get(),
      .edge_dst = owned_p_.edge_dst.get(),
      .edge_label_idx = owned_p_.edge_label_idx.get(),
      .n_edges = owned_p_.n_edges,
  };
  p_ = gar_pattern_arrays_;

  // Allocate GARMatch output arrays owned by this class.
  owned_out_.row_capacity = 4096;
  owned_out_.match_capacity = 16384;
  owned_out_.row_pivot_id =
      std::make_unique<uint32_t[]>(owned_out_.row_capacity);
  owned_out_.row_cond_j = std::make_unique<int32_t[]>(owned_out_.row_capacity);
  owned_out_.row_pos = std::make_unique<int32_t[]>(owned_out_.row_capacity);
  owned_out_.row_offset = std::make_unique<int32_t[]>(owned_out_.row_capacity);
  owned_out_.row_count = std::make_unique<int32_t[]>(owned_out_.row_capacity);
  owned_out_.matched_v_ids =
      std::make_unique<uint32_t[]>(owned_out_.match_capacity);
  std::fill_n(owned_out_.row_pivot_id.get(), owned_out_.row_capacity, 0);
  std::fill_n(owned_out_.row_cond_j.get(), owned_out_.row_capacity, 0);
  std::fill_n(owned_out_.row_pos.get(), owned_out_.row_capacity, 0);
  std::fill_n(owned_out_.row_offset.get(), owned_out_.row_capacity, 0);
  std::fill_n(owned_out_.row_count.get(), owned_out_.row_capacity, 0);
  std::fill_n(owned_out_.matched_v_ids.get(), owned_out_.match_capacity, 0);
  owned_out_.num_conditions = 0;
  owned_out_.row_size = 0;
  owned_out_.match_size = 0;

  gar_match_arrays_ = GARMatchArrays{
      .num_conditions = &owned_out_.num_conditions,
      .row_pivot_id = owned_out_.row_pivot_id.get(),
      .row_cond_j = owned_out_.row_cond_j.get(),
      .row_pos = owned_out_.row_pos.get(),
      .row_offset = owned_out_.row_offset.get(),
      .row_count = owned_out_.row_count.get(),
      .row_capacity = owned_out_.row_capacity,
      .row_size = &owned_out_.row_size,
      .matched_v_ids = owned_out_.matched_v_ids.get(),
      .match_capacity = owned_out_.match_capacity,
      .match_size = &owned_out_.match_size,
  };
  out_ = &gar_match_arrays_;
}

__host__ void GARMatch::ResetOwnedBuffers() {
  owned_g_ = OwnedGraphBuffers{};
  owned_p_ = OwnedPatternBuffers{};
  owned_out_ = OwnedMatchBuffers{};

  g_ = GARGraphArrays{};
  p_ = GARPatternArrays{};
  gar_pattern_arrays_ = GARPatternArrays{};
  gar_match_arrays_ = GARMatchArrays{};
  out_ = nullptr;
}

__host__ void GARMatch::Run() {
  std::cout << "[GARMatch] Run() ..." << std::endl;
  if (out_ != nullptr) {
    status_ = GARMatchKernelWrapper::GARMatch(g_, p_, out_);
    return;
  }
  std::cout << "[GARMatch] config_path: " << config_path_ << std::endl;
  std::cout << "[GARMatch] output_path: " << output_path_ << std::endl;
  LoadData();
  status_ = GARMatchKernelWrapper::GARMatch(g_, p_, &gar_match_arrays_);
  std::cout << "[GARMatch] result row_size=" << owned_out_.row_size
            << ", match_size=" << owned_out_.match_size << std::endl;
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
