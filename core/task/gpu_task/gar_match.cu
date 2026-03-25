#include "core/task/gpu_task/gar_match.cuh"
#include "core/task/gpu_task/kernel/kernel_gar_match.cuh"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
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

bool CommandExists(const std::string& command_name) {
  const std::string cmd = "command -v " + command_name + " >/dev/null 2>&1; echo $?";
  const std::string out = ExecCommand(cmd);
  return !out.empty() && out[0] == '0';
}

std::string EscapeForJSDoubleQuote(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    if (c == '\\') {
      out += "\\\\";
    } else if (c == '"') {
      out += "\\\"";
    } else if (c == '\n') {
      out += "\\n";
    } else if (c == '\r') {
      out += "\\r";
    } else if (c == '\t') {
      out += "\\t";
    } else {
      out += c;
    }
  }
  return out;
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

uint32_t ParseVertexIDOrFallback(const std::string& s, uint32_t fallback) {
  if (s.empty()) return fallback;
  uint64_t v = 0;
  for (char c : s) {
    if (!std::isdigit(static_cast<unsigned char>(c))) return fallback;
    v = v * 10 + static_cast<uint64_t>(c - '0');
    if (v > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
      return fallback;
  }
  return static_cast<uint32_t>(v);
}

uint32_t ParseVertexIDStable(const std::string& s) {
  if (s.empty()) return 0u;
  bool all_digit = true;
  for (char c : s) {
    if (!std::isdigit(static_cast<unsigned char>(c))) {
      all_digit = false;
      break;
    }
  }
  if (all_digit) return ParseVertexIDOrFallback(s, 0u);

  // FNV-1a 32-bit for stable non-numeric id mapping.
  uint32_t h = 2166136261u;
  for (char c : s) {
    h ^= static_cast<uint8_t>(c);
    h *= 16777619u;
  }
  return h;
}

int LowerBoundUInt32(const uint32_t* arr, int n, uint32_t target) {
  int l = 0, r = n;
  while (l < r) {
    int m = l + ((r - l) >> 1);
    if (arr[m] < target)
      l = m + 1;
    else
      r = m;
  }
  return l;
}

int UniqueInPlaceUInt32(uint32_t* arr, int n) {
  if (arr == nullptr || n <= 0) return 0;
  int w = 1;
  for (int i = 1; i < n; ++i) {
    if (arr[i] != arr[w - 1]) {
      arr[w++] = arr[i];
    }
  }
  return w;
}

void TrimEOL(std::string* line) {
  if (line == nullptr) return;
  while (!line->empty() &&
         (line->back() == '\n' || line->back() == '\r')) {
    line->pop_back();
  }
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
  const std::string graph_id =
      cfg["graph_query"]["graph_id"].as<std::string>("");
  const std::string business_id =
      cfg["graph_query"]["business_id"].as<std::string>("");
  const int pivot_limit = cfg["graph_query"]["pivot_limit"].as<int>(0);

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

  std::string data_dump_cmd;
  const bool has_arangosh = CommandExists("arangosh");
  if (has_arangosh) {

  std::ostringstream js;
  js << "const c=db._collection(\"" << EscapeForJSDoubleQuote(pivot_graphs_collection)
     << "\");"
     << "if(!c){throw new Error(\"collection not found\");}"
     << "let q='FOR d IN " << EscapeForJSDoubleQuote(pivot_graphs_collection)
     << "';"
     << "const hasG=" << (graph_id.empty() ? "false" : "true") << ";"
     << "const hasB=" << (business_id.empty() ? "false" : "true") << ";"
     << "if(hasG&&hasB){q+=' FILTER d.graph_id==@g AND d.business_id==@b';}"
     << "else if(hasG){q+=' FILTER d.graph_id==@g';}"
     << "else if(hasB){q+=' FILTER d.business_id==@b';}"
     << (pivot_limit > 0 ? " LIMIT @l" : "") << " RETURN d';"
     << "const bind={g:\""
     << EscapeForJSDoubleQuote(graph_id) << "\",b:\""
     << EscapeForJSDoubleQuote(business_id) << "\""
     << (pivot_limit > 0 ? ",l:" + std::to_string(pivot_limit) : "") << "};"
     << "const cur=db._query(q,bind);"
     << "let docs=0;"
     << "while(cur.hasNext()){"
     << "docs++;"
     << "const d=cur.next();"
     << "const vs=d.vertices||[];"
     << "for(let i=0;i<vs.length;i++){const v=vs[i];"
     << "print(\"V\\t\"+String(v.id||\"\")+\"\\t\"+String(v.label||\"\"));}"
     << "const es=d.edges||[];"
     << "for(let i=0;i<es.length;i++){const e=es[i];"
     << "print(\"E\\t\"+String(e.src_id||\"\")+\"\\t\"+String(e.dst_id||\"\")+\"\\t\"+String(e.label||\"\"));}"
     << "}"
     << "print(\"M\\tDOCS\\t\"+String(docs));";

  const std::string arangosh_cmd =
      "arangosh --server.endpoint 'http+tcp://" + EscapeShellSingleQuotes(host) +
      ":" + std::to_string(port) + "' --server.username '" +
      EscapeShellSingleQuotes(user) + "' --server.password '" +
      EscapeShellSingleQuotes(pass) + "' --server.database '" +
      EscapeShellSingleQuotes(db) +
      "' --quiet --javascript.execute-string \"" +
      EscapeForJSDoubleQuote(js.str()) + "\" 2>&1";
  data_dump_cmd = arangosh_cmd;
  } else if (CommandExists("python3")) {
    std::ostringstream py_cmd;
    py_cmd
        << "ARANGO_SCHEME='" << EscapeShellSingleQuotes(scheme) << "' "
        << "ARANGO_HOST='" << EscapeShellSingleQuotes(host) << "' "
        << "ARANGO_PORT='" << std::to_string(port) << "' "
        << "ARANGO_DB='" << EscapeShellSingleQuotes(db) << "' "
        << "ARANGO_USER='" << EscapeShellSingleQuotes(user) << "' "
        << "ARANGO_PASS='" << EscapeShellSingleQuotes(pass) << "' "
        << "ARANGO_COLLECTION='"
        << EscapeShellSingleQuotes(pivot_graphs_collection) << "' "
        << "ARANGO_GRAPH_ID='" << EscapeShellSingleQuotes(graph_id) << "' "
        << "ARANGO_BUSINESS_ID='" << EscapeShellSingleQuotes(business_id) << "' "
        << "ARANGO_PIVOT_LIMIT='" << std::to_string(pivot_limit) << "' "
        << "python3 - <<'PY'\n"
        << "import os, json, base64, urllib.request\n"
        << "scheme=os.environ.get('ARANGO_SCHEME','http')\n"
        << "host=os.environ.get('ARANGO_HOST','127.0.0.1')\n"
        << "port=os.environ.get('ARANGO_PORT','8529')\n"
        << "db=os.environ.get('ARANGO_DB','_system')\n"
        << "user=os.environ.get('ARANGO_USER','root')\n"
        << "pwd=os.environ.get('ARANGO_PASS','')\n"
        << "coll=os.environ.get('ARANGO_COLLECTION','pivot_graphs')\n"
        << "gid=os.environ.get('ARANGO_GRAPH_ID','')\n"
        << "bid=os.environ.get('ARANGO_BUSINESS_ID','')\n"
        << "limit=int(os.environ.get('ARANGO_PIVOT_LIMIT','0') or 0)\n"
        << "base=f\"{scheme}://{host}:{port}/_db/{db}/_api/cursor\"\n"
        << "auth='Basic '+base64.b64encode(f\"{user}:{pwd}\".encode()).decode()\n"
        << "q=f\"FOR d IN {coll}\"\n"
        << "if gid and bid: q += \" FILTER d.graph_id==@g AND d.business_id==@b\"\n"
        << "elif gid: q += \" FILTER d.graph_id==@g\"\n"
        << "elif bid: q += \" FILTER d.business_id==@b\"\n"
        << "if limit>0: q += \" LIMIT @l\"\n"
        << "q += \" RETURN d\"\n"
        << "bind={}\n"
        << "if gid: bind['g']=gid\n"
        << "if bid: bind['b']=bid\n"
        << "if limit>0: bind['l']=limit\n"
        << "payload={'query':q,'bindVars':bind,'batchSize':64}\n"
        << "def req(url, method='POST', body=None):\n"
        << "  data=None if body is None else json.dumps(body).encode()\n"
        << "  r=urllib.request.Request(url, data=data, method=method)\n"
        << "  r.add_header('Authorization', auth)\n"
        << "  r.add_header('Content-Type','application/json')\n"
        << "  with urllib.request.urlopen(r, timeout=60) as resp:\n"
        << "    return json.loads(resp.read().decode())\n"
        << "def emit(doc):\n"
        << "  for v in (doc.get('vertices') or []):\n"
        << "    print('V\\t'+str(v.get('id',''))+'\\t'+str(v.get('label','')))\n"
        << "  for e in (doc.get('edges') or []):\n"
        << "    print('E\\t'+str(e.get('src_id',''))+'\\t'+str(e.get('dst_id',''))+'\\t'+str(e.get('label','')))\n"
        << "try:\n"
        << "  docs=0\n"
        << "  first=req(base,'POST',payload)\n"
        << "  for d in (first.get('result') or []): emit(d); docs+=1\n"
        << "  while first.get('hasMore'):\n"
        << "    cid=first.get('id')\n"
        << "    if not cid: break\n"
        << "    first=req(base+'/'+cid,'PUT',{})\n"
        << "    for d in (first.get('result') or []): emit(d); docs+=1\n"
        << "  print('M\\tDOCS\\t'+str(docs))\n"
        << "except Exception as ex:\n"
        << "  print('ERR\\t'+str(ex))\n"
        << "  raise\n"
        << "PY 2>&1";
    data_dump_cmd = py_cmd.str();
    std::cout << "[GARMatch] arangosh not found, using python3 cursor loader."
              << std::endl;
  } else {
    throw std::runtime_error(
        "Neither arangosh nor python3 is available for ArangoDB data loading.");
  }

  FILE* pipe = popen(data_dump_cmd.c_str(), "r");
  if (pipe == nullptr) {
    throw std::runtime_error("Failed to execute ArangoDB data dump command");
  }

  int vertex_token_count = 0;
  int edge_count = 0;
  int doc_count = -1;
  std::string load_error_line;

  std::array<char, 4096> line_buf{};
  while (fgets(line_buf.data(), static_cast<int>(line_buf.size()), pipe) !=
         nullptr) {
    std::string line(line_buf.data());
    TrimEOL(&line);
    if (line.empty()) continue;
    if (line.rfind("ERR\t", 0) == 0) {
      load_error_line = line;
      continue;
    }
    if (line.rfind("M\tDOCS\t", 0) == 0) {
      doc_count = std::atoi(line.substr(7).c_str());
      continue;
    }
    if (line.rfind("V\t", 0) == 0) {
      size_t p1 = line.find('\t', 2);
      if (p1 == std::string::npos) continue;
      std::string vid = line.substr(2, p1 - 2);
      std::string vlabel = line.substr(p1 + 1);
      (void)vlabel;
      if (vid.empty()) continue;
      ++vertex_token_count;
      continue;
    }
    if (line.rfind("E\t", 0) == 0) {
      size_t p1 = line.find('\t', 2);
      if (p1 == std::string::npos) continue;
      size_t p2 = line.find('\t', p1 + 1);
      if (p2 == std::string::npos) continue;
      std::string src = line.substr(2, p1 - 2);
      std::string dst = line.substr(p1 + 1, p2 - p1 - 1);
      std::string elabel = line.substr(p2 + 1);
      (void)elabel;
      if (src.empty() || dst.empty()) continue;
      vertex_token_count += 2;
      ++edge_count;
      continue;
    }
  }
  pclose(pipe);
  if (!load_error_line.empty()) {
    throw std::runtime_error("ArangoDB loader error: " + load_error_line);
  }
  if (doc_count == 0) {
    throw std::runtime_error(
        "No pivot_graph documents matched graph_id/business_id filters. "
        "Please check configs/arangodb.yaml.");
  }

  if (vertex_token_count <= 0 || edge_count <= 0) {
    throw std::runtime_error(
        "No vertices/edges loaded from ArangoDB pivot_graphs");
  }

  // Allocate raw space for all seen vertex tokens, then deduplicate in-place.
  std::unique_ptr<uint32_t[]> vertex_tokens(
      new uint32_t[vertex_token_count]());

  pipe = popen(data_dump_cmd.c_str(), "r");
  if (pipe == nullptr) {
    throw std::runtime_error("Failed to execute data dump command (2nd pass)");
  }
  int v_write = 0;
  while (fgets(line_buf.data(), static_cast<int>(line_buf.size()), pipe) !=
         nullptr) {
    std::string line(line_buf.data());
    TrimEOL(&line);
    if (line.empty()) continue;
    if (line.rfind("V\t", 0) == 0) {
      size_t p1 = line.find('\t', 2);
      if (p1 == std::string::npos) continue;
      std::string vid = line.substr(2, p1 - 2);
      if (vid.empty()) continue;
      if (v_write < vertex_token_count) {
        vertex_tokens[v_write++] = ParseVertexIDStable(vid);
      }
      continue;
    }
    if (line.rfind("E\t", 0) == 0) {
      size_t p1 = line.find('\t', 2);
      if (p1 == std::string::npos) continue;
      size_t p2 = line.find('\t', p1 + 1);
      if (p2 == std::string::npos) continue;
      std::string src = line.substr(2, p1 - 2);
      std::string dst = line.substr(p1 + 1, p2 - p1 - 1);
      if (src.empty() || dst.empty()) continue;
      if (v_write + 1 < vertex_token_count) {
        vertex_tokens[v_write++] = ParseVertexIDStable(src);
        vertex_tokens[v_write++] = ParseVertexIDStable(dst);
      }
      continue;
    }
  }
  pclose(pipe);
  if (v_write <= 0) {
    throw std::runtime_error("Failed to collect vertex ids from ArangoDB data");
  }

  std::sort(vertex_tokens.get(), vertex_tokens.get() + v_write);
  const int unique_vertex_count = UniqueInPlaceUInt32(vertex_tokens.get(), v_write);

  owned_g_.n_vertices = unique_vertex_count;
  owned_g_.n_edges = edge_count;
  owned_g_.v_id = std::make_unique<uint32_t[]>(owned_g_.n_vertices);
  owned_g_.v_label_idx = std::make_unique<int32_t[]>(owned_g_.n_vertices);
  owned_g_.e_src = std::make_unique<uint32_t[]>(owned_g_.n_edges);
  owned_g_.e_dst = std::make_unique<uint32_t[]>(owned_g_.n_edges);
  owned_g_.e_id = std::make_unique<uint32_t[]>(owned_g_.n_edges);
  owned_g_.e_label_idx = std::make_unique<int32_t[]>(owned_g_.n_edges);
  for (int i = 0; i < owned_g_.n_vertices; ++i) {
    owned_g_.v_id[i] = vertex_tokens[i];
    owned_g_.v_label_idx[i] = 0;
  }

  // Third pass: stream edges directly into pre-allocated arrays.
  pipe = popen(data_dump_cmd.c_str(), "r");
  if (pipe == nullptr) {
    throw std::runtime_error("Failed to execute data dump command (3rd pass)");
  }
  int e_write = 0;
  while (fgets(line_buf.data(), static_cast<int>(line_buf.size()), pipe) !=
         nullptr) {
    std::string line(line_buf.data());
    TrimEOL(&line);
    if (line.rfind("E\t", 0) != 0) continue;

    size_t p1 = line.find('\t', 2);
    if (p1 == std::string::npos) continue;
    size_t p2 = line.find('\t', p1 + 1);
    if (p2 == std::string::npos) continue;
    std::string src = line.substr(2, p1 - 2);
    std::string dst = line.substr(p1 + 1, p2 - p1 - 1);
    std::string elabel = line.substr(p2 + 1);
    (void)elabel;
    if (src.empty() || dst.empty()) continue;
    const uint32_t src_id = ParseVertexIDStable(src);
    const uint32_t dst_id = ParseVertexIDStable(dst);

    if (e_write >= owned_g_.n_edges) break;
    int src_local = LowerBoundUInt32(vertex_tokens.get(), unique_vertex_count, src_id);
    int dst_local = LowerBoundUInt32(vertex_tokens.get(), unique_vertex_count, dst_id);
    if (src_local >= unique_vertex_count || vertex_tokens[src_local] != src_id ||
        dst_local >= unique_vertex_count || vertex_tokens[dst_local] != dst_id) {
      continue;
    }
    owned_g_.e_src[e_write] = static_cast<uint32_t>(src_local);
    owned_g_.e_dst[e_write] = static_cast<uint32_t>(dst_local);
    owned_g_.e_id[e_write] = static_cast<uint32_t>(e_write);
    owned_g_.e_label_idx[e_write] = 0;
    ++e_write;
  }
  pclose(pipe);
  if (e_write != owned_g_.n_edges) {
    owned_g_.n_edges = e_write;
  }

  std::cout << "[GARMatch] loaded real graph from ArangoDB: n_vertices="
            << owned_g_.n_vertices << ", n_edges=" << owned_g_.n_edges
            << std::endl;

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
