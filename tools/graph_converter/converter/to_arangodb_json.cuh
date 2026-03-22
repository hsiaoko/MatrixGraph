#ifndef MATRIXGRAPH_TOOLS_GRAPH_CONVERTER_CONVERTER_TO_ARANGODB_JSON_CUH_
#define MATRIXGRAPH_TOOLS_GRAPH_CONVERTER_CONVERTER_TO_ARANGODB_JSON_CUH_

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/data_structures/edgelist.h"

namespace sics {
namespace matrixgraph {
namespace tools {
namespace converter {

struct CsvEdge {
  std::string src;
  std::string dst;
};

struct ArangoExportOptions {
  std::string graph_id = "demo_graph_id";
  std::string business_id = "demo_business_id";
  std::string pivot_mode = "single";  // single | source
  std::string default_vertex_label = "vertex";
  std::string default_edge_label = "relationship";
  bool random_vertex_labels = false;
  unsigned label_range = 1;
};

static std::string EscapeJSON(const std::string& in) {
  std::string out;
  out.reserve(in.size());
  for (char c : in) {
    switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
        break;
    }
  }
  return out;
}

static bool BuildCSVArraysFromEdgelistCSV(const std::string& input_path,
                                          const std::string& sep,
                                          bool compressed,
                                          std::vector<CsvEdge>* edges,
                                          std::vector<std::string>* vertices) {
  if (edges == nullptr || vertices == nullptr) return false;
  edges->clear();
  vertices->clear();

  sics::matrixgraph::core::data_structures::Edges edgelist;
  edgelist.ReadFromCSV(input_path, sep, compressed);
  auto metadata = edgelist.get_metadata();
  edgelist.ShowGraph();
  auto* localid_to_globalid = edgelist.get_localid_to_globalid_ptr();
  std::unordered_set<std::string> vset;

  for (size_t i = 0; i < metadata.num_edges; ++i) {
    auto src = edgelist.get_src_by_index(i);
    auto dst = edgelist.get_dst_by_index(i);
    if (localid_to_globalid != nullptr) {
      src = localid_to_globalid[src];
      dst = localid_to_globalid[dst];
    }
    const std::string src_str = std::to_string(src);
    const std::string dst_str = std::to_string(dst);
    edges->push_back({src_str, dst_str});
    vset.insert(src_str);
    vset.insert(dst_str);
  }
  if (edges->empty() && metadata.num_edges > 0) return false;
  vertices->assign(vset.begin(), vset.end());
  std::sort(vertices->begin(), vertices->end());
  return true;
}

struct PivotGraphDoc {
  std::string pivot_id;
  std::vector<std::string> vertices;
  std::vector<CsvEdge> edges;
};

static std::string PickVertexLabel(std::mt19937* rng,
                                   const ArangoExportOptions& opt) {
  if (!opt.random_vertex_labels) return opt.default_vertex_label;
  unsigned range = std::max(1u, opt.label_range);
  if (range == 1 || rng == nullptr) return opt.default_vertex_label + "_0";
  std::uniform_int_distribution<unsigned> dist(0, range - 1);
  return opt.default_vertex_label + "_" + std::to_string(dist(*rng));
}

static bool WriteArangoDBJSON(const std::string& out_dir,
                              const std::vector<CsvEdge>& edges,
                              const std::vector<std::string>& vertices,
                              const ArangoExportOptions& opt) {
  const std::string graph_structure_path = out_dir + "/graph_structure.json";
  const std::string pivot_ids_path = out_dir + "/pivot_graph_ids.jsonl";
  const std::string pivot_graphs_path = out_dir + "/pivot_graphs.jsonl";
  const std::string readme_path = out_dir + "/README_arangodb_import.txt";

  std::mt19937 rng(42);
  std::unordered_map<std::string, std::string> vlabel;
  for (const auto& v : vertices) vlabel[v] = PickVertexLabel(&rng, opt);

  std::vector<PivotGraphDoc> pivots;
  if (opt.pivot_mode == "source") {
    std::unordered_map<std::string, PivotGraphDoc> by_pivot;
    for (const auto& e : edges) {
      auto& doc = by_pivot[e.src];
      if (doc.pivot_id.empty()) doc.pivot_id = "pg_" + e.src;
      doc.vertices.push_back(e.src);
      doc.vertices.push_back(e.dst);
      doc.edges.push_back(e);
    }
    for (auto& kv : by_pivot) {
      auto& vs = kv.second.vertices;
      std::sort(vs.begin(), vs.end());
      vs.erase(std::unique(vs.begin(), vs.end()), vs.end());
      pivots.push_back(std::move(kv.second));
    }
    std::sort(pivots.begin(), pivots.end(),
              [](const PivotGraphDoc& a, const PivotGraphDoc& b) {
                return a.pivot_id < b.pivot_id;
              });
  } else {
    PivotGraphDoc doc;
    doc.pivot_id = "pg_0";
    doc.vertices = vertices;
    doc.edges = edges;
    pivots.push_back(std::move(doc));
  }

  {
    std::unordered_set<std::string> vertex_labels;
    for (const auto& kv : vlabel) vertex_labels.insert(kv.second);
    std::unordered_set<std::string> edge_triples;
    for (const auto& e : edges) {
      edge_triples.insert(vlabel[e.src] + "|" + opt.default_edge_label + "|" +
                          vlabel[e.dst]);
    }

    std::ofstream fout(graph_structure_path);
    if (!fout.is_open()) return false;
    fout << "{\n";
    fout << "  \"graph_id\": \"" << EscapeJSON(opt.graph_id) << "\",\n";
    fout << "  \"vertices\": [\n";
    bool first = true;
    for (const auto& vl : vertex_labels) {
      if (!first) fout << ",\n";
      first = false;
      fout << "    {\"label\":\"" << EscapeJSON(vl) << "\",\"attrs\":[]}";
    }
    fout << "\n  ],\n";
    fout << "  \"edges\": [\n";
    first = true;
    for (const auto& et : edge_triples) {
      auto p1 = et.find('|');
      auto p2 = et.find('|', p1 + 1);
      std::string sl = et.substr(0, p1);
      std::string el = et.substr(p1 + 1, p2 - p1 - 1);
      std::string dl = et.substr(p2 + 1);
      if (!first) fout << ",\n";
      first = false;
      fout << "    {\"src_label\":\"" << EscapeJSON(sl) << "\",\"label\":\""
           << EscapeJSON(el) << "\",\"dst_label\":\"" << EscapeJSON(dl)
           << "\",\"attrs\":[]}";
    }
    fout << "\n  ]\n";
    fout << "}\n";
  }

  {
    std::ofstream fout(pivot_ids_path);
    if (!fout.is_open()) return false;
    for (const auto& pg : pivots) {
      fout << "{\"graph_id\":\"" << EscapeJSON(opt.graph_id)
           << "\",\"business_id\":\"" << EscapeJSON(opt.business_id)
           << "\",\"pivot_graph_id\":\"" << EscapeJSON(pg.pivot_id)
           << "\"}\n";
    }
  }

  {
    std::ofstream fout(pivot_graphs_path);
    if (!fout.is_open()) return false;
    int edge_auto = 0;
    for (const auto& pg : pivots) {
      fout << "{\"graph_id\":\"" << EscapeJSON(opt.graph_id)
           << "\",\"business_id\":\"" << EscapeJSON(opt.business_id)
           << "\",\"pivot_graph_id\":\"" << EscapeJSON(pg.pivot_id)
           << "\",\"vertices\":[";
      for (size_t i = 0; i < pg.vertices.size(); ++i) {
        const auto& v = pg.vertices[i];
        if (i) fout << ",";
        fout << "{\"id\":\"" << EscapeJSON(v)
             << "\",\"time\":\"1970-01-01T00:00:00Z\",\"label\":\""
             << EscapeJSON(vlabel[v])
             << "\",\"attrs\":[{\"key\":\"placeholder\",\"value\":\"0\"}]}";
      }
      fout << "],\"edges\":[";
      for (size_t i = 0; i < pg.edges.size(); ++i) {
        const auto& e = pg.edges[i];
        if (i) fout << ",";
        fout << "{\"id\":\"e_" << edge_auto++
             << "\",\"time\":\"1970-01-01T00:00:00Z\",\"src_id\":\""
             << EscapeJSON(e.src) << "\",\"dst_id\":\"" << EscapeJSON(e.dst)
             << "\",\"src_label\":\"" << EscapeJSON(vlabel[e.src])
             << "\",\"label\":\"" << EscapeJSON(opt.default_edge_label)
             << "\",\"dst_label\":\"" << EscapeJSON(vlabel[e.dst])
             << "\",\"attrs\":[{\"key\":\"placeholder\",\"value\":\"0\"}]}";
      }
      fout << "]}\n";
    }
  }

  {
    std::ofstream fout(readme_path);
    if (!fout.is_open()) return false;
    fout << "Generated files for ArangoDB import:\n";
    fout << "1) graph_structure.json\n";
    fout << "2) pivot_graph_ids.jsonl\n";
    fout << "3) pivot_graphs.jsonl\n\n";
    fout << "The export contains placeholder labels/attrs/time where source data is missing.\n";
  }
  return true;
}

static bool ConvertEdgelistCSV2ArangoDBJSON(const std::string& input_path,
                                            const std::string& output_path,
                                            const std::string& sep,
                                            bool compressed,
                                            const ArangoExportOptions& opt) {
  if (!std::filesystem::exists(output_path))
    std::filesystem::create_directory(output_path);

  std::vector<CsvEdge> edges;
  std::vector<std::string> vertices;
  if (!BuildCSVArraysFromEdgelistCSV(input_path, sep, compressed, &edges,
                                     &vertices)) {
    return false;
  }
  return WriteArangoDBJSON(output_path, edges, vertices, opt);
}

}  // namespace converter
}  // namespace tools
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_TOOLS_GRAPH_CONVERTER_CONVERTER_TO_ARANGODB_JSON_CUH_
