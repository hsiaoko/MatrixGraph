#ifndef MATRIXGRAPH_TOOLS_GRAPH_CONVERTER_CONVERTER_TO_ARANGODB_JSON_CUH_
#define MATRIXGRAPH_TOOLS_GRAPH_CONVERTER_CONVERTER_TO_ARANGODB_JSON_CUH_

#include <algorithm>
#include <cstdint>
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

using VertexID = sics::matrixgraph::core::common::VertexID;
using Edge = sics::matrixgraph::core::data_structures::Edge;

struct ArangoExportOptions {
  uint64_t graph_id = 1;
  uint64_t business_id = 1;
  std::string pivot_mode = "single";  // single | source | k_hop
  std::string default_vertex_label = "vertex";
  std::string default_edge_label = "relationship";
  std::string import_time = "1970-01-01T00:00:00Z";  // _time: import time
  std::string pivot_time = "1970-01-01T00:00:00Z";   // _pivot_time: business time
  bool random_vertex_labels = false;
  unsigned label_range = 1;
  VertexID k_hop = 0;  // k-hop distance for k_hop pivot mode (0 means unlimited/all)
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
                                          bool keep_original_vid,
                                          sics::matrixgraph::core::data_structures::Edges* out_edgelist) {
  if (out_edgelist == nullptr) return false;
  out_edgelist->ReadFromCSV(input_path, sep, keep_original_vid);
  out_edgelist->ShowGraph();
  return true;
}

struct PivotGraphDoc {
  std::string pivot_id;
  std::vector<VertexID> vertices;
  std::vector<Edge> edges;
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
                              const sics::matrixgraph::core::data_structures::Edges& edgelist,
                              const ArangoExportOptions& opt) {
  const std::string graph_structure_path = out_dir + "/graph_structure.json";
  const std::string pivot_ids_path = out_dir + "/pivot_graph_ids.jsonl";
  const std::string pivot_graphs_path = out_dir + "/pivot_graphs.jsonl";
  const std::string readme_path = out_dir + "/README_arangodb_import.txt";

  const auto metadata = edgelist.get_metadata();
  auto to_global = [&edgelist](VertexID local_id) {
    return edgelist.get_globalid_by_localid(local_id);
  };

  std::mt19937 rng(42);
  std::unordered_map<VertexID, std::string> vlabel;
  for (size_t i = 0; i < metadata.num_vertices; ++i) {
    auto vid = to_global(static_cast<VertexID>(i));
    vlabel[vid] = PickVertexLabel(&rng, opt);
  }

  std::vector<PivotGraphDoc> pivots;
  if (opt.pivot_mode == "source") {
    std::unordered_map<VertexID, PivotGraphDoc> by_pivot;
    for (size_t i = 0; i < metadata.num_edges; ++i) {
      auto e = edgelist.get_edge_by_index(i);
      Edge global_edge(to_global(e.src), to_global(e.dst));
      auto& doc = by_pivot[global_edge.src];
      if (doc.pivot_id.empty()) {
        doc.pivot_id = "pg_" + std::to_string(global_edge.src);
      }
      doc.vertices.push_back(global_edge.src);
      doc.vertices.push_back(global_edge.dst);
      doc.edges.push_back(global_edge);
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
  } else if (opt.pivot_mode == "k_hop") {
    // Build k-hop subgraphs for each pivot vertex using BuildKHopOutSubgraphs
    VertexID k = opt.k_hop > 0 ? opt.k_hop : metadata.max_vid + 1;
    auto subgraphs = edgelist.BuildKHopOutSubgraphs(k);
    
    // Get all unique vertices as centers (same order as BuildKHopOutSubgraphs)
    std::vector<VertexID> centers;
    centers.reserve(subgraphs.size());
    for (size_t i = 0; i < subgraphs.size(); ++i) {
      centers.push_back(to_global(static_cast<VertexID>(i)));
    }
    
    for (size_t i = 0; i < subgraphs.size(); ++i) {
      const auto& subgraph = subgraphs[i];
      const auto& sub_meta = subgraph.get_metadata();
      
      PivotGraphDoc doc;
      doc.pivot_id = "pg_" + std::to_string(centers[i]);
      
      // Collect vertices from subgraph
      std::unordered_set<VertexID> vertex_set;
      for (size_t j = 0; j < sub_meta.num_edges; ++j) {
        auto e = subgraph.get_edge_by_index(j);
        VertexID global_src = to_global(e.src);
        VertexID global_dst = to_global(e.dst);
        vertex_set.insert(global_src);
        vertex_set.insert(global_dst);
        doc.edges.emplace_back(global_src, global_dst);
      }
      
      doc.vertices.assign(vertex_set.begin(), vertex_set.end());
      std::sort(doc.vertices.begin(), doc.vertices.end());
      
      if (!doc.edges.empty()) {
        pivots.push_back(std::move(doc));
      }
    }
    
    std::sort(pivots.begin(), pivots.end(),
              [](const PivotGraphDoc& a, const PivotGraphDoc& b) {
                return a.pivot_id < b.pivot_id;
              });
  } else {
    // Default: single pivot containing all vertices and edges
    PivotGraphDoc doc;
    doc.pivot_id = "pg_0";
    doc.vertices.reserve(metadata.num_vertices);
    for (size_t i = 0; i < metadata.num_vertices; ++i) {
      doc.vertices.push_back(to_global(static_cast<VertexID>(i)));
    }
    doc.edges.reserve(metadata.num_edges);
    for (size_t i = 0; i < metadata.num_edges; ++i) {
      auto e = edgelist.get_edge_by_index(i);
      doc.edges.emplace_back(to_global(e.src), to_global(e.dst));
    }
    pivots.push_back(std::move(doc));
  }

  {
    std::unordered_set<std::string> vertex_labels;
    for (const auto& kv : vlabel) vertex_labels.insert(kv.second);
    std::unordered_set<std::string> edge_triples;
    for (size_t i = 0; i < metadata.num_edges; ++i) {
      auto e = edgelist.get_edge_by_index(i);
      auto src = to_global(e.src);
      auto dst = to_global(e.dst);
      edge_triples.insert(vlabel[src] + "|" + opt.default_edge_label + "|" +
                          vlabel[dst]);
    }
    std::vector<std::string> vertex_label_list(vertex_labels.begin(),
                                               vertex_labels.end());
    std::sort(vertex_label_list.begin(), vertex_label_list.end());
    std::vector<std::string> edge_triple_list(edge_triples.begin(),
                                              edge_triples.end());
    std::sort(edge_triple_list.begin(), edge_triple_list.end());

    std::ofstream fout(graph_structure_path);
    if (!fout.is_open()) return false;
    fout << "{\n";
    fout << "  \"graph_id\": " << opt.graph_id << ",\n";
    fout << "  \"business_id\": " << opt.business_id << ",\n";
    fout << "  \"num_vertices\": " << metadata.num_vertices << ",\n";
    fout << "  \"num_edges\": " << metadata.num_edges << ",\n";
    fout << "  \"num_vertex_labels\": " << vertex_label_list.size() << ",\n";
    fout << "  \"num_edge_label_triples\": " << edge_triple_list.size()
         << ",\n";
    fout << "  \"vertices\": [\n";
    bool first = true;
    for (const auto& vl : vertex_label_list) {
      if (!first) fout << ",\n";
      first = false;
      fout << "    {\"label\":\"" << EscapeJSON(vl) << "\",\"attrs\":[]}";
    }
    fout << "\n  ],\n";
    fout << "  \"edges\": [\n";
    first = true;
    for (const auto& et : edge_triple_list) {
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
      fout << "{\"graph_id\":" << opt.graph_id
           << ",\"business_id\":" << opt.business_id
           << ",\"pivot_graph_id\":\"" << EscapeJSON(pg.pivot_id)
           << "\"}\n";
    }
  }

  {
    std::ofstream fout(pivot_graphs_path);
    if (!fout.is_open()) return false;
    int edge_auto = 0;
    for (const auto& pg : pivots) {
      fout << "{";
      fout << "\"graph_id\":" << opt.graph_id << ",";
      fout << "\"business_id\":" << opt.business_id << ",";
      fout << "\"pivot_graph_id\":\"" << EscapeJSON(pg.pivot_id) << "\",";
      fout << "\"_time\":\"" << EscapeJSON(opt.import_time) << "\",";
      fout << "\"_pivot_time\":\"" << EscapeJSON(opt.pivot_time) << "\",";
      fout << "\"graph\":{\"main\":{\"vertices\":[";
      for (size_t i = 0; i < pg.vertices.size(); ++i) {
        const auto& v = pg.vertices[i];
        if (i) fout << ",";
        fout << "{\"id\":\"" << std::to_string(v)
             << "\",\"time\":\"" << EscapeJSON(opt.import_time)
             << "\",\"label\":\""
             << EscapeJSON(vlabel[v])
             << "\",\"attrs\":[{\"key\":\"placeholder\",\"value\":\"0\"}]}";
      }
      fout << "],\"edges\":[";
      for (size_t i = 0; i < pg.edges.size(); ++i) {
        const auto& e = pg.edges[i];
        if (i) fout << ",";
        fout << "{\"id\":\"e_" << edge_auto++
             << "\",\"label\":\"" << EscapeJSON(opt.default_edge_label)
             << "\",\"srcId\":\"" << std::to_string(e.src)
             << "\",\"dstId\":\"" << std::to_string(e.dst)
             << "\",\"time\":\"" << EscapeJSON(opt.import_time)
             << "\",\"dstLabel\":\"" << EscapeJSON(vlabel[e.dst])
             << "\",\"srcLabel\":\"" << EscapeJSON(vlabel[e.src])
             << "\",\"attrs\":[{\"key\":\"placeholder\",\"value\":\"0\"}]}";
      }
      fout << "]}}}\n";
    }
  }

  {
    std::ofstream fout(readme_path);
    if (!fout.is_open()) return false;
    fout << "Generated files for ArangoDB import:\n";
    fout << "1) graph_structure.json\n";
    fout << "2) pivot_graph_ids.jsonl\n";
    fout << "3) pivot_graphs.jsonl\n\n";
    fout << "pivot_graphs.jsonl shape: {graph_id,business_id,pivot_graph_id,_time,_pivot_time,graph:{main:{vertices,edges}}}\n";
    fout << "Each edge fields: label,id,srcId,dstId,time,dstLabel,srcLabel,attrs.\n";
    fout << "The export contains placeholder labels/attrs where source data is missing.\n";
  }
  
  return true;
}

static bool ConvertEdgelistCSV2ArangoDBJSON(const std::string& input_path,
                                            const std::string& output_path,
                                            const std::string& sep,
                                            bool keep_original_vid,
                                            const ArangoExportOptions& opt) {
  if (!std::filesystem::exists(output_path))
    std::filesystem::create_directory(output_path);

  sics::matrixgraph::core::data_structures::Edges edgelist;
  if (!BuildCSVArraysFromEdgelistCSV(input_path, sep, keep_original_vid, &edgelist)) {
    return false;
  }
  return WriteArangoDBJSON(output_path, edgelist, opt);
}

}  // namespace converter
}  // namespace tools
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_TOOLS_GRAPH_CONVERTER_CONVERTER_TO_ARANGODB_JSON_CUH_
