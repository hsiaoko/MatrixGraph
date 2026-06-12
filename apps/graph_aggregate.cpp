#include <gflags/gflags.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "core/common/types.h"
#include "core/components/scheduler/scheduler.h"
#include "core/matrixgraph.cuh"
#include "core/task/gpu_task/graph_aggregate.cuh"
#include "core/task/gpu_task/task_base.cuh"

// Synthetic graph parameters
DEFINE_uint32(n_graphs, 1, "Number of synthetic graphs to create");
DEFINE_string(n, "100",
              "Comma-separated number of vertices per graph (e.g. 100,200). "
              "If single value, reused for all graphs.");
DEFINE_string(deg, "3",
              "Comma-separated out-degree per vertex per graph (e.g. 3,4). "
              "If single value, reused for all graphs.");

// Feature selection (comma-separated list of primitives)
DEFINE_string(prims, "Mean,Sum,Count,PercentTrue",
              "Comma-separated aggregation primitives to compute"
              " (e.g., Mean,Sum,Count,Min,Max,Variance,Std,Median,"
              "PercentTrue,CountGreaterThanMean)");

// Fused compute-all mode: compute every primitive in a single kernel launch.
DEFINE_bool(compute_all, false,
            "If true, ignore -prims and compute all aggregation primitives "
            "in one fused kernel launch.");

// System configuration
DEFINE_string(
    scheduler, "CHBL",
    "Scheduler type (options: CHBL, EvenSplit, RoundRobin, default: CHBL)");

using sics::matrixgraph::core::components::scheduler::SchedulerType;
using sics::matrixgraph::core::task::GraphAggregate;
using sics::matrixgraph::core::task::kernel::AggPrim;
using sics::matrixgraph::core::task::kernel::AllFeatures;
using sics::matrixgraph::core::task::kernel::FeatureRequest;
using sics::matrixgraph::core::task::kernel::FeatureValue;
using sics::matrixgraph::core::data_structures::AttributeName;

SchedulerType Scheduler2Enum(const std::string& s) {
  if (s == "EvenSplit")
    return sics::matrixgraph::core::components::scheduler::kEvenSplit;
  else if (s == "CHBL")
    return sics::matrixgraph::core::components::scheduler::kCHBL;
  else if (s == "RoundRobin")
    return sics::matrixgraph::core::components::scheduler::kRoundRobin;
  return sics::matrixgraph::core::components::scheduler::kCHBL;
}

static AggPrim StringToAggPrim(const std::string& s) {
  if (s == "Count") return AggPrim::kCount;
  if (s == "Sum") return AggPrim::kSum;
  if (s == "Mean") return AggPrim::kMean;
  if (s == "Min") return AggPrim::kMin;
  if (s == "Max") return AggPrim::kMax;
  if (s == "Median") return AggPrim::kMedian;
  if (s == "Mode") return AggPrim::kMode;
  if (s == "NumUnique") return AggPrim::kNumUnique;
  if (s == "Entropy") return AggPrim::kEntropy;
  if (s == "Quarter") return AggPrim::kQuarter;
  if (s == "Quartile3") return AggPrim::kQuartile3;
  if (s == "PercentTrue") return AggPrim::kPercentTrue;
  if (s == "Skew") return AggPrim::kSkew;
  if (s == "Variance") return AggPrim::kVariance;
  if (s == "Std") return AggPrim::kStd;
  if (s == "CountGreaterThanMean") return AggPrim::kCountGreaterThanMean;
  std::cerr << "Unknown aggregation primitive: " << s << std::endl;
  return AggPrim::kCount;
}

static std::vector<std::string> SplitByComma(const std::string& s) {
  std::vector<std::string> parts;
  size_t start = 0;
  while (start < s.size()) {
    size_t end = s.find(',', start);
    if (end == std::string::npos) end = s.size();
    std::string part = s.substr(start, end - start);
    size_t ws_front = 0;
    while (ws_front < part.size() && std::isspace(part[ws_front])) ++ws_front;
    size_t ws_back = part.size();
    while (ws_back > ws_front && std::isspace(part[ws_back - 1])) --ws_back;
    if (ws_front < ws_back) parts.emplace_back(part.substr(ws_front, ws_back - ws_front));
    start = end + 1;
  }
  return parts;
}

static std::vector<uint32_t> ParseUintList(const std::string& s) {
  std::vector<uint32_t> vals;
  for (const auto& p : SplitByComma(s)) {
    vals.push_back(static_cast<uint32_t>(std::stoul(p)));
  }
  return vals;
}

void PrintConfig() {
  std::cout << "\n=== GraphAggregate Configuration ===" << std::endl;
  std::cout << "Graphs: " << FLAGS_n_graphs << std::endl;
  std::cout << "Vertices per graph: " << FLAGS_n << std::endl;
  std::cout << "Out-degree per graph: " << FLAGS_deg << std::endl;
  std::cout << "Primitives: " << FLAGS_prims << std::endl;
  std::cout << "ComputeAll: " << (FLAGS_compute_all ? "true" : "false")
            << std::endl;
  std::cout << "Scheduler: " << FLAGS_scheduler << std::endl;
  std::cout << "=====================================\n" << std::endl;
}

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "GraphAggregate synthetic feature computation using MatrixGraph\n"
      "Usage: " +
      std::string(argv[0]) +
      " -n_graphs <count> -n <v1,v2,...> -deg <d1,d2,...> -prims <primitives>");

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  PrintConfig();

  try {
    auto scheduler_type = Scheduler2Enum(FLAGS_scheduler);
    sics::matrixgraph::core::MatrixGraph system(scheduler_type);

    // 1. Parse per-graph parameters.
    std::vector<uint32_t> n_vertices_list = ParseUintList(FLAGS_n);
    std::vector<uint32_t> out_deg_list = ParseUintList(FLAGS_deg);
    uint32_t n_graphs = FLAGS_n_graphs;

    if (n_vertices_list.size() == 1 && n_graphs > 1) {
      n_vertices_list.resize(n_graphs, n_vertices_list[0]);
    }
    if (out_deg_list.size() == 1 && n_graphs > 1) {
      out_deg_list.resize(n_graphs, out_deg_list[0]);
    }
    if (n_vertices_list.size() != n_graphs || out_deg_list.size() != n_graphs) {
      std::cerr << "Error: -n and -deg must have either 1 value or exactly "
                << n_graphs << " values." << std::endl;
      return EXIT_FAILURE;
    }

    // 2. Create task and add synthetic graphs.
    auto* task = new GraphAggregate(std::vector<std::string>{});
    for (uint32_t g = 0; g < n_graphs; ++g) {
      task->AddSyntheticGraph(n_vertices_list[g], out_deg_list[g]);
    }

    // 3. Build feature requests from comma-separated flag (or use fused mode).
    auto prim_names = SplitByComma(FLAGS_prims);
    std::vector<FeatureRequest> requests;
    requests.reserve(prim_names.size());
    for (const auto& name : prim_names) {
      AggPrim prim = StringToAggPrim(name);
      // PercentTrue operates on the "flag" attribute; everything else
      // operates on the "score" attribute in this synthetic demo.
      const char* attr = (prim == AggPrim::kPercentTrue) ? "flag" : "score";
      requests.push_back({AttributeName(attr), 0, true, prim});
    }

    // 4. All vertices of all graphs are pivots.
    std::vector<uint32_t> pivot_gids;
    std::vector<uint32_t> pivot_vids;
    for (uint32_t g = 0; g < n_graphs; ++g) {
      for (uint32_t v = 0; v < n_vertices_list[g]; ++v) {
        pivot_gids.push_back(g);
        pivot_vids.push_back(v);
      }
    }

    // 5. Compute features.
    if (FLAGS_compute_all) {
      std::vector<AllFeatures> all_results;
      task->ComputeAll(pivot_gids, pivot_vids, AttributeName("score"), true,
                       &all_results);

      // 6. Print first few fused results per graph.
      uint32_t print_n = 5;
      std::cout << "Fused ComputeAll results (first " << print_n
                << " vertices per graph):" << std::endl;
      size_t global_idx = 0;
      for (uint32_t g = 0; g < n_graphs; ++g) {
        std::cout << "  Graph " << g << " (|V|=" << n_vertices_list[g]
                  << ", deg=" << out_deg_list[g] << "):" << std::endl;
        for (uint32_t v = 0; v < n_vertices_list[g]; ++v, ++global_idx) {
          if (v >= print_n) continue;
          const auto& a = all_results[global_idx];
          std::cout << "    V" << v
                    << ": count=" << a.count.i64
                    << " count_gtm=" << a.count_greater_than_mean.i64
                    << " num_unique=" << a.num_unique.i64
                    << " sum=" << a.sum.ToDouble()
                    << " mean=" << a.mean.ToDouble()
                    << " variance=" << a.variance.ToDouble()
                    << " std=" << a.std.ToDouble()
                    << " mode=" << a.mode.ToDouble()
                    << " min=" << a.min.ToDouble()
                    << " max=" << a.max.ToDouble()
                    << " median=" << a.median.ToDouble()
                    << " q1=" << a.quarter.ToDouble()
                    << " q3=" << a.quartile3.ToDouble()
                    << " entropy=" << a.entropy.ToDouble()
                    << " percent_true=" << a.percent_true.ToDouble()
                    << " skew=" << a.skew.ToDouble() << std::endl;
        }
      }
    } else {
      std::vector<FeatureValue> results;
      task->ComputeFeatures(pivot_gids, pivot_vids, requests, &results);

      // 6. Print first few results per graph.
    uint32_t print_n = 5;
    std::cout << "Results (first " << print_n << " vertices per graph):"
              << std::endl;
    size_t global_idx = 0;
    for (uint32_t g = 0; g < n_graphs; ++g) {
      std::cout << "  Graph " << g << " (|V|=" << n_vertices_list[g]
                << ", deg=" << out_deg_list[g] << "):" << std::endl;
      for (uint32_t v = 0; v < n_vertices_list[g]; ++v, ++global_idx) {
        if (v >= print_n) continue;
        std::cout << "    V" << v << ":";
        for (size_t r = 0; r < requests.size(); ++r) {
          const auto& val = results[global_idx * requests.size() + r];
          std::cout << " " << prim_names[r] << "=" << val.ToDouble();
        }
        std::cout << std::endl;
      }
    }
    }

    delete task;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
