#include <gflags/gflags.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "core/common/types.h"
#include "core/components/scheduler/scheduler.h"
#include "core/matrixgraph.cuh"
#include "core/task/gpu_task/graph_aggregate.cuh"
#include "core/task/gpu_task/task_base.cuh"

// Synthetic graph parameters
DEFINE_uint32(n, 100, "Number of vertices in the synthetic graph");
DEFINE_uint32(deg, 3, "Out-degree per vertex");

// Feature selection (comma-separated list of primitives)
DEFINE_string(prims, "Mean,Sum,Count,PercentTrue",
              "Comma-separated aggregation primitives to compute"
              " (e.g., Mean,Sum,Count,Min,Max,Variance,Std,Median,"
              "PercentTrue,CountGreaterThanMean)");

// System configuration
DEFINE_string(
    scheduler, "CHBL",
    "Scheduler type (options: CHBL, EvenSplit, RoundRobin, default: CHBL)");

using sics::matrixgraph::core::components::scheduler::SchedulerType;
using sics::matrixgraph::core::task::GraphAggregate;
using sics::matrixgraph::core::task::kernel::AggPrim;
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
    // trim whitespace
    size_t ws_front = 0;
    while (ws_front < part.size() && std::isspace(part[ws_front])) ++ws_front;
    size_t ws_back = part.size();
    while (ws_back > ws_front && std::isspace(part[ws_back - 1])) --ws_back;
    if (ws_front < ws_back) parts.emplace_back(part.substr(ws_front, ws_back - ws_front));
    start = end + 1;
  }
  return parts;
}

void PrintConfig() {
  std::cout << "\n=== GraphAggregate Configuration ===" << std::endl;
  std::cout << "Vertices: " << FLAGS_n << std::endl;
  std::cout << "Out-degree: " << FLAGS_deg << std::endl;
  std::cout << "Primitives: " << FLAGS_prims << std::endl;
  std::cout << "Scheduler: " << FLAGS_scheduler << std::endl;
  std::cout << "=====================================\n" << std::endl;
}

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "GraphAggregate synthetic feature computation using MatrixGraph\n"
      "Usage: " +
      std::string(argv[0]) +
      " -n <vertices> -deg <out_degree> -prims <primitives>");

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  PrintConfig();

  try {
    auto scheduler_type = Scheduler2Enum(FLAGS_scheduler);
    sics::matrixgraph::core::MatrixGraph system(scheduler_type);

    // 1. Create task and load synthetic data.
    auto* task = new GraphAggregate(std::vector<std::string>{});
    task->LoadSyntheticData(FLAGS_n, FLAGS_deg);

    // 2. Build feature requests from comma-separated flag.
    auto prim_names = SplitByComma(FLAGS_prims);
    std::vector<FeatureRequest> requests;
    requests.reserve(prim_names.size());
    for (const auto& name : prim_names) {
      requests.push_back({AttributeName("score"), 0, true, StringToAggPrim(name)});
    }

    // 3. All vertices are pivots.
    std::vector<uint32_t> pivot_gids(FLAGS_n, 0);
    std::vector<uint32_t> pivot_vids(FLAGS_n);
    for (uint32_t i = 0; i < FLAGS_n; ++i) pivot_vids[i] = i;

    // 4. Compute features.
    std::vector<FeatureValue> results;
    task->ComputeFeatures(pivot_gids, pivot_vids, requests, &results);

    // 5. Print first few results.
    uint32_t print_n = std::min<uint32_t>(FLAGS_n, 10);
    std::cout << "Results (first " << print_n << " vertices):" << std::endl;
    for (uint32_t v = 0; v < print_n; ++v) {
      std::cout << "  V" << v << ":";
      for (size_t r = 0; r < requests.size(); ++r) {
        const auto& val = results[v * requests.size() + r];
        std::cout << " " << prim_names[r] << "=" << val.ToDouble();
      }
      std::cout << std::endl;
    }

    delete task;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
