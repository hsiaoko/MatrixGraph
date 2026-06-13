#include <gflags/gflags.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "core/task/gpu_task/graph_aggregate.cuh"

// Graph input: either an existing ImmutableCSR directory or synthetic parameters.
DEFINE_string(g, "", "Path to an existing ImmutableCSR graph directory");
DEFINE_uint32(n, 100, "Number of vertices in the synthetic graph (used when -g is empty)");
DEFINE_uint32(deg, 3, "Out-degree per vertex (used when -g is empty)");

// Feature selection (comma-separated list of primitives)
DEFINE_string(prims, "Mean,Sum,Count,PercentTrue",
              "Comma-separated aggregation primitives to compute"
              " (e.g., Mean,Sum,Count,Min,Max,Variance,Std,Median,"
              "PercentTrue,CountGreaterThanMean)");

// Fused compute-all mode: compute every primitive in a single kernel launch.
DEFINE_bool(compute_all, false,
            "If true, ignore -prims and compute all aggregation primitives "
            "in one fused kernel launch.");

// Stream parallelism configuration
DEFINE_uint32(n_streams, 0,
              "Number of CUDA streams to use (0 = use MATRIXGRAPH_CUDA_STREAMS "
              "env or default 2)");

using sics::matrixgraph::core::task::GraphAggregate;
using sics::matrixgraph::core::task::kernel::AggPrim;
using sics::matrixgraph::core::task::kernel::AllFeatures;
using sics::matrixgraph::core::task::kernel::FeatureRequest;
using sics::matrixgraph::core::task::kernel::FeatureValue;
using sics::matrixgraph::core::data_structures::AttributeName;

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

using Clock = std::chrono::high_resolution_clock;

static double SecondsSince(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
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

void PrintConfig(uint32_t actual_n) {
  std::cout << "\n=== GraphAggregate Configuration ===" << std::endl;
  if (!FLAGS_g.empty()) {
    std::cout << "Graph path: " << FLAGS_g << std::endl;
    std::cout << "Actual vertices: " << actual_n << std::endl;
  } else {
    std::cout << "Vertices: " << FLAGS_n << std::endl;
    std::cout << "Out-degree: " << FLAGS_deg << std::endl;
  }
  std::cout << "Primitives: " << FLAGS_prims << std::endl;
  std::cout << "ComputeAll: " << (FLAGS_compute_all ? "true" : "false")
            << std::endl;
  std::cout << "Streams: " << FLAGS_n_streams
            << " (0 means use env/default)" << std::endl;
  std::cout << "=====================================\n" << std::endl;
}

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "GraphAggregate feature computation using MatrixGraph\n"
      "Usage: " +
      std::string(argv[0]) +
      " [-g <immutable_csr_dir> | -n <vertices> -deg <out_degree>] "
      "[-prims <primitives>]");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  try {
    // 1. Create task and load graph topology + synthetic attributes.
    Clock::time_point t0 = Clock::now();
    GraphAggregate task;
    if (FLAGS_n_streams > 0) {
      task.SetNumStreams(FLAGS_n_streams);
    }

    if (!FLAGS_g.empty()) {
      task.LoadGraph(FLAGS_g);
      std::cout << "[Timing] LoadGraph: " << SecondsSince(t0) << " s" << std::endl;
      Clock::time_point t1 = Clock::now();
      task.GenerateSyntheticAttributes();
      std::cout << "[Timing] GenerateSyntheticAttributes: " << SecondsSince(t1)
                << " s" << std::endl;
    } else {
      task.LoadSyntheticData(FLAGS_n, FLAGS_deg);
      std::cout << "[Timing] LoadSyntheticData: " << SecondsSince(t0) << " s"
                << std::endl;
    }

    // 2. All graph vertices are pivots.
    const uint32_t n = task.GetNumVertices();
    PrintConfig(n);
    Clock::time_point t2 = Clock::now();
    std::vector<uint32_t> pivot_vids(n);
    for (uint32_t v = 0; v < n; ++v) pivot_vids[v] = v;
    std::cout << "[Timing] Build pivot list: " << SecondsSince(t2) << " s"
              << std::endl;

    // 3. Compute features.
    if (FLAGS_compute_all) {
      std::vector<AllFeatures> all_results =
          task.ComputeAll(pivot_vids, AttributeName("score"), true);

      // 4. Print first few fused results.
      uint32_t print_n = std::min<uint32_t>(FLAGS_n, 5);
      std::cout << "Fused ComputeAll results (first " << print_n
                << " vertices):" << std::endl;
      for (uint32_t v = 0; v < print_n; ++v) {
        const auto& a = all_results[v];
        std::cout << "  V" << v
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
    } else {
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

      std::vector<FeatureValue> results =
          task.ComputeFeatures(pivot_vids, requests);

      // 4. Print first few results.
      uint32_t print_n = std::min<uint32_t>(FLAGS_n, 5);
      std::cout << "Results (first " << print_n << " vertices):" << std::endl;
      for (uint32_t v = 0; v < print_n; ++v) {
        std::cout << "  V" << v << ":";
        for (size_t r = 0; r < requests.size(); ++r) {
          const auto& val = results[v * requests.size() + r];
          std::cout << " " << prim_names[r] << "=" << val.ToDouble();
        }
        std::cout << std::endl;
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
