#include <gflags/gflags.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "core/task/gpu_task/execute_agg_prim.cuh"

// Input values as a comma-separated list.
DEFINE_string(values, "1,2,3,4,5",
              "Comma-separated input values for aggregation (numeric)");

// Multiple input lists separated by ';'. When set, -values is ignored.
DEFINE_string(lists, "",
              "Multiple value lists separated by ';' for batch/stream "
              "processing (e.g., '1,2,3;4,5,6;7,8,9'). Overrides -values.");

// Feature selection (comma-separated list of primitives)
DEFINE_string(prims, "Sum,Mean,Count,Median,Min,Max,Variance,Std",
              "Comma-separated aggregation primitives to compute"
              " (e.g., Count,Sum,Mean,Median,Mode,Min,Max,Variance,Std,"
              "Skew,Entropy,NumUnique,PercentTrue,Quarter,Quartile3,"
              "CountGreaterThanMean,DFeat)");

// Fused compute-all mode.
DEFINE_bool(compute_all, false,
            "If true, ignore -prims and compute all aggregation primitives "
            "in one fused kernel launch.");

// CUDA stream parallelism for batch processing.
DEFINE_uint32(n_streams, 1,
              "Number of CUDA streams to use for batch processing. "
              "1 means batch without inter-list stream parallelism (default). "
              "Values >1 enable concurrent stream processing.");

// Multi-primitive batch: when -lists is set, compute these primitives for
// every input list in one batch (instead of a single primitive with -prims).
DEFINE_string(prims_batch, "",
              "Comma-separated primitives to compute for each list in batch "
              "mode (used with -lists). If empty, -prims is used for a single "
              "primitive per list.");

// One-by-one submission: process each list independently with single-list
// Compute/ComputeAll calls instead of fused batch kernels.
DEFINE_bool(one_by_one, false,
            "If true, process batch lists one by one using single-list "
            "Compute/ComputeAll calls (no batch kernel fusion). Used with -lists.");

using sics::matrixgraph::core::task::AggPrim;
using sics::matrixgraph::core::task::AllFeatures;
using sics::matrixgraph::core::task::ExecuteAggPrim;
using sics::matrixgraph::core::task::FeatureValue;
using sics::matrixgraph::core::task::MakeBoolValue;
using sics::matrixgraph::core::task::MakeFloat64Value;
using sics::matrixgraph::core::task::MakeIntValue;

static AggPrim StringToAggPrim(const std::string& s) {
  if (s == "Count") return AggPrim::kCount;
  if (s == "Sum") return AggPrim::kSum;
  if (s == "Mean") return AggPrim::kMean;
  if (s == "Median") return AggPrim::kMedian;
  if (s == "Mode") return AggPrim::kMode;
  if (s == "Min") return AggPrim::kMin;
  if (s == "Max") return AggPrim::kMax;
  if (s == "Variance") return AggPrim::kVariance;
  if (s == "Std") return AggPrim::kStd;
  if (s == "Skew") return AggPrim::kSkew;
  if (s == "Entropy") return AggPrim::kEntropy;
  if (s == "NumUnique") return AggPrim::kNumUnique;
  if (s == "PercentTrue") return AggPrim::kPercentTrue;
  if (s == "Quarter") return AggPrim::kQuarter;
  if (s == "Quartile3") return AggPrim::kQuartile3;
  if (s == "CountGreaterThanMean") return AggPrim::kCountGreaterThanMean;
  if (s == "DFeat") return AggPrim::kDFeat;
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

static std::vector<std::string> SplitBySemicolon(const std::string& s) {
  std::vector<std::string> parts;
  size_t start = 0;
  while (start < s.size()) {
    size_t end = s.find(';', start);
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

static std::vector<FeatureValue> ParseValues(const std::string& s) {
  std::vector<FeatureValue> values;
  auto parts = SplitByComma(s);
  for (const auto& part : parts) {
    // Try bool first.
    if (part == "true" || part == "True" || part == "TRUE") {
      values.push_back(MakeBoolValue(true));
      continue;
    }
    if (part == "false" || part == "False" || part == "FALSE") {
      values.push_back(MakeBoolValue(false));
      continue;
    }

    // Try integer.
    bool is_int = !part.empty();
    size_t idx = 0;
    if (part[0] == '-' && part.size() > 1) idx = 1;
    for (; idx < part.size(); ++idx) {
      if (!std::isdigit(part[idx])) {
        is_int = false;
        break;
      }
    }
    if (is_int) {
      values.push_back(MakeIntValue(std::stoll(part)));
      continue;
    }

    // Fallback to float64.
    values.push_back(MakeFloat64Value(std::stod(part)));
  }
  return values;
}

static void PrintValue(const FeatureValue& v) {
  switch (v.type) {
    case sics::matrixgraph::core::data_structures::ValueType::kInt:
    case sics::matrixgraph::core::data_structures::ValueType::kTime:
      std::cout << v.i64;
      break;
    case sics::matrixgraph::core::data_structures::ValueType::kFloat64:
      std::cout << v.f64;
      break;
    case sics::matrixgraph::core::data_structures::ValueType::kFloat32:
      std::cout << v.f32;
      break;
    case sics::matrixgraph::core::data_structures::ValueType::kBool:
      std::cout << (v.b ? "true" : "false");
      break;
    default:
      std::cout << "Invalid";
  }
}

using Clock = std::chrono::high_resolution_clock;

static double SecondsSince(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "ExecuteAggPrim: GPU aggregation primitives aligned with featurelib\n"
      "Usage: " +
      std::string(argv[0]) +
      " -values <comma-separated-values> [-prims <primitives>] "
      "[-compute_all]\n"
      "       " +
      std::string(argv[0]) +
      " -lists '<list1>;<list2>;...' [-prims <primitive>] "
      "[-prims_batch <prims>] [-compute_all] [-n_streams <N>] [-one_by_one]");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  try {
    ExecuteAggPrim task;
    if (FLAGS_n_streams > 0) {
      task.SetNumStreams(static_cast<uint32_t>(FLAGS_n_streams));
    }

    // Batch mode: multiple value lists.
    if (!FLAGS_lists.empty()) {
      auto list_strs = SplitBySemicolon(FLAGS_lists);
      std::vector<std::vector<FeatureValue>> lists;
      for (const auto& s : list_strs) {
        lists.push_back(ParseValues(s));
      }
      std::cout << "Batch input: " << lists.size() << " list(s), "
                << FLAGS_n_streams << " stream(s)"
                << (FLAGS_one_by_one ? ", one-by-one" : "")
                << std::endl;

      Clock::time_point t0 = Clock::now();
      if (FLAGS_one_by_one) {
        // One-by-one submission: each list is computed independently.
        if (FLAGS_compute_all) {
          std::vector<AllFeatures> all_results;
          all_results.reserve(lists.size());
          for (size_t i = 0; i < lists.size(); ++i) {
            all_results.push_back(
                task.ComputeAll(lists[i].data(), lists[i].size()));
          }
          std::cout << "[Timing] ComputeAll (one-by-one): " << SecondsSince(t0)
                    << " s" << std::endl;
          for (size_t i = 0; i < all_results.size(); ++i) {
            std::cout << "  list[" << i << "] sum=";
            PrintValue(all_results[i].sum);
            std::cout << " mean=";
            PrintValue(all_results[i].mean);
            std::cout << " count=";
            PrintValue(all_results[i].count);
            std::cout << std::endl;
          }
        } else if (!FLAGS_prims_batch.empty()) {
          auto prim_names = SplitByComma(FLAGS_prims_batch);
          std::vector<AggPrim> prims;
          for (const auto& name : prim_names) {
            prims.push_back(StringToAggPrim(name));
          }
          std::vector<std::vector<FeatureValue>> results(
              lists.size(), std::vector<FeatureValue>(prims.size()));
          for (size_t i = 0; i < lists.size(); ++i) {
            for (size_t j = 0; j < prims.size(); ++j) {
              results[i][j] = task.Compute(prims[j], lists[i].data(),
                                           lists[i].size());
            }
          }
          std::cout << "[Timing] Compute (one-by-one): " << SecondsSince(t0)
                    << " s" << std::endl;
          for (size_t i = 0; i < results.size(); ++i) {
            std::cout << "  list[" << i << "]";
            for (size_t j = 0; j < results[i].size(); ++j) {
              std::cout << " " << prim_names[j] << "=";
              PrintValue(results[i][j]);
            }
            std::cout << std::endl;
          }
        } else {
          auto prim_names = SplitByComma(FLAGS_prims);
          if (prim_names.size() != 1) {
            std::cerr << "One-by-one batch mode supports exactly one primitive "
                      << "via -prims; got: " << FLAGS_prims << std::endl;
            return EXIT_FAILURE;
          }
          AggPrim prim = StringToAggPrim(prim_names[0]);
          std::vector<FeatureValue> results(lists.size());
          for (size_t i = 0; i < lists.size(); ++i) {
            results[i] = task.Compute(prim, lists[i].data(), lists[i].size());
          }
          std::cout << "[Timing] Compute (one-by-one): " << SecondsSince(t0)
                    << " s" << std::endl;
          for (size_t i = 0; i < results.size(); ++i) {
            std::cout << "  list[" << i << "] " << prim_names[0] << "=";
            PrintValue(results[i]);
            std::cout << std::endl;
          }
        }
      } else if (FLAGS_compute_all) {
        auto all_results = task.ComputeAllBatch(lists);
        std::cout << "[Timing] ComputeAllBatch: " << SecondsSince(t0) << " s"
                  << std::endl;
        for (size_t i = 0; i < all_results.size(); ++i) {
          std::cout << "  list[" << i << "] sum=";
          PrintValue(all_results[i].sum);
          std::cout << " mean=";
          PrintValue(all_results[i].mean);
          std::cout << " count=";
          PrintValue(all_results[i].count);
          std::cout << std::endl;
        }
      } else if (!FLAGS_prims_batch.empty()) {
        // Multi-primitive batch: 2D matrix input, compute all prims per row.
        auto prim_names = SplitByComma(FLAGS_prims_batch);
        std::vector<AggPrim> prims;
        for (const auto& name : prim_names) {
          prims.push_back(StringToAggPrim(name));
        }
        auto results = task.ComputeBatchMultiPrim(lists, prims);
        std::cout << "[Timing] ComputeBatchMultiPrim: " << SecondsSince(t0)
                  << " s" << std::endl;
        for (size_t i = 0; i < results.size(); ++i) {
          std::cout << "  list[" << i << "]";
          for (size_t j = 0; j < results[i].size(); ++j) {
            std::cout << " " << prim_names[j] << "=";
            PrintValue(results[i][j]);
          }
          std::cout << std::endl;
        }
      } else {
        auto prim_names = SplitByComma(FLAGS_prims);
        if (prim_names.size() != 1) {
          std::cerr << "Batch mode without -prims_batch supports exactly one "
                    << "primitive; got: " << FLAGS_prims << std::endl;
          return EXIT_FAILURE;
        }
        AggPrim prim = StringToAggPrim(prim_names[0]);
        auto results = task.ComputeBatch(prim, lists);
        std::cout << "[Timing] ComputeBatch: " << SecondsSince(t0) << " s"
                  << std::endl;
        for (size_t i = 0; i < results.size(); ++i) {
          std::cout << "  list[" << i << "] " << prim_names[0] << "=";
          PrintValue(results[i]);
          std::cout << std::endl;
        }
      }
      gflags::ShutDownCommandLineFlags();
      return EXIT_SUCCESS;
    }

    // Single-list mode.
    auto values = ParseValues(FLAGS_values);
    if (values.empty()) {
      std::cerr << "Error: no input values provided" << std::endl;
      return EXIT_FAILURE;
    }

    std::cout << "Input values (" << values.size() << "): " << FLAGS_values
              << std::endl;

    Clock::time_point t0 = Clock::now();

    if (FLAGS_compute_all) {
      AllFeatures all = task.ComputeAll(values.data(), values.size());
      std::cout << "[Timing] ComputeAll: " << SecondsSince(t0) << " s"
                << std::endl;

      std::cout << "All primitives:" << std::endl;
      std::cout << "  count="; PrintValue(all.count); std::cout << std::endl;
      std::cout << "  sum="; PrintValue(all.sum); std::cout << std::endl;
      std::cout << "  mean="; PrintValue(all.mean); std::cout << std::endl;
      std::cout << "  median="; PrintValue(all.median); std::cout << std::endl;
      std::cout << "  mode="; PrintValue(all.mode); std::cout << std::endl;
      std::cout << "  min="; PrintValue(all.min); std::cout << std::endl;
      std::cout << "  max="; PrintValue(all.max); std::cout << std::endl;
      std::cout << "  variance="; PrintValue(all.variance); std::cout << std::endl;
      std::cout << "  std="; PrintValue(all.std); std::cout << std::endl;
      std::cout << "  skew="; PrintValue(all.skew); std::cout << std::endl;
      std::cout << "  entropy="; PrintValue(all.entropy); std::cout << std::endl;
      std::cout << "  num_unique="; PrintValue(all.num_unique); std::cout << std::endl;
      std::cout << "  percent_true="; PrintValue(all.percent_true); std::cout << std::endl;
      std::cout << "  quarter="; PrintValue(all.quarter); std::cout << std::endl;
      std::cout << "  quartile3="; PrintValue(all.quartile3); std::cout << std::endl;
      std::cout << "  count_greater_than_mean=";
      PrintValue(all.count_greater_than_mean);
      std::cout << std::endl;
      std::cout << "  dfeat="; PrintValue(all.dfeat); std::cout << std::endl;
    } else {
      auto prim_names = SplitByComma(FLAGS_prims);
      std::cout << "Primitives: " << FLAGS_prims << std::endl;

      for (const auto& name : prim_names) {
        AggPrim prim = StringToAggPrim(name);
        FeatureValue result = task.Compute(prim, values.data(), values.size());
        std::cout << "  " << name << "=";
        PrintValue(result);
        std::cout << std::endl;
      }
      std::cout << "[Timing] Compute: " << SecondsSince(t0) << " s"
                << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
