#include "core/task/gpu_task/graph_aggregate.cuh"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using sics::matrixgraph::core::task::GraphAggregate;
using sics::matrixgraph::core::task::kernel::AggPrim;
using sics::matrixgraph::core::task::kernel::AllFeatures;
using sics::matrixgraph::core::task::kernel::FeatureRequest;
using sics::matrixgraph::core::task::kernel::FeatureValue;
using sics::matrixgraph::core::data_structures::AttributeName;
using VertexID = sics::matrixgraph::core::common::VertexID;

static bool float_eq(double a, double b, double eps = 1e-4) {
  return std::fabs(a - b) < eps;
}

int main() {
  std::cout << "[GraphAggregate Synthetic Test] Starting..." << std::endl;

  const uint32_t n_vertices = 100;
  const uint32_t out_deg = 3;

  // 1. Create task and load synthetic data.
  GraphAggregate task;
  task.LoadSyntheticData(n_vertices, out_deg);

  // 2. All vertices are pivots.
  std::vector<uint32_t> pivot_vids(n_vertices);
  for (uint32_t i = 0; i < n_vertices; ++i) pivot_vids[i] = i;

  // 3. Feature requests.
  std::vector<FeatureRequest> requests = {
      {AttributeName("score"), 0, true, AggPrim::kSum},
      {AttributeName("score"), 0, true, AggPrim::kMin},
      {AttributeName("score"), 0, true, AggPrim::kMax},
      {AttributeName("score"), 0, true, AggPrim::kMean},
      {AttributeName("score"), 0, true, AggPrim::kVariance},
      {AttributeName("score"), 0, true, AggPrim::kStd},
      {AttributeName("score"), 0, true, AggPrim::kMedian},
      {AttributeName("score"), 0, true, AggPrim::kCountGreaterThanMean},
      {AttributeName("score"), 0, true, AggPrim::kCount},
      {AttributeName("flag"), 0, true, AggPrim::kPercentTrue}};
  const uint32_t n_req = requests.size();

  // 4. Run feature computation.
  std::vector<FeatureValue> results =
      task.ComputeFeatures(pivot_vids, requests);

  // 5. Verify vertices without wrap-around (v + out_deg < n_vertices).
  bool pass = true;
  for (uint32_t v = 0; v + out_deg < n_vertices; ++v) {
    uint32_t base = v * n_req;

    // Neighbor IDs and scores.
    double s0 = (v + 1) * 0.5;
    double s1 = (v + 2) * 0.5;
    double s2 = (v + 3) * 0.5;

    // Expected values.
    double expected_sum = s0 + s1 + s2;
    double expected_min = s0;
    double expected_max = s2;
    double expected_mean = s1;
    double expected_variance = 2.0 * 0.5 * 0.5 / 3.0;  // population variance
    double expected_std = std::sqrt(expected_variance);
    double expected_median = s1;
    int64_t expected_count_gtm = 1;  // only max > mean
    int64_t expected_count = 3;

    uint32_t true_count = 0;
    for (uint32_t e = 0; e < out_deg; ++e) {
      VertexID neighbor = (v + 1 + e) % n_vertices;
      true_count += (neighbor % 2 == 0) ? 1 : 0;
    }
    double expected_pct = static_cast<double>(true_count) / out_deg;

    auto check = [&](const char* name, double expected, double actual) {
      if (!float_eq(expected, actual)) {
        std::cerr << "FAIL " << name << " at v=" << v
                  << " expected=" << expected << " got=" << actual << std::endl;
        pass = false;
        return false;
      }
      return true;
    };

    if (!check("sum", expected_sum, results[base + 0].ToDouble())) break;
    if (!check("min", expected_min, results[base + 1].ToDouble())) break;
    if (!check("max", expected_max, results[base + 2].ToDouble())) break;
    if (!check("mean", expected_mean, results[base + 3].ToDouble())) break;
    if (!check("variance", expected_variance, results[base + 4].ToDouble())) break;
    if (!check("std", expected_std, results[base + 5].ToDouble())) break;
    if (!check("median", expected_median, results[base + 6].ToDouble())) break;
    if (results[base + 7].i64 != expected_count_gtm) {
      std::cerr << "FAIL count_gtm at v=" << v << " expected=" << expected_count_gtm
                << " got=" << results[base + 7].i64 << std::endl;
      pass = false;
      break;
    }
    if (results[base + 8].i64 != expected_count) {
      std::cerr << "FAIL count at v=" << v << " expected=" << expected_count
                << " got=" << results[base + 8].i64 << std::endl;
      pass = false;
      break;
    }
    if (!check("percent_true", expected_pct, results[base + 9].ToDouble())) break;
  }

  if (pass) {
    std::cout << "[GraphAggregate Synthetic Test] PASSED (" << n_vertices
              << " vertices, " << n_req << " requests)" << std::endl;
  } else {
    return 1;
  }

  // -------------------------------------------------------------------------
  // Fused compute_all test: verify it matches the per-request results above.
  // -------------------------------------------------------------------------
  std::vector<AllFeatures> all_results =
      task.ComputeAll(pivot_vids, AttributeName("score"), true);

  bool all_pass = true;
  for (uint32_t v = 0; v + out_deg < n_vertices; ++v) {
    uint32_t base = v * n_req;
    const AllFeatures& a = all_results[v];

    auto check_all = [&](const char* name, double expected,
                         const FeatureValue& actual) {
      if (!float_eq(expected, actual.ToDouble())) {
        std::cerr << "FAIL compute_all " << name << " at v=" << v
                  << " expected=" << expected << " got=" << actual.ToDouble()
                  << std::endl;
        all_pass = false;
        return false;
      }
      return true;
    };

    if (!check_all("sum", results[base + 0].ToDouble(), a.sum)) break;
    if (!check_all("min", results[base + 1].ToDouble(), a.min)) break;
    if (!check_all("max", results[base + 2].ToDouble(), a.max)) break;
    if (!check_all("mean", results[base + 3].ToDouble(), a.mean)) break;
    if (!check_all("variance", results[base + 4].ToDouble(), a.variance)) break;
    if (!check_all("std", results[base + 5].ToDouble(), a.std)) break;
    if (!check_all("median", results[base + 6].ToDouble(), a.median)) break;
    if (results[base + 7].i64 != a.count_greater_than_mean.i64) {
      std::cerr << "FAIL compute_all count_gtm at v=" << v
                << " expected=" << results[base + 7].i64
                << " got=" << a.count_greater_than_mean.i64 << std::endl;
      all_pass = false;
      break;
    }
    if (results[base + 8].i64 != a.count.i64) {
      std::cerr << "FAIL compute_all count at v=" << v
                << " expected=" << results[base + 8].i64
                << " got=" << a.count.i64 << std::endl;
      all_pass = false;
      break;
    }
    // percent_true on a non-bool attribute is defined as 0.
    if (!check_all("percent_true", 0.0, a.percent_true)) break;
  }

  if (!all_pass) return 1;

  // Test ComputeAll on the bool attribute "flag" for PercentTrue.
  std::vector<AllFeatures> flag_results =
      task.ComputeAll(pivot_vids, AttributeName("flag"), true);
  for (uint32_t v = 0; v + out_deg < n_vertices; ++v) {
    uint32_t base = v * n_req;
    double expected_pct = results[base + 9].ToDouble();
    if (!float_eq(expected_pct, flag_results[v].percent_true.ToDouble())) {
      std::cerr << "FAIL compute_all flag percent_true at v=" << v
                << " expected=" << expected_pct
                << " got=" << flag_results[v].percent_true.ToDouble()
                << std::endl;
      all_pass = false;
      break;
    }
  }

  if (all_pass) {
    std::cout << "[GraphAggregate ComputeAll Test] PASSED (" << n_vertices
              << " vertices)" << std::endl;
    return 0;
  }
  return 1;
}
