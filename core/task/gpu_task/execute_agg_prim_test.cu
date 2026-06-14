#include "core/task/gpu_task/execute_agg_prim.cuh"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using sics::matrixgraph::core::task::AggPrim;
using sics::matrixgraph::core::task::AllFeatures;
using sics::matrixgraph::core::task::ExecuteAggPrim;
using sics::matrixgraph::core::task::FeatureValue;
using sics::matrixgraph::core::task::MakeBoolValue;
using sics::matrixgraph::core::task::MakeFloat64Value;
using sics::matrixgraph::core::task::MakeIntValue;

static bool float_eq(double a, double b, double eps = 1e-4) {
  return std::fabs(a - b) < eps;
}

static bool check(const char* name, double expected, double actual) {
  if (!float_eq(expected, actual)) {
    std::cerr << "FAIL " << name << " expected=" << expected
              << " got=" << actual << std::endl;
    return false;
  }
  return true;
}

static bool test_single_primitives() {
  std::cout << "[ExecuteAggPrim Test] Single primitives..." << std::endl;

  ExecuteAggPrim task;
  bool pass = true;

  // Numeric list: [1, 2, 3, 4, 5]
  std::vector<FeatureValue> nums;
  for (int i = 1; i <= 5; ++i) nums.push_back(MakeIntValue(i));

  auto count = task.Compute(AggPrim::kCount, nums.data(), nums.size());
  auto sum = task.Compute(AggPrim::kSum, nums.data(), nums.size());
  auto mean = task.Compute(AggPrim::kMean, nums.data(), nums.size());
  auto median = task.Compute(AggPrim::kMedian, nums.data(), nums.size());
  auto mode = task.Compute(AggPrim::kMode, nums.data(), nums.size());
  auto minv = task.Compute(AggPrim::kMin, nums.data(), nums.size());
  auto maxv = task.Compute(AggPrim::kMax, nums.data(), nums.size());
  auto var = task.Compute(AggPrim::kVariance, nums.data(), nums.size());
  auto stdv = task.Compute(AggPrim::kStd, nums.data(), nums.size());
  auto q1 = task.Compute(AggPrim::kQuarter, nums.data(), nums.size());
  auto q3 = task.Compute(AggPrim::kQuartile3, nums.data(), nums.size());
  auto cgm = task.Compute(AggPrim::kCountGreaterThanMean, nums.data(),
                          nums.size());
  auto nu = task.Compute(AggPrim::kNumUnique, nums.data(), nums.size());

  // Expected values for [1,2,3,4,5]:
  // count=5, sum=15, mean=3, median=3 (upper median), min=1, max=5
  // variance = ((1-3)^2 + ... + (5-3)^2) / 5 = 10/5 = 2
  // std = sqrt(2), q1 = 2 (pos=1), q3 = 4 (pos=3)
  // count_greater_than_mean: values > 3 are {4,5} -> 2
  // num_unique = 5
  if (count.i64 != 5) pass = false;
  pass &= check("sum", 15.0, sum.ToDouble());
  pass &= check("mean", 3.0, mean.ToDouble());
  pass &= check("median", 3.0, median.ToDouble());
  pass &= check("min", 1.0, minv.ToDouble());
  pass &= check("max", 5.0, maxv.ToDouble());
  pass &= check("variance", 2.0, var.ToDouble());
  pass &= check("std", std::sqrt(2.0), stdv.ToDouble());
  pass &= check("q1", 2.0, q1.ToDouble());
  pass &= check("q3", 4.0, q3.ToDouble());
  if (cgm.i64 != 2) pass = false;
  if (nu.i64 != 5) pass = false;

  // Mode test: [1, 2, 2, 3, 3] -> mode = 2 (first record-breaking)
  std::vector<FeatureValue> mode_nums = {
      MakeIntValue(1), MakeIntValue(2), MakeIntValue(2),
      MakeIntValue(3), MakeIntValue(3)};
  auto mode_result = task.Compute(AggPrim::kMode, mode_nums.data(),
                                  mode_nums.size());
  if (mode_result.i64 != 2) {
    std::cerr << "FAIL mode expected=2 got=" << mode_result.i64 << std::endl;
    pass = false;
  }

  // Entropy test: [1,1,2,2] -> p=0.5 each -> entropy=1.0
  std::vector<FeatureValue> entropy_nums = {
      MakeIntValue(1), MakeIntValue(1), MakeIntValue(2), MakeIntValue(2)};
  auto entropy = task.Compute(AggPrim::kEntropy, entropy_nums.data(),
                              entropy_nums.size());
  pass &= check("entropy", 1.0, entropy.ToDouble());

  // PercentTrue test: [true, false, true, true] -> 0.75
  std::vector<FeatureValue> bools = {
      MakeBoolValue(true), MakeBoolValue(false),
      MakeBoolValue(true), MakeBoolValue(true)};
  auto pct = task.Compute(AggPrim::kPercentTrue, bools.data(), bools.size());
  pass &= check("percent_true", 0.75, pct.ToDouble());

  // DFeat test
  std::vector<FeatureValue> single = {MakeIntValue(42)};
  auto dfeat = task.Compute(AggPrim::kDFeat, single.data(), single.size());
  if (dfeat.i64 != 42) {
    std::cerr << "FAIL dfeat expected=42 got=" << dfeat.i64 << std::endl;
    pass = false;
  }

  // Empty input -> Invalid
  auto empty = task.Compute(AggPrim::kSum, nullptr, 0);
  if (empty.IsValid()) {
    std::cerr << "FAIL empty sum should be Invalid" << std::endl;
    pass = false;
  }

  if (pass) {
    std::cout << "[ExecuteAggPrim Test] Single primitives PASSED" << std::endl;
  }
  return pass;
}

static bool test_batch() {
  std::cout << "[ExecuteAggPrim Test] Batch Sum..." << std::endl;

  ExecuteAggPrim task;
  std::vector<std::vector<FeatureValue>> lists = {
      {MakeIntValue(1), MakeIntValue(2), MakeIntValue(3)},
      {MakeIntValue(10), MakeIntValue(20)}};

  auto results = task.ComputeBatch(AggPrim::kSum, lists);
  bool pass = true;
  pass &= check("batch[0]", 6.0, results[0].ToDouble());
  pass &= check("batch[1]", 30.0, results[1].ToDouble());

  if (pass) {
    std::cout << "[ExecuteAggPrim Test] Batch PASSED" << std::endl;
  }
  return pass;
}

static bool test_compute_all() {
  std::cout << "[ExecuteAggPrim Test] ComputeAll..." << std::endl;

  ExecuteAggPrim task;
  std::vector<FeatureValue> nums;
  for (int i = 1; i <= 5; ++i) nums.push_back(MakeIntValue(i));

  AllFeatures all = task.ComputeAll(nums.data(), nums.size());
  bool pass = true;

  if (all.count.i64 != 5) pass = false;
  pass &= check("all.sum", 15.0, all.sum.ToDouble());
  pass &= check("all.mean", 3.0, all.mean.ToDouble());
  pass &= check("all.median", 3.0, all.median.ToDouble());
  pass &= check("all.min", 1.0, all.min.ToDouble());
  pass &= check("all.max", 5.0, all.max.ToDouble());
  pass &= check("all.variance", 2.0, all.variance.ToDouble());
  pass &= check("all.std", std::sqrt(2.0), all.std.ToDouble());
  pass &= check("all.q1", 2.0, all.quarter.ToDouble());
  pass &= check("all.q3", 4.0, all.quartile3.ToDouble());
  if (all.count_greater_than_mean.i64 != 2) pass = false;
  if (all.num_unique.i64 != 5) pass = false;

  // Compare ComputeAll with individual Compute calls.
  auto entropy = task.Compute(AggPrim::kEntropy, nums.data(), nums.size());
  pass &= check("all.entropy", entropy.ToDouble(), all.entropy.ToDouble());

  if (pass) {
    std::cout << "[ExecuteAggPrim Test] ComputeAll PASSED" << std::endl;
  }
  return pass;
}

static bool test_batch_multi_prim() {
  std::cout << "[ExecuteAggPrim Test] ComputeBatchMultiPrim..." << std::endl;

  ExecuteAggPrim task;
  task.SetNumStreams(2);
  std::vector<std::vector<FeatureValue>> inputs = {
      {MakeIntValue(1), MakeIntValue(2), MakeIntValue(3)},
      {MakeIntValue(4), MakeIntValue(5), MakeIntValue(6), MakeIntValue(7)}};
  std::vector<AggPrim> prims = {
      AggPrim::kSum, AggPrim::kMean, AggPrim::kCount, AggPrim::kMin};

  auto results = task.ComputeBatchMultiPrim(inputs, prims);
  bool pass = true;

  // list[0]: [1,2,3] -> sum=6, mean=2, count=3, min=1
  pass &= check("bmp[0].sum", 6.0, results[0][0].ToDouble());
  pass &= check("bmp[0].mean", 2.0, results[0][1].ToDouble());
  if (results[0][2].i64 != 3) pass = false;
  pass &= check("bmp[0].min", 1.0, results[0][3].ToDouble());

  // list[1]: [4,5,6,7] -> sum=22, mean=5.5, count=4, min=4
  pass &= check("bmp[1].sum", 22.0, results[1][0].ToDouble());
  pass &= check("bmp[1].mean", 5.5, results[1][1].ToDouble());
  if (results[1][2].i64 != 4) pass = false;
  pass &= check("bmp[1].min", 4.0, results[1][3].ToDouble());

  if (pass) {
    std::cout << "[ExecuteAggPrim Test] ComputeBatchMultiPrim PASSED"
              << std::endl;
  }
  return pass;
}

int main() {
  bool pass = true;
  pass &= test_single_primitives();
  pass &= test_batch();
  pass &= test_compute_all();
  pass &= test_batch_multi_prim();

  if (pass) {
    std::cout << "[ExecuteAggPrim Test] ALL PASSED" << std::endl;
    return 0;
  }
  std::cerr << "[ExecuteAggPrim Test] FAILED" << std::endl;
  return 1;
}
