// Quick sanity test for the ExecuteAggPrim C API (used by Go CGO).
// Build & run:
//   g++ -std=c++17 -I../../core/go_api -I../.. execute_agg_prim_go_api_test.cpp \
//       -L../../../lib -lmatrixgraph_goapi -lcudart -o /tmp/execute_agg_prim_go_api_test
//   LD_LIBRARY_PATH=../../../lib /tmp/execute_agg_prim_go_api_test

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "matrixgraph_go_api.h"

static MatrixGraphFeatureValue IntValue(int64_t v) {
  MatrixGraphFeatureValue r;
  r.type = MG_VALUE_INT;
  r.i64 = v;
  return r;
}

static MatrixGraphFeatureValue FloatValue(double v) {
  MatrixGraphFeatureValue r;
  r.type = MG_VALUE_FLOAT64;
  r.f64 = v;
  return r;
}

static int CheckDouble(const char* name, double expected, double actual) {
  if (std::fabs(expected - actual) > 1e-6) {
    std::fprintf(stderr, "FAIL %s: expected %.3f, got %.3f\n", name, expected,
                 actual);
    return 1;
  }
  std::printf("PASS %s = %.3f\n", name, actual);
  return 0;
}

static int CheckInt64(const char* name, int64_t expected, int64_t actual) {
  if (expected != actual) {
    std::fprintf(stderr, "FAIL %s: expected %ld, got %ld\n", name, expected,
                 actual);
    return 1;
  }
  std::printf("PASS %s = %ld\n", name, actual);
  return 0;
}

static int CheckValue(const char* name, double expected,
                      const MatrixGraphFeatureValue& v) {
  if (v.type == MG_VALUE_INT || v.type == MG_VALUE_TIME) {
    return CheckInt64(name, static_cast<int64_t>(expected), v.i64);
  }
  if (v.type == MG_VALUE_BOOL) {
    return CheckInt64(name, static_cast<int64_t>(expected), v.b);
  }
  return CheckDouble(name, expected, v.f64);
}

int main() {
  int failures = 0;

  void* handle = matrixgraph_execute_agg_prim_create();
  if (!handle) {
    std::fprintf(stderr, "FAIL: create\n");
    return 1;
  }
  matrixgraph_execute_agg_prim_set_num_streams(handle, 2);

  // Single list, single primitive.
  MatrixGraphFeatureValue values[] = {IntValue(1), IntValue(2), IntValue(3),
                                      IntValue(4), IntValue(5)};
  MatrixGraphFeatureValue out;
  if (matrixgraph_execute_agg_prim_compute(handle, MG_EXEC_AGG_SUM, values, 5,
                                           &out) != 0) {
    std::fprintf(stderr, "FAIL: compute\n");
    failures++;
  } else {
    failures += CheckDouble("sum", 15.0, out.f64);
  }

  // Single list, all primitives.
  MatrixGraphExecuteAllFeatures all;
  if (matrixgraph_execute_agg_prim_compute_all(handle, values, 5, &all) != 0) {
    std::fprintf(stderr, "FAIL: compute_all\n");
    failures++;
  } else {
    failures += CheckDouble("all.sum", 15.0, all.sum.f64);
    failures += CheckDouble("all.mean", 3.0, all.mean.f64);
    failures += CheckInt64("all.count", 5, all.count.i64);
  }

  // Batch: two lists, single primitive (Sum).
  MatrixGraphFeatureValue flat[] = {IntValue(1), IntValue(2), IntValue(3),
                                    IntValue(4), IntValue(5), IntValue(6),
                                    IntValue(7)};
  uint32_t offsets[] = {0, 3, 7};
  MatrixGraphFeatureValue batch_out[2];
  if (matrixgraph_execute_agg_prim_compute_batch(handle, MG_EXEC_AGG_SUM, flat,
                                                 offsets, 2, batch_out) != 0) {
    std::fprintf(stderr, "FAIL: compute_batch\n");
    failures++;
  } else {
    failures += CheckDouble("batch[0].sum", 6.0, batch_out[0].f64);
    failures += CheckDouble("batch[1].sum", 22.0, batch_out[1].f64);
  }

  // Batch multi-primitive: two lists x {Sum, Mean, Count, Min, Max}.
  int32_t prims[] = {MG_EXEC_AGG_SUM, MG_EXEC_AGG_MEAN, MG_EXEC_AGG_COUNT,
                     MG_EXEC_AGG_MIN, MG_EXEC_AGG_MAX};
  MatrixGraphFeatureValue mp_out[2 * 5];
  if (matrixgraph_execute_agg_prim_compute_batch_multi_prim(
          handle, flat, offsets, 2, prims, 5, mp_out) != 0) {
    std::fprintf(stderr, "FAIL: compute_batch_multi_prim\n");
    failures++;
  } else {
    failures += CheckDouble("mp[0].sum", 6.0, mp_out[0].f64);
    failures += CheckDouble("mp[0].mean", 2.0, mp_out[1].f64);
    failures += CheckInt64("mp[0].count", 3, mp_out[2].i64);
    failures += CheckValue("mp[0].min", 1.0, mp_out[3]);
    failures += CheckValue("mp[0].max", 3.0, mp_out[4]);

    failures += CheckDouble("mp[1].sum", 22.0, mp_out[5].f64);
    failures += CheckDouble("mp[1].mean", 5.5, mp_out[6].f64);
    failures += CheckInt64("mp[1].count", 4, mp_out[7].i64);
    failures += CheckValue("mp[1].min", 4.0, mp_out[8]);
    failures += CheckValue("mp[1].max", 7.0, mp_out[9]);
  }

  matrixgraph_execute_agg_prim_destroy(handle);

  if (failures == 0) {
    std::printf("ExecuteAggPrim Go API test: ALL PASSED\n");
    return 0;
  }
  std::fprintf(stderr, "ExecuteAggPrim Go API test: %d failure(s)\n", failures);
  return 1;
}
