// Branch-only probe for the GraphAggregate C ABI conversion (H1 fix).
// Verifies type tags use MG_VALUE_* numbering and the union member matches
// the tag. Value expectations for sum/percent_true depend on the separate
// aggregation bug (sum=0 at HEAD 4cec1d5); tag and count checks do not.
#include <cstdio>
#include <cstring>

#include "core/go_api/matrixgraph_go_api.h"

int main() {
  void* h = matrixgraph_graph_aggregate_create();
  if (!h) {
    std::printf("FAIL create\n");
    return 1;
  }
  if (matrixgraph_graph_aggregate_load_synthetic(h, 100, 3) != 0) {
    std::printf("FAIL load_synthetic\n");
    return 1;
  }

  const uint32_t pivot = 0;
  MatrixGraphFeatureRequest reqs[3];
  std::memset(reqs, 0, sizeof(reqs));
  std::strncpy(reqs[0].attr_name, "score", sizeof(reqs[0].attr_name) - 1);
  reqs[0].use_outgoing = 1;
  reqs[0].prim = 0;  // featurelib kCount
  std::strncpy(reqs[1].attr_name, "score", sizeof(reqs[1].attr_name) - 1);
  reqs[1].use_outgoing = 1;
  reqs[1].prim = 1;  // featurelib kSum
  std::strncpy(reqs[2].attr_name, "flag", sizeof(reqs[2].attr_name) - 1);
  reqs[2].use_outgoing = 1;
  reqs[2].prim = 12;  // featurelib kPercentTrue

  MatrixGraphFeatureValue out[3];
  std::memset(out, 0, sizeof(out));
  int rc = matrixgraph_graph_aggregate_compute_features(h, &pivot, 1, reqs, 3,
                                                        out);
  std::printf("compute_features rc=%d\n", rc);
  std::printf("count: type=%d i64=%lld (want type=%d i64=3)\n", out[0].type,
              (long long)out[0].i64, MG_VALUE_INT);
  std::printf("sum:   type=%d f64=%g (want type=%d f64=3)\n", out[1].type,
              out[1].f64, MG_VALUE_FLOAT64);
  std::printf("pct:   type=%d f64=%g (want type=%d f64=0.333333)\n",
              out[2].type, out[2].f64, MG_VALUE_FLOAT64);

  bool tags_ok = rc == 0 && out[0].type == MG_VALUE_INT &&
                 out[1].type == MG_VALUE_FLOAT64 &&
                 out[2].type == MG_VALUE_FLOAT64;
  bool count_ok = rc == 0 && out[0].i64 == 3;
  bool sum_ok = rc == 0 && out[1].f64 == 3.0;
  bool pct_ok = rc == 0 && out[2].f64 > 0.3333 && out[2].f64 < 0.3334;
  std::printf("TAGS_%s COUNT_%s SUM_%s PCT_%s\n", tags_ok ? "OK" : "FAIL",
              count_ok ? "OK" : "FAIL", sum_ok ? "OK" : "FAIL",
              pct_ok ? "OK" : "FAIL");

  MatrixGraphAllFeatures all;
  std::memset(&all, 0, sizeof(all));
  rc = matrixgraph_graph_aggregate_compute_all(h, &pivot, 1, "score", 1, &all);
  std::printf("compute_all rc=%d count.type=%d count.i64=%lld sum.type=%d "
              "sum.f64=%g dfeat.type=%d\n",
              rc, all.count.type, (long long)all.count.i64, all.sum.type,
              all.sum.f64, all.dfeat.type);
  bool all_tags_ok = rc == 0 && all.count.type == MG_VALUE_INT &&
                     all.sum.type == MG_VALUE_FLOAT64;
  std::printf("ALL_TAGS_%s\n", all_tags_ok ? "OK" : "FAIL");

  matrixgraph_graph_aggregate_destroy(h);
  return (tags_ok && all_tags_ok) ? 0 : 1;
}
