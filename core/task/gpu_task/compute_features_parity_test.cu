/**
 * @file compute_features_parity_test.cu
 * @brief Parity harness for the ComputeFeatures plan interpreter.
 *
 * Every aggregation primitive runs through the plan interpreter on a fixed
 * ring graph, and each result is compared with a host reference that mirrors
 * block_agg_prims (the semantic anchor). The same table must pass before and
 * after the value-type unification, so a diff here means behavior moved.
 */

#include "core/data_structures/immutable_csr.cuh"
#include "core/task/gpu_task/compute_features.cuh"
#include "core/util/cuda_check.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using VertexID = sics::matrixgraph::core::common::VertexID;
using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using data_structures::ImmutableCSR;

namespace {

constexpr uint32_t kNVertices = 8;
constexpr uint32_t kOutDegree = 3;

std::vector<VertexID> Neighbors(uint32_t v) {
  std::vector<VertexID> out;
  for (uint32_t e = 1; e <= kOutDegree; ++e) out.push_back((v + e) % kNVertices);
  return out;
}

std::unique_ptr<ImmutableCSR> BuildRingCSR(uint32_t n_vertices) {
  auto csr = std::make_unique<ImmutableCSR>();
  const uint32_t n_edges = n_vertices * kOutDegree;
  const uint32_t max_vid = n_vertices - 1;

  const size_t buf_size = sizeof(VertexID) * n_vertices * 3 +
                          sizeof(EdgeIndex) * (n_vertices + 1) * 2 +
                          sizeof(VertexID) * n_edges * 2 +
                          sizeof(VertexID) * (max_vid + 1) * 2;
  uint8_t* buf = new uint8_t[buf_size]();
  csr->SetNumVertices(n_vertices);
  csr->SetNumIncomingEdges(n_edges);
  csr->SetNumOutgoingEdges(n_edges);
  csr->SetMaxVid(max_vid);
  csr->SetMinVid(0);
  csr->SetGraphBuffer(buf);
  csr->SetVertexLabelBuffer(n_vertices);
  csr->ParseBasePtr(buf);

  VertexID* outdegree = csr->GetOutDegreeBasePointer();
  EdgeIndex* out_offset = csr->GetOutOffsetBasePointer();
  VertexID* outgoing_edges = csr->GetOutgoingEdgesBasePointer();
  VertexID* indegree = csr->GetInDegreeBasePointer();
  EdgeIndex* in_offset = csr->GetInOffsetBasePointer();
  VertexID* incoming_edges = csr->GetIncomingEdgesBasePointer();

  out_offset[0] = 0;
  for (uint32_t v = 0; v < n_vertices; ++v) {
    auto nbrs = Neighbors(v);
    outdegree[v] = static_cast<VertexID>(nbrs.size());
    out_offset[v + 1] = out_offset[v] + outdegree[v];
    for (uint32_t i = 0; i < nbrs.size(); ++i) {
      outgoing_edges[out_offset[v] + i] = nbrs[i];
    }
  }

  in_offset[0] = 0;
  for (uint32_t v = 0; v < n_vertices; ++v) {
    uint32_t deg = 0;
    for (uint32_t u = 0; u < n_vertices; ++u) {
      auto nbrs = Neighbors(u);
      if (std::find(nbrs.begin(), nbrs.end(), v) != nbrs.end()) {
        incoming_edges[in_offset[v] + deg] = u;
        ++deg;
      }
    }
    indegree[v] = deg;
    in_offset[v + 1] = in_offset[v] + deg;
  }

  VertexID* globalid = csr->GetGloablIDBasePointer();
  for (uint32_t v = 0; v < n_vertices; ++v) globalid[v] = v;
  return csr;
}

MatrixGraphPlanNode AttrNode(const char* key) {
  MatrixGraphPlanNode n{};
  n.type = MG_EXPR_ATTR;
  std::strncpy(n.key, key, sizeof(n.key) - 1);
  return n;
}

MatrixGraphPlanNode AggNode(int32_t op, int32_t src_idx, int32_t nav_idx) {
  MatrixGraphPlanNode n{};
  n.type = MG_EXPR_AGG;
  n.op = op;
  n.src_idx = src_idx;
  n.nav_idx = nav_idx;
  return n;
}

MatrixGraphPlanNode NeighborNav() {
  MatrixGraphPlanNode n{};
  n.type = MG_NAV_NEIGHBOR;
  n.direction = 0;
  return n;
}

MatrixGraphPlanNode SelfNav() {
  MatrixGraphPlanNode n{};
  n.type = MG_NAV_SELF;
  return n;
}

ComputeFeaturesAttributeColumn MakeColumn(const char* key, int32_t type,
                                          uint32_t n_values, void* values) {
  ComputeFeaturesAttributeColumn col{};
  std::strncpy(col.key, key, sizeof(col.key) - 1);
  col.value_type = type;
  col.n_values = n_values;
  col.values = values;
  return col;
}

// Host references mirroring block_agg_prims. Values are sorted in place.
double RefMedian(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

double RefMode(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  size_t best = 1, curr = 1;
  double mode = v[0];
  for (size_t i = 1; i < v.size(); ++i) {
    curr = (v[i] == v[i - 1]) ? curr + 1 : 1;
    if (curr > best) {
      best = curr;
      mode = v[i];
    }
  }
  return mode;
}

double RefEntropy(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  double entropy = 0.0;
  size_t i = 0;
  while (i < v.size()) {
    size_t j = i + 1;
    while (j < v.size() && v[j] == v[i]) ++j;
    double p = static_cast<double>(j - i) / static_cast<double>(v.size());
    entropy -= p * std::log2(p);
    i = j;
  }
  return entropy;
}

double RefQuantile(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  double pos = q * static_cast<double>(v.size() - 1);
  size_t lower = static_cast<size_t>(std::floor(pos));
  size_t upper = static_cast<size_t>(std::ceil(pos));
  double w = pos - static_cast<double>(lower);
  return v[lower] * (1.0 - w) + v[upper] * w;
}

}  // namespace

TEST(ComputeFeaturesParity, AllPrimitivesMatchHostReference) {
  const std::string tmp_path = "/tmp/matrixgraph_compute_features_parity_ring";
  auto graph = BuildRingCSR(kNVertices);
  graph->Write(tmp_path, 0);

  ComputeFeaturesTask task(tmp_path);
  task.LoadGraph(tmp_path);

  std::vector<double> score(kNVertices);
  std::vector<int64_t> rank(kNVertices);
  std::vector<uint8_t> flag(kNVertices);
  for (uint32_t v = 0; v < kNVertices; ++v) {
    score[v] = 0.5 + 0.25 * v;
    rank[v] = static_cast<int64_t>(v) * 10;
    flag[v] = (v % 2) ? 1 : 0;
  }
  std::vector<ComputeFeaturesAttributeColumn> cols;
  cols.push_back(MakeColumn("score", MG_VALUE_FLOAT64, kNVertices, score.data()));
  cols.push_back(MakeColumn("rank", MG_VALUE_INT, kNVertices, rank.data()));
  cols.push_back(MakeColumn("flag", MG_VALUE_BOOL, kNVertices, flag.data()));
  task.LoadAttributes(static_cast<uint32_t>(cols.size()), cols.data());

  // Plan holds expressions only; navigators travel separately and are
  // referenced by nav_idx. 0=score attr, 1=rank attr, 2=flag attr.
  std::vector<MatrixGraphPlanNode> plan;
  plan.push_back(AttrNode("score"));   // 0
  plan.push_back(AttrNode("rank"));    // 1
  plan.push_back(AttrNode("flag"));    // 2

  std::vector<MatrixGraphPlanNode> navs;
  navs.push_back(NeighborNav());  // 0
  navs.push_back(SelfNav());      // 1

  struct PrimCase {
    int32_t op;
    int32_t src_idx;
    int32_t nav_idx;
    const char* name;
  };
  const std::vector<PrimCase> cases = {
      {MG_AGG_COUNT, 0, 0, "count"},
      {MG_AGG_SUM, 0, 0, "sum"},
      {MG_AGG_MEAN, 0, 0, "mean"},
      {MG_AGG_MEDIAN, 0, 0, "median"},
      {MG_AGG_MODE, 0, 0, "mode"},
      {MG_AGG_MIN, 0, 0, "min"},
      {MG_AGG_MAX, 0, 0, "max"},
      {MG_AGG_VARIANCE, 0, 0, "variance"},
      {MG_AGG_STD, 0, 0, "std"},
      {MG_AGG_SKEW, 0, 0, "skew"},
      {MG_AGG_ENTROPY, 0, 0, "entropy"},
      {MG_AGG_NUM_UNIQUE, 0, 0, "num_unique"},
      {MG_AGG_PERCENT_TRUE, 2, 0, "percent_true"},
      {MG_AGG_QUARTER, 0, 0, "quarter"},
      {MG_AGG_QUARTILE3, 0, 0, "quartile3"},
      {MG_AGG_COUNT_GREATER_THAN_MEAN, 0, 0, "count_gt_mean"},
      {MG_AGG_SUM, 1, 0, "sum_int"},
      {MG_AGG_MEAN, 1, 0, "mean_int"},
  };

  std::vector<int32_t> outputs;
  for (size_t i = 0; i < cases.size(); ++i) {
    plan.push_back(AggNode(cases[i].op, cases[i].src_idx, cases[i].nav_idx));
    outputs.push_back(static_cast<int32_t>(3 + i));
  }

  std::vector<uint32_t> pivots;
  for (uint32_t v = 0; v < kNVertices; ++v) pivots.push_back(v);
  std::vector<MatrixGraphCondNode> conds;

  auto result = task.Compute(pivots, plan, navs, conds, outputs);
  ASSERT_EQ(result.size(), pivots.size() * cases.size());

  for (size_t p = 0; p < pivots.size(); ++p) {
    uint32_t v = pivots[p];
    std::vector<double> vals;
    std::vector<double> ranks;
    for (VertexID u : Neighbors(v)) {
      vals.push_back(score[u]);
      ranks.push_back(static_cast<double>(rank[u]));
    }
    double mean = 0.0;
    for (double x : vals) mean += x;
    mean /= static_cast<double>(vals.size());
    double var = 0.0;
    for (double x : vals) var += (x - mean) * (x - mean);
    var /= static_cast<double>(vals.size());
    double stdv = std::sqrt(var);
    double skew = 0.0;
    for (double x : vals) {
      double z = (x - mean) / stdv;
      skew += z * z * z;
    }
    skew /= static_cast<double>(vals.size());
    int true_count = 0;
    for (VertexID u : Neighbors(v)) true_count += flag[u] ? 1 : 0;
    int count_gt_mean = 0;
    for (double x : vals) count_gt_mean += (x > mean) ? 1 : 0;
    std::vector<double> sorted = vals;
    std::sort(sorted.begin(), sorted.end());
    int num_unique = 1;
    for (size_t i = 1; i < sorted.size(); ++i) {
      if (sorted[i] != sorted[i - 1]) ++num_unique;
    }

    auto expect_near = [&](size_t idx, double want, const char* name) {
      SCOPED_TRACE(testing::Message() << "pivot=" << v << " prim=" << name);
      EXPECT_EQ(result[idx].type, MG_VALUE_FLOAT64);
      EXPECT_NEAR(result[idx].f64, want, 1e-9);
    };

    const size_t base = p * cases.size();
    expect_near(base + 1, mean * 3.0, "sum");
    expect_near(base + 2, mean, "mean");
    expect_near(base + 3, RefMedian(vals), "median");
    expect_near(base + 4, RefMode(vals), "mode");
    expect_near(base + 5, *std::min_element(vals.begin(), vals.end()), "min");
    expect_near(base + 6, *std::max_element(vals.begin(), vals.end()), "max");
    expect_near(base + 7, var, "variance");
    expect_near(base + 8, stdv, "std");
    expect_near(base + 9, skew, "skew");
    expect_near(base + 10, RefEntropy(vals), "entropy");
    expect_near(base + 13, RefQuantile(vals, 0.25), "quarter");
    expect_near(base + 14, RefQuantile(vals, 0.75), "quartile3");
    expect_near(base + 16, ranks[0] + ranks[1] + ranks[2], "sum_int");
    expect_near(base + 17, (ranks[0] + ranks[1] + ranks[2]) / 3.0, "mean_int");

    EXPECT_EQ(result[base + 0].type, MG_VALUE_INT);
    EXPECT_EQ(result[base + 0].i64, 3);
    EXPECT_EQ(result[base + 11].type, MG_VALUE_INT);
    EXPECT_EQ(result[base + 11].i64, num_unique);
    expect_near(base + 12,
                static_cast<double>(true_count) / static_cast<double>(kOutDegree),
                "percent_true");
    EXPECT_EQ(result[base + 15].type, MG_VALUE_INT);
    EXPECT_EQ(result[base + 15].i64, count_gt_mean);
  }
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics