/**
 * @file compute_features_test.cu
 * @brief Unit tests for ComputeFeaturesTask (requires googletest).
 *
 * These tests are compiled only when TEST=ON and third_party/googletest is
 * populated.  They mirror the checks in apps/compute_features.cpp.
 */

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/data_structures/immutable_csr.cuh"
#include "core/task/gpu_task/compute_features.cuh"
#include "core/util/cuda_check.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using VertexID = sics::matrixgraph::core::common::VertexID;
using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using kernel::kAggMean;
using kernel::kAggSum;

namespace {

constexpr uint32_t kNVertices = 4;

std::unique_ptr<ImmutableCSR> BuildRingCSR(uint32_t n_vertices) {
  auto csr = std::make_unique<ImmutableCSR>();
  uint32_t n_edges = n_vertices;
  uint32_t max_vid = n_vertices - 1;

  size_t buf_size =
      sizeof(VertexID) * n_vertices +         // globalid
      sizeof(VertexID) * n_vertices +         // indegree
      sizeof(VertexID) * n_vertices +         // outdegree
      sizeof(EdgeIndex) * (n_vertices + 1) +  // in_offset
      sizeof(EdgeIndex) * (n_vertices + 1) +  // out_offset
      sizeof(VertexID) * n_edges +            // incoming_edges
      sizeof(VertexID) * n_edges +            // outgoing_edges
      sizeof(VertexID) * (max_vid + 1) +      // edges_globalid
      sizeof(VertexID) * (max_vid + 1);       // localid_by_globalid

  uint8_t* buf = new uint8_t[buf_size]();
  csr->SetNumVertices(n_vertices);
  csr->SetNumIncomingEdges(n_edges);
  csr->SetNumOutgoingEdges(n_edges);
  csr->SetMaxVid(max_vid);
  csr->SetMinVid(0);
  csr->SetGraphBuffer(buf);
  csr->SetVertexLabelBuffer(n_vertices);
  csr->ParseBasePtr(buf);

  VertexID* globalid = csr->GetGloablIDBasePointer();
  VertexID* indegree = csr->GetInDegreeBasePointer();
  VertexID* outdegree = csr->GetOutDegreeBasePointer();
  EdgeIndex* in_offset = csr->GetInOffsetBasePointer();
  EdgeIndex* out_offset = csr->GetOutOffsetBasePointer();
  VertexID* incoming_edges = csr->GetIncomingEdgesBasePointer();
  VertexID* outgoing_edges = csr->GetOutgoingEdgesBasePointer();

  std::vector<std::vector<VertexID>> out_edges_host(n_vertices);
  std::vector<std::vector<VertexID>> in_edges_host(n_vertices);

  for (uint32_t v = 0; v < n_vertices; ++v) {
    globalid[v] = v;
    VertexID dst = (v + 1) % n_vertices;
    out_edges_host[v].push_back(dst);
    in_edges_host[dst].push_back(v);
  }

  out_offset[0] = 0;
  for (uint32_t v = 0; v < n_vertices; ++v) {
    outdegree[v] = static_cast<VertexID>(out_edges_host[v].size());
    out_offset[v + 1] = out_offset[v] + outdegree[v];
    for (uint32_t i = 0; i < outdegree[v]; ++i) {
      outgoing_edges[out_offset[v] + i] = out_edges_host[v][i];
    }
  }

  in_offset[0] = 0;
  for (uint32_t v = 0; v < n_vertices; ++v) {
    indegree[v] = static_cast<VertexID>(in_edges_host[v].size());
    in_offset[v + 1] = in_offset[v] + indegree[v];
    for (uint32_t i = 0; i < indegree[v]; ++i) {
      incoming_edges[in_offset[v] + i] = in_edges_host[v][i];
    }
  }

  VertexID* edges_globalid = csr->GetEdgesGloablIDBasePointer();
  VertexID* localid = csr->GetLocalIDBasePointer();
  for (uint32_t i = 0; i <= max_vid; ++i) {
    edges_globalid[i] = i;
    localid[i] = i;
  }

  return csr;
}

MatrixGraphPlanNode AttrNode(const char* key) {
  MatrixGraphPlanNode n{};
  n.type = MG_EXPR_ATTR;
  std::strncpy(n.key, key, sizeof(n.key) - 1);
  return n;
}

MatrixGraphPlanNode ConstFloatNode(double v) {
  MatrixGraphPlanNode n{};
  n.type = MG_EXPR_CONST;
  n.const_type = MG_VALUE_FLOAT64;
  n.const_f64 = v;
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

MatrixGraphPlanNode OutNeighborNav() {
  MatrixGraphPlanNode n{};
  n.type = MG_NAV_NEIGHBOR;
  n.direction = 0;  // outgoing
  return n;
}

}  // namespace

TEST(ComputeFeaturesTest, SelfAttributeAndConstantAndNeighborAgg) {
  const std::string tmp_path = "/tmp/matrixgraph_compute_features_test_ring";
  auto graph = BuildRingCSR(kNVertices);
  graph->Write(tmp_path, 0);

  ComputeFeaturesTask task(tmp_path);
  task.LoadGraph(tmp_path);

  std::vector<double> scores(kNVertices);
  for (uint32_t v = 0; v < kNVertices; ++v) scores[v] = v * 1.5;
  ComputeFeaturesAttributeColumn score_col{};
  std::strncpy(score_col.key, "score", sizeof(score_col.key) - 1);
  score_col.value_type = MG_VALUE_FLOAT64;
  score_col.n_values = kNVertices;
  score_col.values = scores.data();

  task.LoadAttributes(1, &score_col);

  std::vector<MatrixGraphPlanNode> plan;
  plan.push_back(AttrNode("score"));        // 0
  plan.push_back(ConstFloatNode(3.14));     // 1
  plan.push_back(AggNode(MG_AGG_SUM, 0, 0));   // 2
  plan.push_back(AggNode(MG_AGG_MEAN, 0, 0));  // 3

  std::vector<MatrixGraphPlanNode> navs;
  navs.push_back(OutNeighborNav());

  std::vector<int32_t> outputs = {0, 1, 2, 3};
  std::vector<uint32_t> pivots = {0, 1, 2, 3};

  std::vector<MatrixGraphCondNode> conds;
  auto result = task.Compute(pivots, plan, navs, conds, outputs);
  ASSERT_EQ(result.size(), pivots.size() * outputs.size());

  for (size_t i = 0; i < pivots.size(); ++i) {
    uint32_t v = pivots[i];
    uint32_t neighbor = (v + 1) % kNVertices;

    EXPECT_EQ(result[i * 4 + 0].type, MG_VALUE_FLOAT64);
    EXPECT_DOUBLE_EQ(result[i * 4 + 0].f64, scores[v]);

    EXPECT_EQ(result[i * 4 + 1].type, MG_VALUE_FLOAT64);
    EXPECT_DOUBLE_EQ(result[i * 4 + 1].f64, 3.14);

    EXPECT_EQ(result[i * 4 + 2].type, MG_VALUE_FLOAT64);
    EXPECT_DOUBLE_EQ(result[i * 4 + 2].f64, scores[neighbor]);

    EXPECT_EQ(result[i * 4 + 3].type, MG_VALUE_FLOAT64);
    EXPECT_DOUBLE_EQ(result[i * 4 + 3].f64, scores[neighbor]);
  }
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
