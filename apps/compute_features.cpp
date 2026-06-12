/**
 * @file compute_features.cpp
 * @brief Standalone smoke test / demo binary for ComputeFeaturesTask.
 *
 * This executable builds a small directed ring graph, loads a few synthetic
 * attribute columns, and evaluates a set of feature expressions covering
 * AttrExpr, ConstExpr, AggExpr, TransExpr, nested aggregations and FilterNav.
 *
 * Build target: compute_features_exec
 * Usage:        ./bin/compute_features_exec
 *
 * It exits with 0 when all checks pass and non-zero otherwise.
 */

#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "core/data_structures/immutable_csr.cuh"
#include "core/task/gpu_task/compute_features.cuh"

using sics::matrixgraph::core::common::EdgeIndex;
using sics::matrixgraph::core::common::VertexID;
using sics::matrixgraph::core::data_structures::ImmutableCSR;
using sics::matrixgraph::core::task::ComputeFeaturesTask;


namespace {

constexpr uint32_t kNVertices = 4;
constexpr uint32_t kOutDegree = 2;

/**
 * @brief Build a small directed ring CSR in host memory.
 *
 * Each vertex has outgoing edges to (v+1)%n and (v+2)%n.  The buffer layout
 * matches ImmutableCSR exactly so that Write()/Read() round-trips work.
 */
std::unique_ptr<ImmutableCSR> BuildRingCSR(uint32_t n_vertices) {
  auto csr = std::make_unique<ImmutableCSR>();
  uint32_t n_edges = n_vertices * kOutDegree;
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
    for (uint32_t d = 1; d <= 2; ++d) {
      VertexID dst = (v + d) % n_vertices;
      out_edges_host[v].push_back(dst);
      in_edges_host[dst].push_back(v);
    }
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

/** @brief Build an attribute expression node. */
MatrixGraphPlanNode AttrNode(const char* key) {
  MatrixGraphPlanNode n{};
  n.type = MG_EXPR_ATTR;
  std::strncpy(n.key, key, sizeof(n.key) - 1);
  return n;
}

/** @brief Build a floating-point constant expression node. */
MatrixGraphPlanNode ConstFloatNode(double v) {
  MatrixGraphPlanNode n{};
  n.type = MG_EXPR_CONST;
  n.const_type = MG_VALUE_FLOAT64;
  n.const_f64 = v;
  return n;
}

/** @brief Build an integer constant expression node. */
MatrixGraphPlanNode ConstIntNode(int64_t v) {
  MatrixGraphPlanNode n{};
  n.type = MG_EXPR_CONST;
  n.const_type = MG_VALUE_INT;
  n.const_i64 = v;
  return n;
}

/** @brief Build an aggregation expression node. */
MatrixGraphPlanNode AggNode(int32_t op, int32_t src_idx, int32_t nav_idx) {
  MatrixGraphPlanNode n{};
  n.type = MG_EXPR_AGG;
  n.op = op;
  n.src_idx = src_idx;
  n.nav_idx = nav_idx;
  return n;
}

/** @brief Build a transformation (arithmetic) expression node. */
MatrixGraphPlanNode TransNode(int32_t op, int32_t child_a, int32_t child_b) {
  MatrixGraphPlanNode n{};
  n.type = MG_EXPR_TRANS;
  n.op = op;
  n.child_a = child_a;
  n.child_b = child_b;
  return n;
}

/** @brief Build an outgoing-neighbor navigator node. */
MatrixGraphPlanNode OutNeighborNav() {
  MatrixGraphPlanNode n{};
  n.type = MG_NAV_NEIGHBOR;
  n.direction = 0;
  return n;
}

/** @brief Build a FilterNav wrapping an inner navigator and a condition. */
MatrixGraphPlanNode FilterNavNode(int32_t inner_nav_idx, int32_t cond_idx) {
  MatrixGraphPlanNode n{};
  n.type = MG_NAV_FILTER;
  n.inner_nav_idx = inner_nav_idx;
  n.cond_idx = cond_idx;
  return n;
}

/** @brief Build a binary condition node. */
MatrixGraphCondNode CondNode(int32_t op, int32_t left, int32_t right) {
  MatrixGraphCondNode n{};
  n.op = op;
  n.left_expr = left;
  n.right_expr = right;
  return n;
}

/** @brief Build a float64 attribute column descriptor. */
ComputeFeaturesAttributeColumn MakeColumn(const char* key,
                                          const std::vector<double>& values) {
  ComputeFeaturesAttributeColumn col{};
  std::strncpy(col.key, key, sizeof(col.key) - 1);
  col.value_type = MG_VALUE_FLOAT64;
  col.n_values = static_cast<uint32_t>(values.size());
  col.values = const_cast<void*>(static_cast<const void*>(values.data()));
  return col;
}

/** @brief Build an int64 attribute column descriptor. */
ComputeFeaturesAttributeColumn MakeColumn(const char* key,
                                          const std::vector<int64_t>& values) {
  ComputeFeaturesAttributeColumn col{};
  std::strncpy(col.key, key, sizeof(col.key) - 1);
  col.value_type = MG_VALUE_INT;
  col.n_values = static_cast<uint32_t>(values.size());
  col.values = const_cast<void*>(static_cast<const void*>(values.data()));
  return col;
}

/** @brief Build an int32 attribute column descriptor. */
ComputeFeaturesAttributeColumn MakeColumn(const char* key,
                                          const std::vector<int32_t>& values) {
  ComputeFeaturesAttributeColumn col{};
  std::strncpy(col.key, key, sizeof(col.key) - 1);
  col.value_type = MG_VALUE_INT;
  col.n_values = static_cast<uint32_t>(values.size());
  col.values = const_cast<void*>(static_cast<const void*>(values.data()));
  return col;
}

}  // namespace

/**
 * @brief Smoke test entry point.
 *
 * Phase 0/1/2 expressions are checked against hand-computed expected values.
 */
int main(int argc, char** argv) {
  const std::string graph_path = "/tmp/matrixgraph_compute_features_app_ring";
  auto graph = BuildRingCSR(kNVertices);
  graph->Write(graph_path, 0);

  ComputeFeaturesTask task(graph_path);
  task.LoadGraph(graph_path);

  std::vector<double> scores(kNVertices);
  for (uint32_t v = 0; v < kNVertices; ++v) scores[v] = v * 1.5;
  auto score_col = MakeColumn("score", scores);
  task.LoadAttributes(1, &score_col);

  std::vector<MatrixGraphPlanNode> plan;
  plan.push_back(AttrNode("score"));           // 0
  plan.push_back(ConstFloatNode(3.14));        // 1
  plan.push_back(ConstFloatNode(10.0));        // 2
  plan.push_back(AggNode(MG_AGG_SUM, 0, 0));   // 3
  plan.push_back(AggNode(MG_AGG_MEAN, 0, 0));  // 4
  plan.push_back(TransNode(MG_TRANS_ADD, 0, 2));  // 5: score + 10
  plan.push_back(AggNode(MG_AGG_SUM, 0, 0));   // 6: inner sum over neighbors
  plan.push_back(AggNode(MG_AGG_MEAN, 6, 0));  // 7: mean of neighbor sums

  std::vector<MatrixGraphPlanNode> navs;
  navs.push_back(OutNeighborNav());

  std::vector<MatrixGraphCondNode> conds;

  std::vector<int32_t> outputs = {0, 1, 3, 4, 5, 7};
  std::vector<uint32_t> pivots = {0, 1, 2, 3};

  auto result = task.Compute(pivots, plan, navs, conds, outputs);

  bool ok = true;
  for (size_t i = 0; i < pivots.size(); ++i) {
    uint32_t v = pivots[i];
    double expected_sum = 0.0;
    for (uint32_t d = 1; d <= kOutDegree; ++d) {
      expected_sum += scores[(v + d) % kNVertices];
    }
    double expected_mean = expected_sum / static_cast<double>(kOutDegree);

    // Nested: mean of neighbor sums.
    double expected_nested = 0.0;
    for (uint32_t d = 1; d <= kOutDegree; ++d) {
      uint32_t u = (v + d) % kNVertices;
      double inner_sum = 0.0;
      for (uint32_t dd = 1; dd <= kOutDegree; ++dd) {
        inner_sum += scores[(u + dd) % kNVertices];
      }
      expected_nested += inner_sum;
    }
    expected_nested /= static_cast<double>(kOutDegree);

    auto check = [&](const char* name, int idx, double expected) {
      double got = result[i * 6 + idx].f64;
      if (std::fabs(got - expected) > 1e-6) {
        std::cerr << "Mismatch pivot " << v << " " << name << ": expected "
                  << expected << " got " << got << std::endl;
        ok = false;
      }
    };

    check("attr", 0, scores[v]);
    check("const", 1, 3.14);
    check("sum", 2, expected_sum);
    check("mean", 3, expected_mean);
    check("trans_add", 4, scores[v] + 10.0);
    check("nested_mean", 5, expected_nested);
  }

  if (!ok) return 1;

  // ---------------------------------------------------------------------------
  // FilterNav test: sum scores of outgoing neighbors whose flag == true.
  // ---------------------------------------------------------------------------
  std::vector<int64_t> flags(kNVertices);
  for (uint32_t v = 0; v < kNVertices; ++v) flags[v] = (v % 2 == 0) ? 1 : 0;
  auto flag_col = MakeColumn("flag", flags);
  task.LoadAttributes(1, &flag_col);

  std::vector<MatrixGraphPlanNode> plan2;
  plan2.push_back(AttrNode("score"));     // 0
  plan2.push_back(AttrNode("flag"));      // 1
  plan2.push_back(ConstIntNode(1));       // 2
  plan2.push_back(AggNode(MG_AGG_SUM, 0, 1));  // 3: sum score of filtered neighbors

  std::vector<MatrixGraphPlanNode> navs2;
  navs2.push_back(OutNeighborNav());       // 0: out neighbor
  navs2.push_back(FilterNavNode(0, 0));    // 1: filter where flag == 1

  std::vector<MatrixGraphCondNode> conds2;
  conds2.push_back(CondNode(MG_COND_EQ, 1, 2));

  std::vector<int32_t> outputs2 = {3};
  auto result2 = task.Compute(pivots, plan2, navs2, conds2, outputs2);

  for (size_t i = 0; i < pivots.size(); ++i) {
    uint32_t v = pivots[i];
    double expected = 0.0;
    for (uint32_t d = 1; d <= kOutDegree; ++d) {
      uint32_t u = (v + d) % kNVertices;
      if (flags[u]) expected += scores[u];
    }
    double got = result2[i].f64;
    if (std::fabs(got - expected) > 1e-6) {
      std::cerr << "FilterNav mismatch pivot " << v << ": expected " << expected
                << " got " << got << std::endl;
      ok = false;
    }
  }

  if (ok) {
    std::cout << "ComputeFeatures smoke test passed." << std::endl;
    return 0;
  }
  return 1;
}
