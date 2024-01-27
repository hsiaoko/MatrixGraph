#ifndef _MATRIXGRAPH_CORE_COMPONENTS_EXECUTIONPLAN_GENERATOR_H_
#define _MATRIXGRAPH_CORE_COMPONENTS_EXECUTIONPLAN_GENERATOR_H_

#include <memory>
#include <stack>
#include <string>

#include "core/data_structures/immutable_csr.h"
#include "core/util/bitmap.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace components {

class ExecutionPlan {
private:
  using ImmutableCSRVertex =
      sics::matrixgraph::core::data_structures::ImmutableCSRVertex;

public:
  ExecutionPlan() = default;

  void AddPath(std::list<ImmutableCSRVertex> *p_path) {
    path_.push_back(p_path);
  }

private:
  std::vector<std::list<ImmutableCSRVertex> *> path_;
};

class ExecutionPlanGenerator {
private:
  using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;
  using ImmutableCSRVertex =
      sics::matrixgraph::core::data_structures::ImmutableCSRVertex;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
  using Bitmap = sics::matrixgraph::core::util::Bitmap;

public:
  ExecutionPlanGenerator(const std::string &root_path) : root_path_(root_path) {
    immutable_csr_ = std::make_unique<ImmutableCSR>();
    immutable_csr_->Read(root_path_);
  }

  ExecutionPlan GenerateExecutionPlan() {
    auto &&exp = SplitPlttern();
    return exp;
  }

  friend class ExecutionPlan;

private:
  ExecutionPlan SplitPlttern() {

    ExecutionPlan exp;

    Bitmap visited(immutable_csr_->get_num_vertices());

    std::cout << "visited init : " << visited.Count() << " "
              << immutable_csr_->get_num_vertices() << std::endl;
    while (visited.Count() < immutable_csr_->get_num_vertices()) {

      // Find vertex with highest degree
      VertexID max_degree_vertex_id = 0;
      VertexID max_degree = 0;
      for (int i = 0; i < immutable_csr_->get_num_vertices(); i++) {
        if (visited.GetBit(i))
          continue;
        if (max_degree < immutable_csr_->GetOutDegreeByLocalID(i)) {
          max_degree_vertex_id = i;
          max_degree = immutable_csr_->GetOutDegreeByLocalID(i);
        }
      }
      auto root = immutable_csr_->GetVertexByLocalID(max_degree_vertex_id);
      std::stack<VertexID> unvisited_stack;

      // DFS traverse to generate path manner subgraph in which each subgraph is
      // an execution plan.
      auto path = new std::list<ImmutableCSRVertex>();
      unvisited_stack.push(max_degree_vertex_id);
      while (!unvisited_stack.empty()) {
        auto current_id = unvisited_stack.top();
        unvisited_stack.pop();
        visited.SetBit(current_id);

        auto u = immutable_csr_->GetVertexByLocalID(current_id);
        path->push_back(u);
        for (int nbr_id = 0; nbr_id < u.outdegree; nbr_id++) {
          unvisited_stack.push(u.outgoing_edges[nbr_id]);
        }
      }
      exp.AddPath(path);
    }

    return exp;
  }

  const std::string root_path_;
  std::unique_ptr<ImmutableCSR> immutable_csr_;
};

} // namespace components
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // MATRIXGRAPH_CORE_COMPONENTS_EXECUTIONPLAN_GENERATOR_H_
