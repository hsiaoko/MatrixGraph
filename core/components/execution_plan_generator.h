#ifndef HYPERBLOCKER_CORE_COMPONENTS_EXECUTIONPLANGENERATOR_H_
#define HYPERBLOCKER_CORE_COMPONENTS_EXECUTIONPLANGENERATOR_H_

#include <experimental/filesystem>
#include <string>

#include <cuda_runtime.h>

#include "core/common/types.h"
#include "core/util/atomic.h"
#include "core/util/bitmap.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace components {

using sics::matrixgraph::core::data_structures::Rule;

enum EPGStrategy {
  kEqualitiesFirst, // default
  kSimFirst
};

struct SerializedExecutionPlan {
  int n_rules;
  int length;
  int *pred_index;
  char *pred_type;
  float *pred_threshold;
};

class ExecutionPlanGenerator {
  using Bitmap = sics::matrixgraph::core::util::Bitmap;

public:
  ExecutionPlanGenerator(const std::string &rule_dir) {
    std::vector<std::string> file_list;
    for (auto &iter :
         std::experimental::filesystem::directory_iterator(rule_dir)) {
      auto rule_path = iter.path();
      file_list.push_back(rule_path.string());
      YAML::Node yaml_node;
      yaml_node = YAML::LoadFile(rule_path.string());
      auto rule = yaml_node.as<Rule>();
      rule_vec_.push_back(rule);
    }
    std::sort(rule_vec_.begin(), rule_vec_.end(),
              [](const auto &rule1, const auto &rule2) {
                return rule1.pre.eq.size() < rule2.pre.eq.size();
                // return rule1.pre.sim.size() > rule2.pre.sim.size();
              });
    for (auto &iter : rule_vec_)
      iter.Show();
  }

  SerializedExecutionPlan
  GetExecutionPlan(EPGStrategy strategy = kEqualitiesFirst) {
    SerializedExecutionPlan ep;
    ep.n_rules = rule_vec_.size();

    switch (strategy) {
      // CASE 1: Equalities first
    case kEqualitiesFirst: {
      // TODO: DFS traversal preconditions to get the serialized execution plan

      // Pre-compute the max predicate index.
      unsigned max_pred_index = 0;
      for (size_t i = 0; i < rule_vec_.size(); i++) {
        std::for_each(rule_vec_[i].pre.relation_l.begin(),
                      rule_vec_[i].pre.relation_l.end(),
                      [&max_pred_index](auto &pred_index) {
                        sics::matrixgraph::core::util::WriteMax(&max_pred_index,
                                                                pred_index);
                      });
      }

      std::vector<int> serialized_pred_index_vec;
      std::vector<char> serialized_pred_type_vec;
      std::vector<float> serialized_pred_threshold_vec;
      serialized_pred_index_vec.reserve(max_pred_index);
      serialized_pred_type_vec.reserve(max_pred_index);
      serialized_pred_threshold_vec.reserve(max_pred_index);

      for (size_t i = 0; i < rule_vec_.size(); i++) {
        Bitmap visited(max_pred_index);
        std::for_each(
            rule_vec_[i].pre.eq.begin() + 2, rule_vec_[i].pre.eq.end(),
            [&visited, &serialized_pred_index_vec, &serialized_pred_type_vec,
             &serialized_pred_threshold_vec](auto &pred_index) {
              visited.SetBit(pred_index);
              serialized_pred_index_vec.push_back(pred_index);
              serialized_pred_type_vec.push_back(EQUALITIES);
              serialized_pred_threshold_vec.push_back(1);
            });

        visited.Clear();
        std::for_each(
            rule_vec_[i].pre.sim.begin() + 2, rule_vec_[i].pre.sim.end(),
            [&, i](auto &pred_index) {
              visited.SetBit(pred_index);
              serialized_pred_index_vec.push_back(pred_index);
              serialized_pred_type_vec.push_back(SIM);
              serialized_pred_threshold_vec.push_back(
                  rule_vec_[i].pre.threshold.find(pred_index)->second);
            });

        serialized_pred_index_vec.push_back(CHECK_POINT);
        serialized_pred_type_vec.push_back(CHECK_POINT_CHAR);
        serialized_pred_threshold_vec.push_back(1);
      }

      cudaHostAlloc(&(ep.pred_index),
                    sizeof(int) * serialized_pred_index_vec.size(),
                    cudaHostAllocDefault);
      cudaHostAlloc(&(ep.pred_type),
                    sizeof(char) * serialized_pred_type_vec.size(),
                    cudaHostAllocDefault);
      cudaHostAlloc(&(ep.pred_threshold),
                    sizeof(float) * serialized_pred_threshold_vec.size(),
                    cudaHostAllocDefault);
      for (size_t i = 0; i < serialized_pred_index_vec.size(); i++) {
        ep.pred_index[i] = serialized_pred_index_vec[i];
        ep.pred_type[i] = serialized_pred_type_vec[i];
        ep.pred_threshold[i] = serialized_pred_threshold_vec[i];
      }
      ep.length = serialized_pred_index_vec.size();
      break;
    }
    case kSimFirst:
      unsigned max_pred_index = 0;
      for (size_t i = 0; i < rule_vec_.size(); i++) {
        std::for_each(rule_vec_[i].pre.relation_l.begin(),
                      rule_vec_[i].pre.relation_l.end(),
                      [&max_pred_index](auto &pred_index) {
                        sics::matrixgraph::core::util::WriteMax(&max_pred_index,
                                                                pred_index);
                      });
      }

      std::vector<int> serialized_pred_index_vec;
      std::vector<char> serialized_pred_type_vec;
      std::vector<float> serialized_pred_threshold_vec;
      serialized_pred_index_vec.reserve(max_pred_index);
      serialized_pred_type_vec.reserve(max_pred_index);
      serialized_pred_threshold_vec.reserve(max_pred_index);

      for (size_t i = 0; i < rule_vec_.size(); i++) {
        Bitmap visited(max_pred_index);

        std::for_each(
            rule_vec_[i].pre.sim.begin() + 2, rule_vec_[i].pre.sim.end(),
            [&, i](auto &pred_index) {
              visited.SetBit(pred_index);
              serialized_pred_index_vec.push_back(pred_index);
              serialized_pred_type_vec.push_back(SIM);
              serialized_pred_threshold_vec.push_back(
                  rule_vec_[i].pre.threshold.find(pred_index)->second);
            });

        std::for_each(
            rule_vec_[i].pre.eq.begin() + 2, rule_vec_[i].pre.eq.end(),
            [&visited, &serialized_pred_index_vec, &serialized_pred_type_vec,
             &serialized_pred_threshold_vec](auto &pred_index) {
              visited.SetBit(pred_index);
              serialized_pred_index_vec.push_back(pred_index);
              serialized_pred_type_vec.push_back(EQUALITIES);
              serialized_pred_threshold_vec.push_back(1);
            });

        serialized_pred_index_vec.push_back(CHECK_POINT);
        serialized_pred_type_vec.push_back(CHECK_POINT_CHAR);
        serialized_pred_threshold_vec.push_back(1);
      }

      cudaHostAlloc(&(ep.pred_index),
                    sizeof(int) * serialized_pred_index_vec.size(),
                    cudaHostAllocDefault);
      cudaHostAlloc(&(ep.pred_type),
                    sizeof(char) * serialized_pred_type_vec.size(),
                    cudaHostAllocDefault);
      cudaHostAlloc(&(ep.pred_threshold),
                    sizeof(float) * serialized_pred_threshold_vec.size(),
                    cudaHostAllocDefault);

      for (size_t i = 0; i < serialized_pred_index_vec.size(); i++) {
        ep.pred_index[i] = serialized_pred_index_vec[i];
        ep.pred_type[i] = serialized_pred_type_vec[i];
        ep.pred_threshold[i] = serialized_pred_threshold_vec[i];
      }
      ep.length = serialized_pred_index_vec.size();
      break;
    }
    return ep;
  }

private:
  void Prioritizer();

  std::vector<Rule> rule_vec_;
};

} // namespace components
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // HYPERBLOCKER_CORE_COMPONENTS_EXECUTIONPLANGENERATOR_H_
