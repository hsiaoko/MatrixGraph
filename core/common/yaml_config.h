#ifndef SICS_GRAPH_SYSTEMS_TOOLS_COMMON_YAML_CONFIG_H_
#define SICS_GRAPH_SYSTEMS_TOOLS_COMMON_YAML_CONFIG_H_

#include <yaml-cpp/yaml.h>

#include "data_structures/rule.h"

namespace YAML {

using sics::matrixgraph::core::data_structures::Rule;

template <> struct convert<Rule> {
  static Node encode(const Rule &rule) {
    Node node;
    // TODO (hsiaoko): to add encode function.
    return node;
  }

  static bool decode(const Node &node, Rule &rule) {
    if (node.size() < 2) {
      std::cout << "Invalid rule's metadata format" << std::endl;
      return false;
    }

    auto relation_l =
        node["Preconditions"]["Relations"][0].as<std::vector<std::string>>();
    auto relation_r =
        node["Preconditions"]["Relations"][1].as<std::vector<std::string>>();

    if (node["Preconditions"].size() == 4) {
      for (size_t i = 0; i < relation_l.size(); i++)
        rule.pre.relation_l.emplace_back(std::stoi(relation_l[i]));
      for (size_t i = 0; i < relation_r.size(); i++)
        rule.pre.relation_r.emplace_back(std::stoi(relation_r[i]));

      auto eq =
          node["Preconditions"]["Equalities"][0].as<std::vector<std::string>>();
      for (size_t i = 0; i < eq.size(); i++)
        rule.pre.eq.emplace_back(std::stoi(eq[i]));

      auto sim = node["Preconditions"]["Sim"][0].as<std::vector<std::string>>();
      for (size_t i = 0; i < sim.size(); i++)
        rule.pre.sim.emplace_back(std::stoi(sim[i]));

      auto threshold =
          node["Preconditions"]["Threshold"][0].as<std::vector<std::string>>();
      for (size_t i = 2; i < threshold.size(); i++) {
        rule.pre.threshold.insert(std::make_pair(rule.pre.sim[2 + (i - 2) * 2],
                                                 std::stof(threshold[i])));
      }

      auto equality = node["Conseq"]["Equality"].as<std::vector<std::string>>();
      for (size_t i = 0; i < equality.size(); i++)
        rule.con.eq.emplace_back(std::stoi(equality[i]));
    }
    return true;
  }
};

} // namespace YAML
#endif // SICS_GRAPH_SYSTEMS_TOOLS_COMMON_YAML_CONFIG_H_
