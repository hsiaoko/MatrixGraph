#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "core/common/consts.h"
#include "core/common/types.h"
#include "core/data_structures/metadata.h"
#include "core/task/cpu_task/cpu_subiso.cuh"
#include "core/util/format_converter.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using VertexID = sics::matrixgraph::core::common::VertexID;
using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
using ImmutableCSR = sics::matrixgraph::core::data_structures::ImmutableCSR;
using Edges = sics::matrixgraph::core::data_structures::Edges;
using Edge = sics::matrixgraph::core::data_structures::Edge;
using GraphMetadata = sics::matrixgraph::core::data_structures::GraphMetadata;

void CPUSubIso::LoadData() {
  std::cout << "[CPUSubIso] LoadData() ..." << std::endl;

  p_.Read(pattern_path_);
  p_.PrintGraph(3);

  g_.Read(data_graph_path_);
  g_.PrintGraph(3);
}

void CPUSubIso::Matching(const ImmutableCSR& p, const ImmutableCSR& g) {
  std::cout << "CPU Matching with " << num_threads_ << " threads..."
            << std::endl;
}

void CPUSubIso::ParallelMatching(const ImmutableCSR& p, const ImmutableCSR& g,
                                 size_t thread_id, size_t total_threads) {}

void CPUSubIso::WOJMatching(const ImmutableCSR& p, const ImmutableCSR& g) {
  std::cout << "CPU WOJ Matching not implemented yet" << std::endl;
  // TODO: Implement CPU version of WOJ matching if needed
}

void CPUSubIso::Run() {
  auto start_time_0 = std::chrono::system_clock::now();
  LoadData();
  auto start_time_1 = std::chrono::system_clock::now();

  std::cout << "Data loading time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   start_time_1 - start_time_0)
                   .count()
            << " ms" << std::endl;

  Matching(p_, g_);

  auto end_time = std::chrono::system_clock::now();
  std::cout << "Total execution time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   end_time - start_time_0)
                   .count()
            << " ms" << std::endl;
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
