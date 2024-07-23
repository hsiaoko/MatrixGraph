// This file belongs to the SICS graph-systems project, a C++ library for
// exploiting parallelism graph computing.
#include <iostream>

#include <gflags/gflags.h>

#include "tools/common/types.h"
#include "tools/graph_partitioner/partitioner/grid_cut.cuh"

using GridCutPartitioner =
    sics::matrixgraph::tools::partitioner::GridCutPartitioner;
using StoreStrategy = sics::matrixgraph::tools::common::StoreStrategy;

// using enum sics::matrixgraph::tools::common::StoreStrategy;
//     using kIncomingOnly = sics::matrixgraph::tools::common::kIncomingOnly;
//  using kOutgoingOnly = sics::matrixgraph::tools::common::kOutgoingOnly;
//  using kUnconstrained = sics::matrixgraph::tools::common::kUnconstrained;

DEFINE_string(partitioner, "", "partitioner type.");
DEFINE_string(i, "", "input path.");
DEFINE_string(o, "", "output path.");
DEFINE_uint64(n_partitions, 1, "the number of partitions");
DEFINE_string(store_strategy, "unconstrained",
              "graph-systems adopted three strategies to store edges: "
              "kUnconstrained, incoming, and outgoing.");
DEFINE_bool(biggraph, false, "for big graphs.");

enum Partitioner {
  kHashEdgeCut, // default
  kHashVertexCut,
  kHybridCut,
  kPlanarVertexCut,
  kGridCut,
  kUndefinedPartitioner
};

static Partitioner Partitioner2Enum(const std::string &s) {
  if (s == "gridcut")
    return kGridCut;
  else
    std::cout << "Unknown partitioner type: " << s << std::endl;
  return kUndefinedPartitioner;
};

static StoreStrategy StoreStrategy2Enum(const std::string &s) {
  if (s == "incoming_only")
    return StoreStrategy::kIncomingOnly;
  else if (s == "outgoing_only")
    return StoreStrategy::kOutgoingOnly;
  else if (s == "unconstrained")
    return StoreStrategy::kUnconstrained;
  return StoreStrategy::kUndefinedStrategy;
};

int main(int argc, char **argv) {
  gflags::SetUsageMessage(
      "\n USAGE: graph-partitioner -partitioner [options] -i [input "
      "path] "
      "-o [output path] -n_partitions [number of partitions]\n"
      " General options:\n"
      "\t hashedgecut: - Using hash-based edge cut partitioner "
      "\n"
      "\t hashvertexcut: - Using hash-based vertex cut partitioner"
      "\n"
      "\t hybridcut:   - Using hybrid cut partitioner "
      "\n");

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_i == "" || FLAGS_o == "")
    exit(EXIT_FAILURE);

  switch (Partitioner2Enum(FLAGS_partitioner)) {
  case kGridCut: {

    GridCutPartitioner partitioner(FLAGS_i, FLAGS_o,
                                   StoreStrategy2Enum(FLAGS_store_strategy),
                                   FLAGS_n_partitions);
    partitioner.RunPartitioner();
    break;
  }
  case kUndefinedPartitioner: {
    break;
  }
  default: {
    std::cout << "Unknown partitioner type: " << FLAGS_partitioner;
    exit(EXIT_FAILURE);
  }
  }

  gflags::ShutDownCommandLineFlags();
  return 0;
}