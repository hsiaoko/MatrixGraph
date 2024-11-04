// This file belongs to the SICS graph-systems project, a C++ library for
// exploiting parallelism graph computing.
//
// We store graphs in a binary format. Other formats such as edge-list can be
// converted to our format with graph_convert tool. You can use graph_convert as
// follows: e.g. convert a graph from edgelist CSV to edgelist bin file.
// USAGE: graph-convert --convert_mode=[options] -i <input file path> -o <output
// file path> --sep=[separator]

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <thread>
#include <type_traits>

#include <gflags/gflags.h>
#include <yaml-cpp/yaml.h>

#include "core/common/types.h"
#include "core/data_structures/bit_tiled_matrix.cuh"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/util/atomic.h"
#include "core/util/bitmap.h"
#include "core/util/cuda_check.cuh"
#include "core/util/format_converter.cuh"
#include "tools/graph_converter/converter/to_bit_tiled_matrix.cuh"
#include "tools/graph_converter/converter/to_csr_tiled_matrix.cuh"
#include "tools/graph_converter/converter/to_edgelist.cuh"
#include "tools/graph_converter/converter/to_immutable_csr.cuh"

using sics::matrixgraph::core::common::EdgeIndex;
using sics::matrixgraph::core::common::VertexID;
using sics::matrixgraph::core::data_structures::ImmutableCSR;
using sics::matrixgraph::core::util::Bitmap;
using sics::matrixgraph::core::util::atomic::WriteMax;

DEFINE_string(i, "", "input path.");
DEFINE_string(o, "", "output path.");
DEFINE_string(convert_mode, "", "Conversion mode");
DEFINE_string(sep, "", "separator to split a line of csv file.");
DEFINE_bool(compressed, false, "compressed vid");
DEFINE_uint32(tile_size, 64, "the size of single tile");

enum ConvertMode {
  kEdgelistCSV2TiledMatrix, // default
  kEdgelistBin2TiledMatrix,
  kEdgelistCSV2BitTiledMatrix,
  kEdgelistBin2BitTiledMatrix,
  kEdgelistBin2TransposedEdgelistBin,
  kGridEdgelistBin2BitTiledMatrix,
  kGridEdgelistBin2CSRTiledMatrix,
  kCSRBin2BitTiledMatrix,
  kEdgelistBin2CSRBin,
  kEdgelistCSV2CSRBin,
  kEdgelistCSV2EdgelistBin,
  kCSRBin2EdgelistBin,
  kEdgelistCSV2CGGraphCSR,
  kEdgelistBin2CGGraphCSR,
  kUndefinedMode
};

static inline ConvertMode ConvertMode2Enum(const std::string &s) {
  if (s == "edgelistcsv2tiledmatrix")
    return kEdgelistCSV2TiledMatrix;
  if (s == "edgelistcsv2bittiledmatrix")
    return kEdgelistCSV2BitTiledMatrix;
  if (s == "edgelistbin2bittiledmatrix")
    return kEdgelistBin2BitTiledMatrix;
  if (s == "edgelistbin2csrbin")
    return kEdgelistBin2CSRBin;
  if (s == "csrbin2bittiledmatrix")
    return kCSRBin2BitTiledMatrix;
  if (s == "csrbin2edgelistbin")
    return kCSRBin2EdgelistBin;
  if (s == "edgelistbin2tiledmatrix")
    return kEdgelistBin2TiledMatrix;
  if (s == "edgelistbin2transposededgelistbin")
    return kEdgelistBin2TransposedEdgelistBin;
  if (s == "edgelistcsv2edgelistbin")
    return kEdgelistCSV2EdgelistBin;
  if (s == "edgelistcsv2csrbin")
    return kEdgelistCSV2CSRBin;
  if (s == "gridedgelistbin2bittiledmatrix")
    return kGridEdgelistBin2BitTiledMatrix;
  if (s == "gridedgelistbin2csrtiledmatrix")
    return kGridEdgelistBin2CSRTiledMatrix;
  if (s == "edgelistcsv2cggraphcsr")
    return kEdgelistCSV2CGGraphCSR;
  if (s == "edgelistbin2cggraphcsr")
    return kEdgelistBin2CGGraphCSR;
  return kUndefinedMode;
};

int main(int argc, char **argv) {
  gflags::SetUsageMessage(
      "\n USAGE: graph-convert --convert_mode=[options] -i <input file path> "
      "-o <output file path> --sep=[separator] \n"
      " General options:\n"
      "\t edgelistcsv2edgelistbin:  - Convert edge list txt to binary edge \n");

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_i == "" || FLAGS_o == "") {
    std::cout << "Input (output) path is empty." << std::endl;
    exit(EXIT_FAILURE);
  }

  switch (ConvertMode2Enum(FLAGS_convert_mode)) {
  case kEdgelistBin2CSRBin:
    sics::matrixgraph::tools::converter::ConvertEdgelistBin2CSRBin(FLAGS_i,
                                                                   FLAGS_o);
    break;
  case kEdgelistBin2TransposedEdgelistBin:
    sics::matrixgraph::tools::converter::
        ConvertEdgelistBin2TransposedEdgelistBin(FLAGS_i, FLAGS_o);
    break;
  case kEdgelistCSV2CSRBin:
    sics::matrixgraph::tools::converter::ConvertEdgelistCSV2ImmutableCSR(
        FLAGS_i, FLAGS_o, FLAGS_sep);
    break;
  case kEdgelistCSV2EdgelistBin:
    sics::matrixgraph::tools::converter::ConvertEdgelistCSV2EdgelistBin(
        FLAGS_i, FLAGS_o, FLAGS_sep, FLAGS_compressed);
    break;
  case kCSRBin2EdgelistBin:
    sics::matrixgraph::tools::converter::ConvertImmutableCSR2EdgelistBin(
        FLAGS_i, FLAGS_o, FLAGS_compressed);
    break;
  case kGridEdgelistBin2BitTiledMatrix:
    sics::matrixgraph::tools::converter::ConvertGridGraph2BitTiledMatrix(
        FLAGS_i, FLAGS_o, FLAGS_tile_size);
    break;
  case kGridEdgelistBin2CSRTiledMatrix:
    sics::matrixgraph::tools::converter::ConvertGridGraph2CSRTiledMatrix(
        FLAGS_i, FLAGS_o, FLAGS_tile_size);
    break;
  case kEdgelistCSV2CGGraphCSR:
    sics::matrixgraph::tools::converter::ConvertEdgelistCSV2CGGraphCSR(
        FLAGS_i, FLAGS_o, FLAGS_sep);
    break;
  case kEdgelistBin2CGGraphCSR:
    sics::matrixgraph::tools::converter::ConvertEdgelistBin2CGGraphCSR(FLAGS_i,
                                                                       FLAGS_o);
    break;
  default:
    exit(EXIT_FAILURE);
  }

  gflags::ShutDownCommandLineFlags();
  return 0;
}