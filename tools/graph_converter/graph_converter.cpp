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
#include "core/data_structures/edgelist.h"
#include "core/data_structures/immutable_csr.h"
#include "core/util/atomic.h"
#include "core/util/bitmap.h"
#include "tools/common/format_converter.h"

using sics::matrixgraph::core::common::EdgeIndex;
using sics::matrixgraph::core::common::VertexID;
using sics::matrixgraph::core::data_structures::ImmutableCSR;
using sics::matrixgraph::core::data_structures::TiledMatrix;
using sics::matrixgraph::core::util::Bitmap;
using sics::matrixgraph::core::util::atomic::WriteMax;

DEFINE_string(i, "", "input path.");
DEFINE_string(o, "", "output path.");
DEFINE_string(convert_mode, "", "Conversion mode");
DEFINE_string(sep, "", "separator to split a line of csv file.");

enum ConvertMode {
  kEdgelistCSV2TiledMatrix, // default
  kEdgelistBin2TiledMatrix,
  kEdgelistCSV2CSR,
  kUndefinedMode
};

static inline ConvertMode ConvertMode2Enum(const std::string &s) {
  if (s == "edgelistcsv2tiledmatrix")
    return kEdgelistCSV2TiledMatrix;
  if (s == "edgelistbin2tiledmatrix")
    return kEdgelistBin2TiledMatrix;
  if (s == "edgelistcsv2csr")
    return kEdgelistCSV2CSR;
  return kUndefinedMode;
};

void ConvertEdgelistBin2TiledMatrix(const std::string &input_path,
                                    const std::string &output_path) {

  YAML::Node node = YAML::LoadFile(input_path + "meta.yaml");

  sics::matrixgraph::core::data_structures::EdgelistMetadata edgelist_metadata =
      {node["EdgelistBin"]["num_vertices"].as<VertexID>(),
       node["EdgelistBin"]["num_edges"].as<VertexID>(),
       node["EdgelistBin"]["max_vid"].as<VertexID>()};

  auto buffer_edges =
      new sics::matrixgraph::core::data_structures::Edge[edgelist_metadata
                                                             .num_edges]();

  std::ifstream in_file(input_path + "edgelist.bin");
  if (!in_file) {

    std::cout << "Open file failed: " + input_path + "edgelist.bin"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  in_file.read(reinterpret_cast<char *>(buffer_edges),
               sizeof(sics::matrixgraph::core::data_structures::Edge) *
                   edgelist_metadata.num_edges);
  sics::matrixgraph::core::data_structures::Edges edgelist(edgelist_metadata,
                                                           buffer_edges);
  edgelist.SortBySrc();

  auto p_immutable_csr =
      sics::matrixgraph::tools::format_converter::Edgelist2ImmutableCSR(
          edgelist);

  auto p_tiled_matrix =
      sics::matrixgraph::tools::format_converter::ImmutableCSR2TiledMatrix(
          *p_immutable_csr);
  p_tiled_matrix->Write(output_path + "origin/");

  auto p_tiled_matrix_t = sics::matrixgraph::tools::format_converter::
      ImmutableCSR2TransposedTiledMatrix(*p_immutable_csr);
  p_tiled_matrix_t->Write(output_path + "transposed/");

  p_tiled_matrix->Show();
  p_tiled_matrix_t->Show();

  delete p_immutable_csr;
  delete p_tiled_matrix_t;
}

// @DESCRIPTION: convert a edgelist graph from csv file to binary file. Here the
// compression operations is default in ConvertEdgelist.
// @PARAMETER: input_path and output_path indicates the input and output path
// respectively, sep determines the separator for the csv file, read_head
// indicates whether to read head.
void ConvertEdgelistCSV2TiledMatrix(const std::string &input_path,
                                    const std::string &output_path,
                                    const std::string &sep) {
  if (!std::filesystem::exists(output_path))
    std::filesystem::create_directory(output_path);

  sics::matrixgraph::core::data_structures::Edges edgelist;
  edgelist.ReadFromCSV(input_path, sep);

  auto p_immutable_csr =
      sics::matrixgraph::tools::format_converter::Edgelist2ImmutableCSR(
          edgelist);

  auto p_tiled_matrix =
      sics::matrixgraph::tools::format_converter::ImmutableCSR2TiledMatrix(
          *p_immutable_csr);
  p_tiled_matrix->Write(output_path + "origin/");

  auto p_tiled_matrix_t = sics::matrixgraph::tools::format_converter::
      ImmutableCSR2TransposedTiledMatrix(*p_immutable_csr);
  p_tiled_matrix_t->Write(output_path + "transposed/");

  p_tiled_matrix->Show();
  p_tiled_matrix_t->Show();

  delete p_immutable_csr;
  delete p_tiled_matrix_t;
}

// @DESCRIPTION: convert a edgelist graph from csv file to binary file. Here the
// compression operations is default in ConvertEdgelist.
// @PARAMETER: input_path and output_path indicates the input and output path
// respectively, sep determines the separator for the csv file, read_head
// indicates whether to read head.
void ConvertEdgelistCSV2ImmutableCSR(const std::string &input_path,
                                     const std::string &output_path,
                                     const std::string &sep) {
  if (!std::filesystem::exists(output_path))
    std::filesystem::create_directory(output_path);

  sics::matrixgraph::core::data_structures::Edges edgelist;
  edgelist.ReadFromCSV(input_path, sep);
  auto p_immutable_csr =
      sics::matrixgraph::tools::format_converter::Edgelist2ImmutableCSR(
          edgelist);

  p_immutable_csr->Write(output_path);
  delete p_immutable_csr;
}

int main(int argc, char **argv) {
  gflags::SetUsageMessage(
      "\n USAGE: graph-convert --convert_mode=[options] -i <input file path> "
      "-o <output file path> --sep=[separator] \n"
      " General options:\n"
      "\t edgelistcsv2edgelistbin:  - Convert edge list txt to binary edge \n");

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_i == "" || FLAGS_o == "") {
    std::cout << "Input (output) path is empty." << std::endl;
    return -1;
  }

  switch (ConvertMode2Enum(FLAGS_convert_mode)) {
  case kEdgelistCSV2TiledMatrix:
    ConvertEdgelistCSV2TiledMatrix(FLAGS_i, FLAGS_o, FLAGS_sep);
    break;
  case kEdgelistBin2TiledMatrix:
    ConvertEdgelistBin2TiledMatrix(FLAGS_i, FLAGS_o);
    break;
  case kEdgelistCSV2CSR:
    ConvertEdgelistCSV2ImmutableCSR(FLAGS_i, FLAGS_o, FLAGS_sep);
    break;
  default:
    return -1;
  }

  gflags::ShutDownCommandLineFlags();
  return 0;
}