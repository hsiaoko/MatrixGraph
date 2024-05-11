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
#include "core/data_structures/immutable_csr.cuh"
#include "core/util/atomic.h"
#include "core/util/bitmap.h"
#include "core/util/cuda_check.cuh"
#include "tools/common/format_converter.h"

using sics::matrixgraph::core::common::EdgeIndex;
using sics::matrixgraph::core::common::VertexID;
using sics::matrixgraph::core::data_structures::ImmutableCSR;
using sics::matrixgraph::core::data_structures::kOrigin;
using sics::matrixgraph::core::data_structures::kTransposed;
using sics::matrixgraph::core::data_structures::TiledMatrix;
using sics::matrixgraph::core::util::Bitmap;
using sics::matrixgraph::core::util::atomic::WriteMax;

DEFINE_string(i, "", "input path.");
DEFINE_string(o, "", "output path.");
DEFINE_string(convert_mode, "", "Conversion mode");
DEFINE_string(sep, "", "separator to split a line of csv file.");
DEFINE_uint32(tile_size, 64, "the size of single tile");

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
                                    const std::string &output_path,
                                    size_t tile_size) {

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

  CUDA_LOG_INFO("Converting Edgelist to ImmutableCSR done.");
  auto p_immutable_csr =
      sics::matrixgraph::tools::format_converter::Edgelist2ImmutableCSR(
          edgelist);
  p_immutable_csr->ShowGraph(3);

  auto start_time = std::chrono::system_clock::now();

  auto tiled_matrix_ptr = new TiledMatrix(*p_immutable_csr, tile_size);
  CUDA_LOG_INFO("Converting ImmutableCSR to TiledMatrix done.");

  tiled_matrix_ptr->Write(output_path + "origin/");
  tiled_matrix_ptr->Show();

  auto end_time = std::chrono::system_clock::now();
  std::cout << "Generating TiledMatrix elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                     start_time)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;

  delete p_immutable_csr;
  delete tiled_matrix_ptr;
}

// @DESCRIPTION: convert a edgelist graph from csv file to binary file. Here the
// compression operations is default in ConvertEdgelist.
// @PARAMETER: input_path and output_path indicates the input and output path
// respectively, sep determines the separator for the csv file, read_head
// indicates whether to read head.
void ConvertEdgelistCSV2TiledMatrix(const std::string &input_path,
                                    const std::string &output_path,
                                    const std::string &sep, size_t tile_size) {
  if (!std::filesystem::exists(output_path))
    std::filesystem::create_directory(output_path);

  sics::matrixgraph::core::data_structures::Edges edgelist;
  edgelist.ReadFromCSV(input_path, sep);

  CUDA_LOG_INFO("Converting Edgelist to ImmutableCSR done.");
  auto p_immutable_csr =
      sics::matrixgraph::tools::format_converter::Edgelist2ImmutableCSR(
          edgelist);
  p_immutable_csr->ShowGraph(3);

  auto start_time = std::chrono::system_clock::now();

  auto tiled_matrix_ptr = new TiledMatrix(*p_immutable_csr, tile_size);
  CUDA_LOG_INFO("Converting ImmutableCSR to TiledMatrix done.");

  tiled_matrix_ptr->Write(output_path + "origin/");
  tiled_matrix_ptr->Show();

  auto end_time = std::chrono::system_clock::now();
  std::cout << "Generating TiledMatrix elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                     start_time)
                       .count() /
                   (double)CLOCKS_PER_SEC
            << std::endl;

  delete p_immutable_csr;
  delete tiled_matrix_ptr;
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
    ConvertEdgelistCSV2TiledMatrix(FLAGS_i, FLAGS_o, FLAGS_sep,
                                   FLAGS_tile_size);
    break;
  case kEdgelistBin2TiledMatrix:
    ConvertEdgelistBin2TiledMatrix(FLAGS_i, FLAGS_o, FLAGS_tile_size);
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