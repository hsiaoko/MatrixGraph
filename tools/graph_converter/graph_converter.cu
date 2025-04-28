// This file belongs to the SICS graph-systems project, a C++ library for
// exploiting parallelism graph computing.
//
// We store graphs in a binary format. Other formats such as edge-list can be
// converted to our format with graph_convert tool. You can use graph_convert as
// follows: e.g. convert a graph from edgelist CSV to edgelist bin file.
// USAGE: graph-convert --convert_mode=[options] -i <input file path> -o <output
// file path> --sep=[separator]

#include <gflags/gflags.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <type_traits>

#include "core/common/types.h"
#include "core/data_structures/bit_tiled_matrix.cuh"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/util/atomic.h"
#include "core/util/bitmap.h"
#include "core/util/cuda_check.cuh"
#include "core/util/format_converter.cuh"
#include "tools/graph_converter/converter/to_bit_tiled_matrix.cuh"
#include "tools/graph_converter/converter/to_edgelist.cuh"
#include "tools/graph_converter/converter/to_egsm_graph.cuh"
#include "tools/graph_converter/converter/to_grid_csr_tiled_matrix.cuh"
#include "tools/graph_converter/converter/to_immutable_csr.cuh"

using sics::matrixgraph::core::common::EdgeIndex;
using sics::matrixgraph::core::common::VertexID;
using sics::matrixgraph::core::data_structures::ImmutableCSR;
using sics::matrixgraph::core::util::Bitmap;
using sics::matrixgraph::core::util::atomic::WriteMax;

// Command line flags
DEFINE_string(i, "", "Input file path");
DEFINE_string(o, "", "Output file path");
DEFINE_string(convert_mode, "",
              "Conversion mode (see help for available modes)");
DEFINE_string(sep, ",", "Separator for CSV files (default: comma)");
DEFINE_bool(compressed, false, "Use compressed vertex IDs");
DEFINE_uint32(tile_size, 64, "Size of a single tile");
DEFINE_uint32(label_range, 1, "label range for initialization");

// Conversion modes
enum class ConvertMode {
  kEdgelistCSV2TiledMatrix,     // Convert edge list CSV to tiled matrix
  kEdgelistBin2TiledMatrix,     // Convert binary edge list to tiled matrix
  kEdgelistCSV2BitTiledMatrix,  // Convert edge list CSV to bit-tiled matrix
  kEdgelistBin2BitTiledMatrix,  // Convert binary edge list to bit-tiled matrix
  kEdgelistBin2TransposedEdgelistBin,  // Convert binary edge list to transposed
                                       // binary edge list
  kGridEdgelistBin2BitTiledMatrix,     // Convert grid graph binary edge list to
                                       // bit-tiled matrix
  kGridEdgelistBin2CSRTiledMatrix,     // Convert grid graph binary edge list to
                                       // CSR tiled matrix
  kCSRBin2BitTiledMatrix,              // Convert binary CSR to bit-tiled matrix
  kEdgelistBin2CSRBin,                 // Convert binary edge list to binary CSR
  kEdgelistCSV2CSRBin,                 // Convert edge list CSV to binary CSR
  kEdgelistCSV2EdgelistBin,  // Convert edge list CSV to binary edge list
  kCSRBin2EdgelistBin,       // Convert binary CSR to binary edge list
  kEdgelistCSV2CGGraphCSR,   // Convert edge list CSV to CG graph CSR
  kEdgelistBin2CGGraphCSR,   // Convert binary edge list to CG graph CSR
  kCSRBin2EGSM,              // Convert binary CSR to EGSM format
  kEGSM2EdgelistBin,         // Convert EGSM format to binary edge list
  kEGSM2CSRBin,              // Convert EGSM format to binary csr
  kUndefined                 // Undefined conversion mode
};

// Convert string to ConvertMode enum
ConvertMode ConvertMode2Enum(const std::string& mode) {
  static const std::unordered_map<std::string, ConvertMode> mode_map = {
      {"edgelistcsv2tiledmatrix", ConvertMode::kEdgelistCSV2TiledMatrix},
      {"edgelistbin2tiledmatrix", ConvertMode::kEdgelistBin2TiledMatrix},
      {"edgelistcsv2bittiledmatrix", ConvertMode::kEdgelistCSV2BitTiledMatrix},
      {"edgelistbin2bittiledmatrix", ConvertMode::kEdgelistBin2BitTiledMatrix},
      {"edgelistbin2transposededgelistbin",
       ConvertMode::kEdgelistBin2TransposedEdgelistBin},
      {"gridedgelistbin2bittiledmatrix",
       ConvertMode::kGridEdgelistBin2BitTiledMatrix},
      {"gridedgelistbin2csrtiledmatrix",
       ConvertMode::kGridEdgelistBin2CSRTiledMatrix},
      {"csrbin2bittiledmatrix", ConvertMode::kCSRBin2BitTiledMatrix},
      {"edgelistbin2csrbin", ConvertMode::kEdgelistBin2CSRBin},
      {"edgelistcsv2csrbin", ConvertMode::kEdgelistCSV2CSRBin},
      {"edgelistcsv2edgelistbin", ConvertMode::kEdgelistCSV2EdgelistBin},
      {"csrbin2edgelistbin", ConvertMode::kCSRBin2EdgelistBin},
      {"edgelistcsv2cggraphcsr", ConvertMode::kEdgelistCSV2CGGraphCSR},
      {"edgelistbin2cggraphcsr", ConvertMode::kEdgelistBin2CGGraphCSR},
      {"csrbin2egsm", ConvertMode::kCSRBin2EGSM},
      {"egsm2edgelistbin", ConvertMode::kEGSM2EdgelistBin},
      {"egsm2csrbin", ConvertMode::kEGSM2CSRBin}};

  auto it = mode_map.find(mode);
  return it != mode_map.end() ? it->second : ConvertMode::kUndefined;
}

// Print usage information
void PrintUsage() {
  std::cout
      << "\nGraph Converter Tool\n"
      << "Usage: graph-convert --convert_mode=<mode> -i <input_path> -o "
         "<output_path> [options]\n\n"
      << "Available conversion modes:\n"
      << "  edgelistcsv2edgelistbin    - Convert edge list CSV to binary edge "
         "list\n"
      << "  edgelistcsv2csrbin         - Convert edge list CSV to binary CSR\n"
      << "  edgelistbin2csrbin         - Convert binary edge list to binary "
         "CSR\n"
      << "  edgelistbin2cggraphbin     - Convert binary edge list to binary "
         "CGGraph CSR\n"
      << "  edgelistcsv2cggraphbin     - Convert edge list CSV to binary "
         "CGGraph CSR\n"
      << "  edgelistbin2tiledmatrix    - Convert binary edge list to tiled "
         "matrix\n"
      << "  csrbin2edgelistbin         - Convert binary CSR to binary edge "
         "list\n"
      << "  csrbin2egsm                - Convert binary CSR to EGSM format\n"
      << "  egsm2edgelistbin           - Convert EGSM format to binary edge "
         "list\n"
      << "  egsm2csrbin                - Convert EGSM format to binary csr\n "
      << "\nOptions:\n"
      << "  --sep=<separator>           - Separator for CSV files (default: "
         "comma)\n"
      << "  --compressed                - Use compressed vertex IDs\n"
      << "  --tile_size=<size>         - Size of a single tile (default: 64)\n"
      << std::endl;
}

// Validate input parameters
bool ValidateParameters() {
  if (FLAGS_i.empty() || FLAGS_o.empty()) {
    std::cerr << "Error: Input and output paths are required." << std::endl;
    return false;
  }

  if (FLAGS_convert_mode.empty()) {
    std::cerr << "Error: Conversion mode is required." << std::endl;
    return false;
  }

  if (!std::filesystem::exists(FLAGS_i)) {
    std::cerr << "Error: Input file does not exist: " << FLAGS_i << std::endl;
    return false;
  }

  if (ConvertMode2Enum(FLAGS_convert_mode) == ConvertMode::kUndefined) {
    std::cerr << "Error: Invalid conversion mode: " << FLAGS_convert_mode
              << std::endl;
    return false;
  }

  return true;
}

int main(int argc, char** argv) {
  gflags::SetUsageMessage("Graph format converter tool");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!ValidateParameters()) {
    PrintUsage();
    return EXIT_FAILURE;
  }

  try {
    switch (ConvertMode2Enum(FLAGS_convert_mode)) {
      case ConvertMode::kEdgelistBin2CSRBin:
        sics::matrixgraph::tools::converter::ConvertEdgelistBin2CSRBin(
            FLAGS_i, FLAGS_o, FLAGS_compressed, FLAGS_label_range);
        break;
      case ConvertMode::kEdgelistBin2TransposedEdgelistBin:
        sics::matrixgraph::tools::converter::
            ConvertEdgelistBin2TransposedEdgelistBin(FLAGS_i, FLAGS_o);
        break;
      case ConvertMode::kEdgelistCSV2CSRBin:
        sics::matrixgraph::tools::converter::ConvertEdgelistCSV2ImmutableCSR(
            FLAGS_i, FLAGS_o, FLAGS_sep, FLAGS_compressed, FLAGS_label_range);
        break;
      case ConvertMode::kEdgelistCSV2EdgelistBin:
        sics::matrixgraph::tools::converter::ConvertEdgelistCSV2EdgelistBin(
            FLAGS_i, FLAGS_o, FLAGS_sep, FLAGS_compressed, FLAGS_label_range);
        break;
      case ConvertMode::kCSRBin2EdgelistBin:
        sics::matrixgraph::tools::converter::ConvertImmutableCSR2EdgelistBin(
            FLAGS_i, FLAGS_o, FLAGS_compressed);
        break;
      case ConvertMode::kGridEdgelistBin2BitTiledMatrix:
        sics::matrixgraph::tools::converter::ConvertGridGraph2BitTiledMatrix(
            FLAGS_i, FLAGS_o, FLAGS_tile_size);
        break;
      case ConvertMode::kGridEdgelistBin2CSRTiledMatrix:
        sics::matrixgraph::tools::converter::ConvertGridGraph2CSRTiledMatrix(
            FLAGS_i, FLAGS_o, FLAGS_tile_size);
        break;
      case ConvertMode::kEdgelistCSV2CGGraphCSR:
        sics::matrixgraph::tools::converter::ConvertEdgelistCSV2CGGraphCSR(
            FLAGS_i, FLAGS_o, FLAGS_sep);
        break;
      case ConvertMode::kEdgelistBin2CGGraphCSR:
        sics::matrixgraph::tools::converter::ConvertEdgelistBin2CGGraphCSR(
            FLAGS_i, FLAGS_o);
        break;
      case ConvertMode::kCSRBin2EGSM:
        sics::matrixgraph::tools::converter::ConvertCSRBin2EGSMGraph(FLAGS_i,
                                                                     FLAGS_o);
        break;
      case ConvertMode::kEGSM2EdgelistBin:
        sics::matrixgraph::tools::converter::ConvertEGSMGraph2EdgelistBin(
            FLAGS_i, FLAGS_o);
        break;
      case ConvertMode::kEGSM2CSRBin:
        sics::matrixgraph::tools::converter::ConvertEGSMGraph2CSRBin(FLAGS_i,
                                                                     FLAGS_o);
        break;
      default:
        std::cerr << "Error: Unsupported conversion mode" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Conversion completed successfully." << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error during conversion: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}