// This file belongs to the SICS graph-systems project, a C++ library for
// exploiting parallelism graph computing.
//
// We store graphs in a binary format. Other formats such as edge-list can be
// converted to our format with graph_convert tool. You can use graph_convert as
// follows: e.g. convert a graph from edgelist CSV to edgelist bin file.
// USAGE: graph-convert --convert_mode=[options] -i <input file path> -o <output
// file path> --sep=[separator]

#include <gflags/gflags.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/types.h"
#include "core/data_structures/edgelist.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/util/atomic.h"
#include "core/util/bitmap.h"
#include "core/util/format_converter.cuh"
#include "tools/graph_converter/converter/to_bit_tiled_matrix.cuh"
#include "tools/graph_converter/converter/to_arangodb_json.cuh"
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
DEFINE_bool(keep_original_vid, false, "Keep original vertex IDs (no compression to contiguous range)");
DEFINE_uint32(tile_size, 64, "Size of a single tile");
DEFINE_uint32(label_range, 1, "label range for initialization");
DEFINE_string(graph_id, "1", "Graph ID for ArangoDB export (numeric)");
DEFINE_string(business_id, "1",
              "Business ID for ArangoDB export");
DEFINE_string(import_time, "1970-01-01T00:00:00Z",
              "Import time for ArangoDB export (_time)");
DEFINE_string(pivot_time, "1970-01-01T00:00:00Z",
              "Business time for ArangoDB export (_pivot_time)");
DEFINE_string(pivot_mode, "single",
              "Pivot graph generation mode for ArangoDB export: single|source|k_hop");
DEFINE_uint32(k_hop, 2, "K-hop distance for k_hop pivot mode (default: 2)");
DEFINE_string(default_vertex_label, "vertex",
              "Default vertex label for ArangoDB export");
DEFINE_string(default_edge_label, "relationship",
              "Default edge label for ArangoDB export");
DEFINE_bool(random_vertex_labels, false,
            "Randomly assign vertex labels within --label_range");

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
  kCSRBin2CECI,              // Convert binary CSR to CECI format
  kCSRBin2GNNPE,             // Convert binary CSR to GNNPE format
  kCSRBin2VF3,               // Convert binary CSR to VF3 format
  kEGSM2EdgelistBin,         // Convert EGSM format to binary edge list
  kEGSM2CSRBin,              // Convert EGSM format to binary csr
  kEdgelistCSV2ArangoDBJSON, // Convert edge list CSV to ArangoDB JSON files
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
      {"csrbin2ceci", ConvertMode::kCSRBin2CECI},
      {"csrbin2gnnpe", ConvertMode::kCSRBin2GNNPE},
      {"csrbin2vf3", ConvertMode::kCSRBin2VF3},
      {"egsm2edgelistbin", ConvertMode::kEGSM2EdgelistBin},
      {"egsm2csrbin", ConvertMode::kEGSM2CSRBin},
      {"edgelistcsv2arangodbjson", ConvertMode::kEdgelistCSV2ArangoDBJSON}};

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
      << "  csrbin2ceci                - Convert binary CSR to CECI format\n"
      << "  csrbin2gnnpe                - Convert binary CSR to GNNPE format\n"
      << "  egsm2edgelistbin           - Convert EGSM format to binary edge "
         "list\n"
      << "  egsm2csrbin                - Convert EGSM format to binary csr\n "
      << "  edgelistcsv2arangodbjson   - Convert edge list CSV to ArangoDB JSON "
         "files\n"
      << "\nOptions:\n"
      << "  --sep=<separator>           - Separator for CSV files (default: "
         "comma)\n"
      << "  --keep_original_vid         - Keep original vertex IDs (no compression to contiguous range)\n"
      << "  --tile_size=<size>         - Size of a single tile (default: 64)\n"
      << "  --graph_id=<id>             - Graph ID for ArangoDB export\n"
      << "  --business_id=<id>          - Business ID for ArangoDB export\n"
      << "  --import_time=<ts>          - Import time (_time)\n"
      << "  --pivot_time=<ts>           - Business time (_pivot_time)\n"
      << "  --pivot_mode=single|source|k_hop  - Pivot generation mode\n"
      << "  --k_hop=<N>                 - K-hop distance for k_hop mode (default: 2)\n"
      << "  --default_vertex_label=<l>  - Default vertex label\n"
      << "  --default_edge_label=<l>    - Default edge label\n"
      << "  --random_vertex_labels      - Random labels within --label_range\n"
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

  if (ConvertMode2Enum(FLAGS_convert_mode) ==
      ConvertMode::kEdgelistCSV2ArangoDBJSON) {
    auto is_numeric = [](const std::string& s) {
      return !s.empty() &&
             std::all_of(s.begin(), s.end(),
                         [](unsigned char c) { return std::isdigit(c) != 0; });
    };
    if (!is_numeric(FLAGS_graph_id) || !is_numeric(FLAGS_business_id)) {
      std::cerr << "Error: --graph_id and --business_id must be numeric."
                << std::endl;
      return false;
    }
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
            FLAGS_i, FLAGS_o, FLAGS_keep_original_vid, FLAGS_label_range);
        break;
      case ConvertMode::kEdgelistBin2TransposedEdgelistBin:
        sics::matrixgraph::tools::converter::
            ConvertEdgelistBin2TransposedEdgelistBin(FLAGS_i, FLAGS_o);
        break;
      case ConvertMode::kEdgelistCSV2CSRBin:
        sics::matrixgraph::tools::converter::ConvertEdgelistCSV2ImmutableCSR(
            FLAGS_i, FLAGS_o, FLAGS_sep, FLAGS_keep_original_vid, FLAGS_label_range);
        break;
      case ConvertMode::kEdgelistCSV2EdgelistBin:
        sics::matrixgraph::tools::converter::ConvertEdgelistCSV2EdgelistBin(
            FLAGS_i, FLAGS_o, FLAGS_sep, FLAGS_keep_original_vid, FLAGS_label_range);
        break;
      case ConvertMode::kCSRBin2EdgelistBin:
        sics::matrixgraph::tools::converter::ConvertImmutableCSR2EdgelistBin(
            FLAGS_i, FLAGS_o, FLAGS_keep_original_vid);
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
      case ConvertMode::kCSRBin2CECI:
        sics::matrixgraph::tools::converter::ConvertCSRBin2CECIGraph(FLAGS_i,
                                                                     FLAGS_o);
        break;
      case ConvertMode::kCSRBin2GNNPE:
        sics::matrixgraph::tools::converter::ConvertCSRBin2GNNPEGraph(FLAGS_i,
                                                                      FLAGS_o);
        break;
      case ConvertMode::kCSRBin2VF3:
        sics::matrixgraph::tools::converter::ConvertCSRBin2VF3Graph(FLAGS_i,
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
      case ConvertMode::kEdgelistCSV2ArangoDBJSON: {
        sics::matrixgraph::tools::converter::ArangoExportOptions opt;
        opt.graph_id = std::stoull(FLAGS_graph_id);
        opt.business_id = std::stoull(FLAGS_business_id);
        opt.import_time = FLAGS_import_time;
        opt.pivot_time = FLAGS_pivot_time;
        opt.pivot_mode = FLAGS_pivot_mode;
        opt.k_hop = FLAGS_k_hop;
        opt.default_vertex_label = FLAGS_default_vertex_label;
        opt.default_edge_label = FLAGS_default_edge_label;
        opt.random_vertex_labels = FLAGS_random_vertex_labels;
        opt.label_range = FLAGS_label_range;
        if (!sics::matrixgraph::tools::converter::ConvertEdgelistCSV2ArangoDBJSON(
                FLAGS_i, FLAGS_o, FLAGS_sep, FLAGS_keep_original_vid, opt)) {
          std::cerr << "Error: write ArangoDB JSON failed." << std::endl;
          return EXIT_FAILURE;
        }
        break;
      }
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