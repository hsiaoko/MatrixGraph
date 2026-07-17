// GraphRAG wrapper: run MatrixGraph BFS and write per-vertex levels to disk.
#include <gflags/gflags.h>

#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "core/common/types.h"
#include "core/components/scheduler/scheduler.h"
#include "core/data_structures/host_buffer.cuh"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/unified_buffer.cuh"
#include "core/matrixgraph.cuh"
#include "core/task/gpu_task/kernel/kernel_bfs.cuh"
#include "core/task/gpu_task/task_base.cuh"
#include "core/util/cuda_check.cuh"
#include "core/util/cuda_device.cuh"
#include "core/util/cuda_prefetch.cuh"

DEFINE_string(g, "", "Path to the input CSR graph directory (required)");
DEFINE_string(o, "", "Path to the output directory (required)");
DEFINE_int32(src, 0, "Source vertex local id (default: 0)");
DEFINE_string(scheduler, "CHBL",
              "Scheduler type (options: CHBL, EvenSplit, RoundRobin)");

using sics::matrixgraph::core::common::EdgeIndex;
using sics::matrixgraph::core::common::VertexID;
using sics::matrixgraph::core::common::VertexLabel;
using sics::matrixgraph::core::components::scheduler::SchedulerType;
using sics::matrixgraph::core::data_structures::ImmutableCSR;
using sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer;
using sics::matrixgraph::core::task::kernel::BFSKernelWrapper;

SchedulerType Scheduler2Enum(const std::string& s) {
  if (s == "EvenSplit")
    return sics::matrixgraph::core::components::scheduler::kEvenSplit;
  else if (s == "CHBL")
    return sics::matrixgraph::core::components::scheduler::kCHBL;
  else if (s == "RoundRobin")
    return sics::matrixgraph::core::components::scheduler::kRoundRobin;
  return sics::matrixgraph::core::components::scheduler::kCHBL;
}

bool ValidateParameters() {
  bool is_valid = true;
  if (FLAGS_g.empty()) {
    std::cerr << "Error: Input graph path (-g) is required" << std::endl;
    is_valid = false;
  }
  if (FLAGS_o.empty()) {
    std::cerr << "Error: Output path (-o) is required" << std::endl;
    is_valid = false;
  }
  return is_valid;
}

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "GraphRAG BFS wrapper: compute BFS and write levels to disk\n"
      "Usage: " +
      std::string(argv[0]) +
      " -g <csr_dir> -o <output_dir> -src <local_vertex_id>");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!ValidateParameters()) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "apps/graphrag_bfs.cpp");
    return EXIT_FAILURE;
  }

  try {
    ImmutableCSR g;
    g.Read(FLAGS_g);

    auto scheduler_type = Scheduler2Enum(FLAGS_scheduler);
    sics::matrixgraph::core::MatrixGraph system(scheduler_type);

    // Replicate BFS::ExecuteBFS buffer setup.
    sics::matrixgraph::core::data_structures::Buffer<uint8_t> data_g;
    data_g.data = g.GetGraphBuffer();
    data_g.size = sizeof(VertexID) * g.get_num_vertices() * 3 +
                  sizeof(EdgeIndex) * (g.get_num_vertices() + 1) * 2 +
                  sizeof(VertexID) *
                      (g.get_num_incoming_edges() + g.get_num_outgoing_edges() +
                       g.get_max_vid() + 1);

    UnifiedOwnedBuffer<uint8_t> unified_data_g;
    unified_data_g.Init(data_g);

    sics::matrixgraph::core::data_structures::Buffer<VertexLabel> v_label_g;
    v_label_g.data = g.GetVLabelBasePointer();
    v_label_g.size = sizeof(VertexLabel) * g.get_num_vertices();

    UnifiedOwnedBuffer<VertexLabel> unified_v_label_g;
    unified_v_label_g.Init(v_label_g);

    const std::vector<int> cuda_devices =
        sics::matrixgraph::core::util::MatrixGraphCudaDeviceList();
    const int primary_gpu = cuda_devices.empty() ? 0 : cuda_devices[0];
    CUDA_CHECK(cudaSetDevice(primary_gpu));

    std::vector<std::pair<void*, size_t>> prefetch = {
        {reinterpret_cast<void*>(unified_data_g.GetPtr()),
         unified_data_g.GetSize()},
        {reinterpret_cast<void*>(unified_v_label_g.GetPtr()),
         unified_v_label_g.GetSize()}};
    sics::matrixgraph::core::util::MatrixGraphPrefetchManagedToDevice(
        primary_gpu, sics::matrixgraph::core::util::MatrixGraphCudaStreamsPerGpu(),
        prefetch);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    BFSKernelWrapper::BFS(stream, g.get_num_vertices(),
                          g.get_num_outgoing_edges(), FLAGS_src, unified_data_g,
                          unified_v_label_g);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results from managed memory back to the host label buffer.
    CUDA_CHECK(cudaMemcpy(g.GetVLabelBasePointer(),
                          unified_v_label_g.GetPtr(),
                          sizeof(VertexLabel) * g.get_num_vertices(),
                          cudaMemcpyDefault));

    std::filesystem::create_directories(FLAGS_o);
    std::ofstream level_file(FLAGS_o + "/bfs_levels.bin", std::ios::binary);
    level_file.write(
        reinterpret_cast<const char*>(g.GetVLabelBasePointer()),
        sizeof(VertexLabel) * g.get_num_vertices());
    level_file.close();

    VertexLabel max_level = 0;
    VertexLabel unreachable = std::numeric_limits<VertexLabel>::max();
    VertexID reached = 0;
    for (VertexID i = 0; i < g.get_num_vertices(); i++) {
      VertexLabel lvl = g.GetVLabelBasePointer()[i];
      if (lvl != unreachable) {
        reached++;
        if (lvl > max_level) max_level = lvl;
      }
    }

    std::ofstream summary_file(FLAGS_o + "/bfs_summary.txt");
    summary_file << "num_vertices " << g.get_num_vertices() << "\n";
    summary_file << "source " << FLAGS_src << "\n";
    summary_file << "reached " << reached << "\n";
    summary_file << "max_level " << max_level << "\n";
    summary_file.close();

    std::cout << "[GraphRAG BFS] Wrote " << g.get_num_vertices()
              << " levels to " << FLAGS_o << "/bfs_levels.bin"
              << ", reached=" << reached << ", max_level=" << max_level
              << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
