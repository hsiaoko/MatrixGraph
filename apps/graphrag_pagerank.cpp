// GraphRAG wrapper: run MatrixGraph PageRank and write per-vertex scores to disk.
#include <gflags/gflags.h>

#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "core/common/types.h"
#include "core/components/scheduler/scheduler.h"
#include "core/data_structures/host_buffer.cuh"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/unified_buffer.cuh"
#include "core/matrixgraph.cuh"
#include "core/task/gpu_task/kernel/kernel_pagerank.cuh"
#include "core/task/gpu_task/task_base.cuh"
#include "core/util/cuda_check.cuh"
#include "core/util/cuda_device.cuh"
#include "core/util/cuda_prefetch.cuh"

DEFINE_string(g, "", "Path to the input CSR graph directory (required)");
DEFINE_string(o, "", "Path to the output directory (required)");
DEFINE_double(damping, 0.85, "Damping factor (default: 0.85)");
DEFINE_double(epsilon, 1e-6, "Convergence threshold (default: 1e-6)");
DEFINE_int32(max_iter, 10, "Maximum iterations (default: 10)");
DEFINE_string(scheduler, "CHBL",
              "Scheduler type (options: CHBL, EvenSplit, RoundRobin)");

using sics::matrixgraph::core::common::EdgeIndex;
using sics::matrixgraph::core::common::VertexID;
using sics::matrixgraph::core::common::VertexLabel;
using sics::matrixgraph::core::components::scheduler::SchedulerType;
using sics::matrixgraph::core::data_structures::ImmutableCSR;
using sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer;
using sics::matrixgraph::core::task::kernel::PageRankKernelWrapper;

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
      "GraphRAG PageRank wrapper: compute PageRank and write scores to disk\n"
      "Usage: " +
      std::string(argv[0]) +
      " -g <csr_dir> -o <output_dir> [-damping <f>] [-epsilon <f>] "
      "[-max_iter <n>]");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!ValidateParameters()) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "apps/graphrag_pagerank.cpp");
    return EXIT_FAILURE;
  }

  try {
    ImmutableCSR g;
    g.Read(FLAGS_g);

    auto scheduler_type = Scheduler2Enum(FLAGS_scheduler);
    sics::matrixgraph::core::MatrixGraph system(scheduler_type);

    // Replicate PageRank::ComputePageRank buffer setup.
    sics::matrixgraph::core::data_structures::Buffer<uint8_t> data_g;
    data_g.data = g.GetGraphBuffer();
    data_g.size = sizeof(VertexID) * g.get_num_vertices() +
                  sizeof(VertexID) * g.get_num_vertices() +
                  sizeof(VertexID) * g.get_num_vertices() +
                  sizeof(EdgeIndex) * (g.get_num_vertices() + 1) +
                  sizeof(EdgeIndex) * (g.get_num_vertices() + 1) +
                  sizeof(VertexID) * g.get_num_incoming_edges() +
                  sizeof(VertexID) * g.get_num_outgoing_edges() +
                  sizeof(VertexID) * (g.get_max_vid() + 1);

    UnifiedOwnedBuffer<uint8_t> unified_data_g;
    unified_data_g.Init(data_g);

    sics::matrixgraph::core::data_structures::Buffer<VertexLabel> v_label_g;
    v_label_g.data = g.GetVLabelBasePointer();
    v_label_g.size = sizeof(VertexLabel) * g.get_num_vertices();

    UnifiedOwnedBuffer<VertexLabel> unified_v_label_g;
    unified_v_label_g.Init(v_label_g);

    // PageRank result buffer. The original MatrixGraph app sizes it to max_vid,
    // which is off-by-one for compact IDs. We allocate num_vertices to be safe.
    float* page_ranks_ptr = new float[g.get_num_vertices()]();
    sics::matrixgraph::core::data_structures::Buffer<float> page_ranks;
    page_ranks.data = page_ranks_ptr;
    page_ranks.size = sizeof(float) * g.get_num_vertices();

    UnifiedOwnedBuffer<float> unified_page_ranks;
    unified_page_ranks.Init(page_ranks);

    const std::vector<int> cuda_devices =
        sics::matrixgraph::core::util::MatrixGraphCudaDeviceList();
    const int primary_gpu = cuda_devices.empty() ? 0 : cuda_devices[0];
    CUDA_CHECK(cudaSetDevice(primary_gpu));

    std::vector<std::pair<void*, size_t>> prefetch = {
        {reinterpret_cast<void*>(unified_data_g.GetPtr()),
         unified_data_g.GetSize()},
        {reinterpret_cast<void*>(unified_v_label_g.GetPtr()),
         unified_v_label_g.GetSize()},
        {reinterpret_cast<void*>(unified_page_ranks.GetPtr()),
         unified_page_ranks.GetSize()}};
    sics::matrixgraph::core::util::MatrixGraphPrefetchManagedToDevice(
        primary_gpu, sics::matrixgraph::core::util::MatrixGraphCudaStreamsPerGpu(),
        prefetch);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    PageRankKernelWrapper::PageRank(
        stream, g.get_num_vertices(), g.get_num_outgoing_edges(), unified_data_g,
        unified_page_ranks, static_cast<float>(FLAGS_damping),
        static_cast<float>(FLAGS_epsilon), FLAGS_max_iter);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results from managed memory back to the host buffer.
    CUDA_CHECK(cudaMemcpy(page_ranks_ptr, unified_page_ranks.GetPtr(),
                          sizeof(float) * g.get_num_vertices(),
                          cudaMemcpyDefault));

    std::filesystem::create_directories(FLAGS_o);

    // Write scores for every local vertex.
    std::ofstream score_file(FLAGS_o + "/pagerank_scores.bin", std::ios::binary);
    score_file.write(reinterpret_cast<const char*>(page_ranks_ptr),
                     sizeof(float) * g.get_num_vertices());
    score_file.close();

    // Write summary.
    double sum = 0.0;
    float max_score = 0.0f;
    for (VertexID i = 0; i < g.get_num_vertices(); i++) {
      sum += page_ranks_ptr[i];
      if (page_ranks_ptr[i] > max_score) max_score = page_ranks_ptr[i];
    }

    std::ofstream summary_file(FLAGS_o + "/pagerank_summary.txt");
    summary_file << "num_vertices " << g.get_num_vertices() << "\n";
    summary_file << "damping " << FLAGS_damping << "\n";
    summary_file << "epsilon " << FLAGS_epsilon << "\n";
    summary_file << "max_iter " << FLAGS_max_iter << "\n";
    summary_file << "sum_scores " << sum << "\n";
    summary_file << "max_score " << max_score << "\n";
    summary_file.close();

    delete[] page_ranks_ptr;

    std::cout << "[GraphRAG PageRank] Wrote " << g.get_num_vertices()
              << " scores to " << FLAGS_o << "/pagerank_scores.bin"
              << ", sum=" << sum << ", max=" << max_score << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
