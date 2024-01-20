#ifndef MATRIXGRAPH_CORE_GPU_HOST_FUNC_CUH_
#define MATRIXGRAPH_CORE_GPU_HOST_FUNC_CUH_

#include <cublas_v2.h>
#include <cutlass/gemm/device/gemm.h>

#include "core/data_structures/tiled_matrix.cuh"
#include "core/util/matrix.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace gpu {

using sics::matrixgraph::core::common::TileIndex;
using sics::matrixgraph::core::common::VertexID;
using sics::matrixgraph::core::data_structures::Tile;
using sics::matrixgraph::core::data_structures::TiledMatrix;
using sics::matrixgraph::core::util::AllocateMatrix;
using sics::matrixgraph::core::util::Bitmap;
using sics::matrixgraph::core::util::InitializeMatrix_kernel;

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(int M, int N, int K, float alpha, float const *A,
                           int lda, float const *B, int ldb, float beta,
                           float *C, int ldc) {

  // Define type definition for single-precision CUTLASS GEMM with
  // column-major input matrices and 128x128x8 threadblock tile size (chosen
  // by default).
  //
  // To keep the interface manageable, several helpers are defined for
  // plausible compositions including the following example for
  // single-precision GEMM. Typical values are used as default template
  // arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for
  // more details.
  //
  // To view the full gemm device API interface, see
  // `cutlass/gemm/device/gemm.h`

  using ColumnMajor = cutlass::layout::ColumnMajor;

  using CutlassGemm =
      cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                  ColumnMajor,  // Layout of A matrix
                                  float,        // Data-type of B matrix
                                  ColumnMajor,  // Layout of B matrix
                                  float,        // Data-type of C matrix
                                  ColumnMajor>; // Layout of C matrix

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that
  // are constructible in host code and passed to kernels by value. These may
  // include pointers, strides, scalars, and other arguments needed by Gemm
  // and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy
  // for passing host-constructible arguments to kernels and (2.) minimized
  // initialization overhead on kernel entry.
  //
  CutlassGemm::Arguments args(
      {M, N, K},      // Gemm Problem dimensions
      {A, lda},       // Tensor-ref for source matrix A
      {B, ldb},       // Tensor-ref for source matrix B
      {C, ldc},       // Tensor-ref for source matrix C
      {C, ldc},       // Tensor-ref for destination matrix D (may be different
                      // memory than source C matrix)
      {alpha, beta}); // Scalars used in the Epilogue

  // Launch the CUTLASS GEMM kernel.

  cutlass::Status status = gemm_operator(args);

  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

/// Naive GEMM computation.
cudaError_t NaiveGemm(int M, int N, int K, float alpha, float const *A, int lda,
                      float const *B, int ldb, float beta, float *C, int ldc) {

  dim3 block(16, 16);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  NaiveGemm_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, C,
                                    ldc);

  return cudaGetLastError();
}

/// cuda Tensor core GEMM computation.
cudaError_t TensorCoreGemm(int M, int N, int K, float alpha, half const *A,
                           int lda, half const *B, int ldb, float beta,
                           float *C, int ldc) {

  dim3 block(16, 16);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  TensorCoreGemm_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta,
                                         C, ldc);

  return cudaGetLastError();
}

// Allocate several matrices in GPU device memory and call a single-precision
// CUTLASS GEMM kernel.
cudaError_t SubmitCutlassGemm(int M, int N, int K, float alpha, float beta) {
  cudaError_t result;

  //
  // Define several matrices to be used as operands to GEMM kernels.
  //

  // Compute leading dimensions for each matrix.
  int lda = M;
  int ldb = K;
  int ldc = M;

  // Compute size in bytes of the C matrix.
  size_t sizeof_C = sizeof(float) * ldc * N;

  // Define pointers to matrices in GPU device memory.
  float *A;
  float *B;
  float *C_cutlass;

  //
  // Allocate matrices in GPU device memory with arbitrary seeds.
  //

  result = AllocateMatrix(&A, M, K, 0);

  if (result != cudaSuccess) {
    return result;
  }

  result = AllocateMatrix(&B, K, N, 17);

  if (result != cudaSuccess) {
    cudaFree(A);
    return result;
  }

  result = AllocateMatrix(&C_cutlass, M, N, 101);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    return result;
  }

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    cudaFree(C_cutlass);
    return result;
  }

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy C_cutlass matrix to C_reference: "
              << cudaGetErrorString(result) << std::endl;
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  //
  // Launch CUTLASS GEMM.
  //

  result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc);

  if (result != cudaSuccess) {
    std::cerr << "CUTLASS GEMM kernel failed: " << cudaGetErrorString(result)
              << std::endl;

    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  // Copy to host and verify equivalence.
  std::vector<float> host_cutlass(ldc * N, 0);

  result = cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C,
                      cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < host_cutlass.size(); i++) {
    std::cout << host_cutlass[i] << ", ";
  }

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy CUTLASS GEMM results: "
              << cudaGetErrorString(result) << std::endl;

    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  //
  // Free device memory allocations.
  //

  cudaFree(C_cutlass);
  cudaFree(B);
  cudaFree(A);
  return cudaSuccess;
}

// Allocate several matrices in GPU device memory and call a single-precision
// CUTLASS GEMM kernel.
cudaError_t NaiveGemm_host(int M, int N, int K, float alpha, float beta) {
  cudaError_t result;

  std::cout << "Naive GEMM" << std::endl;
  //
  // Define several matrices to be used as operands to GEMM kernels.
  //

  // Compute leading dimensions for each matrix.
  int lda = M;
  int ldb = K;
  int ldc = M;

  // Compute size in bytes of the C matrix.
  size_t sizeof_C = sizeof(float) * ldc * N;

  // Define pointers to matrices in GPU device memory.
  float *A;
  float *B;
  float *C_reference;

  //
  // Allocate matrices in GPU device memory with arbitrary seeds.
  //

  result = AllocateMatrix(&A, M, K, 0);

  if (result != cudaSuccess) {
    return result;
  }

  result = AllocateMatrix(&B, K, N, 17);

  if (result != cudaSuccess) {
    cudaFree(A);
    return result;
  }

  result = AllocateMatrix(&C_reference, M, N, 101);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    return result;
  }

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy C_cutlass matrix to C_reference: "
              << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  // Launch Naive GEMM
  result = NaiveGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

  if (result != cudaSuccess) {
    std::cerr << "Reference GEMM kernel failed: " << cudaGetErrorString(result)
              << std::endl;

    cudaFree(C_reference);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  // Copy to host and verify equivalence.
  std::vector<float> host_reference(ldc * N, 0);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy CUTLASS GEMM results: "
              << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  result = cudaMemcpy(host_reference.data(), C_reference, sizeof_C,
                      cudaMemcpyDeviceToHost);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy Reference GEMM results: "
              << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  //
  // Free device memory allocations.
  //
  cudaFree(C_reference);
  cudaFree(B);
  cudaFree(A);

  std::cout << "snap shot: " << host_reference[0] << std::endl;

  return cudaSuccess;
}

cudaError_t cuBLASGemm_host(int M, int N, int K, float alpha, float beta) {
  std::cout << "cuBLAS GEMM" << std::endl;
  int m = M;
  int n = N;
  int k = K;

  printf("shape: (%d, %d) x (%d, %d)\n", m, k, k, n);

  int alg_id = -1;
  int start_algo = CUBLAS_GEMM_DEFAULT;
  // int start_algo = alg_id;
  int end_algo = CUBLAS_GEMM_ALGO23;
  // int end_algo = alg_id;
  int start_algo_t_op = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  // int start_algo_t_op = alg_id;
  int end_algo_t_op = CUBLAS_GEMM_ALGO15_TENSOR_OP;
  // int end_algo_t_op = alg_id;
  int iteration = 1;

  float *fA, *fB, *fC;
  __half *hA, *hB, *hC;
  int8_t *iA, *iB;
  int32_t *iC;
  float f_alpha = 1, f_beta = 0;
  __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
  int32_t i_alpha = 1, i_beta = 0;
  sics::matrixgraph::core::util::AllocateDeviceMemory(m, n, k, &fA, &fB, &fC);
  sics::matrixgraph::core::util::AllocateDeviceMemory(m, n, k, &hA, &hB, &hC);
  sics::matrixgraph::core::util::AllocateDeviceMemory(m, n, k, &iA, &iB, &iC);
  for (int i = 0; i < m * k; ++i) {
    fA[i] = float(i % 255 - 127) / 127;
    hA[i] = __float2half_rn(fA[i]);
    iA[i] = sics::matrixgraph::core::util::float2int8(fA[i], 127);
  }
  for (int i = 0; i < k * n; ++i) {
    fB[i] = float(i % 255 - 127) / 127;
    hB[i] = __float2half_rn(fB[i]);
    iB[i] = sics::matrixgraph::core::util::float2int8(fB[i], 127);
  }
  cublasHandle_t handle;
  cublasCreate(&handle);

  printf(">>>>>>>>>>>>>>>>> test fp32 >>>>>>>>>>>>>>>>>\n");
  for (int algo = start_algo; algo < end_algo; ++algo)
    sics::matrixgraph::core::util::TestcuBLASGemm(
        handle, m, n, k, fA, fB, fC, &f_alpha, &f_beta, algo, iteration);
  for (int algo = start_algo_t_op; algo < end_algo_t_op; ++algo)
    sics::matrixgraph::core::util::TestcuBLASGemm(
        handle, m, n, k, fA, fB, fC, &f_alpha, &f_beta, algo, iteration);

  printf(">>>>>>>>>>>>>>>>> test fp16 >>>>>>>>>>>>>>>>>\n");
  for (int algo = start_algo; algo < end_algo; ++algo)
    sics::matrixgraph::core::util::TestcuBLASGemm(
        handle, m, n, k, hA, hB, hC, &h_alpha, &h_beta, algo, iteration);
  for (int algo = start_algo_t_op; algo <= end_algo_t_op; ++algo)
    sics::matrixgraph::core::util::TestcuBLASGemm(
        handle, m, n, k, hA, hB, hC, &h_alpha, &h_beta, algo, iteration);

  printf(">>>>>>>>>>>>>>>>> test int8 >>>>>>>>>>>>>>>>>\n");
  for (int algo = start_algo; algo <= end_algo; ++algo)
    sics::matrixgraph::core::util::TestcuBLASGemm(
        handle, m, n, k, iA, iB, iC, &i_alpha, &i_beta, algo, iteration);
  for (int algo = start_algo_t_op; algo <= end_algo_t_op; ++algo)
    sics::matrixgraph::core::util::TestcuBLASGemm(
        handle, m, n, k, iA, iB, iC, &i_alpha, &i_beta, algo, iteration);

  printf(">>>>>>>>>>>>>>>>> compare result >>>>>>>>>>>>>>>>>\n");
  printf("fp32: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", fC[i], " \n"[i == 9]);
  printf("fp16: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(hC[i]), " \n"[i == 9]);
  printf("int8: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(iC[i]) / 127 / 127, " \n"[i == 9]);

  std::cout << "#cuBLAS GEMM" << std::endl;

  sics::matrixgraph::core::util::FreeDeviceMemory(iA, iB, iC);
  sics::matrixgraph::core::util::FreeDeviceMemory(fA, fB, fC);
  return cudaSuccess;
}

cudaError_t TiledMatrixGemm_host(const Tile &tile, const Tile &tile_t,
                                 const cudaStream_t &stream) {
  // std::cout
  //     << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>TiledMatrix GEMM"
  //     << std::endl;

  tile.Show();
  tile_t.Show();
  cudaError_t result;

  // 1. Allocate device memory.
  auto tile_device = new Tile();
  auto tile_t_device = new Tile();
  auto *tile_output_device = new Tile();

  auto tile_output_host = new Tile();

  result = tile_device->InitDevice(tile.get_tile_size(), tile.get_tile_x(),
                                   tile.get_tile_y(), tile.get_n_nz());

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: " << cudaGetErrorString(result)
              << std::endl;
    return result;
  }

  result =
      tile_t_device->InitDevice(tile_t.get_tile_size(), tile_t.get_tile_x(),
                                tile_t.get_tile_y(), tile_t.get_n_nz());
  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: " << cudaGetErrorString(result)
              << std::endl;
    return result;
  }

  tile_output_host->InitAsOutput(tile, tile_t);
  result = tile_output_device->InitDevice(
      tile_output_host->get_tile_size(), tile_output_host->get_tile_x(),
      tile_output_host->get_tile_y(), tile_output_host->get_n_nz());

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: " << cudaGetErrorString(result)
              << std::endl;
    return result;
  }

  result =
      tile_output_device->MemcpyAsyncHost2Device(*tile_output_host, stream);
  if (result != cudaSuccess) {
    std::cerr << "Failed to copy memory from host to device: "
              << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // 2. Transfer data from host to device.
  result = tile_device->MemcpyAsyncHost2Device(tile, stream);
  if (result != cudaSuccess) {
    std::cerr << "Failed to copy memory from host to device: "
              << cudaGetErrorString(result) << std::endl;
    return result;
  }
  result = tile_t_device->MemcpyAsyncHost2Device(tile_t, stream);
  if (result != cudaSuccess) {
    std::cerr << "Failed to copy memory from host to device: "
              << cudaGetErrorString(result) << std::endl;
    return result;
  }

  dim3 dimBlock(4, 4);
  dim3 dimGrid(1);

  int *offset_d, *offset_h;
  cudaMalloc(reinterpret_cast<void **>(&offset_d), sizeof(int));
  offset_h = new int();
  cudaMemcpyAsync(offset_d, offset_h, sizeof(int), cudaMemcpyHostToDevice,
                  stream);

  int *n_nz_for_each_row_d, *n_nz_for_each_row_h;
  cudaMalloc(reinterpret_cast<void **>(&n_nz_for_each_row_d),
             sizeof(int) * tile_output_host->get_tile_size());
  n_nz_for_each_row_h = new int[tile_output_host->get_tile_size()]();
  cudaMemcpyAsync(n_nz_for_each_row_d, n_nz_for_each_row_h,
                  sizeof(int) * tile_output_host->get_tile_size(),
                  cudaMemcpyHostToDevice, stream);

  // 3. Lunch the kernel function
  TileGemm_kernel<<<dimGrid, dimBlock, 48 * 1024, stream>>>(
      tile_device->get_n_nz(), tile_t_device->get_n_nz(),
      tile_output_device->get_tile_size(), tile_device->get_tile_size(),
      offset_d, n_nz_for_each_row_d, tile_device->GetBarOffsetPtr(),
      tile_t_device->GetBarOffsetPtr(), tile_output_device->GetBarOffsetPtr(),
      tile_device->GetRowIdxPtr(), tile_t_device->GetRowIdxPtr(),
      tile_output_device->GetRowIdxPtr(), tile_device->GetColIdxPtr(),
      tile_t_device->GetColIdxPtr(), tile_output_device->GetColIdxPtr(),
      tile_device->GetDataPtr(), tile_t_device->GetDataPtr(),
      tile_output_device->GetDataPtr(),
      tile_device->GetMaskPtr()->GetDataPtr()->GetDataBasePointer(),
      tile_t_device->GetMaskPtr()->GetDataPtr()->GetDataBasePointer(),
      tile_output_device->GetMaskPtr()->GetDataPtr()->GetDataBasePointer());

  // 4. Transfer data from device to host.
  tile_output_host->MemcpyAsyncDevice2Host(*tile_output_device, stream);

  // 5. Compute row_ptr for the output tile.
  cudaMemcpyAsync(n_nz_for_each_row_h, n_nz_for_each_row_d,
                  sizeof(int) * tile_output_host->get_tile_size(),
                  cudaMemcpyDeviceToHost, stream);
  auto output_tile_ptr_ptr = tile_output_host->GetBarOffsetPtr();
  for (size_t i = 0; i < tile_output_host->get_tile_size() - 1; i++) {
    output_tile_ptr_ptr[i + 1] =
        output_tile_ptr_ptr[i] + n_nz_for_each_row_h[i];
  }

  std::cout << "\nOUTPUT: " << std::endl;
  tile_output_host->Show();

  // 6. Synchronize the stream.
  cudaStreamSynchronize(stream);
  std::cout << "GPU finished" << std::endl;

  delete offset_h;
  delete[] n_nz_for_each_row_h;

  tile_t_device->FreeDevice();
  tile_device->FreeDevice();
  tile_output_device->FreeDevice();
  cudaFree(offset_d);
  cudaFree(n_nz_for_each_row_d);
  return cudaSuccess;
}

} // namespace gpu
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // MATRIXGRAPH_CORE_GPU_HOST_FUNC_CUH_