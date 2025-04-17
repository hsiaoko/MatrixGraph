
#ifndef MATRIX_CORE_COMMON_CONSTS_H_
#define MATRIX_CORE_COMMON_CONSTS_H_

#include <cstdint>

namespace sics {
namespace matrixgraph {
namespace core {
namespace common {

static const uint32_t kLabelRange = 5;

static const uint32_t kDefaultTileSize = 64;
static const uint32_t kMaxVertexID = std::numeric_limits<uint32_t>::max();
static const uint32_t kDefaultTileNum = 1024;
static const uint32_t kMaxNChunks = 256;
static const uint64_t kDefalutNumEdgesPerBlock = 65536;
static const uint64_t kMaxNumEdgesPerBlock = 2108783;
static const uint64_t kMaxNumEdges = 1073741824;
static const uint64_t kDefalutNumEdgesPerTile = 65536;
static const uint64_t kDefalutOutputBufferSize = 256;
static const uint32_t kDefalutNumVerticesPerTile = 64;
static const uint64_t kMaxNumCandidates = 65536;

// Recursive-based
static const uint64_t kMaxNumCandidatesPerThread = 1024;
static const uint64_t kMaxNumLocalWeft = 1 << 10;

// WOJ SubIso
static const uint64_t kMaxNumWeft = 1 << 10;

// GPU configure.
static const uint32_t kSharedMemoryCapacity = 65536;  // 64kb per SM for V100
static const uint32_t kNCUDACoresPerSM = 64;  // 64 CUDA cores per SM for V100
static const uint32_t kNSMsPerGPU = 80;       // V100 have 80 SMs.
static const uint32_t kNWarpPerCUDACore = 2;  // 2 warps per CUDA core.

static const uint32_t kSharedMemorySize = 1024;
static const uint32_t kGridDim = 512;
static const uint32_t kBlockDim = 128;
// static const uint32_t kGridDim = 64;
// static const uint32_t kBlockDim = 64;
static const uint32_t kWarpSize = 32;
static const uint32_t kLogWarpSize = 5;

static const uint32_t kDefaultHeapCapacity = 7;

}  // namespace common
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_COMMON_TYPES_H_