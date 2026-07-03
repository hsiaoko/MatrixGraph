
#ifndef MATRIX_CORE_COMMON_CONSTS_H_
#define MATRIX_CORE_COMMON_CONSTS_H_

#include <cstdint>
#include <limits>

namespace sics {
namespace matrixgraph {
namespace core {
namespace common {

static const uint32_t kMaxVertexID = std::numeric_limits<uint32_t>::max();
static const uint32_t kMaxNChunks = 256;
static const uint64_t kDefalutNumEdgesPerBlock = 65536;
static const uint64_t kMaxNumEdgesPerBlock = 2108783;
static const uint64_t kMaxNumEdges = 1073741824;
static const uint64_t kDefalutNumEdgesPerTile = 65536;
static const uint64_t kDefalutOutputBufferSize = 256;
static const uint32_t kDefalutNumVerticesPerTile = 64;

// Recursive-based
static const uint64_t kMaxNumCandidatesPerThread = 128;
static const uint64_t kMaxNumLocalWeft = 1 << 8;

// SubIso / DFS / Matches: max weft bundles (non-WOJ paths, GAR row cap, etc.).
static const uint64_t kMaxNumWeft = 1 << 16;

// SubIso validation/backtracking tuning knobs.
static const uint64_t kSubIsoMaxBacktrackNodes = 1000000;
static const uint64_t kSubIsoMaxMsPerWeft = 10;
static const uint64_t kSubIsoMaxValidateWefts = 165536;
static const uint64_t kSubIsoValidateMatchingTimeoutSec = 10;
static const uint64_t kSubIsoProgressPrintInterval = 10;
static const uint64_t kSubIsoLocalMatchesSizeBuffer = 10;

// WOJ Filter+Join only: max rows in each `WOJMatches` table (`Init(..., y)`).
static const uint64_t kMaxMatchTableRows = 1 << 28;

static const uint32_t kSharedMemorySize = 1024;
static const uint32_t kGridDim = 512;
static const uint32_t kBlockDim = 1024;
// static const uint32_t kGridDim = 64;
// static const uint32_t kBlockDim = 64;
static const uint32_t kWarpSize = 32;
static const uint32_t kLogWarpSize = 5;

// Specialized block dimensions for shared-memory-heavy aggregation kernels.
// kBlockDim (1024) is the generic default; these are per-kernel tunings.
static const uint32_t kAllFeaturesBlockDim = 512;    // fused all-primitives kernels
static const uint32_t kFilterAggBlockDim = 256;      // conditional filter+aggregate
static const uint32_t kStreamingAggBlockDim = 256;   // streaming reduction primitives
static const uint32_t kNumUniqueHashBlockDim = 256;  // hash-based unique count

// Memory-bounded chunking for GraphFilterAggregate::Compute. A batch of
// (pivot x feature) requests is split into chunks dispatched round-robin across
// the CUDA streams, so peak device memory (per-request buffers + flattened
// conditions + NumUnique hash scratch) stays bounded by n_streams * per-chunk
// budget regardless of how many total requests the batch has. A chunk is closed
// when either cap is reached; a single request that alone exceeds the hash-slot
// budget still forms its own chunk.
static const uint32_t kFilterMaxChunkRequests = 1u << 17;      // 131072 requests/chunk
static const size_t kFilterMaxChunkHashSlots = 1ull << 25;     // 33.5M NumUnique slots/chunk

static const uint32_t kDefaultHeapCapacity = 7;

}  // namespace common
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_COMMON_TYPES_H_
