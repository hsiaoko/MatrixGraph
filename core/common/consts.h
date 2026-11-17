
#ifndef MATRIXGRAPH_CORE_COMMON_CONSTS_H_
#define MATRIXGRAPH_CORE_COMMON_CONSTS_H_

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
static const uint32_t kStreamingAggBlockDim = 256;   // streaming reduction primitives
static const uint32_t kNumUniqueHashBlockDim = 256;  // hash-based unique count
// Per-request feature kernel: 1024 threads would cap registers at 64/thread,
// which even the release build cannot meet (~130 needed) and debug (-G)
// provably cannot (its CollectNeighborValuesBlock callee alone needs 86).
// 512 threads give a 128-register budget; the block reductions are
// blockDim-generic and the workspace is sized for kBlockDim, so the smaller
// launch is safe. Retune with measurements, not guesses.
static const uint32_t kFeatureRequestBlockDim = 512;

static const uint32_t kDefaultHeapCapacity = 32;

}  // namespace common
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_COMMON_TYPES_H_
