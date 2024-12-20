
#ifndef MATRIX_CORE_COMMON_CONSTS_H_
#define MATRIX_CORE_COMMON_CONSTS_H_

namespace sics {
namespace matrixgraph {
namespace core {
namespace common {

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

static const uint64_t kMaxNumCandidatesPerThread = 32;
static const uint64_t kMaxNumWeft = 655360;

} // namespace common
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_COMMON_TYPES_H_