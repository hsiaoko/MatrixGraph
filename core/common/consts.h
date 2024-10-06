
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
static const uint64_t KDefalutNumEdgesPerTile = 65536;
static const uint32_t KDefalutNumVerticesPerTile = 128;

} // namespace common
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_COMMON_TYPES_H_