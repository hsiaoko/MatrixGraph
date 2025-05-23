#ifndef HYPERBLOCKER_CORE_COMMON_TYPES_H_
#define HYPERBLOCKER_CORE_COMMON_TYPES_H_

#include <climits>
#include <stdint.h>

#define EQUALITIES 'e'
#define SIM 's'

#define MAX_CANDIDATE_COUNT 8192
#define MAX_EID_COL_SIZE 64

#define CHECK_POINT INT_MAX
#define CHECK_POINT_CHAR ' '

#define MAX_HASH_TIMES 1

namespace sics {
namespace matrixgraph {
namespace core {
namespace common {

typedef uint32_t GraphID;  // uint32_t: 0 ~ 4,294,967,295
typedef uint32_t VertexID;
typedef uint32_t VertexIndex;
typedef uint16_t TileIndex;
typedef uint32_t VertexLabel;
typedef uint32_t VertexDegree;
typedef uint32_t VertexCount;
typedef uint32_t EdgeIndex;

#define MAX_VERTEX_ID std::numeric_limits<VertexID>::max()
#define DEFAULT_TILE_NUM 1024

#define CHECK MAX_VERTEX_ID

enum StoreStrategy {
  kUnconstrained,  // default
  kIncomingOnly,
  kOutgoingOnly,
  kUndefinedStrategy
};

// Enum defining different GPU task types
enum TaskType {
  kGEMM,  // Default task type
  kMatrixAnalysis,
  kPPRQuery,
  kSubIso,
  kCPUSubIso,
  kWCC,
  kSSSP,
  kBFS,
  kPageRank,
  kGEMV,
  // Other types.
};

}  // namespace common
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_COMMON_TYPES_H_