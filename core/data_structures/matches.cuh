#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_MATCHES_CUH_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_MATCHES_CUH_

#include "core/common/consts.h"
#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/host_buffer.cuh"
#include "core/data_structures/unified_buffer.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

using sics::matrixgraph::core::common::kMaxNumCandidates;

class Matches {
private:
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;
  using DeviceOwnedBufferUint64 =
      sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<uint64_t>;
  using DeviceOwnedBufferUint32 =
      sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<uint32_t>;
  using DeviceOwnedBufferUint8 =
      sics::matrixgraph::core::data_structures::DeviceOwnedBuffer<uint8_t>;
  using UnifiedOwnedBufferUint32 =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint32_t>;
  using UnifiedOwnedBufferUint64 =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint64_t>;
  using UnifiedOwnedBufferEdgeIndex =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<EdgeIndex>;
  using UnifiedOwnedBufferVertexID =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexID>;
  using UnifiedOwnedBufferVertexLabel =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexLabel>;
  using UnifiedOwnedBufferUint8 =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<uint8_t>;
  using BufferUint64 =
      sics::matrixgraph::core::data_structures::Buffer<uint64_t>;
  using BufferUint8 = sics::matrixgraph::core::data_structures::Buffer<uint8_t>;
  using BufferUint32 =
      sics::matrixgraph::core::data_structures::Buffer<uint32_t>;
  using BufferEdgeIndex =
      sics::matrixgraph::core::data_structures::Buffer<EdgeIndex>;
  using BufferVertexID =
      sics::matrixgraph::core::data_structures::Buffer<VertexID>;
  using BufferVertexLabel =
      sics::matrixgraph::core::data_structures::Buffer<VertexLabel>;

public:
  Matches(VertexID n_vertices, VertexID max_n_weft)
      : n_vertices_(n_vertices), max_n_weft_(max_n_weft) {

    v_candidate_offset_for_each_weft_.Init(sizeof(VertexID) * n_vertices);
    matches_data_.Init(sizeof(VertexID) * 2 * n_vertices * kMaxNumCandidates);
    weft_offset_.Init(sizeof(EdgeIndex) * max_n_weft);
    weft_size_.Init(sizeof(VertexID) * max_n_weft);
    matches_count_.Init(sizeof(VertexID));
  }

  UnifiedOwnedBufferVertexID matches_count_;
  UnifiedOwnedBufferEdgeIndex weft_offset_;
  UnifiedOwnedBufferVertexID weft_size_;
  UnifiedOwnedBufferVertexID v_candidate_offset_for_each_weft_;
  UnifiedOwnedBufferVertexID matches_data_;

  VertexID n_vertices_ = 0;
  VertexID max_n_weft_ = 0;
};

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // INC_51_11_GRAPH_COMPUTING_MATRIXGRAPH_CORE_DATA_STRUCTURES_MATHES_CUH_