#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_MATCHES_CUH_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_MATCHES_CUH_

#include <algorithm>

#include "core/common/consts.h"
#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/host_buffer.cuh"
#include "core/data_structures/unified_buffer.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

using sics::matrixgraph::core::common::kMaxNumCandidates;
using sics::matrixgraph::core::common::kMaxNumCandidatesPerThread;
using sics::matrixgraph::core::common::kMaxNumWeft;

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
  using UnifiedOwnedBufferVertexIDPtr =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexID *>;
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

    v_candidate_offset_for_each_weft_.Init(sizeof(VertexID) * (n_vertices + 1) *
                                           max_n_weft * 10);

    weft_offset_.Init(sizeof(EdgeIndex) * max_n_weft);
    weft_size_.Init(sizeof(VertexID) * max_n_weft);
    weft_count_.Init(sizeof(VertexID));

    matches_data_.Init(sizeof(VertexID) * 2 * n_vertices *
                       kMaxNumCandidatesPerThread * max_n_weft);
  }

  void Print(VertexID n_matches = 3) const {
    VertexID min_n_matches = std::min(*weft_count_.GetPtr(), n_matches);

    std::cout << "[Matches] Print n_matches:" << *weft_count_.GetPtr()
              << std::endl;
    for (VertexID weft_id = 0; weft_id < min_n_matches; weft_id++) {
      if (weft_id > max_n_weft_)
        break;
      std::cout << "Weft " << weft_id << std::endl;
      VertexID weft_offset =
          weft_id * 2 * n_vertices_ * kMaxNumCandidatesPerThread;

      for (auto i = 0; i < n_vertices_; i++) {
        VertexID v_candidate_offset =
            v_candidate_offset_for_each_weft_
                .GetPtr()[weft_id * (n_vertices_ + 1) + i];
        VertexID v_candidate_size =
            v_candidate_offset_for_each_weft_
                .GetPtr()[weft_id * (n_vertices_ + 1) + i + 1] -
            v_candidate_offset_for_each_weft_
                .GetPtr()[weft_id * (n_vertices_ + 1) + i];
        std::cout << "\t u" << i << " offset:" << v_candidate_offset
                  << " size: " << v_candidate_size << ": ";
        for (VertexID candidate_id = 0; candidate_id < v_candidate_size;
             candidate_id++) {
          std::cout << *(matches_data_.GetPtr() + weft_offset +
                         v_candidate_offset * 2 + 2 * candidate_id)
                    << "->"
                    << *(matches_data_.GetPtr() + weft_offset +
                         v_candidate_offset * 2 + 2 * candidate_id + 1)
                    << ",";
        }
        std::cout << std::endl;
      }
    }
  }

  UnifiedOwnedBufferVertexID weft_count_;
  UnifiedOwnedBufferVertexID v_candidate_offset_for_each_weft_;
  UnifiedOwnedBufferVertexID weft_size_;
  UnifiedOwnedBufferVertexIDPtr matches_weft_ptr_;

  UnifiedOwnedBufferEdgeIndex weft_offset_;

  UnifiedOwnedBufferVertexID matches_data_;

  VertexID n_vertices_ = 0;
  VertexID max_n_weft_ = 0;
};

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_DATA_STRUCTURES_MATHES_CUH_