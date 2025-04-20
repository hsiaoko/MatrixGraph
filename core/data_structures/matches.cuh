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
namespace data_structures {

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
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexID*>;
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
  Matches(VertexID n_vertices,
          VertexID max_n_weft = sics::matrixgraph::core::common::kMaxNumWeft,
          VertexID max_n_local_weft =
              sics::matrixgraph::core::common::kMaxNumLocalWeft)
      : n_vertices_(n_vertices),
        max_n_weft_(max_n_weft),
        max_n_local_weft_(max_n_local_weft) {
    v_candidate_offset_for_each_weft_.Init(sizeof(VertexID) * (n_vertices + 1) *
                                           max_n_weft);

    weft_offset_.Init(sizeof(EdgeIndex) * max_n_weft_);
    weft_size_.Init(sizeof(VertexID) * max_n_weft_);
    weft_count_.Init(sizeof(VertexID));

    matches_data_.Init(sizeof(VertexID) * 2 * n_vertices * max_n_weft_ *
                       max_n_weft);
  }

  void Print(VertexID n_matches = 3) const {
    VertexID min_n_matches = std::min(*weft_count_.GetPtr(), n_matches);

    std::cout << "[Matches] Print n_matches:" << *weft_count_.GetPtr()
              << " cols: " << n_vertices_ << std::endl;
    for (VertexID weft_id = 0; weft_id < min_n_matches; weft_id++) {
      if (weft_id > max_n_weft_) break;
      std::cout << "Weft " << weft_id << std::endl;

      for (auto i = 0; i < n_vertices_; i++) {
        auto v_candidate_offset =
            GetVCandidateOffsetPtr()[weft_id * (n_vertices_ + 1) + i];
        auto v_candidate_size =
            GetVCandidateOffsetPtr()[weft_id * (n_vertices_ + 1) + i + 1] -
            GetVCandidateOffsetPtr()[weft_id * (n_vertices_ + 1) + i];
        std::cout << "\t u" << i << " offset:" << v_candidate_offset
                  << " size: " << v_candidate_size << ": ";
        for (VertexID candidate_id = 0; candidate_id < v_candidate_size;
             candidate_id++) {
          std::cout << *(matches_data_.GetPtr() +
                         weft_id * n_vertices_ * 2 * max_n_local_weft_ +
                         i * 2 * max_n_weft_ + 2 * candidate_id)
                    << "->"
                    << *(matches_data_.GetPtr() +
                         weft_id * n_vertices_ * 2 * max_n_local_weft_ +
                         i * 2 * max_n_weft_ + 2 * candidate_id + 1)
                    << ",";
        }
        std::cout << std::endl;
      }
    }
  }

  VertexID* GetWeftCountPtr() const { return weft_count_.GetPtr(); }

  VertexID* GetWeftSizePtr() const { return weft_size_.GetPtr(); }

  VertexID* GetDataPtr() const { return matches_data_.GetPtr(); }

  VertexID* GetVCandidateOffsetPtr() const {
    return v_candidate_offset_for_each_weft_.GetPtr();
  }

  VertexID get_weft_count() const { return *weft_count_.GetPtr(); }

  VertexID get_n_vertices() const { return n_vertices_; }

  VertexID get_max_n_weft() const { return max_n_weft_; }

  VertexID get_max_n_local_weft() const { return max_n_local_weft_; }

  UnifiedOwnedBufferVertexID weft_count_;
  UnifiedOwnedBufferVertexID v_candidate_offset_for_each_weft_;
  UnifiedOwnedBufferVertexID weft_size_;
  UnifiedOwnedBufferVertexIDPtr matches_weft_ptr_;

  UnifiedOwnedBufferEdgeIndex weft_offset_;

  UnifiedOwnedBufferVertexID matches_data_;

  VertexID n_vertices_ = 0;
  VertexID max_n_weft_ = sics::matrixgraph::core::common::kMaxNumWeft;
  VertexID max_n_local_weft_ =
      sics::matrixgraph::core::common::kMaxNumLocalWeft;
  ;
};

}  // namespace data_structures
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_DATA_STRUCTURES_MATHES_CUH_