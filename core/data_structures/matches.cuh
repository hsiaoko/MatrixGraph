#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_MATCHES_CUH_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_MATCHES_CUH_

#include <algorithm>

#include "core/common/consts.h"
#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/host_buffer.cuh"
#include "core/data_structures/unified_buffer.cuh"
#include "core/util/bitmap_ownership.h"

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
  using BitmapOwnership = sics::matrixgraph::core::util::BitmapOwnership;

 public:
  Matches(VertexID n_vertices,
          VertexID max_n_weft = sics::matrixgraph::core::common::kMaxNumWeft,
          VertexID max_n_local_weft =
              sics::matrixgraph::core::common::kMaxNumLocalWeft,
          VertexID g_n_vertices = 65536)
      : n_vertices_(n_vertices),
        max_n_weft_(max_n_weft),
        max_n_local_weft_(max_n_local_weft) {
    v_candidate_offset_for_each_weft_.Init(sizeof(VertexID) * (n_vertices + 1) *
                                           max_n_weft);

    weft_offset_.Init(sizeof(EdgeIndex) * max_n_weft_);
    weft_size_.Init(sizeof(VertexID) * max_n_weft_);
    weft_count_.Init(sizeof(VertexID));

    matches_data_.Init(sizeof(VertexID) * 2 * n_vertices * max_n_local_weft *
                       max_n_weft);
    invalid_match_ = new BitmapOwnership(max_n_weft);
    header_.resize(n_vertices_);

    // src_visited_vec_.resize(n_vertices, g_n_vertices);
    // dst_visited_vec_.resize(n_vertices, g_n_vertices);
  }

  void Print(VertexID n_matches = 3) const {
    std::cout << "[Matches] Print n_matches:" << *weft_count_.GetPtr()
              << " cols: " << n_vertices_
              << " invalid: " << GetInvalidMatchesCount() << std::endl;
    VertexID count = 0;
    for (VertexID weft_id = 0; weft_id < *weft_count_.GetPtr(); weft_id++) {
      if (invalid_match_->GetBit(weft_id)) continue;
      if (count++ > n_matches) break;
      std::cout << "Weft " << weft_id << std::endl;

      for (auto i = 0; i < n_vertices_; i++) {
        auto v_candidate_offset =
            GetVCandidateOffsetPtr()[weft_id * (n_vertices_ + 1) + i];
        auto v_candidate_size =
            GetVCandidateOffsetPtr()[weft_id * (n_vertices_ + 1) + i + 1] -
            GetVCandidateOffsetPtr()[weft_id * (n_vertices_ + 1) + i];
        std::cout << "\t " << header_[i].first << "->" << header_[i].second
                  << " offset:" << v_candidate_offset
                  << " size: " << v_candidate_size << ": ";
        for (VertexID candidate_id = 0; candidate_id < v_candidate_size;
             candidate_id++) {
          if (*(matches_data_.GetPtr() +
                weft_id * n_vertices_ * 2 * max_n_local_weft_ +
                i * 2 * max_n_local_weft_ + 2 * candidate_id) != kMaxVertexID &&
              *(matches_data_.GetPtr() +
                weft_id * n_vertices_ * 2 * max_n_local_weft_ +
                i * 2 * max_n_local_weft_ + 2 * candidate_id + 1) !=
                  kMaxVertexID) {
            std::cout << *(matches_data_.GetPtr() +
                           weft_id * n_vertices_ * 2 * max_n_local_weft_ +
                           i * 2 * max_n_local_weft_ + 2 * candidate_id)
                      << "->"
                      << *(matches_data_.GetPtr() +
                           weft_id * n_vertices_ * 2 * max_n_local_weft_ +
                           i * 2 * max_n_local_weft_ + 2 * candidate_id + 1)
                      << ",";
          }
        }
        std::cout << std::endl;
      }
    }
  }

  size_t ComputeNMatches() const {
    size_t count = 0;
    for (VertexID weft_id = 0; weft_id < *weft_count_.GetPtr(); weft_id++) {
      if (invalid_match_->GetBit(weft_id)) continue;

      auto tag = false;
      size_t weft_matches_count = 1;
      for (auto i = 0; i < n_vertices_; i++) {
        auto v_candidate_offset =
            GetVCandidateOffsetPtr()[weft_id * (n_vertices_ + 1) + i];
        auto v_candidate_size =
            GetVCandidateOffsetPtr()[weft_id * (n_vertices_ + 1) + i + 1] -
            GetVCandidateOffsetPtr()[weft_id * (n_vertices_ + 1) + i];
        size_t tmp_count = 0;
        for (VertexID candidate_id = 0; candidate_id < v_candidate_size;
             candidate_id++) {
          if (*(matches_data_.GetPtr() +
                weft_id * n_vertices_ * 2 * max_n_local_weft_ +
                i * 2 * max_n_local_weft_ + 2 * candidate_id) != kMaxVertexID &&
              *(matches_data_.GetPtr() +
                weft_id * n_vertices_ * 2 * max_n_local_weft_ +
                i * 2 * max_n_local_weft_ + 2 * candidate_id + 1) !=
                  kMaxVertexID) {
            tmp_count++;
          }
        }
        if (tmp_count != 0) {
          tag = true;
          weft_matches_count *= tmp_count;
        }
      }
      if (tag) count += weft_matches_count;
    }
    return count;
  }

  void UpdateInvalidMatches() {
    for (VertexID weft_id = 0; weft_id < *weft_count_.GetPtr(); weft_id++) {
      for (auto i = 0; i < n_vertices_; i++) {
        auto v_candidate_offset =
            GetVCandidateOffsetPtr()[weft_id * (n_vertices_ + 1) + i];
        auto v_candidate_size =
            GetVCandidateOffsetPtr()[weft_id * (n_vertices_ + 1) + i + 1] -
            GetVCandidateOffsetPtr()[weft_id * (n_vertices_ + 1) + i];
        auto invalid_count = 0;
        for (VertexID candidate_id = 0; candidate_id < v_candidate_size;
             candidate_id++) {
          if (*(matches_data_.GetPtr() +
                weft_id * n_vertices_ * 2 * max_n_local_weft_ +
                i * 2 * max_n_local_weft_ + 2 * candidate_id) == kMaxVertexID ||
              *(matches_data_.GetPtr() +
                weft_id * n_vertices_ * 2 * max_n_local_weft_ +
                i * 2 * max_n_local_weft_ + 2 * candidate_id + 1) ==
                  kMaxVertexID) {
            invalid_count++;
          }
        }

        if (invalid_count == v_candidate_size) {
          invalid_match_->SetBit(weft_id);
          break;
        }
      }
    }
  }

  void Write(const std::string& path) const {
    std::cout << "[Matches::Write]" << get_weft_count() << " save at" << path
              << std::endl;

    uint64_t* weft_root_buf =
        new uint64_t[get_weft_count() - GetInvalidMatchesCount()]();
    auto count = 0;
    for (auto _ = 0; _ < get_weft_count(); _++) {
      if (invalid_match_->GetBit(_)) continue;
      weft_root_buf[count++] = *(matches_data_.GetPtr() +
                                 _ * n_vertices_ * 2 * max_n_local_weft_ + 1);
    }
    std::ofstream out_file(path);
    out_file.write(
        reinterpret_cast<char*>(weft_root_buf),
        sizeof(uint64_t) * (get_weft_count() - GetInvalidMatchesCount()));
    out_file.close();

    delete[] weft_root_buf;
  }

  void Read(const std::string& path) const {
    std::ifstream data_file(path);
    if (!data_file) throw std::runtime_error("Read matches error.");

    data_file.seekg(0, std::ios::end);
    size_t file_size = data_file.tellg();
    data_file.seekg(0, std::ios::beg);

    uint64_t* weft_root_buf = new uint64_t[file_size / sizeof(uint64_t)]();
    data_file.read(reinterpret_cast<char*>(weft_root_buf), file_size);
    data_file.close();
    for (auto _ = 0; _ < get_weft_count(); _++) {
      std::cout << "weft_root_buf: " << weft_root_buf[_] << std::endl;
    }

    delete[] weft_root_buf;
  }

  size_t GetInvalidMatchesCount() const { return invalid_match_->Count(); }

  VertexID* GetWeftCountPtr() const { return weft_count_.GetPtr(); }

  VertexID* GetWeftSizePtr() const { return weft_size_.GetPtr(); }

  VertexID* GetDataPtr() const { return matches_data_.GetPtr(); }

  VertexID* GetVCandidateOffsetPtr() const {
    return v_candidate_offset_for_each_weft_.GetPtr();
  }

  void SetHeader(VertexID pos, std::pair<VertexID, VertexID> e) {
    assert(pos < n_vertices_);
    header_[pos].first = e.first;
    header_[pos].second = e.second;
  }

  std::vector<std::pair<VertexID, VertexID>> GetHeader() const {
    return header_;
  }

  VertexID get_weft_count() const { return *weft_count_.GetPtr(); }

  VertexID get_n_vertices() const { return n_vertices_; }

  VertexID get_max_n_weft() const { return max_n_weft_; }

  VertexID get_max_n_local_weft() const { return max_n_local_weft_; }

  std::vector<std::pair<VertexID, VertexID>> header_;

  UnifiedOwnedBufferVertexID weft_count_;
  UnifiedOwnedBufferVertexID v_candidate_offset_for_each_weft_;
  UnifiedOwnedBufferVertexID weft_size_;
  UnifiedOwnedBufferVertexIDPtr matches_weft_ptr_;

  UnifiedOwnedBufferEdgeIndex weft_offset_;

  UnifiedOwnedBufferVertexID matches_data_;

  std::vector<BitmapOwnership> src_visited_vec_;
  std::vector<BitmapOwnership> dst_visited_vec_;

  VertexID n_vertices_ = 0;
  VertexID max_n_weft_ = sics::matrixgraph::core::common::kMaxNumWeft;
  VertexID max_n_local_weft_ =
      sics::matrixgraph::core::common::kMaxNumLocalWeft;

  BitmapOwnership* invalid_match_ = nullptr;
};

}  // namespace data_structures
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_DATA_STRUCTURES_MATHES_CUH_