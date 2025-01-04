#ifndef SICS_MATRIXGRAPH_CORE_DATA_STRUCTURES_EDGELIST_H_
#define SICS_MATRIXGRAPH_CORE_DATA_STRUCTURES_EDGELIST_H_

#include <cstring>
#include <execution>
#include <iostream>

#include <yaml-cpp/yaml.h>

#include "core/common/consts.h"
#include "core/common/types.h"
#include "core/util/atomic.h"
#include "core/util/bitmap.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

struct EdgelistMetadata {
private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;

public:
  VertexID num_vertices = 0;
  EdgeIndex num_edges = 0;
  VertexID max_vid = 0;
  VertexID min_vid = sics::matrixgraph::core::common::kMaxVertexID;
};

struct Edge {
private:
  using VertexID = sics::matrixgraph::core::common::VertexID;

public:
  Edge(VertexID src, VertexID dst) : src(src), dst(dst) {}
  Edge() = default;

  bool operator<(const Edge &b) const { return src < b.src; }

  VertexID src;
  VertexID dst;
};

class Edges {
private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
  using EdgelistMetadata =
      sics::matrixgraph::core::data_structures::EdgelistMetadata;
  using Bitmap = sics::matrixgraph::core::util::Bitmap;
  using VertexLabel = sics::matrixgraph::core::common::VertexLabel;

public:
  struct Iterator {
    // Iterator tags
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = Edge;
    using pointer = Edge *;
    using reference = Edge &;

    // Iterator constructors
    Iterator(pointer ptr) : base_ptr_(ptr) {}

    // Iterator operators
    reference operator*() const { return *base_ptr_; }
    pointer operator->() { return base_ptr_; }
    Iterator &operator++() {
      base_ptr_++;
      return *this;
    }
    // used for clang++ compiling
    Iterator &operator+=(int n) {
      base_ptr_ += n;
      return *this;
    }
    Iterator &operator--() {
      base_ptr_--;
      return *this;
    }
    Iterator &operator+(int n) {
      base_ptr_ += n;
      return *this;
    }
    Iterator &operator-(int n) {
      base_ptr_ -= n;
      return *this;
    }
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }
    bool operator-(const Iterator &b) const {
      return (*base_ptr_).src - (*b).src;
    }
    bool operator<(const Iterator &b) const {
      return (*base_ptr_).src < (*b).src;
    }
    bool operator>(const Iterator &b) const {
      return (*base_ptr_).src > (*b).src;
    }
    // used for clang++ compiling
    bool operator<=(const Iterator &b) const {
      return (*base_ptr_).src <= (*b).src;
    }
    // used for clang++ compiling
    bool operator>=(const Iterator &b) const {
      return (*base_ptr_).src >= (*b).src;
    }

    friend bool operator==(const Iterator &a, const Iterator &b) {
      return a.base_ptr_ == b.base_ptr_;
    };
    friend bool operator!=(const Iterator &a, const Iterator &b) {
      return a.base_ptr_ != b.base_ptr_;
    };

    pointer get_base_ptr() const { return base_ptr_; };

  private:
    pointer base_ptr_;
  };

  Edges() = default;

  Edges(const EdgelistMetadata &edgelist_metadata)
      : edgelist_metadata_(edgelist_metadata) {
    edges_ptr_ = new Edge[edgelist_metadata.num_edges]();
    vertex_label_base_pointer_ =
        new VertexLabel[edgelist_metadata.num_vertices]();
  }

  Edges(const EdgelistMetadata &edgelist_metadata, Edge *edges_ptr)
      : edgelist_metadata_(edgelist_metadata), edges_ptr_(edges_ptr) {}

  Edges(const EdgelistMetadata &edgelist_metadata, Edge *edges_ptr,
        VertexID *localid_to_globalid)
      : edgelist_metadata_(edgelist_metadata), edges_ptr_(edges_ptr),
        localid_to_globalid_(localid_to_globalid) {}

  Edges(const EdgelistMetadata &edgelist_metadata, Edge *edges_ptr,
        VertexID *localid_to_globalid, VertexLabel *vertex_label_base_pointer)
      : edgelist_metadata_(edgelist_metadata), edges_ptr_(edges_ptr),
        localid_to_globalid_(localid_to_globalid),
        vertex_label_base_pointer_(vertex_label_base_pointer) {}

  Edges(EdgeIndex n_edges, VertexID *edges_buf,
        VertexID *localid2globalid = nullptr);

  void Init(EdgeIndex n_edges, VertexID *edges_buf,
            VertexID *localid2globalid = nullptr);

  Edges(const Edges &edges);

  ~Edges() { delete[] edges_ptr_; }

  Iterator begin() { return Iterator(&edges_ptr_[0]); }

  Iterator end() {
    return Iterator(&edges_ptr_[edgelist_metadata_.num_edges - 1]);
  }

  void WriteToBinary(const std::string &filename);

  // Read From CSv
  void ReadFromCSV(const std::string &filename, const std::string &sep,
                   bool compressed = true);

  // Read From Bin
  void ReadFromBin(const std::string &filename);

  void GenerateLocalID2GlobalID();

  void Compacted();

  void Transpose();

  // Sort edges by source.
  void SortBySrc();

  void ShowGraph(EdgeIndex n_edges = 3) const;

  void GenerateVLabel(VertexID range = 5);

  Edge *get_base_ptr() const { return edges_ptr_; }

  VertexID *get_localid_to_globalid_ptr() const { return localid_to_globalid_; }

  VertexLabel *get_vertex_label_ptr() const {
    return vertex_label_base_pointer_;
  }

  EdgelistMetadata get_metadata() const { return edgelist_metadata_; }

  VertexID get_src_by_index(size_t i) const { return edges_ptr_[i].src; }

  VertexID get_dst_by_index(size_t i) const { return edges_ptr_[i].dst; }

  Edge get_edge_by_index(size_t i) const { return edges_ptr_[i]; }

  size_t get_index_by_iter(const Iterator &iter) {
    return (iter.get_base_ptr() - begin().get_base_ptr());
  };

  VertexID get_globalid_by_localid(VertexID localid) const;

  void SetLocalIDToGlobalID(VertexID *localid_to_globalid);

private:
  VertexID *localid_to_globalid_ = nullptr;

  VertexLabel *vertex_label_base_pointer_ = nullptr;

  Edge *edges_ptr_;
  EdgelistMetadata edgelist_metadata_;
};

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // SICS_MATRIXGRAPH_CORE_DATA_STRUCTURES_EDGELIST_H_