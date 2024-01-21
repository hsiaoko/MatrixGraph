#ifndef SICS_MATRIXGRAPH_CORE_DATA_STRUCTURES_EDGELIST_H_
#define SICS_MATRIXGRAPH_CORE_DATA_STRUCTURES_EDGELIST_H_

#include <cstring>

#ifdef TBB_FOUND
#include <execution>
#endif

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

public:
  VertexID num_vertices;
  EdgeIndex num_edges;
  VertexID max_vid;
};

struct Edge {
private:
  using VertexID = sics::matrixgraph::core::common::VertexID;

public:
  Edge(VertexID src, VertexID dst) : src(src), dst(dst) {}
  Edge() = default;

  bool operator<(const Edge &b) const { return src < b.src; }

public:
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

  Edges(const EdgelistMetadata &edgelist_metadata, Edge *edges_ptr)
      : edgelist_metadata_(edgelist_metadata), edges_ptr_(edges_ptr) {}

  Edges(const EdgelistMetadata &edgelist_metadata)
      : edgelist_metadata_(edgelist_metadata) {
    edges_ptr_ = new Edge[edgelist_metadata.num_edges]();
  }

  Edges(const Edges &edges) {
    edgelist_metadata_ = edges.get_metadata();
    edges_ptr_ = new Edge[edgelist_metadata_.num_edges]();
    memcpy(edges_ptr_, edges.get_base_ptr(),
           sizeof(Edge) * edgelist_metadata_.num_edges);
  }

  ~Edges() { delete[] edges_ptr_; }

  Iterator begin() { return Iterator(&edges_ptr_[0]); }
  Iterator end() {
    return Iterator(&edges_ptr_[edgelist_metadata_.num_edges - 1]);
  }

  // Read From CSv
  void ReadFromCSV(const std::string &filename, const std::string &sep,
                   bool compressed = false) {
    std::ifstream in_file(filename);

    in_file.seekg(0, std::ios::end);
    size_t length = in_file.tellg();
    in_file.seekg(0, std::ios::beg);

    char *buff = new char[length]();
    in_file.read(buff, length);
    std::string content(buff, length);

    EdgeIndex n_edges = count(content.begin(), content.end(), '\n');
    auto buffer_edges = new VertexID[n_edges * 2]();
    std::stringstream ss(content);
    delete[] buff;

    EdgeIndex index = 0;
    VertexID max_vid = 0, compressed_vid = 0;
    std::string line, vid_str;

    while (getline(ss, line, '\n')) {
      if (*line.c_str() == '\0')
        break;
      std::stringstream ss_line(line);
      while (getline(ss_line, vid_str, *sep.c_str())) {
        VertexID vid = stoll(vid_str);
        sics::matrixgraph::core::util::atomic::WriteMax(&max_vid, vid);
        buffer_edges[index++] = vid;
        std::cout << "vid: " << vid << std::endl;
      }
    }
    content.clear();
    in_file.close();

    auto aligned_max_vid = (((max_vid + 1) >> 6) << 6) + 64;
    edges_ptr_ = new Edge[n_edges]();
    Bitmap bitmap(aligned_max_vid);

    auto vid_map = new VertexID[aligned_max_vid]();
    auto compressed_buffer_edges = new VertexID[n_edges * 2]();

    // Compute the mapping between origin vid to compressed vid.
    for (EdgeIndex index = 0; index < n_edges * 2; index++) {
      if (!bitmap.GetBit(buffer_edges[index])) {
        bitmap.SetBit(buffer_edges[index]);
        vid_map[buffer_edges[index]] = compressed_vid++;
      }
    }

    if (compressed) {
      // Compress vid and buffer graph.
      for (EdgeIndex i = 0; i < n_edges * 2; i++) {
        compressed_buffer_edges[i] = vid_map[buffer_edges[i]];
      }
      for (EdgeIndex i = 0; i < n_edges; i++) {
        edges_ptr_[i].src = compressed_buffer_edges[2 * i];
        edges_ptr_[i].dst = compressed_buffer_edges[2 * i + 1];
      }
    } else {
      for (EdgeIndex i = 0; i < n_edges; i++) {
        edges_ptr_[i].src = buffer_edges[2 * i];
        edges_ptr_[i].dst = buffer_edges[2 * i + 1];
      }
    }

    delete[] buffer_edges;
    delete[] vid_map;
    delete[] compressed_buffer_edges;

    std::cout << "Loaded graph with " << n_edges << std::endl;
    // Compute metadata.
    edgelist_metadata_.num_edges = n_edges;
    edgelist_metadata_.num_vertices = bitmap.Count();
    edgelist_metadata_.max_vid = max_vid;
  }

  // Sort edges by source.
  void SortBySrc() {
#ifdef TBB_FOUND
    std::sort(std::execution::par, edges_ptr_,
              edges_ptr_ + edgelist_metadata_.num_edges);
#else
    std::sort(edges_ptr_, edges_ptr_ + edgelist_metadata_.num_edges);
#endif
  }

  void ShowGraph() {
    for (EdgeIndex i = 0; i < edgelist_metadata_.num_edges; i++) {
      std::cout << edges_ptr_[i].src << " " << edges_ptr_[i].dst << std::endl;
    }
  }

  Edge *get_base_ptr() const { return edges_ptr_; }
  EdgelistMetadata get_metadata() const { return edgelist_metadata_; }

  VertexID get_src_by_index(size_t i) const { return edges_ptr_[i].src; }
  VertexID get_dst_by_index(size_t i) const { return edges_ptr_[i].dst; }
  Edge get_edge_by_index(size_t i) const { return edges_ptr_[i]; }
  size_t get_index_by_iter(const Iterator &iter) {
    return (iter.get_base_ptr() - begin().get_base_ptr());
  };

private:
  Edge *edges_ptr_;
  EdgelistMetadata edgelist_metadata_;
};

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // SICS_MATRIXGRAPH_CORE_DATA_STRUCTURES_EDGELIST_H_
