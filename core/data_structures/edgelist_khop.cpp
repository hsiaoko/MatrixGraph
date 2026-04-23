// k-hop subgraph build: host-only; compiled as CXX so <execution> parallel STL works.
// nvcc cannot reliably compile <execution> / some parallel runtimes (intrinsic conflicts).

#include "core/data_structures/edgelist.h"
#include "core/util/atomic.h"
#include "core/util/bitmap_ownership.h"

#include <algorithm>
#include <execution>
#include <memory>
#include <numeric>
#include <queue>
#include <vector>

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

using VertexID = sics::matrixgraph::core::common::VertexID;
using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using sics::matrixgraph::core::util::atomic::WriteMax;

std::vector<Edges> Edges::BuildKHopOutSubgraphs(VertexID k) const {
  const EdgeIndex num_edges = edgelist_metadata_.num_edges;
  if (num_edges == 0) return {};

  std::vector<Edge> sorted_e(edges_ptr_, edges_ptr_ + num_edges);
  std::sort(std::execution::par_unseq, sorted_e.begin(), sorted_e.end(),
            [](const Edge& a, const Edge& b) {
              if (a.src != b.src) return a.src < b.src;
              return a.dst < b.dst;
            });

  VertexID max_v = 0;
  std::for_each(std::execution::par, sorted_e.begin(), sorted_e.end(),
                [&max_v](const Edge& e) {
                  WriteMax(&max_v, e.src);
                  WriteMax(&max_v, e.dst);
                });
  const size_t bm_sz = static_cast<size_t>(max_v) + 1;

  std::vector<VertexID> centers(static_cast<size_t>(num_edges) * 2);
  {
    std::vector<EdgeIndex> er(static_cast<size_t>(num_edges));
    std::iota(er.begin(), er.end(), static_cast<EdgeIndex>(0));
    std::for_each(std::execution::par, er.begin(), er.end(),
                  [&](EdgeIndex i) {
                    const size_t si = static_cast<size_t>(i);
                    const Edge& e = sorted_e[si];
                    centers[si * 2] = e.src;
                    centers[si * 2 + 1] = e.dst;
                  });
  }
  std::sort(std::execution::par, centers.begin(), centers.end());
  centers.erase(std::unique(centers.begin(), centers.end()), centers.end());

  VertexID stub_pair[2] = {0, 0};
  const size_t ncenters = centers.size();
  std::vector<std::unique_ptr<Edges>> subgraph_slots(ncenters);

  std::vector<size_t> cidx(ncenters);
  std::iota(cidx.begin(), cidx.end(), size_t{0});
  std::for_each(std::execution::par, cidx.begin(), cidx.end(),
                [&](size_t idx) {
                  BitmapOwnership visited_bm(bm_sz);
                  const VertexID center = centers[idx];
                  visited_bm.Clear();
                  visited_bm.SetBit(static_cast<size_t>(center));
                  std::queue<std::pair<VertexID, VertexID>> q;
                  q.emplace(center, 0);
                  std::vector<Edge> picked_edges;

                  while (!q.empty()) {
                    auto cur = q.front();
                    q.pop();
                    const VertexID u = cur.first;
                    const VertexID depth = cur.second;
                    if (depth >= k) continue;

                    auto lo = std::lower_bound(
                        sorted_e.begin(), sorted_e.end(), u,
                        [](const Edge& e, VertexID x) { return e.src < x; });
                    for (auto it = lo; it != sorted_e.end() && it->src == u;
                         ++it) {
                      picked_edges.push_back(*it);
                      const size_t di = static_cast<size_t>(it->dst);
                      if (!visited_bm.GetBit(di)) {
                        visited_bm.SetBit(di);
                        q.emplace(it->dst, depth + 1);
                      }
                    }
                  }

                  if (picked_edges.empty()) {
                    subgraph_slots[idx] = std::make_unique<Edges>(
                        static_cast<EdgeIndex>(0), stub_pair);
                    return;
                  }

                  const size_t pe = picked_edges.size();
                  std::vector<VertexID> edge_buf(pe * 2);
                  std::vector<size_t> pj(pe);
                  std::iota(pj.begin(), pj.end(), size_t{0});
                  std::for_each(std::execution::par, pj.begin(), pj.end(),
                                [&](size_t j) {
                                  const Edge& e = picked_edges[j];
                                  edge_buf[j * 2] = e.src;
                                  edge_buf[j * 2 + 1] = e.dst;
                                });

                  subgraph_slots[idx] = std::make_unique<Edges>(
                      static_cast<EdgeIndex>(pe), edge_buf.data());
                });

  std::vector<Edges> subgraphs;
  subgraphs.reserve(ncenters);
  for (size_t i = 0; i < ncenters; ++i) {
    subgraphs.push_back(std::move(*subgraph_slots[i]));
  }
  return subgraphs;
}

}  // namespace data_structures
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
