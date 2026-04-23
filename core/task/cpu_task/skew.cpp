#include "core/task/cpu_task/skew.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <execution>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using VertexID = sics::matrixgraph::core::common::VertexID;
using ImmutableCSRGraph =
    sics::matrixgraph::core::data_structures::ImmutableCSR;

namespace {

uint32_t BfsMaxDistanceUndirected(const ImmutableCSRGraph& g, VertexID src) {
  const VertexID n = g.get_num_vertices();
  constexpr uint32_t kInf = std::numeric_limits<uint32_t>::max();
  std::vector<uint32_t> dist(n, kInf);
  std::vector<VertexID> q;
  q.reserve(n);
  dist[src] = 0;
  q.push_back(src);
  std::size_t head = 0;
  uint32_t maxd = 0;
  while (head < q.size()) {
    const VertexID u = q[head++];
    const uint32_t du = dist[u];
    if (du > maxd) maxd = du;

    auto relax = [&](VertexID v) {
      if (dist[v] == kInf) {
        dist[v] = du + 1u;
        q.push_back(v);
      }
    };

    for (VertexID k = 0; k < g.GetOutDegreeByLocalID(u); ++k) {
      relax(g.GetOutgoingEdgesByLocalID(u)[k]);
    }
    for (VertexID k = 0; k < g.GetInDegreeByLocalID(u); ++k) {
      relax(g.GetIncomingEdgesByLocalID(u)[k]);
    }
  }
  return maxd;
}

}  // namespace

void Skew::LoadData() {
  std::cout << "[Skew] LoadData()" << std::endl;
  g_.Read(data_graph_path_);
}

void Skew::ComputeSkew(const ImmutableCSRGraph& g) {
  const VertexID n = g.get_num_vertices();
  if (n == 0) {
    std::cout << "[Skew] empty graph; skew undefined." << std::endl;
    return;
  }

  const double n_d = static_cast<double>(n);
  const double m_out = static_cast<double>(g.get_num_outgoing_edges());
  const double m_in = static_cast<double>(g.get_num_incoming_edges());
  const double d_bar = (m_out + m_in) / n_d;

  const size_t n_sz = static_cast<size_t>(n);
  const bool exact = (sample_sources_ == 0);
  size_t k = exact ? n_sz : std::min(sample_sources_, n_sz);

  std::vector<VertexID> sources(k);
  if (k == n_sz) {
    std::iota(sources.begin(), sources.end(), static_cast<VertexID>(0));
  } else {
    std::vector<VertexID> pool(n_sz);
    std::iota(pool.begin(), pool.end(), static_cast<VertexID>(0));
    std::mt19937_64 gen(random_seed_);
    std::sample(pool.begin(), pool.end(), sources.begin(), k, gen);
  }

  std::atomic<uint32_t> d_hat_atomic{0};
  std::for_each(std::execution::par, sources.begin(), sources.end(),
                [&](VertexID s) {
                  const uint32_t ecc = BfsMaxDistanceUndirected(g, s);
                  uint32_t cur = d_hat_atomic.load(std::memory_order_relaxed);
                  while (ecc > cur &&
                         !d_hat_atomic.compare_exchange_weak(
                             cur, ecc, std::memory_order_relaxed,
                             std::memory_order_relaxed)) {
                  }
                });

  const uint32_t d_hat = d_hat_atomic.load();

  std::cout << "[Skew] d_bar = (|E_out|+|E_in|)/n = " << d_bar << std::endl;
  if (exact) {
    std::cout << "[Skew] d_hat(G) (exact, " << k << " BFS sources): " << d_hat
              << std::endl;
  } else {
    std::cout << "[Skew] d_hat(G) (approximate, " << k
              << " random BFS sources, seed=" << random_seed_
              << "): " << d_hat << std::endl;
    std::cout << "[Skew] note: d_hat ≤ true undirected diameter (same as "
                  "Diameter app)."
              << std::endl;
  }

  if (d_bar <= 0.0) {
    std::cout << "[Skew] skew(G) undefined (mean total degree is 0)."
              << std::endl;
    return;
  }

  const double skew = static_cast<double>(d_hat) / d_bar;
  std::cout << "[Skew] skew(G) ≈ d_hat / d_bar = " << skew << std::endl;
}

void Skew::Run() {
  auto t0 = std::chrono::system_clock::now();
  LoadData();
  auto t1 = std::chrono::system_clock::now();
  ComputeSkew(g_);
  auto t2 = std::chrono::system_clock::now();

  std::cout << "[Skew] LoadData() elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                       .count() /
                   static_cast<double>(CLOCKS_PER_SEC)
            << std::endl;
  std::cout << "[Skew] Compute() elapsed: "
            << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                       .count() /
                   static_cast<double>(CLOCKS_PER_SEC)
            << std::endl;
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
