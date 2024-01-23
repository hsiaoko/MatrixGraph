#ifndef CORE_UTIL_SET_H_
#define CORE_UTIL_SET_H_

#include <numeric>
#include <set>
#include <vector>

namespace sics {
namespace matrixgraph {
namespace core {
namespace util {
namespace set {

using VertexID = sics::matrixgraph::core::common::VertexID;
using Bitmap = sics::matrixgraph::core::util::Bitmap;

// @DESCRIPTION
// GetIntersection tasks two sets of vertex IDs and returns the "index" of
// intersection.
// @PARAM
// p_vec_a, p_vec_b: The two sets of vertex IDs.
// len_a, len_b: The length of the two sets of vertex IDs.
// max_val: The maximum vertex ID.
// TODO (hsiaoko): implement GetIntersection in parallel by using TBB.
std::vector<std::pair<VertexID, VertexID>>
GetIntersection(const VertexID *p_vec_a, VertexID len_a,
                const VertexID *p_vec_b, VertexID len_b, VertexID max_val) {

  auto parallelism = std::thread::hardware_concurrency();
  std::vector<VertexID> parallel_scope(parallelism);
  Bitmap visited(max_val);

  std::iota(parallel_scope.begin(), parallel_scope.end(), 0);
  std::vector<std::pair<VertexID, VertexID>> intersection;

  std::cout << "GetIntersection" << std::endl;

  std::cout << "A: " << std::endl << "    ";

  for (size_t i = 0; i < len_a; i++) {
    std::cout << p_vec_a[i] << ", ";
  }

  std::cout << "\nB: " << std::endl << "    ";
  for (size_t i = 0; i < len_b; i++) {
    std::cout << p_vec_b[i] << ", ";
  }
  std::cout << std::endl;

  if (len_b < len_a) {
    VertexID idx_by_val_b[max_val] = {0};
    std::for_each(
        parallel_scope.begin(), parallel_scope.end(),
        [parallelism, &visited, &p_vec_b, &idx_by_val_b, len_b](auto i) {
          auto j = i;
          for (j; j < len_b; j += parallelism) {
            idx_by_val_b[p_vec_b[j]] = j;
            visited.SetBit(p_vec_b[j]);
          }
        });

    std::for_each(parallel_scope.begin(), parallel_scope.end(),
                  [parallelism, &visited, &intersection, &p_vec_a,
                   &idx_by_val_b, len_a](auto i) {
                    auto j = i;
                    for (j; j < len_a; j += parallelism) {
                      if (visited.GetBit(p_vec_a[j])) {
                        intersection.push_back(
                            std::make_pair(j, idx_by_val_b[p_vec_a[j]]));
                      }
                    }
                  });

  } else {
    VertexID idx_by_val_a[max_val] = {0};
    std::for_each(
        parallel_scope.begin(), parallel_scope.end(),
        [parallelism, &visited, &p_vec_a, &idx_by_val_a, len_a](auto i) {
          auto j = i;
          for (j; j < len_a; j += parallelism) {
            idx_by_val_a[p_vec_a[j]] = j;
            visited.SetBit(p_vec_a[j]);
          }
        });
    std::for_each(parallel_scope.begin(), parallel_scope.end(),
                  [parallelism, &visited, &intersection, &p_vec_b,
                   &idx_by_val_a, len_b](auto i) {
                    auto j = i;
                    for (j; j < len_b; j += parallelism) {
                      if (visited.GetBit(p_vec_b[j])) {
                        intersection.push_back(
                            std::make_pair(idx_by_val_a[p_vec_b[j]], j));
                      }
                    }
                  });
  }
  for (int t = 0; t < intersection.size(); t++) {
    std::cout << "first: " << intersection[t].first
              << " second: " << intersection[t].second << std::endl;
  }
  return intersection;
}
} // namespace set
} // namespace util
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // CORE_UTIL_ATOMIC_H_