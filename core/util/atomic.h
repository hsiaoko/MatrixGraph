#ifndef MATRIXGRAPH_CORE_UTIL_ATOMIC_H_
#define MATRIXGRAPH_CORE_UTIL_ATOMIC_H_

// CAS and the atomic min/max/add helpers built on it. Host-side only;
// device code uses CUDA atomics.
#include <bit>

namespace sics {
namespace matrixgraph {
namespace core {
namespace util {
namespace atomic {

// @DESCRIPTION
//
//  CAS is an atomic instruction used in multithreading to achieve
//  synchronization.  It compares the contents of a memory location with a given
//  value and, only if they are the same, modifies the contents of that memory
//  location to a new given value.
//
//  The value bits are reinterpreted with std::bit_cast instead of pointer
//  punning, so the comparison operates on a well-defined object copy.
template <class ET>
inline bool CAS(ET* ptr, ET oldv, ET newv) {
  static_assert(sizeof(ET) == 2 || sizeof(ET) == 4 || sizeof(ET) == 8,
                "CAS supports 2/4/8-byte types");
  if constexpr (sizeof(ET) == 8) {
    return __sync_bool_compare_and_swap(reinterpret_cast<long*>(ptr),
                                        std::bit_cast<long>(oldv),
                                        std::bit_cast<long>(newv));
  } else if constexpr (sizeof(ET) == 4) {
    return __sync_bool_compare_and_swap(reinterpret_cast<int*>(ptr),
                                        std::bit_cast<int>(oldv),
                                        std::bit_cast<int>(newv));
  } else {
    return __sync_bool_compare_and_swap(reinterpret_cast<unsigned short*>(ptr),
                                        std::bit_cast<unsigned short>(oldv),
                                        std::bit_cast<unsigned short>(newv));
  }
}

template <class ET>
inline bool WriteMin(ET* a, ET b) {
  ET c;
  bool r = 0;
  do c = *a;
  while (c > b && !(r = CAS(a, c, b)));
  return r;
}

template <class ET>
inline bool WriteMax(ET* a, ET b) {
  ET c;
  bool r = 0;
  do c = *a;
  while (c < b && !(r = CAS(a, c, b)));
  return r;
}

template <class ET>
inline void WriteAdd(ET* a, ET b) {
  volatile ET new_v, old_v;
  do {
    old_v = *a;
    new_v = old_v + b;
  } while (!CAS(a, old_v, new_v));
}

}  // namespace atomic
}  // namespace util
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_UTIL_ATOMIC_H_