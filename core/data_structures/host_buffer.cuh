#ifndef MATRIXGRAPH_CORE_DATA_STRUCTURES_HOST_BUFFER_CUH_
#define MATRIXGRAPH_CORE_DATA_STRUCTURES_HOST_BUFFER_CUH_

#include <stdint.h>

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

// Structure to hold buffer information
template <typename T> struct Buffer {
  // Pointer to data
  T *data;

  // Size of the buffer (byte)
  size_t size;

  T *GetPtr() const { return data; }

  size_t GetSize() const { return size; };

  size_t GetElementSize() const { return sizeof(T); }
};

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_DATA_STRUCTURES_HOST_BUFFER_CUH_