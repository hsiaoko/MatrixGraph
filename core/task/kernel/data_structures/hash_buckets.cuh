#ifndef MATRIXGRAPH_CORE_TASK_KERNEL_DATA_STRUCTURES_HASH_BUCKETS_CUH_
#define MATRIXGRAPH_CORE_TASK_KERNEL_DATA_STRUCTURES_HASH_BUCKETS_CUH_

#include "core/common/types.h"
#include "core/data_structures/device_buffer.cuh"
#include "core/data_structures/host_buffer.cuh"
#include "core/data_structures/metadata.h"
#include "core/data_structures/unified_buffer.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

class HashBuckets {
private:
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using BufferVertexID =
      sics::matrixgraph::core::data_structures::Buffer<VertexID>;
  using UnifiedOwnedBufferVertexID =
      sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<VertexID>;

public:
  __host__ HashBuckets(VertexID x, VertexID y);

  __host__ HashBuckets() = default;

  void __host__ Init(VertexID x, VertexID y);

  void __host__ Free();

  VertexID **data_ = nullptr;
  VertexID *offset_ = nullptr;
  VertexID y_ = 0;
  VertexID x_ = 0;
};

} // namespace kernel
} // namespace task
} // namespace core
} // namespace matrixgraph
} // namespace sics

#endif // MATRIXGRAPH_CORE_TASK_KERNEL_DATA_STRUCTURES_HASH_BUCKETS_CUH_