#ifndef MATRIXGRAPH_CORE_TASK_GPU_TASK_GRAPH_FILTER_AGGREGATE_CUH_
#define MATRIXGRAPH_CORE_TASK_GPU_TASK_GRAPH_FILTER_AGGREGATE_CUH_

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

#include "core/data_structures/attributes.h"
#include "core/task/gpu_task/execute_agg_prim.cuh"
#include "core/task/gpu_task/graph_aggregate.cuh"
#include "core/task/gpu_task/task_base.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using ValueType = sics::matrixgraph::core::data_structures::ValueType;
using AttributeName = sics::matrixgraph::core::data_structures::AttributeName;

// -----------------------------------------------------------------------------
// Condition expression supported on the device.
//
// For now: Compound "and" of CmpExpr conditions.  Left operand can be a single
// attribute read or a Subtract of two attribute reads.  Right operand must be a
// constant.
// -----------------------------------------------------------------------------
struct FilterOperand {
  enum class Kind : uint8_t {
    kConst = 0,
    kAttr = 1,         // read attribute of the pivot vertex
    kPatternAttr = 2,  // read attribute of the neighbor vertex
    kSubtract = 3,     // left_attr - right_attr (both scalar)
  };

  Kind kind = Kind::kConst;

  // For kConst.
  ValueType const_type = ValueType::kInvalid;
  int64_t const_i64 = 0;
  double const_f64 = 0.0;

  // For kAttr / kPatternAttr / kSubtract inputs.
  AttributeName attr_name;
  int32_t pattern_position = -1;  // >=0 for PatternAttr inputs

  // For kSubtract: second operand is another attribute/pattern-attr.
  AttributeName sub_attr_name;
  int32_t sub_pattern_position = -1;
};

struct FilterCondition {
  enum class Op : uint8_t {
    kEq = 0,
    kNeq,
    kGt,
    kGte,
    kLt,
    kLte,
  };

  Op op = Op::kEq;
  FilterOperand left;
  FilterOperand right;
};

struct FilterAggRequest {
  uint32_t pivot_vertex_id = 0;
  uint32_t edge_label = 0;           // 0 = any edge label
  uint32_t target_vertex_label = 0;  // 0 = any neighbor label
  bool use_outgoing = true;    // true = outgoing, false = incoming
  int32_t agg_prim = 0;        // maps to execute_agg_prim::AggPrim
  AttributeName agg_attr_name;

  uint32_t n_conditions = 0;
  const FilterCondition* conditions = nullptr;
};

// -----------------------------------------------------------------------------
// GraphFilterAggregate: per-pivot filtered aggregation over a single graph.
//
// The graph topology and per-vertex attributes are uploaded once.  Each Compute()
// call supplies a list of requests; each request describes (pivot, edge filter,
// conditions, target attribute, primitive).  The kernel evaluates the conditions
// for every neighbor in parallel, keeps the values that pass, and reduces them
// according to the requested primitive.
// -----------------------------------------------------------------------------
class GraphFilterAggregate : public TaskBase {
 public:
  GraphFilterAggregate();
  ~GraphFilterAggregate();

  GraphFilterAggregate(const GraphFilterAggregate&) = delete;
  GraphFilterAggregate& operator=(const GraphFilterAggregate&) = delete;

  GraphFilterAggregate(GraphFilterAggregate&& other) noexcept;
  GraphFilterAggregate& operator=(GraphFilterAggregate&& other) noexcept;

  void SetNumStreams(uint32_t n_streams);

  // Load graph topology from flat CSR arrays.  Both offsets and edges are host
  // pointers.  edge_labels may be null (treated as all zero/any). vertex_labels
  // may be null (treated as all zero/any); when present it has n_vertices
  // entries and is used to filter neighbors by target vertex label.
  __host__ void LoadGraphCSR(uint32_t n_vertices,
                             uint32_t n_edges,
                             const uint32_t* csr_offsets,
                             const uint32_t* csr_edges,
                             const uint32_t* edge_labels,
                             const uint32_t* vertex_labels);

  // Load per-vertex attribute columns.  Reuses GraphAggregate's column format.
  __host__ void LoadVertexAttributes(
      uint32_t n_columns,
      const GraphAggregateAttributeColumn* columns);

  // Compute one batch of requests.  Returns one FeatureValue per request.
  __host__ std::vector<sics::matrixgraph::core::task::FeatureValue> Compute(
      const std::vector<FilterAggRequest>& requests);

  __host__ void Run();

 private:
  using FeatureValue = sics::matrixgraph::core::task::FeatureValue;

  __host__ void EnsureStreams();
  __host__ void DestroyStreams();
  __host__ void FreeBuffers();

  __host__ void BuildAttributesFromColumns(
      uint32_t n_columns,
      const GraphAggregateAttributeColumn* columns);

  // Reusable device buffers, grown on demand.
  __host__ void EnsureRequestBuffers(uint32_t n_requests);
  __host__ void EnsureScratch(uint32_t max_degree);

  uint32_t n_streams_ = 2;
  std::vector<cudaStream_t> streams_;

  // Graph topology (host master + device copy).
  using VertexID = sics::matrixgraph::core::common::VertexID;
  using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;

  uint32_t n_vertices_ = 0;
  uint32_t n_edges_ = 0;

  std::vector<EdgeIndex> h_csr_offsets_;
  std::vector<VertexID> h_csr_edges_;
  std::vector<uint32_t> h_edge_labels_;
  std::vector<uint32_t> h_vertex_labels_;

  EdgeIndex* d_csr_offsets_ = nullptr;
  VertexID* d_csr_edges_ = nullptr;
  uint32_t* d_edge_labels_ = nullptr;
  uint32_t* d_vertex_labels_ = nullptr;

  // Per-vertex attributes on the device.
  std::vector<sics::matrixgraph::core::data_structures::DeviceAttributes>
      per_vertex_attrs_;
  sics::matrixgraph::core::data_structures::Attributes* d_vertex_attrs_ = nullptr;

  // Per-attribute column buffers (owned by us).
  std::vector<uint8_t*> column_buffers_;

  // Request buffers.
  FilterAggRequest* d_requests_ = nullptr;
  uint32_t requests_cap_ = 0;
  FeatureValue* d_outputs_ = nullptr;
  uint32_t outputs_cap_ = 0;

  // Per-request scratch for order-dependent primitives (NumUnique hash table).
  // Layout: request i owns slots [hash_offsets[i], hash_offsets[i+1]).
  unsigned long long* d_hash_scratch_ = nullptr;
  uint32_t* d_hash_offsets_ = nullptr;
  size_t hash_scratch_cap_ = 0;
  uint32_t hash_offsets_cap_ = 0;
};

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_GPU_TASK_GRAPH_FILTER_AGGREGATE_CUH_
