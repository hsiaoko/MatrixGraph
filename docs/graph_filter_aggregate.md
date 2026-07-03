# GraphFilterAggregate — GPU Filtered Neighbor Aggregation

`GraphFilterAggregate` is a GPU task that, for each *pivot* vertex, walks its
outgoing neighbors, keeps the ones that satisfy a set of filter conditions, and
reduces a chosen attribute of the surviving neighbors with an aggregation
primitive — all on the device in a single kernel launch.

It complements `GraphAggregate`: where `GraphAggregate` reduces caller-supplied
value lists, `GraphFilterAggregate` fuses **navigation + per-neighbor filtering +
value gather + reduction** so the host never has to materialize the filtered
value lists.

- Header: `core/task/gpu_task/graph_filter_aggregate.cuh`
- Impl:   `core/task/gpu_task/graph_filter_aggregate.cu`

## Data model

The graph is uploaded **once** and reused across many `Compute` calls:

- **Topology** — flat CSR arrays (dense `uint32` vertex ids):
  - `csr_offsets[n_vertices + 1]`, `csr_edges[n_edges]` (target vertex ids)
  - `edge_labels[n_edges]` — optional; `0` means "any label"
  - `vertex_labels[n_vertices]` — optional; used to filter neighbors by target
    vertex label (`0` = any)
- **Per-vertex attributes** — one column per attribute, reusing
  `GraphAggregateAttributeColumn`:
  - `key` (attribute name), `value_type` (`ValueType`), `n_values` (= n_vertices)
  - `values` — a *tightly packed raw* buffer (not `FeatureValue`): `int64`/`double`
    = 8B, `float` = 4B, `bool` = 1B, `time` = `int64` UnixMilli
  - `valid` — optional `n_values` bytes (`1` = present). A missing/invalid value
    is treated as "condition false" and is dropped from the aggregation, matching
    the CPU semantics.

## API

```cpp
using namespace sics::matrixgraph::core::task;

GraphFilterAggregate task;
task.SetNumStreams(n_streams);

// 1) Upload topology once.
task.LoadGraphCSR(n_vertices, n_edges,
                  csr_offsets, csr_edges,
                  edge_labels,      // may be nullptr
                  vertex_labels);   // may be nullptr

// 2) Upload per-vertex attribute columns once.
task.LoadVertexAttributes(n_columns, columns);

// 3) Run one or more request batches.
std::vector<FilterAggRequest> requests = ...;
std::vector<FeatureValue> results = task.Compute(requests); // one per request
```

### Request

```cpp
struct FilterAggRequest {
  uint32_t pivot_vertex_id;      // dense CSR id of the pivot
  uint32_t edge_label = 0;       // 0 = any edge label
  uint32_t target_vertex_label = 0; // 0 = any neighbor label
  bool     use_outgoing = true;  // only outgoing is supported
  int32_t  agg_prim = 0;         // AggPrim
  AttributeName agg_attr_name;   // attribute to aggregate (read on the neighbor)
  uint32_t n_conditions = 0;
  const FilterCondition* conditions = nullptr; // AND-ed together
};
```

### Conditions

Conditions are an **AND** of comparisons. Each comparison is `left OP right`,
and *both* sides are evaluated on the device, so either side may be a constant,
an attribute read, or a subtract of two attribute reads.

```cpp
struct FilterCondition {
  enum class Op { kEq, kNeq, kGt, kGte, kLt, kLte };
  Op op;
  FilterOperand left, right;
};

struct FilterOperand {
  enum class Kind { kConst, kAttr, kPatternAttr, kSubtract };
  Kind kind;
  // kConst:
  ValueType const_type; int64_t const_i64; double const_f64;
  // kAttr (pivot) / kPatternAttr (neighbor) / kSubtract inputs:
  AttributeName attr_name;      int32_t pattern_position;      // >=0 => neighbor, <0 => pivot
  AttributeName sub_attr_name;  int32_t sub_pattern_position;  // kSubtract second operand
};
```

Operand semantics on the device:

- `kConst` — a numeric constant (`kFloat64` uses `const_f64`, otherwise `const_i64`;
  times are UnixMilli, durations are milliseconds).
- `kAttr` — reads `attr_name` on the **pivot** vertex.
- `kPatternAttr` — reads `attr_name` on the **neighbor** vertex.
- `kSubtract` — `a - b`, where each of `a`/`b` is an attribute read whose vertex
  is the neighbor when its `pattern_position >= 0`, else the pivot. All reads are
  as `double` (int/time in ms), so e.g. `Subtract(pivotTime, neighborTime)`
  compared against a millisecond duration constant reproduces the CPU result
  exactly.

If any operand a condition reads is missing/invalid at its vertex, the whole
comparison is **false** (the neighbor is rejected), matching the CPU path.

## Supported aggregation primitives

`Count`, `Sum`, `Mean`, `Min`, `Max`, `NumUnique`, `PercentTrue`,
`CountGreaterThanMean`. Any other primitive should be handled on the host.

Empty result set (no neighbor passes the filter) returns an **Invalid**
`FeatureValue` for every primitive, matching `ExecuteAggPrim`.

## Kernel

`FilterAggKernel` launches **one block per request**:

1. Threads grid-stride over the pivot's outgoing edges.
2. Per neighbor: apply edge-label + target-vertex-label filters, then evaluate
   the AND-ed conditions; skip neighbors whose aggregation source value is
   missing.
3. Reduce the surviving values: warp-shuffle + shared-memory block reduction for
   `Sum/Count/Mean/PercentTrue`; a cross-warp arg-min/arg-max for `Min/Max`; a
   per-request open-addressing hash table in global scratch for `NumUnique`; a
   second pass for `CountGreaterThanMean`.
4. Thread 0 writes one `FeatureValue` result.

A single shared `Attributes` table maps attribute name → full per-vertex column;
values are read by row (= dense vertex id), so there is no per-vertex hash-map
allocation.

## Limitations

- **Outgoing edges only** (incoming would need a separate in-CSR).
- **AND-only** conditions; comparison operators `eq/neq/gt/gte/lt/lte`.
- Aggregated / condition attributes must be **scalar** (`int`/`float`/`bool`/
  `time`); string and list attributes are not representable in the device columns
  and must be handled on the host.
- The aggregated `Min`/`Max` returns the value type of the winning neighbor's
  attribute (times come back as `int64` milliseconds).

## Performance notes

The dominant cost is not the kernel but the one-time host→device serialization
(building the dense CSR + attribute columns and uploading them). It pays off when
the resident graph is reused across many `Compute` calls, or when the per-neighbor
host work (attribute access, condition evaluation) is expensive relative to the
columnar materialization. For a static graph, build the resident graph once and
issue many request batches against it.
