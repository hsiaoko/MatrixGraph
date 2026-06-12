# ComputeFeatures Application

`apps/compute_features.cpp` is the standalone smoke-test / demo binary for the
`ComputeFeaturesTask`.  It exercises the GPU feature-expression interpreter
without requiring Go or any other language binding.

## Build

From the project root:

```bash
mkdir -p build && cd build
cmake ..
make compute_features_exec -j4
```

The binary is written to `bin/compute_features_exec`.

## Run

```bash
./bin/compute_features_exec
```

Expected output (abridged):

```text
[ComputeFeaturesTask] Loaded graph: 4 vertices, 8 outgoing edges
[ComputeFeaturesTask] Loaded 1 attribute column(s) for 4 vertices
[ComputeFeaturesTask] Loaded 1 attribute column(s) for 4 vertices
ComputeFeatures smoke test passed.
```

A non-zero exit code means one of the checks failed.

## What the binary demonstrates

The binary builds a tiny directed ring graph with 4 vertices and out-degree 2,
loads two attribute columns (`score` and `flag`), and evaluates several
expressions:

| Output | Expression | Purpose |
|--------|------------|---------|
| `score` | `AttrExpr("score")` | Per-pivot attribute read. |
| `const` | `ConstExpr(3.14)` | Constant value. |
| `sum` | `sum(score over outgoing neighbors)` | One-hop aggregation. |
| `mean` | `mean(score over outgoing neighbors)` | One-hop aggregation. |
| `trans_add` | `score + 10` | Transformation. |
| `nested_mean` | `mean(sum(score over neighbors of neighbors))` | Nested aggregation. |
| `filter_sum` | `sum(score of neighbors where flag == 1)` | FilterNav. |

## Runtime model

`ComputeFeaturesTask` is a standalone GPU task:

1. **Load graph**: `LoadGraph(path)` reads a MatrixGraph CSR directory and
   uploads the contiguous graph buffer to the GPU.
2. **Load attributes**: `LoadAttributes(columns)` uploads columnar per-vertex
   attributes and builds a per-vertex attribute map.
3. **Optional labels**: `LoadLabels(labels)` uploads per-vertex labels used for
   label-filter navigators.
4. **Compute**: `Compute(pivots, plan, navs, conds, outputs)` uploads the flat
   plan, launches one CUDA block per pivot, and returns a row-major result
   array.

Result layout:

```text
result[i * n_outputs + j]  = j-th output feature for pivots[i]
```

Each result entry is a `MatrixGraphFeatureValue` (typed union).

## Expression language

The interpreter consumes a flat array of `MatrixGraphPlanNode` nodes.  Each
node has a `type` field that selects one of the expression kinds:

### `MG_EXPR_ATTR`

Read a per-vertex attribute by name.

```cpp
MatrixGraphPlanNode n{};
n.type = MG_EXPR_ATTR;
std::strncpy(n.key, "score", sizeof(n.key));
```

The key must have been loaded via `LoadAttributes()`.

### `MG_EXPR_CONST`

Literal constant.  Only the union member matching `const_type` is valid.

```cpp
n.type = MG_EXPR_CONST;
n.const_type = MG_VALUE_INT;
n.const_i64 = 1;
```

### `MG_EXPR_AGG`

Aggregate a sub-expression over a navigator.

```cpp
n.type = MG_EXPR_AGG;
n.op = MG_AGG_SUM;     // aggregation primitive
n.src_idx = 0;         // index of the expression to aggregate
n.nav_idx = 0;         // index into the navigator array
```

Supported primitives: `COUNT`, `SUM`, `MEAN`, `MIN`, `MAX`, `VARIANCE`, `STD`,
`MEDIAN`, `MODE`, `NUM_UNIQUE`, `ENTROPY`, `QUARTER`, `QUARTILE3`,
`PERCENT_TRUE`, `SKEW`, `COUNT_GREATER_THAN_MEAN`.

### `MG_EXPR_TRANS`

Arithmetic/logical transformation.  Unary operators (`NEG`, `ABS`, `SQRT`) use
only `child_a`; binary operators use both `child_a` and `child_b`.

```cpp
n.type = MG_EXPR_TRANS;
n.op = MG_TRANS_ADD;
n.child_a = 0;   // index of left sub-expression
n.child_b = 2;   // index of right sub-expression
```

## Navigators

Navigators live in a separate `MatrixGraphPlanNode` array and are referenced by
index from aggregation expressions.

### `MG_NAV_SELF`

Single binding: the pivot vertex itself.

```cpp
n.type = MG_NAV_SELF;
```

### `MG_NAV_NEIGHBOR`

Outgoing or incoming neighbors.  `direction == 0` is outgoing, `1` is incoming.

```cpp
n.type = MG_NAV_NEIGHBOR;
n.direction = 0;                 // outgoing
std::strncpy(n.target_label, "", sizeof(n.target_label)); // no label filter
```

### `MG_NAV_FILTER`

Wraps another navigator and keeps only bindings that satisfy a condition.

```cpp
n.type = MG_NAV_FILTER;
n.inner_nav_idx = 0;   // index of the wrapped navigator
n.cond_idx = 0;        // index into the condition array
```

Conditions are `MatrixGraphCondNode` records:

```cpp
MatrixGraphCondNode c{};
c.op = MG_COND_EQ;
c.left_expr = 1;   // index into the expression plan
c.right_expr = 2;  // index into the expression plan
```

## Full example plan

```cpp
// Expression plan
std::vector<MatrixGraphPlanNode> plan;
plan.push_back(AttrNode("score"));                 // 0
plan.push_back(AttrNode("flag"));                  // 1
plan.push_back(ConstIntNode(1));                   // 2
plan.push_back(AggNode(MG_AGG_SUM, 0, 1));         // 3: sum score where flag==1

// Navigator plan
std::vector<MatrixGraphPlanNode> navs;
navs.push_back(OutNeighborNav());                  // 0: outgoing neighbors
navs.push_back(FilterNavNode(0, 0));               // 1: filter with cond 0

// Conditions
std::vector<MatrixGraphCondNode> conds;
conds.push_back(CondNode(MG_COND_EQ, 1, 2));       // flag == 1

// Outputs
std::vector<int32_t> outputs = {3};
std::vector<uint32_t> pivots = {0, 1, 2, 3};

auto result = task.Compute(pivots, plan, navs, conds, outputs);
```

## Limitations

- `MG_NAV_NEIGHBOR` with `direction == 2` (both in and out) is not implemented.
- `MG_NAV_PATTERN` is not implemented.
- `MG_EXPR_PATTERN_ATTR` is not implemented.
- The maximum number of bindings per pivot is `kComputeFeaturesMaxNeighbors`
  (currently 256); larger neighborhoods are silently truncated.
- String attribute values are accepted by `LoadAttributes()` but are not
  returned by aggregation primitives.
