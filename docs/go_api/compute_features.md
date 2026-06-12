# ComputeFeatures Go / CGO API

This document describes the C API exposed in `core/go_api/matrixgraph_go_api.h`
for driving `ComputeFeaturesTask` from Go (or any other language that can call
C functions).

The API is intentionally low-level: Go code builds flat `MatrixGraphPlanNode`,
`MatrixGraphPlanNode` (navigators) and `MatrixGraphCondNode` arrays and passes
them to the C functions.  All pointers are host memory.

## Header

```c
#include "core/task/gpu_task/compute_features_types.h"
#include "core/go_api/matrixgraph_go_api.h"
```

## Lifecycle

### Create a handle

```c
void* matrixgraph_compute_features_create(void);
```

Returns an opaque task handle, or `NULL` on failure.  The handle owns all GPU
resources (graph buffer, attribute columns, labels) loaded through it.

### Destroy a handle

```c
void matrixgraph_compute_features_destroy(void* handle);
```

Frees the handle and all associated device memory.  Passing `NULL` is safe.

## Loading data

All load functions return `0` on success and non-zero on error.

### Graph

```c
int matrixgraph_compute_features_load_graph(void* handle, const char* graph_path);
```

`graph_path` must point to a MatrixGraph CSR directory written by
`ImmutableCSR::Write()`.

### Attributes

```c
int matrixgraph_compute_features_load_attributes(
    void* handle, uint32_t n_columns,
    const ComputeFeaturesAttributeColumn* columns);
```

Each column is:

```c
typedef struct {
  char key[64];       // attribute name used by MG_EXPR_ATTR
  int32_t value_type; // MatrixGraphValueType, e.g. MG_VALUE_FLOAT64
  uint32_t n_values;  // must equal the number of vertices
  void* values;       // host pointer; must stay valid for this call only
} ComputeFeaturesAttributeColumn;
```

Calls are cumulative: previously loaded columns remain available.

### Labels (optional)

```c
int matrixgraph_compute_features_load_labels(
    void* handle, const uint32_t* labels, uint32_t n);
```

Labels enable `NeighborNav` label filters and will be required by pattern
navigators in later phases.

## Computing features

```c
int matrixgraph_compute_features_compute(
    void* handle,
    const uint32_t* pivot_vertex_ids, uint32_t n_pivots,
    const MatrixGraphPlanNode* plan, uint32_t n_plan_nodes,
    const MatrixGraphPlanNode* navs, uint32_t n_navs,
    const MatrixGraphCondNode* conds, uint32_t n_conds,
    const int32_t* output_expr_indices, uint32_t n_outputs,
    MatrixGraphFeatureValue* out_values);
```

Parameters:

| Parameter | Meaning |
|-----------|---------|
| `pivot_vertex_ids` | Array of vertex ids to evaluate, length `n_pivots`. |
| `plan` | Flat expression plan, length `n_plan_nodes`. |
| `navs` | Flat navigator plan, length `n_navs`.  May be `NULL` if `n_navs == 0`. |
| `conds` | Flat condition array, length `n_conds`.  May be `NULL` if `n_conds == 0`. |
| `output_expr_indices` | Indices into `plan` to emit, length `n_outputs`. |
| `out_values` | Pre-allocated output buffer of length `n_pivots * n_outputs`. |

Output layout:

```text
out_values[i * n_outputs + j]
```

is the `j`-th requested feature for `pivot_vertex_ids[i]`.

Each output value is a typed union:

```c
typedef struct {
  int32_t type;  // MatrixGraphValueType
  union {
    int64_t i64;  // MG_VALUE_INT / MG_VALUE_TIME
    double f64;   // MG_VALUE_FLOAT64 / MG_VALUE_FLOAT32
    int32_t b;    // MG_VALUE_BOOL
  };
} MatrixGraphFeatureValue;
```

Only the member corresponding to `type` is valid.

## Go example

```go
package main

/*
#cgo LDFLAGS: -L${SRCDIR}/../../lib -lmatrixgraph_goapi -lcudart
#include "core/go_api/matrixgraph_go_api.h"
*/
import "C"
import (
    "fmt"
    "unsafe"
)

func main() {
    handle := C.matrixgraph_compute_features_create()
    if handle == nil {
        panic("failed to create task")
    }
    defer C.matrixgraph_compute_features_destroy(handle)

    cPath := C.CString("/path/to/graph")
    defer C.free(unsafe.Pointer(cPath))
    if C.matrixgraph_compute_features_load_graph(handle, cPath) != 0 {
        panic("failed to load graph")
    }

    // Load a "score" attribute column.
    scores := []float64{0.0, 1.5, 3.0, 4.5}
    col := C.ComputeFeaturesAttributeColumn{
        value_type: C.MG_VALUE_FLOAT64,
        n_values:   C.uint32_t(len(scores)),
        values:     unsafe.Pointer(&scores[0]),
    }
    C.strncpy(&col.key[0], C.CString("score"), 63)
    if C.matrixgraph_compute_features_load_attributes(handle, 1, &col) != 0 {
        panic("failed to load attributes")
    }

    // Plan: AttrExpr("score") at index 0.
    plan := []C.MatrixGraphPlanNode{
        {type: C.MG_EXPR_ATTR},
    }
    C.strncpy(&plan[0].key[0], C.CString("score"), 63)

    // Navigator: SelfNav at index 0.
    navs := []C.MatrixGraphPlanNode{
        {type: C.MG_NAV_SELF},
    }

    // Output expression index 0.
    outputs := []C.int32_t{0}
    pivots := []C.uint32_t{0, 1, 2, 3}

    out := make([]C.MatrixGraphFeatureValue, len(pivots)*len(outputs))
    if C.matrixgraph_compute_features_compute(
        handle,
        &pivots[0], C.uint32_t(len(pivots)),
        &plan[0], C.uint32_t(len(plan)),
        &navs[0], C.uint32_t(len(navs)),
        nil, 0, // no conditions
        &outputs[0], C.uint32_t(len(outputs)),
        &out[0]) != 0 {
        panic("compute failed")
    }

    for i, pid := range pivots {
        v := out[i*len(outputs)]
        fmt.Printf("pivot %d score = %f\n", pid, float64(v.f64))
    }
}
```

## Error handling

- All C functions return `0` on success and non-zero on error.
- A `NULL` handle or missing required pointer is treated as an error.
- If the plan references an attribute that was never loaded, the device
  interpreter returns `MG_VALUE_INVALID` for that expression rather than
  crashing.

## Thread safety

A single `ComputeFeaturesTask` handle is **not** thread-safe.  If multiple Go
goroutines need to compute features concurrently, create one handle per
goroutine and load the graph/attributes independently, or serialize access to
a shared handle.

## Building the shared library

```bash
cd build
cmake ..
make matrixgraph_goapi -j4
```

This produces `lib/libmatrixgraph_goapi.so` (and the required static
`libmatrixgraph_core.a`).  Link your Go/cgo build against these libraries and
against the CUDA runtime (`-lcudart`).

## Relationship to GraphAggregate

`ComputeFeaturesTask` is independent of `GraphAggregate`.  Use
`matrixgraph_compute_features_*` when you have a pre-built flat expression plan
and want full flexibility (nested expressions, FilterNav, transformations).
Use the `matrixgraph_graph_aggregate_*` API when you only need simple
attribute aggregations over neighbors.
