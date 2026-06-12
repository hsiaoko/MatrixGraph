# Go API (`libmatrixgraph_goapi.so`)

MatrixGraph exposes a C API via `core/go_api/` so that Go programs can call GPU graph kernels through **CGO**. The shared library is built as `lib/libmatrixgraph_goapi.so`.

---

## Build

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_HOST_COMPILER=g++-13 -DTEST=OFF
cmake --build . --target matrixgraph_goapi
```

Output: `lib/libmatrixgraph_goapi.so`

---

## CGO Quick Start

Create a Go package (e.g. `pkg/matrixgraph/matrixgraph.go`) next to the project root:

```go
package matrixgraph

/*
#cgo LDFLAGS: -L${SRCDIR}/../../../lib -lmatrixgraph_goapi -lcudart
#cgo CFLAGS: -I${SRCDIR}/../../../core/go_api
#include "matrixgraph_go_api.h"
*/
import "C"
import (
    "fmt"
    "unsafe"
)
```

> `${SRCDIR}` expands to the directory containing the `.go` file. Adjust relative paths if your package lives elsewhere.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MATRIXGRAPH_CUDA_DEVICE` | GPU device index (e.g. `"0"`). The shared library calls `cudaSetDevice` automatically before every API entry point. |

---

## API Summary

| C Function | Go Wrapper | Description |
|------------|------------|-------------|
| `matrixgraph_matmult` | `MatMult` | Dense matrix multiplication C = A × B |
| `matrixgraph_relu` | `ReLU` | In-place ReLU activation |
| `matrixgraph_matadd` | `MatAdd` | In-place matrix addition B = A + B |
| `matrixgraph_transpose` | `Transpose` | Matrix transpose B = Aᵀ |
| `matrixgraph_graph_aggregate_create` | `GraphAggregateCreate` | Create a GraphAggregate handle |
| `matrixgraph_graph_aggregate_destroy` | `GraphAggregateDestroy` | Destroy the handle |
| `matrixgraph_graph_aggregate_load_synthetic` | `GraphAggregateLoadSynthetic` | Load a synthetic ring graph (clears existing) |
| `matrixgraph_graph_aggregate_add_synthetic` | `GraphAggregateAddSynthetic` | Add an additional synthetic graph (multi-graph) |
| `matrixgraph_graph_aggregate_compute_features` | `GraphAggregateComputeFeatures` | Compute per-vertex aggregated features |
| `matrixgraph_graph_aggregate_compute_all` | `GraphAggregateComputeAll` | Fused compute-all primitives in one kernel launch |
| `matrixgraph_compute_features_create` | `ComputeFeaturesCreate` | Create a ComputeFeatures handle |
| `matrixgraph_compute_features_destroy` | `ComputeFeaturesDestroy` | Destroy the handle |
| `matrixgraph_compute_features_load_graph` | `ComputeFeaturesLoadGraph` | Load a MatrixGraph CSR graph |
| `matrixgraph_compute_features_load_attributes` | `ComputeFeaturesLoadAttributes` | Load per-vertex attribute columns |
| `matrixgraph_compute_features_load_labels` | `ComputeFeaturesLoadLabels` | Load per-vertex labels (optional) |
| `matrixgraph_compute_features_compute` | `ComputeFeaturesCompute` | Evaluate a flat expression plan |
| `matrixgraph_subiso` | `SubIso` | GPU subgraph isomorphism (WOJ) |
| `matrixgraph_gar_match` | `GARMatch` | Graph association rule matching (stub) |

---

## MatrixOps

### `matrixgraph_matmult`

**C signature**
```c
int matrixgraph_matmult(const float* A, const float* B, float* C,
                        int m, int k, int n);
```

Row-major dense GEMM. `A` is `m×k`, `B` is `k×n`, `C` is `m×n`.

**Go example**
```go
func MatMult(A, B []float32, m, k, n int) ([]float32, error) {
    if len(A) != m*k || len(B) != k*n {
        return nil, fmt.Errorf("dimension mismatch")
    }
    C := make([]float32, m*n)
    ret := C.matrixgraph_matmult(
        (*C.float)(unsafe.Pointer(&A[0])),
        (*C.float)(unsafe.Pointer(&B[0])),
        (*C.float)(unsafe.Pointer(&C[0])),
        C.int(m), C.int(k), C.int(n),
    )
    if ret != 0 {
        return nil, fmt.Errorf("matmult failed")
    }
    return C, nil
}
```

### `matrixgraph_relu`

**C signature**
```c
int matrixgraph_relu(float* A, int m, int n);
```

In-place ReLU on an `m×n` row-major matrix.

**Go example**
```go
func ReLU(A []float32, m, n int) error {
    if len(A) != m*n {
        return fmt.Errorf("dimension mismatch")
    }
    ret := C.matrixgraph_relu(
        (*C.float)(unsafe.Pointer(&A[0])), C.int(m), C.int(n))
    if ret != 0 {
        return fmt.Errorf("relu failed")
    }
    return nil
}
```

### `matrixgraph_matadd`

**C signature**
```c
int matrixgraph_matadd(const float* A, float* B, int m, int n);
```

`B = A + B` in-place.

### `matrixgraph_transpose`

**C signature**
```c
int matrixgraph_transpose(const float* A, float* B, int m, int n);
```

`B = Aᵀ`. `A` is `m×n`, `B` must be `n×m`.

---

## GraphAggregate

GraphAggregate computes aggregated neighbor features for a set of pivot vertices. A typical workflow is:

1. `create`
2. `load_synthetic`
3. `compute_features`
4. `destroy`

### C Types

```c
typedef struct {
  char   attr_name[64];
  uint32_t neighbor_label;
  uint8_t  use_outgoing;   // 1 = outgoing, 0 = incoming
  int32_t  prim;           // AggPrim enum value
} MatrixGraphFeatureRequest;

typedef struct {
  int32_t type;   // ValueType
  union {
    int64_t i64;
    double  f64;
    uint8_t b;
  };
} MatrixGraphFeatureValue;
```

> The C `union` means `i64`, `f64`, and `b` share the same 8-byte slot.  Go wrappers must mirror this layout exactly (see `FeatureValue` below).

### AggPrim values

| Value | Name | Output type |
|-------|------|-------------|
| `0` | `kCount` | `int64` |
| `1` | `kCountGreaterThanMean` | `int64` |
| `2` | `kNumUnique` | `int64` |
| `3` | `kSum` | `float64` |
| `4` | `kMean` | `float64` |
| `5` | `kVariance` | `float64` |
| `6` | `kStd` | `float64` |
| `7` | `kMode` | `float64` |
| `8` | `kMin` | `float64` |
| `9` | `kMax` | `float64` |
| `10`| `kMedian` | `float64` |
| `11`| `kQuarter` | `float64` |
| `12`| `kQuartile3` | `float64` |
| `13`| `kEntropy` | `float64` |
| `14`| `kPercentTrue` | `float64` |
| `15`| `kSkew` | `float64` |

### Go example — synthetic graph + feature computation

```go
package matrixgraph

/*
#cgo LDFLAGS: -L${SRCDIR}/../../../lib -lmatrixgraph_goapi -lcudart
#cgo CFLAGS: -I${SRCDIR}/../../../core/go_api
#include "matrixgraph_go_api.h"
*/
import "C"
import (
    "fmt"
    "unsafe"
)

type AggPrim int32

const (
    AggCount                AggPrim = 0
    AggCountGreaterThanMean AggPrim = 1
    AggNumUnique            AggPrim = 2
    AggSum                  AggPrim = 3
    AggMean                 AggPrim = 4
    AggVariance             AggPrim = 5
    AggStd                  AggPrim = 6
    AggMode                 AggPrim = 7
    AggMin                  AggPrim = 8
    AggMax                  AggPrim = 9
    AggMedian               AggPrim = 10
    AggQuarter              AggPrim = 11
    AggQuartile3            AggPrim = 12
    AggEntropy              AggPrim = 13
    AggPercentTrue          AggPrim = 14
    AggSkew                 AggPrim = 15
)

func GraphAggregateCreate() unsafe.Pointer {
    return C.matrixgraph_graph_aggregate_create()
}

func GraphAggregateDestroy(h unsafe.Pointer) {
    C.matrixgraph_graph_aggregate_destroy(h)
}

func GraphAggregateLoadSynthetic(h unsafe.Pointer, nVertices, outDeg uint32) error {
    ret := C.matrixgraph_graph_aggregate_load_synthetic(
        h, C.uint32_t(nVertices), C.uint32_t(outDeg))
    if ret != 0 {
        return fmt.Errorf("load_synthetic failed")
    }
    return nil
}

// FeatureRequest mirrors the C struct.
type FeatureRequest struct {
    AttrName      [64]byte
    NeighborLabel uint32
    UseOutgoing   uint8
    Prim          int32
}

// FeatureValue mirrors the C union layout:
//   int32_t type;  // 4 bytes
//   char padding[4];
//   union { int64_t i64; double f64; uint8_t b; };  // 8 bytes
type FeatureValue struct {
    Type int32
    _    [4]byte
    raw  [8]byte
}

func (v FeatureValue) I64() int64 {
    return *(*int64)(unsafe.Pointer(&v.raw[0]))
}

func (v FeatureValue) F64() float64 {
    return *(*float64)(unsafe.Pointer(&v.raw[0]))
}

func (v FeatureValue) B() uint8 {
    return v.raw[0]
}

func GraphAggregateComputeFeatures(
    h unsafe.Pointer,
    pivotGraphIDs, pivotVertexIDs []uint32,
    requests []FeatureRequest,
) ([]FeatureValue, error) {

    nPivots := len(pivotGraphIDs)
    nReq := len(requests)
    if len(pivotVertexIDs) != nPivots {
        return nil, fmt.Errorf("pivot id count mismatch")
    }

    out := make([]FeatureValue, nPivots*nReq)

    ret := C.matrixgraph_graph_aggregate_compute_features(
        h,
        (*C.uint32_t)(unsafe.Pointer(&pivotGraphIDs[0])),
        (*C.uint32_t)(unsafe.Pointer(&pivotVertexIDs[0])),
        C.uint32_t(nPivots),
        (*C.MatrixGraphFeatureRequest)(unsafe.Pointer(&requests[0])),
        C.uint32_t(nReq),
        (*C.MatrixGraphFeatureValue)(unsafe.Pointer(&out[0])),
    )
    if ret != 0 {
        return nil, fmt.Errorf("compute_features failed")
    }
    return out, nil
}

// --- usage ---
func GraphAggregateAddSynthetic(h unsafe.Pointer, nVertices, outDeg uint32) error {
    ret := C.matrixgraph_graph_aggregate_add_synthetic(
        h, C.uint32_t(nVertices), C.uint32_t(outDeg))
    if ret != 0 {
        return fmt.Errorf("add_synthetic failed")
    }
    return nil
}

func ExampleGraphAggregate() {
    h := GraphAggregateCreate()
    defer GraphAggregateDestroy(h)

    if err := GraphAggregateLoadSynthetic(h, 100, 3); err != nil {
        panic(err)
    }

    pivots := make([]uint32, 100)
    gids := make([]uint32, 100)
    for i := 0; i < 100; i++ {
        pivots[i] = uint32(i)
    }

    reqs := []FeatureRequest{
        {Prim: int32(AggMean), UseOutgoing: 1},     // "score" mean
        {Prim: int32(AggCount), UseOutgoing: 1},    // count
        {Prim: int32(AggPercentTrue), UseOutgoing: 1}, // percent true
    }
    // attr_name defaults to "score" if first request, "flag" if second, etc.
    copy(reqs[0].AttrName[:], "score")
    copy(reqs[1].AttrName[:], "score")
    copy(reqs[2].AttrName[:], "flag")

    results, err := GraphAggregateComputeFeatures(h, gids, pivots, reqs)
    if err != nil {
        panic(err)
    }

    for v := 0; v < 5; v++ {
        base := v * len(reqs)
        fmt.Printf("V%d: Mean=%.3f Count=%d PctTrue=%.3f\n",
            v, results[base].F64(), results[base+1].I64(), results[base+2].F64())
    }
}

// Multi-graph example
func ExampleGraphAggregateMultiGraph() {
    h := GraphAggregateCreate()
    defer GraphAggregateDestroy(h)

    // Graph 0: 100 vertices, deg 3
    GraphAggregateLoadSynthetic(h, 100, 3)
    // Graph 1: 200 vertices, deg 4
    GraphAggregateAddSynthetic(h, 200, 4)

    var pivots, gids []uint32
    for v := 0; v < 100; v++ {
        gids = append(gids, 0)
        pivots = append(pivots, uint32(v))
    }
    for v := 0; v < 200; v++ {
        gids = append(gids, 1)
        pivots = append(pivots, uint32(v))
    }

    reqs := []FeatureRequest{{Prim: int32(AggMean), UseOutgoing: 1}}
    copy(reqs[0].AttrName[:], "score")

    results, err := GraphAggregateComputeFeatures(h, gids, pivots, reqs)
    if err != nil {
        panic(err)
    }
    fmt.Printf("Graph0 V0 Mean=%.3f, Graph1 V0 Mean=%.3f\n",
        results[0].F64(), results[100].F64())
}

// AllFeatures mirrors the C struct returned by the fused compute-all kernel.
type AllFeatures struct {
    Count                FeatureValue
    CountGreaterThanMean FeatureValue
    NumUnique            FeatureValue
    Sum                  FeatureValue
    Mean                 FeatureValue
    Variance             FeatureValue
    Std                  FeatureValue
    Mode                 FeatureValue
    Min                  FeatureValue
    Max                  FeatureValue
    Median               FeatureValue
    Quarter              FeatureValue
    Quartile3            FeatureValue
    Entropy              FeatureValue
    PercentTrue          FeatureValue
    Skew                 FeatureValue
}

func GraphAggregateComputeAll(
    h unsafe.Pointer,
    pivotGraphIDs, pivotVertexIDs []uint32,
    attrName string,
    useOutgoing bool,
) ([]AllFeatures, error) {

    nPivots := len(pivotGraphIDs)
    if len(pivotVertexIDs) != nPivots {
        return nil, fmt.Errorf("pivot id count mismatch")
    }

    cAttrName := C.CString(attrName)
    defer C.free(unsafe.Pointer(cAttrName))

    out := make([]AllFeatures, nPivots)

    var u8 uint8
    if useOutgoing {
        u8 = 1
    }

    ret := C.matrixgraph_graph_aggregate_compute_all(
        h,
        (*C.uint32_t)(unsafe.Pointer(&pivotGraphIDs[0])),
        (*C.uint32_t)(unsafe.Pointer(&pivotVertexIDs[0])),
        C.uint32_t(nPivots),
        cAttrName,
        C.uint8_t(u8),
        (*C.MatrixGraphAllFeatures)(unsafe.Pointer(&out[0])),
    )
    if ret != 0 {
        return nil, fmt.Errorf("compute_all failed")
    }
    return out, nil
}

func ExampleGraphAggregateComputeAll() {
    h := GraphAggregateCreate()
    defer GraphAggregateDestroy(h)

    GraphAggregateLoadSynthetic(h, 100, 3)

    pivots := make([]uint32, 100)
    gids := make([]uint32, 100)
    for i := 0; i < 100; i++ {
        pivots[i] = uint32(i)
    }

    results, err := GraphAggregateComputeAll(h, gids, pivots, "score", true)
    if err != nil {
        panic(err)
    }

    for v := 0; v < 3; v++ {
        r := results[v]
        fmt.Printf("V%d: count=%d sum=%.3f mean=%.3f std=%.3f median=%.3f skew=%.6f\n",
            v, r.Count.I64(), r.Sum.F64(), r.Mean.F64(), r.Std.F64(), r.Median.F64(), r.Skew.F64())
    }
}
```

---

## ComputeFeatures

`ComputeFeatures` evaluates arbitrary feature-expression plans on a GPU.  It is
more flexible than `GraphAggregate`: it supports nested aggregations,
arithmetic transformations, and conditional navigator filters.

See the dedicated page for the full C API and a Go example:

**[ComputeFeatures Go API →](compute_features.md)**

The high-level workflow is:

1. `matrixgraph_compute_features_create`
2. `matrixgraph_compute_features_load_graph`
3. `matrixgraph_compute_features_load_attributes` (and optionally `load_labels`)
4. `matrixgraph_compute_features_compute`
5. `matrixgraph_compute_features_destroy`

---

## SubIso (GPU WOJ)

SubIso performs **subgraph isomorphism** via the GPU WOJ (Worst-Case Optimal Join) pipeline.

### Input — CSR buffer layout

Both **pattern** and **data graph** are passed as flat byte arrays with this exact `uint32_t` layout:

```
[global_id        * n_vertices]
[in_degree        * n_vertices]
[out_degree       * n_vertices]
[in_offset        * (n_vertices + 1)]
[out_offset       * (n_vertices + 1)]
[incoming_edges   * n_in_edges]
[outgoing_edges   * n_out_edges]
[edges_globalid   * (max_vid + 1)]
[localid          * (max_vid + 1)]
```

`labels` is a separate `uint32_t[n_vertices]` array.

### Output — WOJMatches tables

The join result is a list of 2-D tables. Caller pre-allocates flat buffers:

| Buffer | Size | Content |
|--------|------|---------|
| `out_table_cols` | `[max_result_tables]` | Actual column count per table |
| `out_table_rows` | `[max_result_tables]` | Actual row count per table |
| `out_headers_flat` | `[max_result_tables * max_result_cols]` | Column headers (vertex IDs) |
| `out_data_flat` | `[max_result_tables * max_result_rows * max_result_cols]` | Row-major match data |
| `out_num_tables` | `int*` | Number of valid result tables written |

> If a table exceeds `max_result_rows` or `max_result_cols`, it is **truncated**.

### Go example

```go
package matrixgraph

/*
#cgo LDFLAGS: -L${SRCDIR}/../../../lib -lmatrixgraph_goapi -lcudart
#cgo CFLAGS: -I${SRCDIR}/../../../core/go_api
#include "matrixgraph_go_api.h"
*/
import "C"
import (
    "fmt"
    "unsafe"
)

// BuildCSRBuffer packs the 9 uint32 slices into a single contiguous byte buffer.
func BuildCSRBuffer(
    globalID, inDegree, outDegree []uint32,
    inOffset, outOffset []uint32,
    incomingEdges, outgoingEdges []uint32,
    edgesGlobalID, localID []uint32,
) []byte {
    total := len(globalID) + len(inDegree) + len(outDegree) +
        len(inOffset) + len(outOffset) +
        len(incomingEdges) + len(outgoingEdges) +
        len(edgesGlobalID) + len(localID)
    buf := make([]byte, total*4)
    off := 0
    copyU32 := func(src []uint32) {
        for _, v := range src {
            *(*uint32)(unsafe.Pointer(&buf[off])) = v
            off += 4
        }
    }
    copyU32(globalID)
    copyU32(inDegree)
    copyU32(outDegree)
    copyU32(inOffset)
    copyU32(outOffset)
    copyU32(incomingEdges)
    copyU32(outgoingEdges)
    copyU32(edgesGlobalID)
    copyU32(localID)
    return buf
}

func SubIso(
    pNumVertices, pNumInEdges, pNumOutEdges, pMaxVid, pMinVid uint32,
    pCSR []byte, pLabels []uint32,
    gNumVertices, gNumInEdges, gNumOutEdges, gMaxVid, gMinVid uint32,
    gCSR []byte, gLabels []uint32,
    maxTables, maxRows, maxCols int,
) (numTables int, cols, rows []uint32, headers, data [][]uint32, err error) {

    outCols := make([]uint32, maxTables)
    outRows := make([]uint32, maxTables)
    outHeaders := make([]uint32, maxTables*maxCols)
    outData := make([]uint32, maxTables*maxRows*maxCols)
    outNum := C.int(0)

    ret := C.matrixgraph_subiso(
        C.uint32_t(pNumVertices), C.uint32_t(pNumInEdges), C.uint32_t(pNumOutEdges),
        C.uint32_t(pMaxVid), C.uint32_t(pMinVid),
        (*C.uint8_t)(unsafe.Pointer(&pCSR[0])), C.uint64_t(len(pCSR)),
        (*C.uint32_t)(unsafe.Pointer(&pLabels[0])),
        C.uint32_t(gNumVertices), C.uint32_t(gNumInEdges), C.uint32_t(gNumOutEdges),
        C.uint32_t(gMaxVid), C.uint32_t(gMinVid),
        (*C.uint8_t)(unsafe.Pointer(&gCSR[0])), C.uint64_t(len(gCSR)),
        (*C.uint32_t)(unsafe.Pointer(&gLabels[0])),
        C.int(maxTables), C.int(maxRows), C.int(maxCols),
        (*C.uint32_t)(unsafe.Pointer(&outCols[0])),
        (*C.uint32_t)(unsafe.Pointer(&outRows[0])),
        (*C.uint32_t)(unsafe.Pointer(&outHeaders[0])),
        (*C.uint32_t)(unsafe.Pointer(&outData[0])),
        &outNum,
    )
    if ret != 0 {
        return 0, nil, nil, nil, nil, fmt.Errorf("subiso failed")
    }

    numTables = int(outNum)
    cols = outCols[:numTables]
    rows = outRows[:numTables]

    headers = make([][]uint32, numTables)
    data = make([][]uint32, numTables)
    for t := 0; t < numTables; t++ {
        c := int(cols[t])
        r := int(rows[t])
        headers[t] = make([]uint32, c)
        copy(headers[t], outHeaders[t*maxCols:t*maxCols+c])
        data[t] = make([]uint32, r*c)
        for row := 0; row < r; row++ {
            srcOff := t*maxRows*maxCols + row*maxCols
            dstOff := row * c
            copy(data[t][dstOff:dstOff+c], outData[srcOff:srcOff+c])
        }
    }
    return
}
```

---

## GARMatch

`matrixgraph_gar_match` is currently a **stub** that returns empty output. It accepts serialized graph and pattern arrays. See `core/go_api/matrixgraph_go_api.h` for the full 22-parameter signature if you need to integrate with the ArangoDB GAR pipeline.

---

## Runtime — `LD_LIBRARY_PATH`

When running a Go binary that links `libmatrixgraph_goapi.so`:

```bash
export LD_LIBRARY_PATH=/path/to/MatrixGraph/lib:$LD_LIBRARY_PATH
go run ./cmd/myapp
```

Or copy the `.so` next to your Go binary and use `rpath`:

```bash
#cgo LDFLAGS: -Wl,-rpath,$ORIGIN -L. -lmatrixgraph_goapi -lcudart
```
