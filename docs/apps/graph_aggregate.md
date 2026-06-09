# GraphAggregate (`graph_aggregate_exec`)

## Overview

GPU **per-vertex feature aggregation** over a synthetic directed ring graph. Each vertex has two synthetic attributes — `"score"` (`float64`) and `"flag"` (`bool`) — and the binary computes aggregated neighbor features for every vertex in parallel.

This app is primarily a **demonstration / test harness** for the `GraphAggregate` task class (`core/task/gpu_task/graph_aggregate.cuh`). It loads data in-memory (no external graph file required) and prints the first 10 vertices’ results.

## Parameters

| Flag | Description |
|------|-------------|
| `-n` | Number of vertices in the synthetic graph (default `100`). |
| `-deg` | Out-degree per vertex (default `3`). Each vertex `v` has outgoing edges to `(v+1) … (v+deg) mod n`. |
| `-prims` | Comma-separated list of aggregation primitives to compute (see table below). Default: `Mean,Sum,Count,PercentTrue`. |
| `-scheduler` | `CHBL` (default), `EvenSplit`, or `RoundRobin`. |

### Supported primitives

| Primitive | Description | Works on attribute |
|-----------|-------------|-------------------|
| `Count` | Number of neighbors | any |
| `Sum` | Sum of neighbor values | `"score"` |
| `Mean` | Arithmetic mean | `"score"` |
| `Min` | Minimum value | `"score"` |
| `Max` | Maximum value | `"score"` |
| `Median` | Median value | `"score"` |
| `Variance` | Population variance | `"score"` |
| `Std` | Standard deviation | `"score"` |
| `PercentTrue` | Fraction of neighbors where `"flag" == true` | `"flag"` |
| `CountGreaterThanMean` | Count of neighbors whose `"score"` > mean | `"score"` |
| `Mode`, `NumUnique`, `Entropy`, `Quarter`, `Quartile3`, `Skew` | Additional statistical aggregates | `"score"` |

> By default every request targets `"score"` **except** `PercentTrue`, which targets `"flag"`.

## Example

### Basic — 100 vertices, out-degree 3, four primitives

```bash
./bin/graph_aggregate_exec -n 100 -deg 3 -prims "Mean,Sum,Count,PercentTrue"
```

Output:
```
=== GraphAggregate Configuration ===
Vertices: 100
Out-degree: 3
Primitives: Mean,Sum,Count,PercentTrue
Scheduler: CHBL
=====================================

Scheduler: CHBL.
[GraphAggregate] Loading synthetic data: 100 vertices, out-degree=3
[GraphAggregate] Synthetic data ready.
[GraphAggregate] Graph data transferred to device (1 graph(s))
Results (first 10 vertices):
  V0: Mean=1 Sum=3 Count=3 PercentTrue=0.333333
  V1: Mean=1.5 Sum=4.5 Count=3 PercentTrue=0.666667
  V2: Mean=2 Sum=6 Count=3 PercentTrue=0.333333
  ...
```

### Extended statistics — 1 000 vertices, out-degree 5

```bash
./bin/graph_aggregate_exec -n 1000 -deg 5 \
  -prims "Mean,Min,Max,Variance,Std,Median,PercentTrue,CountGreaterThanMean"
```

### All available primitives

```bash
./bin/graph_aggregate_exec -n 100 -deg 3 \
  -prims "Count,Sum,Mean,Min,Max,Median,Mode,NumUnique,Entropy,Quarter,Quartile3,PercentTrue,Skew,Variance,Std,CountGreaterThanMean"
```

## How it works

1. **Synthetic graph generation** (`GraphAggregate::LoadSyntheticData`):
   - Builds a directed ring where each vertex `v` points to `(v+1) … (v+deg) mod n`.
   - Fills per-vertex attributes:
     - `"score"` = `v * 0.5`
     - `"flag"`  = `v % 2 == 0`
2. **Device transfer** — CSR buffers and attribute HashMaps are copied to GPU.
3. **Feature kernel** (`ComputeFeaturesKernel`) — one CUDA thread per pivot vertex:
   - Collects neighbor values via CSR outgoing-edge traversal.
   - Applies the requested `AggPrim` and writes a `FeatureValue`.
4. **Host print** — the app copies results back and prints the first 10 vertices.

## Source

- `apps/graph_aggregate.cpp`
- `core/task/gpu_task/graph_aggregate.cuh`, `graph_aggregate.cu`
- `core/task/gpu_task/kernel/kernel_graph_aggregate.cuh`, `kernel_graph_aggregate.cu`

## See also

- [Applications index](README.md)
- [Go API — GraphAggregate](../go_api/README.md#graphaggregate)
