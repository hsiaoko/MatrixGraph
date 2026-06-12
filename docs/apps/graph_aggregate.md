# GraphAggregate (`graph_aggregate_exec`)

## Overview

GPU **per-vertex feature aggregation** over one or more synthetic directed ring graphs. Each vertex has two synthetic attributes — `"score"` (`float64`) and `"flag"` (`bool`) — and the binary computes aggregated neighbor features for every vertex in parallel.

This app is primarily a **demonstration / test harness** for the `GraphAggregate` task class (`core/task/gpu_task/graph_aggregate.cuh`). It loads data in-memory (no external graph file required) and prints the first few vertices’ results per graph.

## Parameters

| Flag | Description |
|------|-------------|
| `-n_graphs` | Number of synthetic graphs to create (default `1`). |
| `-n` | Number of vertices per graph. **Comma-separated list** when `-n_graphs > 1`, e.g. `-n "100,200,300"`. A single value is reused for all graphs. |
| `-deg` | Out-degree per vertex per graph. **Comma-separated list** when `-n_graphs > 1`, e.g. `-deg "3,4,5"`. A single value is reused for all graphs. |
| `-prims` | Comma-separated list of aggregation primitives to compute (see table below). Default: `Mean,Sum,Count,PercentTrue`. |
| `-compute_all` | If `true`, ignore `-prims` and compute **all** primitives in one fused kernel launch (`ComputeAll`). Default: `false`. |
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

## Examples

### Single graph — 100 vertices, out-degree 3

```bash
./bin/graph_aggregate_exec -n 100 -deg 3 -prims "Mean,Sum,Count,PercentTrue"
```

Output:
```
=== GraphAggregate Configuration ===
Graphs: 1
Vertices per graph: 100
Out-degree per graph: 3
Primitives: Mean,Sum,Count,PercentTrue
Scheduler: CHBL
=====================================

Scheduler: CHBL.
[GraphAggregate] Adding synthetic graph: 100 vertices, out-degree=3
[GraphAggregate] Synthetic graph 0 ready.
[GraphAggregate] Graph data transferred to device (1 graph(s))
Results (first 5 vertices per graph):
  Graph 0 (|V|=100, deg=3):
    V0: Mean=1 Sum=3 Count=3 PercentTrue=0.333333
    V1: Mean=1.5 Sum=4.5 Count=3 PercentTrue=0.666667
    ...
```

### Multi-graph — 3 graphs with different sizes

```bash
./bin/graph_aggregate_exec -n_graphs 3 -n "10,20,50" -deg "3,4,5" \
  -prims "Mean,Count,Variance"
```

Output shows results for graph 0 (10 vertices, deg 3), graph 1 (20 vertices, deg 4), and graph 2 (50 vertices, deg 5).

### Multi-graph — same size for all graphs

```bash
./bin/graph_aggregate_exec -n_graphs 3 -n 100 -deg 3 \
  -prims "Mean,Min,Max,Std,PercentTrue"
```

The single values `-n 100` and `-deg 3` are automatically reused for all 3 graphs.

### Extended statistics

```bash
./bin/graph_aggregate_exec -n 1000 -deg 5 \
  -prims "Mean,Min,Max,Variance,Std,Median,PercentTrue,CountGreaterThanMean"
```

### Fused compute-all

```bash
./bin/graph_aggregate_exec -n 1000 -deg 5 -compute_all=true
```

This launches a single fused kernel (`ComputeAllFeaturesKernel`) that produces all 15 aggregation primitives for every pivot at once.  It avoids redundant neighbor collection, mean/variance recalculation, and sorting that would occur when calling `ComputeFeatures` once per primitive.

## How it works

1. **Synthetic graph generation** (`GraphAggregate::AddSyntheticGraph`):
   - Builds a directed ring where each vertex `v` points to `(v+1) … (v+deg) mod n`.
   - Fills per-vertex attributes:
     - `"score"` = `v * 0.5`
     - `"flag"`  = `v % 2 == 0`
   - Each call appends a new graph; device buffers are invalidated and re-transferred on the next `ComputeFeatures`.
2. **Device transfer** — CSR buffers and attribute HashMaps for *all* graphs are copied to GPU.
3. **Feature kernel** (`ComputeFeaturesKernel`) — one CUDA thread per pivot vertex:
   - Collects neighbor values via CSR outgoing-edge traversal.
   - Applies the requested `AggPrim` and writes a `FeatureValue`.
4. **Host print** — the app copies results back and prints the first 5 vertices per graph.

## Source

- `apps/graph_aggregate.cpp`
- `core/task/gpu_task/graph_aggregate.cuh`, `graph_aggregate.cu`
- `core/task/gpu_task/kernel/kernel_graph_aggregate.cuh`, `kernel_graph_aggregate.cu`

## See also

- [Applications index](README.md)
- [Go API — GraphAggregate](../go_api/README.md#graphaggregate)
