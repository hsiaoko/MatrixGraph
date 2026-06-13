# GraphAggregate (`graph_aggregate_exec`)

## Overview

GPU **per-vertex feature aggregation** over a single graph. The app supports two input modes:

1. **Synthetic mode** (default): builds a directed ring with `n` vertices and out-degree `deg`, and fills two synthetic attributes — `"score"` (`float64`) and `"flag"` (`bool`).
2. **Real graph mode** (`-g <csr_dir>`): loads an existing MatrixGraph CSR directory and injects the same synthetic `"score"` / `"flag"` attributes for demonstration.

The binary computes aggregated neighbor features for **every vertex in the graph** in parallel. It is primarily a **demonstration / test harness** for the `GraphAggregate` task class (`core/task/gpu_task/graph_aggregate.cuh`).

## Parameters

| Flag | Description |
|------|-------------|
| `-g` | Path to an existing MatrixGraph CSR directory. When set, `-n` and `-deg` are ignored and all graph vertices become pivots. |
| `-n` | Number of vertices in the synthetic graph (default `100`). Only used when `-g` is empty. |
| `-deg` | Out-degree per vertex (default `3`). Only used when `-g` is empty. |
| `-prims` | Comma-separated list of aggregation primitives to compute (see table below). Default: `Mean,Sum,Count,PercentTrue`. Ignored when `-compute_all` is `true`. |
| `-compute_all` | If `true`, ignore `-prims` and compute **all** primitives in one fused kernel launch (`ComputeAll`). Default: `false`. |
| `-n_streams` | Number of CUDA streams to use per GPU for pivot-batch parallelism. `0` (default) uses `MATRIXGRAPH_CUDA_STREAMS` env or falls back to `2`. |

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

## Multi-GPU

`graph_aggregate_exec` automatically uses the devices selected by `MATRIXGRAPH_CUDA_DEVICES` (or the other MatrixGraph device-selection rules). Each GPU gets a full replica of the graph and attributes; pivots are split evenly across GPUs and further partitioned across streams per GPU.

```bash
# Use GPUs 2 and 3
MATRIXGRAPH_CUDA_DEVICES=2,3 ./bin/graph_aggregate_exec -g data/csr/ -compute_all

# Use all visible GPUs
MATRIXGRAPH_CUDA_ALL_DEVICES=1 ./bin/graph_aggregate_exec -g data/csr/ -compute_all
```

## Examples

### Synthetic graph — 100 vertices, out-degree 3

```bash
./bin/graph_aggregate_exec -n 100 -deg 3 -prims "Mean,Sum,Count,PercentTrue"
```

Output:
```
=== GraphAggregate Configuration ===
Vertices: 100
Out-degree: 3
Primitives: Mean,Sum,Count,PercentTrue
ComputeAll: false
Streams: 0 (0 means use env/default)
=====================================

[GraphAggregate] Adding synthetic graph: 100 vertices, out-degree=3
[GraphAggregate] Graph data replicated to 1 GPU(s) (5208 bytes each)
[GraphAggregate] Loaded 2 attribute column(s) on 1 GPU(s)
Results (first 5 vertices):
  V0: Mean=1 Sum=3 Count=3 PercentTrue=0.333333
  V1: Mean=1.5 Sum=4.5 Count=3 PercentTrue=0.666667
  ...
```

### Real CSR graph

```bash
# Convert CSV to CSR first (see docs/tools/GraphConverter.md)
./bin/tools/graph_converter -i data/graph.csv -o data/csr/ \
  -convert_mode edgelistcsv2csrbin

# Run aggregation on the real graph
./bin/graph_aggregate_exec -g data/csr/ -compute_all
```

### Extended statistics

```bash
./bin/graph_aggregate_exec -n 1000 -deg 5 \
  -prims "Mean,Min,Max,Variance,Std,Median,PercentTrue,CountGreaterThanMean"
```

### Fused compute-all

```bash
./bin/graph_aggregate_exec -n 1000 -deg 5 -compute_all=true
```

This launches the fused kernel (`ComputeAllFeaturesKernel`) that produces all 15 aggregation primitives for every pivot at once. It avoids redundant neighbor collection, mean/variance recalculation, and sorting that would occur when calling `ComputeFeatures` once per primitive.

### Stream parallelism

Use 4 CUDA streams per GPU to process pivot batches concurrently:

```bash
./bin/graph_aggregate_exec -n 100000 -deg 5 -n_streams 4 -compute_all
```

Or via environment variable:

```bash
MATRIXGRAPH_CUDA_STREAMS=4 ./bin/graph_aggregate_exec -n 100000 -deg 5 -compute_all
```

## How it works

1. **Graph loading**:
   - Synthetic mode: `GraphAggregate::LoadSyntheticData` builds a directed ring where each vertex `v` points to `(v+1) … (v+deg) mod n`.
   - Real graph mode: `GraphAggregate::LoadGraph` reads a MatrixGraph CSR directory (`meta.yaml`, `graphs/0.bin`, `label/0.bin`).
2. **Synthetic attributes**: per-vertex `"score" = v * 0.5` and `"flag" = (v % 2 == 0)` are loaded via `GraphAggregate::LoadAttributes`.
3. **Device transfer**: the CSR buffer and per-vertex attribute HashMap are replicated to every selected GPU.
4. **Multi-GPU + stream parallelism**: pivots are split across GPUs, and each GPU's chunk is further split across CUDA streams.
5. **Feature kernels**: one CUDA block per pivot vertex. All 256 threads in the block cooperatively collect neighbor values and compute reductions / sorting in shared memory.
   - `ComputeFeaturesKernel` applies one primitive per request.
   - `ComputeAllFeaturesKernel` computes every primitive in one pass over the neighbor list.
6. **Host print**: the app copies results back and prints the first 5 vertices.

## Source

- `apps/graph_aggregate.cpp`
- `core/task/gpu_task/graph_aggregate.cuh`, `graph_aggregate.cu`
- `core/task/gpu_task/kernel/kernel_graph_aggregate.cuh`, `kernel_graph_aggregate.cu`

## See also

- [Applications index](README.md)
- [Go API — GraphAggregate](../go_api/README.md#graphaggregate)
- [GPU environment variables](../MATRIXGRAPH_ENV.md)
