# ExecuteAggPrim (`execute_agg_prim_exec`)

## Overview

`execute_agg_prim_exec` is a small standalone harness for the `ExecuteAggPrim` GPU task class. It accepts **lists of numeric values** (not a graph), computes aggregation primitives on each list, and prints the results.

This app is useful for:

- Testing / benchmarking individual `AggPrim` implementations without loading a full graph.
- Comparing GPU primitive results against featurelib / Python references.
- Verifying batch and multi-primitive parallelism behaviour.

The underlying task (`core/task/gpu_task/execute_agg_prim`) implements the same primitive semantics as `GraphAggregate` and `ComputeFeatures`, but with a **flat value-list input** rather than per-vertex neighbor lists.

## Parameters

| Flag | Description |
|------|-------------|
| `-prim` | Single primitive to compute when using **single-list** mode (e.g. `Sum`, `Mean`, `Median`). |
| `-values` | Comma-separated values for single-list mode. Example: `1,2,3,4,5`. |
| `-lists` | **Batch mode**: multiple value lists separated by `;`. Example: `1,2,3;4,5,6,7;8,9`. |
| `-prims` | **Batch single-primitive mode**: one primitive applied to every list in `-lists`. |
| `-prims_batch` | **Batch multi-primitive mode**: comma-separated primitives applied to **every** list in `-lists`. Output is `[n_lists][n_prims]`. |
| `-compute_all` | If `true`, compute **all** primitives for every list in `-lists`. Default `false`. |
| `-n_streams` | Number of CUDA streams for batch parallelism. `1` (default) means **batch without inter-list stream parallelism**. Set to `>1` to enable concurrent stream processing. |
| `-one_by_one` | If `true`, process batch lists one at a time with single-list `Compute` / `ComputeAll` calls instead of fused batch kernels. Used with `-lists`. |

> Exactly one of `-values` or `-lists` must be supplied. When `-lists` is used, exactly one of `-prims`, `-prims_batch`, or `-compute_all` must select what to compute.

## Supported primitives

| Primitive | Output type | Description |
|-----------|-------------|-------------|
| `Count` | `int64` | Number of values in the list. |
| `Sum` | `double` | Sum of all values. |
| `Mean` | `double` | Arithmetic mean. |
| `Median` | passthrough | Middle value after sorting (average of two middles for even length). |
| `Mode` | passthrough | Most frequent value; first record-breaking value wins when tied. |
| `Max` | passthrough | Maximum value. |
| `Min` | passthrough | Minimum value. |
| `Variance` | `double` | Population variance. |
| `Std` | `double` | Population standard deviation. |
| `Skew` | `double` | Skewness of the distribution. |
| `Entropy` | `double` | Shannon entropy of value frequencies. |
| `NumUnique` | `int64` | Number of distinct values. |
| `PercentTrue` | `double` | Fraction of values interpreted as `true` / non-zero. |
| `Quarter` | `double` | First quartile (25th percentile). |
| `Quartile3` | `double` | Third quartile (75th percentile). |
| `CountGreaterThanMean` | `int64` | Count of values strictly greater than the mean. |
| `DFeat` | passthrough | Identity / pass-through primitive (returns the input value for length-1 lists, otherwise first value). |

> "passthrough" means the result keeps the same internal `FeatureValue` type as the input element (int64 / double / bool).

## Submission modes

The app supports four ways to submit work to `ExecuteAggPrim`:

| Mode | CLI pattern | API pattern | What it does |
|------|-------------|-------------|--------------|
| **Single-list** | `-values` + `-prims` / `-compute_all` | `Compute(...)` / `ComputeAll(...)` | One list, one or all primitives. |
| **Batch non-stream** | `-lists` (default `-n_streams 1`) | `ComputeBatch(...)` / `ComputeBatchMultiPrim(...)` / `ComputeAllBatch(...)` with `SetNumStreams(1)` | All lists submitted as one batch, processed on a single CUDA stream (no inter-list overlap). |
| **Batch stream-parallel** | `-lists` + `-n_streams N` (`N>1`) | Same batch APIs with `SetNumStreams(N)` | The same batch is split into chunks and processed across `N` CUDA streams concurrently. Switching from non-stream to stream is just changing this one parameter. |
| **One-by-one** | `-lists` + `-one_by_one` | Loop calling `Compute(...)` / `ComputeAll(...)` per list | Each list is computed independently with single-list kernels; no batch fusion at all. |

## Examples

### Single list, single primitive

```bash
./bin/execute_agg_prim_exec -prim Mean -values "1,2,3,4,5"
```

Output:

```
Input: 1 2 3 4 5
Mean = 3
```

### Batch mode: one primitive over many lists (non-stream, default)

```bash
./bin/execute_agg_prim_exec -lists "1,2,3;4,5,6,7;8,9" -prims Sum
```

Output:

```
Batch input: 3 list(s), 1 stream(s)
[Timing] ComputeBatch: ... s
  list[0] Sum=6
  list[1] Sum=22
  list[2] Sum=17
```

### Switching the same batch to stream parallelism

The same batch command can be switched to stream mode by only changing `-n_streams`:

```bash
./bin/execute_agg_prim_exec -lists "1,2,3;4,5,6,7;8,9" -prims Sum -n_streams 2
```

Output:

```
Batch input: 3 list(s), 2 stream(s)
[Timing] ComputeBatch: ... s
  list[0] Sum=6
  list[1] Sum=22
  list[2] Sum=17
```

For multi-primitive batches, `-n_streams` works the same way:

```bash
./bin/execute_agg_prim_exec \
  -lists "1,2,3;4,5,6,7;8,9" \
  -prims_batch "Sum,Mean,Count,Min,Max" \
  -n_streams 4
```

Output:

```
Batch input: 3 list(s), 4 stream(s)
[Timing] ComputeBatchMultiPrim: 0.31 s
  list[0] Sum=6 Mean=2 Count=3 Min=1 Max=3
  list[1] Sum=22 Mean=5.5 Count=4 Min=4 Max=7
  list[2] Sum=17 Mean=8.5 Count=2 Min=8 Max=9
```

### One-by-one submission

```bash
./bin/execute_agg_prim_exec \
  -lists "1,2,3;4,5,6,7;8,9" \
  -prims_batch "Sum,Mean,Count,Min,Max" \
  -one_by_one
```

Output:

```
Batch input: 3 list(s), 1 stream(s), one-by-one
[Timing] Compute (one-by-one): ... s
  list[0] Sum=6 Mean=2 Count=3 Min=1 Max=3
  list[1] Sum=22 Mean=5.5 Count=4 Min=4 Max=7
  list[2] Sum=17 Mean=8.5 Count=2 Min=8 Max=9
```

### Compute all primitives for every list

```bash
./bin/execute_agg_prim_exec -lists "1,2,3,4,5;10,20,30" -compute_all -n_streams 2
```

## How it works

1. **Parse input**: the CLI builds `std::vector<FeatureValue>` from `-values`, or `std::vector<std::vector<FeatureValue>>` from `-lists`.
2. **Select API**:
   - Single list → `ExecuteAggPrim::Compute(prim, values, n)` / `ComputeAll(values, n)`.
   - Batch non-stream → `ExecuteAggPrim::ComputeBatch(...)` / `ComputeBatchMultiPrim(...)` / `ComputeAllBatch(...)` with `SetNumStreams(1)`.
   - Batch stream-parallel → same batch APIs with `SetNumStreams(N)` (`N>1`).
   - One-by-one → loop calling `ExecuteAggPrim::Compute(...)` / `ComputeAll(...)` per list (selected by `-one_by_one`).
3. **GPU execution**: each list is assigned to one CUDA block. Within the block, 256 threads cooperatively sort / reduce the values in shared memory, then apply the requested primitives.
4. **Stream parallelism**: batch lists are chunked across `n_streams` CUDA streams, so multiple kernels can overlap on the GPU. With `n_streams == 1` the whole batch runs on a single stream.
5. **Copy back & print**: results are transferred to host and printed.

## Source

- `apps/execute_agg_prim.cpp`
- `core/task/gpu_task/execute_agg_prim.cuh`, `execute_agg_prim.cu`

## See also

- [Applications index](README.md)
- [GraphAggregate](graph_aggregate.md) — graph-input aggregation using the same primitives
- [ComputeFeatures](compute_features.md) — flexible expression evaluation over graph features
- [GPU environment variables](../MATRIXGRAPH_ENV.md)
