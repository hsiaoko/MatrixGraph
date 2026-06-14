# MatrixGraph Documentation

## Overview

MatrixGraph is a C++/CUDA library for parallel graph computing. This directory contains documentation for **tools**, command-line **applications** (`docs/apps/`), and the **Go CGO API** (`docs/go_api/`). The [GEMM](apps/gemm.md) page also documents the `MatrixOps` C++ API (`Matmult` / `Activate`) used outside `gemm_exec`.

For a **single YAML summary** of a CSR dataset (|V|, |E|, degree stats, approximate diameter & skew, WCC count), use **[GraphFeatures](tools/GraphFeatures.md)** (`tools/python/graph_features.py`), which shells out to `wcc_exec`, `diameter_exec`, and `skew_exec`.

---

## Go API (`go_api/`)

CGO bindings for `libmatrixgraph_goapi.so` — matrix ops, GraphAggregate feature computation, ComputeFeatures flexible expression evaluation, and SubIso (GPU WOJ).

**[Go API docs →](go_api/README.md)**

---

## Applications (`apps/`)

Shipped executables are built as `<name>_exec` from `apps/`. Full table, build hints, and per-app flags:

**[Applications index →](apps/README.md)**

| App | Doc |
|-----|-----|
| `wcc_exec` | [WCC](apps/wcc.md) |
| `bfs_exec` | [BFS](apps/bfs.md) |
| `pagerank_exec` | [PageRank](apps/pagerank.md) |
| `diameter_exec` | [Diameter](apps/diameter.md) |
| `skew_exec` | [Skew](apps/skew.md) |
| `gemm_exec` | [GEMM](apps/gemm.md) |
| `subiso_exec` | [SubIso (GPU)](apps/subiso_gpu.md) |
| `cpu_subiso_exec` | [SubIso (CPU)](apps/cpu_subiso.md) |
| `gar_match_exec` | [GARMatch](apps/gar_match.md) |
| `graph_aggregate_exec` | [GraphAggregate](apps/graph_aggregate.md) |
| `compute_features_exec` | [ComputeFeatures](apps/compute_features.md) |
| `execute_agg_prim_exec` | [ExecuteAggPrim](apps/execute_agg_prim.md) |

---

## Tools

| Document | Description |
|----------|--------------|
| [GraphConverter](tools/GraphConverter.md) | Format conversion (CSV, edgelist, CSR, tiled matrix, EGSM, VF3, etc.) |
| [GraphPartitioner](tools/GraphPartitioner.md) | Graph partitioning (GridCut) for tiled processing |
| [FormatConverter](tools/FormatConverter.md) | Internal C++ format conversion utilities |
| [Preprocessing4MatrixFilter](tools/Preprocessing4MatrixFilter.md) | Python preprocessing for ML filter (Rapids→torch, embeddings, training) |
| [GenerateRandomGraph](tools/GenerateRandomGraph.md) | Random vertex-labeled graph generator |
| [SubIsoTraining](tools/SubIsoTraining.md) | ML filter training workflow for SubIso |
| [Embedding](tools/Embedding.md) | Graph embedding generator (PyTorch → binary) |
| [ComputeF1](tools/ComputeF1.md) | F1 / Precision / Recall calculator |
| [GraphFeatures](tools/GraphFeatures.md) | CSR graph summary YAML (|V|, |E|, degrees, diameter, skew, WCC) |
| [ArangoDBImport](tools/ArangoDBImport.md) | ArangoDB setup & import for GAR / `gar_match_exec` |

---

## GPU runtime (environment)

| Document | Description |
|----------|-------------|
| [MATRIXGRAPH_ENV](MATRIXGRAPH_ENV.md) | Optional env vars: CUDA device list (`MATRIXGRAPH_CUDA_DEVICES`), launch dims (`MG_GPU_*`, `MG_SUBISO_*`), WOJ tuning. **All optional**; defaults apply when unset. |

---

## CPU graph metrics (via apps)

[Diameter](apps/diameter.md) and [Skew](apps/skew.md) implement host-side BFS statistics; both accept **`-cpu_parallel`** to cap oneTBB-backed parallelism over sources. See also [GraphFeatures](tools/GraphFeatures.md) to batch-export YAML, and [Applications index — batch benchmarks](apps/README.md#batch-benchmarks-autoconfig-yaml) for **`scripts/run_autoconfig_gpu_exp.py`** (maps YAML **`cpu_cores`** to `-cpu_parallel`).

---

## Tool Chain (Typical Workflows)

**Graph → WCC / BFS / PageRank / Diameter:**

```
CSV → graph_converter (edgelistcsv2edgelistbin) → edgelist
edgelist → graph_converter (edgelistbin2csrbin) → CSR → wcc_exec / bfs_exec / pagerank_exec / diameter_exec / skew_exec
```

(`diameter_exec` / `skew_exec` are CPU-only; by default **approximate d_hat** uses 50 random BFS sources—see [Diameter](apps/diameter.md) and [Skew](apps/skew.md).)

**CSR → feature YAML (batch stats):**

```
CSR directory → python3 tools/python/graph_features.py -g <csr_root> -o features.yaml
```

**Graph → GEMM / PPR (tiled):**

```
CSV → graph_converter (edgelistcsv2edgelistbin) → edgelist
edgelist → graph_partitioner (gridcut) → partitions
partitions → graph_converter (gridedgelistbin2csrtiledmatrix) → tiled → gemm_exec
```

(Personalized PageRank / `PPRQuery` lives in `core/task/gpu_task/ppr_query` but is not built as a separate `apps/` executable; use the library API or add an app wrapper if needed.)

**SubIso ML filter training:**

```
Text graph → graph_reader.py → .pt → data.py → embedding
Ground truth (custom) + embeddings → train.py → model
```
