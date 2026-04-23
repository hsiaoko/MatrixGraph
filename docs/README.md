# MatrixGraph Documentation

## Overview

MatrixGraph is a C++/CUDA library for parallel graph computing. This directory contains documentation for tools, GPU tasks, CPU tasks, and command-line **applications**.

For a **single YAML summary** of a CSR dataset (|V|, |E|, degree stats, approximate diameter & skew, WCC count), use **[GraphFeatures](tools/GraphFeatures.md)** (`tools/python/graph_features.py`), which shells out to `wcc_exec`, `diameter_exec`, and `skew_exec`.

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
| `cpu_subiso_exec` | [SubIso (CPU)](cpu_task/subiso.md) |
| `gar_match_exec` | [GARMatch](gpu_task/gar_match.md) |

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

## CPU Tasks

| Document | Description |
|----------|--------------|
| [SubIso](cpu_task/subiso.md) | Subgraph isomorphism (VF3 + ML filter) |

**CPU metrics (via apps):** [Diameter](apps/diameter.md) and [Skew](apps/skew.md) implement host-side BFS statistics; see also [GraphFeatures](tools/GraphFeatures.md) to batch-export YAML.

---

## GPU Tasks

| Document | Description |
|----------|--------------|
| [Matrix Operations](gpu_task/matrix_ops.md) | Matmult, Activate (ReLU) |
| [GARMatch](gpu_task/gar_match.md) | Graph association rule matching (ArangoDB + GPU); app: `gar_match_exec` |

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
