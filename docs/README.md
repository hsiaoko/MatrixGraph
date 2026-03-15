# MatrixGraph Documentation

## Overview

MatrixGraph is a C++/CUDA library for parallel graph computing. This directory contains documentation for tools, GPU tasks, and CPU tasks.

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

---

## CPU Tasks

| Document | Description |
|----------|--------------|
| [SubIso](cpu_task/subiso.md) | Subgraph isomorphism (VF3 + ML filter) |

---

## GPU Tasks

| Document | Description |
|----------|--------------|
| [Matrix Operations](gpu_task/matrix_ops.md) | Matmult, Activate (ReLU) |

---

## Tool Chain (Typical Workflows)

**Graph → WCC / BFS / PageRank:**

```
CSV → graph_converter (edgelistcsv2edgelistbin) → edgelist
edgelist → graph_converter (edgelistbin2csrbin) → CSR → wcc_exec / bfs_exec / pagerank_exec
```

**Graph → GEMM / PPR (tiled):**

```
CSV → graph_converter (edgelistcsv2edgelistbin) → edgelist
edgelist → graph_partitioner (gridcut) → partitions
partitions → graph_converter (gridedgelistbin2csrtiledmatrix) → tiled → gemm_exec / ppr_query_exec
```

**SubIso ML filter training:**

```
Text graph → graph_reader.py → .pt → data.py → embedding
Ground truth (custom) + embeddings → train.py → model
```
