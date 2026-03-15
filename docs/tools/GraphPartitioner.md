# Graph Partitioner

## Overview

Partitions a graph into subgraphs for distributed or tiled processing. Output can be used by `graph_converter` to produce CSR tiled matrices for GEMM, PPR, and similar applications.

## Functionality

- **GridCut**: Grid-based partitioning; produces subgraphs suitable for tiled matrix operations.
- Input: binary edge-list (output of `graph_converter` with `edgelistcsv2edgelistbin` or `edgelistbin2csrbin` → `csrbin2edgelistbin`).

## Parameters

| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `-i` | | (required) | Input path (binary edge-list directory) |
| `-o` | | (required) | Output path for partitions |
| `-partitioner` | | (required) | `gridcut` |
| `-n_partitions` | | `1` | Number of partitions (grid dimension) |
| `-store_strategy` | | `unconstrained` | Edge storage: `incoming_only`, `outgoing_only`, `unconstrained` |
| `-biggraph` | | `false` | Enable optimizations for large graphs |

## Input Format

**Binary edge-list** (from `graph_converter`):

- Directory containing `edgelist.bin`, `localid2globalid.bin`, `vlabel.bin`, `meta.yaml`.

## Output Format

**Partitioned subgraphs**:

- `meta.yaml` — graph metadata and partition info
- `graphs/<gid>.bin` — edge list per partition
- `local_id_to_global_id/<gid>.bin` — vertex ID mapping per partition

This output is the input for `graph_converter` with `gridedgelistbin2csrtiledmatrix` or `gridedgelistbin2bittiledmatrix`.

## Source

`tools/graph_partitioner/graph_partitioner.cu`  
`tools/graph_partitioner/partitioner/grid_cut.cu`

## Example

```bash
# Partition into 4x4 grid
./bin/tools/graph_partitioner -i edgelist/ -o partitions/ \
  -partitioner gridcut -n_partitions 4 -store_strategy unconstrained
```

## Tool Chain

```
CSV edge-list
    → graph_converter (edgelistcsv2edgelistbin)
Binary edge-list
    → graph_partitioner (gridcut)
Partitioned subgraphs
    → graph_converter (gridedgelistbin2csrtiledmatrix)
CSR tiled matrix
    → gemm_exec / ppr_query_exec
```
