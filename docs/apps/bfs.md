# BFS (`bfs_exec`)

## Overview

GPU **breadth-first search** from a single source on a CSR graph (`core/task/gpu_task/bfs.cu`).

## Parameters

| Flag | Description |
|------|-------------|
| `-g` | **Required.** Input graph directory (CSR). |
| `-src` | Source vertex **local ID** (default `0`). |
| `-scheduler` | `CHBL` (default), `EvenSplit`, or `RoundRobin`. |

## Example

```bash
./bin/bfs_exec -g /path/to/csr_graph/ -src 0
```

## Source

- `apps/bfs.cpp`
- `core/task/gpu_task/bfs.cuh`, `bfs.cu`
- `core/task/gpu_task/kernel/kernel_bfs.cuh` (and related CUDA sources)

## See also

- [Applications index](README.md)
