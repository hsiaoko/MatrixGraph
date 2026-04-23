# PageRank (`pagerank_exec`)

## Overview

GPU **PageRank** on a CSR graph (`core/task/gpu_task/pagerank.cu`).

## Parameters

| Flag | Description |
|------|-------------|
| `-g` | **Required.** Input graph directory (CSR). |
| `-o` | Output path for PageRank values (passed to the `PageRank` task; supply a valid directory/file as expected by the implementation). |
| `-damping` | Damping factor in `(0, 1)` (default `0.85`). |
| `-epsilon` | Convergence threshold `> 0` (default `1e-6`). |
| `-max_iter` | Maximum iterations `> 0` (default `10`). |
| `-scheduler` | `CHBL` (default), `EvenSplit`, or `RoundRobin`. |

## Example

```bash
./bin/pagerank_exec -g /path/to/csr_graph/ -o /path/to/pagerank_out/
```

## Source

- `apps/pagerank.cpp`
- `core/task/gpu_task/pagerank.cuh`, `pagerank.cu`

## See also

- [Applications index](README.md)
