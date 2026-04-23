# WCC (`wcc_exec`)

## Overview

Computes **weakly connected components** on a graph in CSR form using the GPU Hash-Min style propagation (`core/task/gpu_task/wcc.cu`, `kernel_wcc.cu`).

## Parameters

| Flag | Description |
|------|-------------|
| `-g` | **Required.** Root path of the input graph (CSR layout + `meta.yaml`, same as other apps). |
| `-scheduler` | `CHBL` (default), `EvenSplit`, or `RoundRobin`. |

**Environment:** `MATRIXGRAPH_CUDA_DEVICE` — 0-based GPU id (default **0**). Required if you only have one GPU; previously the code targeted device 1 and could **crash (SIGSEGV)**.

## Example

```bash
./bin/wcc_exec -g /path/to/csr_graph/
```

## Source

- `apps/wcc.cpp`
- `core/task/gpu_task/wcc.cuh`, `core/task/gpu_task/wcc.cu`
- `core/task/gpu_task/kernel/kernel_wcc.cuh`, `kernel_wcc.cu`

## See also

- [GraphFeatures](../tools/GraphFeatures.md) — YAML summary including WCC component count
- [Applications index](README.md)
