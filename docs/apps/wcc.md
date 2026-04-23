# WCC (`wcc_exec`)

## Overview

Computes **weakly connected components** on a graph in CSR form using the GPU Hash-Min style propagation (`core/task/gpu_task/wcc.cu`, `kernel_wcc.cu`).

## Parameters

| Flag | Description |
|------|-------------|
| `-g` | **Required.** Root path of the input graph (CSR layout + `meta.yaml`, same as other apps). |
| `-scheduler` | `CHBL` (default), `EvenSplit`, or `RoundRobin`. |

## Example

```bash
./bin/wcc_exec -g /path/to/csr_graph/
```

## Source

- `apps/wcc.cpp`
- `core/task/gpu_task/wcc.cuh`, `core/task/gpu_task/wcc.cu`
- `core/task/gpu_task/kernel/kernel_wcc.cuh`, `kernel_wcc.cu`

## See also

- [Applications index](README.md)
