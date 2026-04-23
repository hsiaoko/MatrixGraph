# SubIso — GPU (`subiso_exec`)

## Overview

**Subgraph isomorphism** on the GPU (`core/task/gpu_task/subiso.cu`). Separate from the CPU VF3 / ML pipeline ([SubIso (CPU)](../cpu_task/subiso.md)).

## Parameters

| Flag | Description |
|------|-------------|
| `-p` | **Required.** Pattern graph path (CSR dataset). |
| `-g` | **Required.** Data graph path (CSR dataset). |
| `-e` | Data graph edge-list path (required by task constructor; see implementation for expected format). |
| `-o` | **Required.** Output directory / path for results. |
| `-scheduler` | `CHBL` (default), `EvenSplit`, or `RoundRobin`. |

## Example

```bash
./bin/subiso_exec -p /path/to/pattern_csr/ -g /path/to/data_csr/ \
  -e /path/to/data_edgelist -o /path/to/out/
```

## Source

- `apps/subiso.cu`
- `core/task/gpu_task/subiso.cuh`, `subiso.cu` (and GPU kernels under `core/task/gpu_task/kernel/`)

## See also

- [SubIso (CPU)](../cpu_task/subiso.md) — `cpu_subiso_exec`
- [Applications index](README.md)
