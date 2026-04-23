# GEMM (`gemm_exec`)

## Overview

GPU workload built around **grid / tiled CSR matrices** (`core/task/gpu_task/gemm.cu`): loads partitioned graph data, prepares tiled structures, and runs the GEMM-oriented pipeline (see task implementation for details). This is **not** a generic dense `gemm` CLI; inputs are **graph partition directories** as produced by the partitioning + conversion toolchain.

## Parameters

| Flag | Description |
|------|-------------|
| `-i` | Input directory for the primary tiled / grid CSR data. |
| `-it` | Input directory for the **transposed** graph layout. |
| `-o` | Output path. |
| `-count` | Integer repeat / count parameter (default `1`). |
| `-scheduler` | `CHBL` (default), `EvenSplit`, or `RoundRobin`. |

## Typical workflow

1. Partition: [GraphPartitioner](../tools/GraphPartitioner.md)  
2. Convert to tiled CSR: [GraphConverter](../tools/GraphConverter.md) (`gridedgelistbin2csrtiledmatrix`, etc.)  
3. Run `gemm_exec` with `-i` / `-it` / `-o`.

## Example

```bash
./bin/gemm_exec -i /path/to/grid_A -it /path/to/grid_Bt -o /path/to/out -count 1
```

## Source

- `apps/gemm.cpp`
- `core/task/gpu_task/gemm.cuh`, `gemm.cu`

Lower-level CUDA matmul helpers are described in [Matrix Operations](../gpu_task/matrix_ops.md).

## See also

- [Applications index](README.md)
