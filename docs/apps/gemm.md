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

---

<a id="matrixops-c-api"></a>
## MatrixOps (C++ API)

Lower-level GPU **dense** matmul / activation helpers used by the stack (`core/task/gpu_task/matrix_ops.cuh`). This is the **library** surface, not the `gemm_exec` CLI above.

### `Matmult`

Computes `C = A × B` on GPU.

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | `float*` | Matrix A (row-major), shape `(m × k)` |
| `B` | `float*` | Matrix B (row-major), shape `(k × n)` |
| `C` | `float*` | Output (row-major), shape `(m × n)` |
| `m` | `int` | Rows of A and C |
| `n` | `int` | Columns of B and C |
| `k` | `int` | Columns of A / rows of B |

```c++
#include "core/task/gpu_task/matrix_ops.cuh"
using sics::matrixgraph::core::task::MatrixOps;

auto* task = new MatrixOps();
task->Matmult(A, B, C, m, k, n);
delete task;
```

### `Activate`

In-place ReLU on a GPU array.

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | `float*` | Array (row-major) |
| `n` | `int` | Number of elements |

```c++
auto* task = new MatrixOps();
task->Activate(A, m * n);
delete task;
```

### Usage with `UnifiedOwnedBuffer`

```c++
#include "core/data_structures/unified_buffer.cuh"
#include "core/data_structures/host_buffer.cuh"
#include "core/task/gpu_task/matrix_ops.cuh"

using sics::matrixgraph::core::task::MatrixOps;
using UnifiedOwnedBufferFloat =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<float>;

UnifiedOwnedBufferFloat buf_A, buf_B, buf_C;
// ... init buffers ...

auto* task = new MatrixOps();
task->Matmult(buf_A.GetPtr(), buf_B.GetPtr(), buf_C.GetPtr(), m, k, n);
delete task;
```

## See also

- [Applications index](README.md)
