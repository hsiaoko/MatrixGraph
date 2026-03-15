# Matrix Operations (GPU)

## Overview

GPU-accelerated matrix operations for graph algorithms. Uses shared-memory tiling and CUDA kernels.

## Source

`core/task/gpu_task/matrix_ops.cuh`

---

## Matmult

Computes `C = A × B` on GPU.

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | `float*` | Matrix A (row-major), shape `(m × k)` |
| `B` | `float*` | Matrix B (row-major), shape `(k × n)` |
| `C` | `float*` | Output (row-major), shape `(m × n)` |
| `m` | `int` | Rows of A and C |
| `n` | `int` | Columns of B and C |
| `k` | `int` | Columns of A / rows of B |

**Example:**

```c++
#include "core/task/gpu_task/matrix_ops.cuh"
using sics::matrixgraph::core::task::MatrixOps;

auto* task = new MatrixOps();
task->Matmult(A, B, C, m, k, n);
delete task;
```

---

## Activate

Applies ReLU activation in-place on a GPU array.

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | `float*` | Array (row-major) |
| `n` | `int` | Number of elements |

**Example:**

```c++
auto* task = new MatrixOps();
task->Activate(A, m * n);
delete task;
```

---

## Usage with Unified Buffer

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
