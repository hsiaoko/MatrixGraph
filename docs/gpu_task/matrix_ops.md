# Matrix Operations

**Source**: `core/task/gpu_task/matrix_ops.cuh`

-------------

## void Matmult(float* A, float* B, float* C, int m, int n, int k);

Optimized matrix multiplication kernel using shared memory tiling for GPU acceleration. Computes `C = A × B` where:

- Input matrix `A`: shape `(m × k)`
- Input matrix `B`: shape `(k × n)`
- Output matrix `C`: shape `(m × n)`

### Parameters Structure

| Parameter | Type     | Description                            |
|-----------|----------|----------------------------------------|
| `A`       | `float*` | pointer to matrix A (row-major)        |
| `B`       | `float*` | pointer to matrix B (row-major)        |
| `C`       | `float*` | pointer to output matrix C (row-major) |
| `m`       | `int`    | Rows of A and C                        |
| `n`       | `int`    | Columns of B and C                     |
| `k`       | `int`    | Columns of A / Rows of B               |

### Input/Output Example
**Matrix A (4×2):**
```c++
[0.0, 1.0,
2.0, 3.0,
4.0, 5.0,
6.0, 7.0]
```
**Matrix B (1×1):**
```c++
[-1.0,
-1.0]
```
**Output C (2×2):**
```c++
[-1.0,
-5.0,
-9.0,
-13.0]
```

### Usage
```c++
// ...
#include "core/data_structures/unified_buffer.cuh"
#include "core/data_structures/host_buffer.cuh"
#include "core/task/gpu_task/matrix_ops.cuh"

using sics::matrixgraph::core::task::MatrixOps;
using UnifiedOwnedBufferFloat =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<float>;
using BufferFloat = sics::matrixgraph::core::data_structures::Buffer<float>;

int main(int argc, char* argv[]) {
    // ...
    
    // Data preparing
    int m = 4;
    int k = 2;
    int n = 1;

    BufferFloat buf_A;
    BufferFloat buf_B;
    BufferFloat buf_C;

    buf_A.data = new float[m * k]();
    buf_A.size = sizeof(float) * m * k;
    buf_B.data = new float[k * n]();
    buf_B.size = sizeof(float) * k * n;
    buf_C.data = new float[m * n]();
    buf_C.size = sizeof(float) * m * n;

    std::cout << "A" << std::endl;
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < k; j++) {
        buf_A.data[i * k + j] = i * k + j;
        std::cout << buf_A.data[i * k + j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "B" << std::endl;

    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        buf_B.data[i * n + j] = -1;
        std::cout << buf_B.data[i * n + j] << " ";
      }
      std::cout << std::endl;
    }

    UnifiedOwnedBufferFloat unified_buf_A;
    UnifiedOwnedBufferFloat unified_buf_B;
    UnifiedOwnedBufferFloat unified_buf_C;

    unified_buf_A.Init(buf_A);
    unified_buf_B.Init(buf_B);
    unified_buf_C.Init(sizeof(float) * m * n);
    
    // Create task
    auto* task = new MatrixOps();     
    task->Matmult(unified_buf_A.GetPtr(), unified_buf_B.GetPtr(),
                  unified_buf_C.GetPtr(), m, k, n);

    // Print output
    std::cout << "C" << m << ", " << n << std::endl;
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        std::cout << unified_buf_C.GetPtr()[i * n + j] << " ";
      }
      std::cout << std::endl;
    }
    delete task;
    // ...
}
```
## void Activate(float* A, int n);
Applies ReLU activation in-place on a GPU array

### Parameters Structure

| Parameter | Type     | Description                     |
|-----------|----------|---------------------------------|
| `A`       | `float*` | pointer to matrix A (row-major) |
| `n`       | `int`    | number of elements              |

### Usage

```c++
// ...

int main(int argc, char* argv[]) {
    // ...
    auto* task = new MatrixOps();
        
    task->Activate(unified_buf_A.GetPtr(), m * n);

    delete task;
    // ...
}
```