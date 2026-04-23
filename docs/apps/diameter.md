# Diameter (`diameter_exec`)

## Overview

**Exact undirected diameter** on the CSR graph: each directed edge is treated as undirected by expanding both **out-** and **in-adjacency** during BFS. The implementation is **CPU**-parallel (`std::execution::par`), one BFS per source (`core/task/cpu_task/diameter.cpp`).

**Complexity:** `O(n · (n + m))` time and significant memory when many threads run BFS concurrently—practical for small/medium graphs; very large graphs (e.g. millions of vertices) may be slow or memory-heavy.

## Parameters

| Flag | Description |
|------|-------------|
| `-g` | **Required.** Input graph directory (CSR). |
| `-scheduler` | Passed through `MatrixGraph` like other apps (`CHBL` default). |

## Example

```bash
./bin/diameter_exec -g /path/to/csr_graph/
```

## Source

- `apps/diameter.cpp`
- `core/task/cpu_task/diameter.h`, `diameter.cpp`

## See also

- [Applications index](README.md)
