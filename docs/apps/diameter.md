# Diameter (`diameter_exec`)

## Overview

**Undirected diameter** on the CSR graph: each directed edge is treated as undirected by expanding both **out-** and **in-adjacency** during BFS. The implementation is **CPU**-parallel (`std::execution::par`) (`core/task/cpu_task/diameter.cpp`).

**Default (approximate):** BFS from **50 random** local vertices (without replacement), take the **maximum eccentricity** among them. That value is always **≤** the true undirected diameter (it can underestimate). Use `-diameter_samples 0` for **exact** mode: BFS from every vertex (`O(n · (n + m))`, heavy on large graphs).

## Parameters

| Flag | Description |
|------|-------------|
| `-g` | **Required.** Input graph directory (CSR). |
| `-diameter_samples` | Number of random BFS sources (default **50**). **`0` = exact** (all vertices). |
| `-diameter_seed` | RNG seed for sampling (default **42**). |
| `-scheduler` | Passed through `MatrixGraph` like other apps (`CHBL` default). |

## Example

```bash
# Approximate (default 50 sources)
./bin/diameter_exec -g /path/to/csr_graph/

# More samples
./bin/diameter_exec -g /path/to/csr_graph/ -diameter_samples 200 -diameter_seed 1

# Exact (slow on large graphs)
./bin/diameter_exec -g /path/to/csr_graph/ -diameter_samples 0
```

## Source

- `apps/diameter.cpp`
- `core/task/cpu_task/diameter.h`, `diameter.cpp`

## See also

- [Skew](skew.md) — ratio \(\hat d / \bar d\) using the same \(\hat d\) sampling
- [GraphFeatures](../tools/GraphFeatures.md) — batch YAML export (diameter + skew + WCC + degrees)
- [Applications index](README.md)
