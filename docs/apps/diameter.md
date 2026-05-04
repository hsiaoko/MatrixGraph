# Diameter (`diameter_exec`)

## Overview

**Undirected diameter** on the CSR graph: each directed edge is treated as undirected by expanding both **out-** and **in-adjacency** during BFS. The implementation parallelizes the outer loop over BFS sources with **`std::execution::par`** (GCC routes this through **oneTBB**, already linked by MatrixGraph). You can cap how many workers participate via **`-cpu_parallel`** (implemented with `oneapi::tbb::global_control` in `core/util/cpu_parallel_scope.h`). **`0`** (default) leaves parallelism unconstrained.

**Default (approximate):** BFS from **50 random** local vertices (without replacement), take the **maximum eccentricity** among them. That value is always **≤** the true undirected diameter (it can underestimate). Use `-diameter_samples 0` for **exact** mode: BFS from every vertex (`O(n · (n + m))`, heavy on large graphs).

## Parameters

| Flag | Description |
|------|-------------|
| `-g` | **Required.** Input graph directory (CSR). |
| `-diameter_samples` | Number of random BFS sources (default **50**). **`0` = exact** (all vertices). |
| `-diameter_seed` | RNG seed for sampling (default **42**). |
| `-cpu_parallel` | Max oneTBB workers for the parallel loop over sources (**`0` = default / unlimited**). |
| `-scheduler` | Passed through `MatrixGraph` like other apps (`CHBL` default). |

## Example

```bash
# Approximate (default 50 sources)
./bin/diameter_exec -g /path/to/csr_graph/

# Cap CPU parallelism (e.g. match machine or experiment YAML cpu_cores)
./bin/diameter_exec -g /path/to/csr_graph/ -cpu_parallel=16

# More samples
./bin/diameter_exec -g /path/to/csr_graph/ -diameter_samples 200 -diameter_seed 1

# Exact (slow on large graphs)
./bin/diameter_exec -g /path/to/csr_graph/ -diameter_samples 0
```

## Source

- `apps/diameter.cpp`
- `core/task/cpu_task/diameter.h`, `diameter.cpp`
- `core/util/cpu_parallel_scope.h` — RAII cap on oneTBB parallelism for `-cpu_parallel`

## See also

- [Skew](skew.md) — ratio \(\hat d / \bar d\) using the same \(\hat d\) sampling
- [GraphFeatures](../tools/GraphFeatures.md) — batch YAML export (diameter + skew + WCC + degrees)
- [Applications index](README.md)
