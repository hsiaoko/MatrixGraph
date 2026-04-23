# Skew (`skew_exec`)

## Overview

Reports **skew** as a dimensionless ratio:

\[
\text{skew}(G) \approx \hat{d}(G) / \overline{d}
\]

- **\(\hat{d}(G)\)** — same as [Diameter](diameter.md): undirected BFS (out + in adjacency), **max eccentricity** over BFS sources. Default: **50 random** sources; `0` = exact (all vertices).
- **\(\overline{d}\)** — mean **total** degree per vertex: \((|E_{\text{out}}| + |E_{\text{in}}|) / n\) from the CSR metadata.

CPU-only, parallel over sources (`std::execution::par`).

## Parameters

| Flag | Description |
|------|-------------|
| `-g` | **Required.** Input graph directory (CSR). |
| `-skew_samples` | Random BFS sources for \(\hat{d}\) (default **50**). **`0` = exact** \(\hat{d}\). |
| `-skew_seed` | RNG seed (default **42**). |
| `-scheduler` | `CHBL` (default), `EvenSplit`, or `RoundRobin`. |

## Example

```bash
./bin/skew_exec -g /path/to/csr_graph/
./bin/skew_exec -g /path/to/csr_graph/ -skew_samples 200 -skew_seed 1
./bin/skew_exec -g /path/to/csr_graph/ -skew_samples 0   # exact d_hat
```

## Source

- `apps/skew.cpp`
- `core/task/cpu_task/skew.h`, `skew.cpp`

## See also

- [Diameter](diameter.md)
- [GraphFeatures](../tools/GraphFeatures.md) — batch YAML export
- [Applications index](README.md)
