# GraphFeatures (`tools/python/graph_features.py`)

## Overview

Summarizes a **CSR directory** (same layout as `ImmutableCSR::Read`: `meta.yaml`, `graphs/0.bin`, `label/0.bin`) into a single **YAML** file with basic counts, degree statistics, approximate **diameter**, **skew** (from `skew_exec`), and **WCC component count** (from `wcc_exec`).

Degree **avg / max / min / std** are computed in Python from the **total degree** per local vertex (`indegree + outdegree`) read out of `graphs/0.bin`.  
**`basic.num_edges`** is **`num_outgoing_edges`** from `meta.yaml` (typically the stored directed edge count `|E|` for one orientation).

## Requirements

- Python **3.8+**
- **PyYAML** (`pip install pyyaml`)
- **NumPy** optional (speeds up degree stats on large graphs)
- Built MatrixGraph apps under `bin/`: **`wcc_exec`**, **`diameter_exec`**, **`skew_exec`**

### GPU device for `wcc_exec`

`wcc_exec` (and BFS / PageRank / SubIso) use **`MATRIXGRAPH_CUDA_DEVICE`** (default **0**). Older code used GPU 1 only; on a **single-GPU** machine that caused **SIGSEGV** (Python often reports exit code **-11**). If you need another card, run e.g. `export MATRIXGRAPH_CUDA_DEVICE=1` before `graph_features.py`.

## Usage

```bash
# YAML to stdout
python3 tools/python/graph_features.py -g /path/to/csr_graph/

# Write file
python3 tools/python/graph_features.py -g /path/to/csr_graph/ -o features.yaml

# Custom bin directory and sampling (forwarded to diameter/skew apps)
python3 tools/python/graph_features.py -g /path/to/csr_graph/ \
  --bin-dir /path/to/bin \
  --diameter-samples 50 --skew-samples 50 --seed 42

# Exact diameter / d_hat (very slow on large graphs)
python3 tools/python/graph_features.py -g /path/to/csr_graph/ \
  --diameter-samples 0 --skew-samples 0
```

## Output shape

```yaml
graph_features:
  basic:
    num_edges: ...
    num_vertices: ...
    num_components: ...
  degree:
    avg: ...
    max: ...
    min: ...
    skew: ...    # skew(G) ≈ d_hat / d_bar from skew_exec
    std: ...     # population std of total degree (ddof=0)
  diameter: ... # from diameter_exec (approximate unless --diameter-samples 0)
```

## See also

- [Applications index](../apps/README.md) — all `*_exec` binaries
- [GraphConverter](GraphConverter.md) — produce CSR from edge list
- [Diameter](../apps/diameter.md), [Skew](../apps/skew.md), [WCC](../apps/wcc.md)
