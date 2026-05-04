# Applications (`apps/`)

Executables are built as `<name>_exec` from `apps/<name>.{cpp,cu}` (see `apps/CMakeLists.txt`).

| Binary | Doc | Description |
|--------|-----|--------------|
| `wcc_exec` | [WCC](wcc.md) | Weakly connected components (GPU Hash-Min) |
| `bfs_exec` | [BFS](bfs.md) | Breadth-first traversal from a source (GPU) |
| `pagerank_exec` | [PageRank](pagerank.md) | PageRank on CSR graph (GPU) |
| `diameter_exec` | [Diameter](diameter.md) | Undirected diameter (CPU; default ~50-sample approximate) |
| `skew_exec` | [Skew](skew.md) | skew ≈ d_hat / d_bar (CPU; same d_hat sampling as Diameter) |
| `gemm_exec` | [GEMM](gemm.md) | Tiled graph/matrix pipeline (GPU) |
| `subiso_exec` | [SubIso (GPU)](subiso_gpu.md) | Subgraph isomorphism (GPU) |
| `cpu_subiso_exec` | [SubIso (CPU)](cpu_subiso.md) | Subgraph isomorphism (CPU, VF3 / ML filter) |
| `gar_match_exec` | [GARMatch](gar_match.md) | Graph association rule matching (GPU, ArangoDB) |

## Build

From the project build directory (after `cmake ..`):

```bash
cmake --build . --target wcc_exec bfs_exec pagerank_exec diameter_exec skew_exec gemm_exec subiso_exec cpu_subiso_exec gar_match_exec
```

Or build all app targets via the `apps` project.

## Typical inputs

CSR datasets produced by [GraphConverter](../tools/GraphConverter.md) (`edgelistbin2csrbin` or tiled variants) match what most of these apps expect under `-g` / `-p` paths. See each app’s doc for flags.

## One-shot feature export

To generate a **YAML** with `num_vertices`, `num_edges`, degree **avg/max/min/std**, **diameter**, **skew**, and **WCC components** (calling `wcc_exec`, `diameter_exec`, `skew_exec` for you), use **[GraphFeatures](../tools/GraphFeatures.md)**:

```bash
python3 tools/python/graph_features.py -g /path/to/csr_graph/ -o graph_features.yaml
```

## Batch benchmarks (AutoConfig YAML)

Use **`scripts/run_autoconfig_gpu_exp.py`** for sweeps over AutoConfig GPU YAMLs (`conf_*.yaml` with `grid_size`, `block_size`, **`cpu_cores`**, etc.). Besides **wcc / bfs / pagerank / subiso**, it supports **`diameter`** and **`skew`**. For those CPU apps it passes **`-cpu_parallel=<cpu_cores>`** from each YAML’s `configurations[0].resource.cpu_cores` when that value is positive, so per-conf parallelism matches the sampled catalog. Logs go under **`exp/gpu/<dataset>/conf_xx/`** (`diameter.log`, `skew.log`, …); **`SKIP_GPU`** still runs diameter/skew when they are listed in **`--apps`**.

Example:

```bash
python3 scripts/run_autoconfig_gpu_exp.py \
  --apps diameter,skew \
  --datasets web-sk,livejournal \
  --conf-dir /path/to/AutoConfig/exp/conf/gpu
```

See also [Diameter](diameter.md) and [Skew](skew.md) for **`-cpu_parallel`** when invoking **`diameter_exec`** / **`skew_exec`** manually.
