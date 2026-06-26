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
| `subiso_cpu_exec` | [SubIso (CPU)](subiso_cpu.md) | Subgraph isomorphism (CPU, VF3 / ML filter) |
| `lftj_subiso_cpu_exec` | [LFTJ SubIso](lftj_subiso.md) | Subgraph isomorphism (CPU, LFTJ exact enumeration) |
| `lftj_subiso_exec` | [LFTJ SubIso](lftj_subiso.md) | Same, through MatrixGraph scheduler |
| `gar_match_exec` | [GARMatch](gar_match.md) | Graph association rule matching (GPU, ArangoDB) |
| `graph_aggregate_exec` | [GraphAggregate](graph_aggregate.md) | Per-vertex feature aggregation demo (GPU, synthetic) |
| `execute_agg_prim_exec` | [ExecuteAggPrim](execute_agg_prim.md) | Standalone aggregation primitive harness (GPU, value lists) |

## Build

From the project build directory (after `cmake ..`):

```bash
cmake --build . --target wcc_exec bfs_exec pagerank_exec diameter_exec skew_exec gemm_exec subiso_exec subiso_cpu_exec lftj_subiso_cpu_exec lftj_subiso_exec gar_match_exec
```

Or build all app targets via the `apps` project.

## How to add a new app

The `apps/` directory is auto-discovered by `apps/CMakeLists.txt`:

```cmake
file(GLOB appfiles
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

foreach (appfile ${appfiles})
    get_filename_component(app ${appfile} NAME_WE)
    add_executable("${app}_exec" ${appfile})
    target_link_libraries("${app}_exec" PUBLIC gflags::gflags matrixgraph_core)
    ...
endforeach ()
```

To submit a new app:

1. Place your source file at `apps/<my_app>.cpp` (or `.cu` if it uses CUDA).
2. Link against `gflags::gflags` for CLI flags and `matrixgraph_core` for graph / task utilities.
3. Include `${PROJECT_ROOT_DIR}` and `${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}` if you need CUDA headers.
4. Re-run CMake so the glob picks up the new file:

```bash
cd build && cmake ..
cmake --build . --target <my_app>_exec
```

5. Add a matching doc page at `docs/apps/<my_app>.md` and link it from this `README.md` and from `docs/README.md`.

> Avoid modifying `apps/CMakeLists.txt` for simple additions — the glob already handles new `.cpp` / `.cu` files. Only edit CMake when you need extra link dependencies, custom compile flags, or conditional builds.

## Typical inputs

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
