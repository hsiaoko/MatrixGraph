# Applications (`apps/`)

Executables are built as `<name>_exec` from `apps/<name>.{cpp,cu}` (see `apps/CMakeLists.txt`).

| Binary | Doc | Description |
|--------|-----|--------------|
| `wcc_exec` | [WCC](wcc.md) | Weakly connected components (GPU Hash-Min) |
| `bfs_exec` | [BFS](bfs.md) | Breadth-first traversal from a source (GPU) |
| `pagerank_exec` | [PageRank](pagerank.md) | PageRank on CSR graph (GPU) |
| `diameter_exec` | [Diameter](diameter.md) | Exact undirected diameter (CPU, parallel BFS) |
| `gemm_exec` | [GEMM](gemm.md) | Tiled graph/matrix pipeline (GPU) |
| `subiso_exec` | [SubIso (GPU)](subiso_gpu.md) | Subgraph isomorphism (GPU) |
| `cpu_subiso_exec` | [SubIso (CPU)](../cpu_task/subiso.md) | Subgraph isomorphism (CPU, VF3 / ML filter) |
| `gar_match_exec` | [GARMatch](../gpu_task/gar_match.md) | Graph association rule matching (GPU, ArangoDB) |

## Build

From the project build directory (after `cmake ..`):

```bash
cmake --build . --target wcc_exec bfs_exec pagerank_exec diameter_exec gemm_exec subiso_exec cpu_subiso_exec gar_match_exec
```

Or build all app targets via the `apps` project.

## Typical inputs

CSR datasets produced by [GraphConverter](../tools/GraphConverter.md) (`edgelistbin2csrbin` or tiled variants) match what most of these apps expect under `-g` / `-p` paths. See each app’s doc for flags.
