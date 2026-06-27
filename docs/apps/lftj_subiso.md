# LFTJ SubIso — CPU (`lftj_subiso_cpu_exec`, `lftj_subiso_exec`)

CPU-only subgraph-isomorphism enumerator based on a Leapfrog-Trie-Join
(LFTJ) style depth-first search. It supports exact counting, optional
materialization, a greedy matching order, and several pre-filters
(label-degree / LDF, neighborhood-label-count / NLC, and k-min-wise).

## Source files

`core/task/cpu_task/lftj_subiso.cu`  
`core/task/cpu_task/lftj_subiso.cuh`  
`core/task/cpu_task/min_wise_filter.h`  
`apps/lftj_subiso_cpu.cpp`  
`apps/lftj_subiso.cpp`

## Binaries

| Binary | Description |
|--------|-------------|
| `lftj_subiso_cpu_exec` | Stand-alone CPU executable |
| `lftj_subiso_exec` | Runs through the MatrixGraph `MatrixGraph` scheduler |

## Build

```bash
cd build
cmake --build . --target lftj_subiso_cpu_exec lftj_subiso_exec -j$(nproc)
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `-p` | *required* | Pattern graph CSR directory |
| `-g` | *required* | Data graph CSR directory |
| `-o` | `""` | Output file path; if empty, only the match count is reported |
| `-t` | hardware threads | Number of CPU threads (stand-alone binary) |
| `-limit` | `max` | Stop enumeration after this many matches |
| `-canonical` | false | Enforce strictly increasing data vertices (avoids automorphic duplicates) |
| `-disable_min_wise_filter` | false | Disable the k-min-wise label-hash pre-filter |
| `-filter_hop` | 1 | Hop distance for min-wise neighbor signature |
| `-filter_k` | 3 | Number of minimum hash values kept by k-min-wise filter |
| `-disable_matching_order` | false | Use natural order `0,1,2,...` instead of the greedy matching order |
| `-disable_ldf_filter` | true | Disable label-degree filter (directed out/in degree check). **Default disabled** because LFTJ matches undirected edges; only enable (set to `false`) for symmetric/directed CSR data |
| `-disable_nlc_filter` | false | Disable neighborhood-label-count filter |

## Output

- **Without `-o`**: count-only mode. Prints `Total matches: N` and filter
  statistics, writes no files.
- **With `-o`**: materializes all embeddings (up to `-limit`) to a binary
  file with the following layout:

```text
[uint32_t pn]              // number of pattern vertices
[uint64_t n_matches]       // number of embeddings
[VertexID × pn × n_matches] // row-major embedding table
```

## Filter statistics

At the end of a run the following counters are printed:

```text
=== Filter Counts ===
Label Filters:      N
Degree Filters:     N
LDF Filters:        N
NLC Filters:        N
Min-Wise Filters:   N
Intersection Prune: N
```

- `Label Filters`: data vertices discarded because their label differs from
  the pattern vertex.
- `Degree Filters`: data vertices discarded because their undirected degree
  is too low.
- `LDF Filters`: data vertices discarded because their directed out-degree
  or in-degree is too low (label-degree filter).
- `NLC Filters`: data vertices discarded because they have fewer distinct
  neighbor labels than the pattern vertex (neighborhood-label-count filter).
- `Min-Wise Filters`: data vertices discarded by the k-min-wise
  label-hash pre-filter.
- `Intersection Prune`: candidate vertices removed by backward-neighbor
  intersection during DFS.

## Examples

**Count-only, default settings:**

```bash
./bin/lftj_subiso_cpu_exec \
  -p <pattern_csr_dir>/ -g <data_csr_dir>/ -t 1
```

**Materialize matches to a binary file:**

```bash
./bin/lftj_subiso_cpu_exec \
  -p <pattern_csr_dir>/ -g <data_csr_dir>/ -t 1 \
  -o /tmp/lftj_matches.bin
```

**Disable all pre-filters and use canonical mode:**

```bash
./bin/lftj_subiso_cpu_exec \
  -p <pattern_csr_dir>/ -g <data_csr_dir>/ -t 1 \
  -disable_min_wise_filter -disable_ldf_filter -disable_nlc_filter -canonical
```

**Run through the MatrixGraph scheduler:**

```bash
./bin/lftj_subiso_exec \
  -p <pattern_csr_dir>/ -g <data_csr_dir>/ -t 1
```

## Notes

- The greedy matching order (`-disable_matching_order=false`) usually gives
  the best performance.
- The NLC and k-min-wise filters are enabled by default. LDF is **disabled by
  default** because LFTJ matches undirected edges; it is only sound when the
  CSR graph is symmetric (undirected stored as bidirectional directed edges).
  Use `-disable_ldf_filter=false` to enable LDF for symmetric/directed data.
- All filters add one-time preprocessing cost but can dramatically reduce the
  DFS search space.
- The scheduler version (`lftj_subiso_exec`) has slightly higher overhead
  than the stand-alone binary (`lftj_subiso_cpu_exec`) because it goes
  through the `MatrixGraph` task dispatch path.

## See Also

- [SubIso (CPU)](subiso_cpu.md) — CPU VF3 / ML-filter pipeline
- [SubIso (GPU)](subiso_gpu.md) — GPU WOJ subgraph isomorphism
