# LFTJ SubIso

MatrixGraph now provides two LFTJ-style subgraph-isomorphism counters:

- `lftj_subiso_exec` — CPU LFTJ (count-only, optional materialization).
- `lftj_subiso_gpu_exec` — single-GPU LFTJ (count-only MVP).

Both reuse the same host-side preprocessing: undirected adjacency,
greedy matching order, and the same set of pre-filters (label-degree /
LDF, neighborhood-label-count / NLC, Bloom label-set, and k-min-wise /
k-min-wise-Bloom).  The GPU version moves the enumeration (per-thread
DFS over the matching order) to a CUDA kernel.

## Source files

`core/task/cpu_task/lftj_subiso.cu`  
`core/task/cpu_task/lftj_subiso.cuh`  
`core/task/cpu_task/min_wise_filter.h`  
`core/task/gpu_task/lftj_subiso_gpu.cu`  
`core/task/gpu_task/lftj_subiso_gpu.cuh`  
`core/task/gpu_task/kernel/kernel_lftj_subiso.cu`  
`core/task/gpu_task/kernel/kernel_lftj_subiso.cuh`  
`apps/lftj_subiso.cpp`  
`apps/lftj_subiso_gpu.cu`

## Binaries

| Binary | Description |
|--------|-------------|
| `lftj_subiso_exec` | CPU LFTJ through the MatrixGraph scheduler |
| `lftj_subiso_gpu_exec` | Single-GPU LFTJ counter through the MatrixGraph scheduler |

The old stand-alone `lftj_subiso_cpu_exec` has been removed; its functionality
(auto thread count, `-reject_output`, `-disable_min_wise_bloom_filter`) was
merged into `lftj_subiso_exec`.

## Build

```bash
cd build
cmake --build . --target lftj_subiso_exec lftj_subiso_gpu_exec -j$(nproc)
```

## Parameters (CPU)

| Flag | Default | Description |
|------|---------|-------------|
| `-p` | *required* | Pattern graph CSR directory |
| `-g` | *required* | Data graph CSR directory |
| `-o` | `""` | Output file path; if empty, only the match count is reported |
| `-reject_output` | `""` | CSV path to write rejected `(u,v)` pairs |
| `-t` | hardware threads | Number of CPU threads (`0` = auto) |
| `-limit` | `max` | Stop enumeration after this many matches |
| `-canonical` | false | Enforce strictly increasing data vertices (avoids automorphic duplicates) |
| `-disable_min_wise_filter` | false | Disable the k-min-wise label-hash pre-filter |
| `-filter_hop` | 1 | Hop distance for min-wise neighbor signature |
| `-filter_k` | 3 | Number of minimum hash values kept by k-min-wise filter |
| `-disable_matching_order` | false | Use natural order `0,1,2,...` instead of the greedy matching order |
| `-disable_ldf_filter` | true | Disable label-degree filter (directed out/in degree check). **Default disabled** because LFTJ matches undirected edges; only enable (set to `false`) for symmetric/directed CSR data |
| `-disable_nlc_filter` | false | Disable neighborhood-label-count filter |
| `-disable_bloom_filter` | false | Disable Bloom neighbor-label-set filter |
| `-disable_min_wise_bloom_filter` | false | Disable k-min-wise Bloom filter |

## Parameters (GPU)

The GPU binary accepts the same filter flags.  `-t` controls the total
number of logical CUDA threads used to split root candidates (`0` = default).

| Flag | Default | Description |
|------|---------|-------------|
| `-p` | *required* | Pattern graph CSR directory |
| `-g` | *required* | Data graph CSR directory |
| `-o` | `""` | Unused in the current count-only GPU implementation |
| `-t` | 0 | Total CUDA threads (`0` = default 256) |
| `-canonical` | false | Enforce strictly increasing data vertices |
| `-disable_min_wise_filter` | false | Disable k-min-wise filter |
| `-filter_hop` | 1 | Hop distance for min-wise neighbor signature |
| `-filter_k` | 3 | Number of minimum hash values kept |
| `-disable_matching_order` | false | Use natural matching order |
| `-disable_ldf_filter` | true | Disable label-degree filter |
| `-disable_nlc_filter` | false | Disable NLC filter |
| `-disable_bloom_filter` | false | Disable Bloom filter |
| `-disable_min_wise_bloom_filter` | false | Disable k-min-wise Bloom filter |

## Output

- **CPU without `-o`**: count-only mode. Prints `Total matches: N` and filter
  statistics.
- **CPU with `-o`**: materializes all embeddings (up to `-limit`) to a binary
  file:

```text
[uint32_t pn]              // number of pattern vertices
[uint64_t n_matches]       // number of embeddings
[VertexID × pn × n_matches] // row-major embedding table
```

- **GPU**: count-only. Prints `Total matches: N` and timing breakdown.

## Examples

**CPU count-only, default settings:**

```bash
./bin/lftj_subiso_exec \
  -p <pattern_csr_dir>/ -g <data_csr_dir>/ -t 0
```

**CPU materialize matches to a binary file:**

```bash
./bin/lftj_subiso_exec \
  -p <pattern_csr_dir>/ -g <data_csr_dir>/ -t 0 \
  -o /tmp/lftj_matches.bin
```

**CPU disable all pre-filters and use canonical mode:**

```bash
./bin/lftj_subiso_exec \
  -p <pattern_csr_dir>/ -g <data_csr_dir>/ -t 0 \
  -disable_min_wise_filter -disable_ldf_filter -disable_nlc_filter -canonical
```

**GPU LFTJ with default filters:**

```bash
./bin/lftj_subiso_gpu_exec \
  -p <pattern_csr_dir>/ -g <data_csr_dir>/ -t 512
```

**GPU LFTJ label-only baseline:**

```bash
./bin/lftj_subiso_gpu_exec \
  -p <pattern_csr_dir>/ -g <data_csr_dir>/ -t 512 \
  -disable_min_wise_filter -disable_nlc_filter \
  -disable_bloom_filter -disable_min_wise_bloom_filter
```

## Notes

- The greedy matching order (`-disable_matching_order=false`) usually gives
  the best performance.
- The NLC, Bloom, k-min-wise, and k-min-wise-Bloom filters are enabled by
  default. LDF is **disabled by default** because LFTJ matches undirected
  edges; it is only sound when the CSR graph is symmetric (undirected stored
  as bidirectional directed edges). Use `-disable_ldf_filter=false` to enable
  LDF for symmetric/directed data.
- The GPU version is a count-only MVP. It copies candidate sets and the
  matching plan to the device and runs per-thread DFS enumeration.  It does
  not materialize embeddings.
- GPU match counts are identical to the CPU version when the same filter
  flags are used.

## See Also

- [SubIso (CPU)](subiso_cpu.md) — CPU VF3 / ML-filter pipeline
- [SubIso (GPU)](subiso_gpu.md) — GPU WOJ subgraph isomorphism
