# SubIso — CPU (`subiso_cpu_exec`)

## Overview

CPU subgraph isomorphism application supporting VF3 and ML-enhanced filtering. Finds all embeddings of a pattern graph in a data graph.

## Functionality

- **VF3 mode**: Classic VF3 algorithm.
- **ML filter mode**: Uses pre-trained MLP to filter candidate vertices before VF3, reducing search space.

## Parameters

| Parameter | Description |
|-----------|-------------|
| `-p` | Pattern graph CSR directory |
| `-g` | Data graph CSR directory |
| `-o` | Output file path for materialized matches; if omitted, only the match count is reported |
| `-m1` | Pattern embedding directory |
| `-m2` | Data graph embedding directory |
| `-m3` | MLP layer 1 weights directory |
| `-m4` | MLP layer 1 bias directory |
| `-m5` | MLP layer 2 weights directory |
| `-m6` | MLP layer 2 bias directory |

## Input Format

**Graphs**: Binary CSR format (see [GraphConverter.md](../tools/GraphConverter.md)). Directory must contain valid CSR files and `meta.yaml`.

**ML model** (optional): Pre-trained MLP weights and biases in separate directories; embeddings from [SubIsoTraining.md](../tools/SubIsoTraining.md) pipeline.

## Output

- **Without `-o`**: count-only mode; only `N Matches` is printed, no files are written.
- **With `-o`**: full matches are materialized and written to the specified directory (format depends on application output configuration).

## Source

`core/task/cpu_task/subiso_cpu.cu`  
`apps/subiso_cpu.cpp`

## Examples

**VF3 only, count-only:**

```bash
./bin/subiso_cpu_exec -p <pattern_csr_dir> -g <data_csr_dir>
```

**VF3 only, materialize matches:**

```bash
./bin/subiso_cpu_exec -p <pattern_csr_dir> -g <data_csr_dir> -o <output_file>
```

**With ML filter:**

```bash
./bin/subiso_cpu_exec \
  -p <pattern_csr_dir> -g <data_csr_dir> \
  -m1 <pattern_embedding_dir> -m2 <data_embedding_dir> \
  -m3 <mlp_w1_dir> -m4 <mlp_b1_dir> \
  -m5 <mlp_w2_dir> -m6 <mlp_b2_dir>
```

## See Also

- [Applications index](README.md) — all `*_exec` binaries
- [SubIso (GPU)](subiso_gpu.md) — `subiso_exec`
- [SubIsoTraining.md](../tools/SubIsoTraining.md) — ML model training
- [GraphConverter.md](../tools/GraphConverter.md) — CSR conversion (`csrbin2vf3`, etc.)
