# Graph Converter

## Overview

Converts between graph formats used by MatrixGraph and external tools. MatrixGraph uses a binary CSR format; this tool supports CSV edge-list, binary edge-list, CSR, tiled matrices, and formats for EGSM, VF3, GNN-PE, and CECI.

## Functionality

- CSV edge-list ↔ binary edge-list ↔ binary CSR
- Binary CSR ↔ tiled matrix (for GEMM, PPR)
- Binary CSR ↔ external formats (EGSM, VF3, GNN-PE, CECI)
- Grid-partitioned edge-list → CSR tiled matrix

## Parameters

| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `-i` | | (required) | Input path (file or directory) |
| `-o` | | (required) | Output path |
| `-convert_mode` | | (required) | Conversion mode (see below) |
| `-sep` | | `,` | CSV separator |
| `-compressed` | | `false` | Use compressed vertex IDs |
| `-tile_size` | | `64` | Tile size for tiled matrix |
| `-label_range` | | `1` | Label range for vertex labels |

## Conversion Modes

| Mode | Input | Output |
|------|-------|--------|
| `edgelistcsv2edgelistbin` | CSV edge-list | Binary edge-list |
| `edgelistcsv2csrbin` | CSV edge-list | Binary CSR |
| `edgelistbin2csrbin` | Binary edge-list | Binary CSR |
| `edgelistbin2transposededgelistbin` | Binary edge-list | Transposed binary edge-list |
| `csrbin2edgelistbin` | Binary CSR | Binary edge-list |
| `edgelistbin2tiledmatrix` | Binary edge-list | Tiled matrix |
| `edgelistcsv2tiledmatrix` | CSV edge-list | Tiled matrix |
| `edgelistbin2bittiledmatrix` | Binary edge-list | Bit-tiled matrix |
| `edgelistcsv2bittiledmatrix` | CSV edge-list | Bit-tiled matrix |
| `gridedgelistbin2bittiledmatrix` | Grid-partitioned edge-list | Bit-tiled matrix |
| `gridedgelistbin2csrtiledmatrix` | Grid-partitioned edge-list | CSR tiled matrix |
| `csrbin2bittiledmatrix` | Binary CSR | Bit-tiled matrix |
| `edgelistcsv2cggraphcsr` | CSV edge-list | CGGraph CSR |
| `edgelistbin2cggraphcsr` | Binary edge-list | CGGraph CSR |
| `csrbin2egsm` | Binary CSR | EGSM format |
| `egsm2csrbin` | EGSM format | Binary CSR |
| `egsm2edgelistbin` | EGSM format | Binary edge-list |
| `csrbin2vf3` | Binary CSR | VF3 format |
| `csrbin2gnnpe` | Binary CSR | GNN-PE format |
| `csrbin2ceci` | Binary CSR | CECI format |

## Input/Output Formats

**CSV edge-list**: One edge per line, `src,dst` or `src dst` (configurable via `-sep`).

**Binary edge-list**: Directory with `edgelist.bin`, `localid2globalid.bin`, `vlabel.bin`, `meta.yaml`.

**Binary CSR**: Directory with CSR offset/edge arrays and `meta.yaml` (see `core/data_structures/immutable_csr.cuh`).

**Grid-partitioned edge-list**: Output of `graph_partitioner` with `gridcut`; directory of subgraph bins and `meta.yaml`.

## Source

`tools/graph_converter/graph_converter.cu`

## Examples

```bash
# CSV → binary edge-list
./bin/tools/graph_converter -i graph.csv -o edgelist/ \
  -convert_mode edgelistcsv2edgelistbin -sep ","

# Binary edge-list → CSR
./bin/tools/graph_converter -i edgelist/ -o csr/ \
  -convert_mode edgelistbin2csrbin

# Grid-partitioned edge-list → CSR tiled matrix (for GEMM/PPR)
./bin/tools/graph_converter -i partitions/ -o tiled/ \
  -convert_mode gridedgelistbin2csrtiledmatrix -tile_size 64
```
