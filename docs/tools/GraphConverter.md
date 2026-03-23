# Graph Converter

## Overview

Converts between graph formats used by MatrixGraph and external tools. MatrixGraph uses a binary CSR format; this tool supports CSV edge-list, binary edge-list, CSR, tiled matrices, ArangoDB JSON export, and formats for EGSM, VF3, GNN-PE, and CECI.

## Functionality

- CSV edge-list ↔ binary edge-list ↔ binary CSR
- Binary CSR ↔ tiled matrix (for GEMM, PPR)
- CSV edge-list → ArangoDB JSON files
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
| `-graph_id` | | `demo_graph_id` | Graph ID used in ArangoDB JSON export |
| `-business_id` | | `demo_business_id` | Business ID used in ArangoDB JSON export |
| `-pivot_mode` | | `single` | Pivot graph mode for ArangoDB export: `single` or `source` |
| `-default_vertex_label` | | `vertex` | Default vertex label for ArangoDB export |
| `-default_edge_label` | | `relationship` | Default edge label for ArangoDB export |
| `-random_vertex_labels` | | `false` | Randomly assign vertex labels in `[0, label_range)` |

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
| `edgelistcsv2arangodbjson` | CSV edge-list | ArangoDB JSON files |

## Input/Output Formats

**CSV edge-list**: One edge per line, `src,dst` or `src dst` (configurable via `-sep`).

**Binary edge-list**: Directory with `edgelist.bin`, `localid2globalid.bin`, `vlabel.bin`, `meta.yaml`.

**Binary CSR**: Directory with CSR offset/edge arrays and `meta.yaml` (see `core/data_structures/immutable_csr.cuh`).

**Grid-partitioned edge-list**: Output of `graph_partitioner` with `gridcut`; directory of subgraph bins and `meta.yaml`.

**ArangoDB JSON files** (`edgelistcsv2arangodbjson`):
- `graph_structure.json`
- `pivot_graph_ids.jsonl`
- `pivot_graphs.jsonl`
- `README_arangodb_import.txt`

`graph_structure.json` is schema/meta only (labels and label-relations).  
Full vertex/edge instances are in `pivot_graphs.jsonl`.

**What is a pivot?**  
`pivot` is an ArangoDB-export-only grouping anchor (`edgelistcsv2arangodbjson`), used to generate each `pivot_graph_id` entry in `pivot_graphs.jsonl`.

**`-pivot_mode` notes**:
- `single`: export all vertices/edges into one pivot graph (pivot id is `pg_0`).
- `source`: group by edge source vertex, and export one pivot graph per source (pivot id is `pg_<src_id>`).

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

# CSV edge-list → ArangoDB JSON (single pivot graph)
# pivot_mode=single:
#   One pivot graph only (pg_0), containing the whole input graph.
./bin/tools/graph_converter -i graph.csv -o arangodb_out/ \
  -convert_mode edgelistcsv2arangodbjson -sep "," \
  -graph_id demo_graph -business_id demo_biz \
  -pivot_mode single -default_vertex_label person -default_edge_label relate_to

# CSV edge-list → ArangoDB JSON (source-based pivot graphs, random labels)
# pivot_mode=source:
#   Build many pivot graphs, grouped by src vertex in the edge list.
#   Example: edges (1,2), (1,3), (4,5) -> pivot graphs pg_1 and pg_4.
./bin/tools/graph_converter -i graph.csv -o arangodb_out/ \
  -convert_mode edgelistcsv2arangodbjson -sep "," \
  -pivot_mode source -random_vertex_labels=true -label_range 8
```

## Import to ArangoDB

For JSON-to-ArangoDB import scripts and robust line-by-line import workflow, see:

- `docs/tools/ArangoDBImport.md`
