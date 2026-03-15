# Random Graph Generator

## Overview

Python tool for generating vertex-labeled graphs compatible with subgraph matching (e.g. VF3). Output format matches the input expected by `graph_reader.py` and SubIso training.

## Functionality

Generates random graphs with configurable vertex count, edge count, and label range. Output uses the vertex-labeled format.

## Parameters

| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `--num-graphs` | `-W` | `10` | Number of graphs |
| `--vertices` | `-n` | `100` | Vertices per graph |
| `--edges` | `-m` | `200` | Edges per graph |
| `--label-range` | `-r` | `5` | Label values in [0, r-1] |
| `--output-dir` | `-o` | `.` | Output directory (relative or absolute) |
| `--start-graph-id` | `-i` | `1` | Starting graph ID |

## Output Format

- **Header**: `t N M` (N = vertices, M = edges)
- **Vertex**: `v VertexID LabelId Degree`
- **Edge**: `e SrcId DstId`

```
t 5 6
v 0 0 2
v 1 1 3
v 2 2 3
v 3 1 2
v 4 2 2
e 0 1
e 0 2
e 1 2
e 1 3
e 2 4
e 3 4
```

## Source

`tools/python/generate_random_graph.py`

## Example

```bash
python tools/python/generate_random_graph.py \
  --num-graphs 20 --vertices 50 --edges 100 --label-range 3 -o ./output_dir/
```
