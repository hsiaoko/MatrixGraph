# Random Graph Generator

A Python tool for generating vertex-labeled graphs compatible with graph matching algorithms like VF3.

**Binary Path**: ` $PROJECT_ROOT_DIR/tools/python/generate_random_graph.py`

## File Format Specification

Generated graphs follow the vertex-labeled format:

- **Header**: `t N M` (N = number of vertices, M = number of edges)
- **Vertex**: `v VertexID LabelId Degree`
- **Edge**: `e VertexId VertexId`

**Example:**

```angular2html
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

## Usage

### Command Line Arguments

| Parameter       | Short | Default | Description                           |
|-----------------|-------|---------|---------------------------------------|
| `--num-graphs`  | `-W`  | 10      | Number of graphs to generate          |
| `--vertices`    | `-n`  | 100     | Number of vertices per graph          |
| `--edges`       | `-m`  | 200     | Number of edges per graph             |
| `--label-range` | `-r`  | 5       | Label value range [0, r-1]            |
| `--output-dir`  | `-o`  | `/root` | Output directory for generated graphs |

### Example

```bash
python  $PROJECT_ROOT_DIR/tools/python/generate_random_graph.py --num-graphs 20 --vertices 50 --edges 100 --label-range 3 -o /output_dir/
```