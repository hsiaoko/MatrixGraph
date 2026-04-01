# GARMatch (Graph Association Rule Match)

## Overview

GPU-accelerated Graph Association Rule (GAR) matching application. Queries graph patterns from an ArangoDB database and finds matching subgraphs using CUDA kernels.

## Functionality

- **GAR Pattern Query**: Loads graph patterns and data graphs from ArangoDB collections
- **GPU-Accelerated Matching**: Uses CUDA kernels to find pattern matches in the data graph
- **Config-Driven**: Reads database connection and query parameters from a YAML configuration file

## Parameters

| Parameter | Description |
|-----------|-------------|
| `-config` | Path to GARMatch YAML config file (required) |
| `-o` | Path for output results (required) |
| `-scheduler` | Scheduler type: `CHBL`, `EvenSplit`, `RoundRobin` (default: `CHBL`) |
| `-device` | CUDA device ID to use (default: 2) |

## Input Format

**Config File**: YAML configuration with the following structure:

```yaml
arango:
  scheme: "http"
  host: "127.0.0.1"
  port: 8529
  database: "_system"
  username: "root"
  password: ""

collections:
  pivot_graphs: "pivot_graphs"

graph_query:
  graph_id: ""        # Optional: filter by graph ID
  business_id: ""     # Optional: filter by business ID
  pivot_limit: 0      # Optional: limit number of pivot graphs (0 = no limit)
```

**Database**: Requires a running ArangoDB instance with a collection containing pivot graphs. Each document in the collection should have:
- `graph_id`: Identifier for the graph
- `business_id`: Business identifier
- `vertices`: Array of vertex objects with `id` and `label` fields
- `edges`: Array of edge objects with `src_id`, `dst_id`, and `label` fields

## Output

GAR match results written to the specified output path. The output contains:
- Matched vertex IDs from the data graph
- Condition information for each match
- Row-based indexing structures for efficient query access

## Source

`core/task/gpu_task/gar_match.cuh`
`core/task/gpu_task/gar_match.cu`
`core/task/gpu_task/kernel/kernel_gar_match.cuh`
`apps/gar_match.cu`

## Examples

**Basic usage:**

```bash
./bin/gar_match_exec -config configs/gar_match.yaml -o results/gar_output.txt
```

**With custom scheduler and device:**

```bash
./bin/gar_match_exec \
  -config configs/gar_match.yaml \
  -o results/gar_output.txt \
  -scheduler EvenSplit \
  -device 0
```

**Example config file:**

```yaml
arango:
  scheme: "http"
  host: "localhost"
  port: 8529
  database: "graph_db"
  username: "admin"
  password: "secret"

collections:
  pivot_graphs: "pivot_graphs"

graph_query:
  graph_id: "graph_001"
  business_id: "biz_123"
  pivot_limit: 100
```

## Dependencies

- **ArangoDB**: Running instance accessible via HTTP API
- **arangosh** (optional): ArangoDB shell for more efficient data extraction (falls back to Python3 if not available)
- **Python3** (optional): Used for data extraction if `arangosh` is not available
- **CUDA**: GPU with compute capability 7.0 or higher

## See Also

- [Matrix Operations](matrix_ops.md) — Other GPU-accelerated matrix operations
- [SubIso](../cpu_task/subiso.md) — CPU-based subgraph isomorphism
