# SubIso Training

## Overview

Workflow for training the ML filter used by the CPU subgraph isomorphism (SubIso) application. Produces embeddings and a trained model for filtering candidate vertices.

## Functionality

1. Generate ground truth (binary array of candidate vertex IDs)
2. Convert graphs to PyTorch Geometric (.pt)
3. Generate embeddings from .pt
4. Train similarity model and GNN

## Input Format (graph_reader.py)

- **Header**: `t N M` (N = vertices, M = edges)
- **Vertex**: `v <id> <label> <degree>`
- **Edge**: `e <src> <dst>`

## Tool Chain

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | (custom) | — | Ground truth (uint64 binary) |
| 2 | `graph_reader.py` | Text graph (above format) | PyTorch .pt |
| 3 | `data.py` or `embedding.py` | PyTorch .pt | Embedding binary |
| 4 | `train_config.py` | config.yaml | Trained model |

## Config (config.yaml)

```yaml
paths:
  pattern_paths: ["path/to/pattern1.pt", ...]
  graph_path: "path/to/data_graph.pt"
  gt_paths: ["path/to/gt1", ...]
  output_dir: "path/to/model"

model:
  in_channels: 64
  hidden_channels: 64
  out_channels: 64

training:
  learning_rate: 0.01
  epochs: 10
```

## Source

`tools/python/graph_reader.py`  
`tools/python/data.py`  
`tools/python/embedding.py`  
`tools/python/train_config.py`

## Example

```bash
# Convert graphs to PyTorch
python tools/python/graph_reader.py [input_graph] [output.pt]

# Generate embeddings
python tools/python/data.py [input.pt] [output_embedding]

# Train
python tools/python/train_config.py --config config.yaml
```

## See Also

- [Preprocessing4MatrixFilter.md](Preprocessing4MatrixFilter.md) — preprocessing pipeline
- [subiso.md](../cpu_task/subiso.md) — SubIso application usage
