# Preprocessing for Matrix Filter

## Overview

Python scripts for preprocessing graphs and training the ML filter used by the CPU subgraph isomorphism (SubIso) application. The pipeline converts graph formats, generates embeddings, and trains similarity models.

## Functionality

1. **Rapids format → PyTorch Geometric (.pt)**
2. **PyTorch .pt → MatrixGraph embedding (binary)**
3. **Ground truth generation** (for training)
4. **Model training** (similarity + GNN)

## Tool Chain

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `graph_reader.py` | Rapids-format graph | PyTorch .pt |
| 2 | `data.py` or `embedding.py` | PyTorch .pt | MatrixGraph embedding (binary) |
| 3 | (custom) | — | Ground truth (uint64 array) |
| 4 | `train.py` | Config + embeddings + ground truth | Trained model |

## Parameters

| Script | Parameters | Description |
|--------|------------|-------------|
| `graph_reader.py` | `[input_path] [output_path]` | Convert Rapids graph to PyTorch |
| `data.py` | `[input_path] [output_path]` | Convert .pt to embedding binary |
| `train.py` | (config) | Train similarity model |

## Source

`tools/python/graph_reader.py`  
`tools/python/data.py`  
`tools/python/embedding.py`  
`tools/python/train.py`

## Examples

```bash
# Step 1: Rapids → PyTorch
python tools/python/graph_reader.py [path-to-rapids-graph] [output.pt]

# Step 2: PyTorch → embedding
python tools/python/data.py [path-to-binary-torch] [output-embedding]

# Step 4: Train (configure config.yaml first)
python tools/python/train.py
```

## Ground Truth

Ground truth must be a 1D binary array of `uint64_t` storing candidate vertex IDs for the first query vertex of the pattern. See [SubIsoTraining.md](SubIsoTraining.md) for details.
