# Graph Embedding Generator

## Overview

Python tool for generating C++-compatible binary embeddings from PyTorch Geometric graphs. Used in the SubIso ML filter pipeline.

## Functionality

Generates vertex embeddings for each vertex in a graph. Output is a binary array compatible with MatrixGraph C++ code.

## Parameters

| Parameter | Description |
|-----------|-------------|
| `input_graph_data` | Path to PyTorch Geometric graph (.pt) |
| `output_embedding_path` | Path for output binary embedding |

## Input Format

PyTorch Geometric format (from `graph_reader.py` or similar).

## Output Format

Binary array of embedding vectors, one per vertex. C++ compatible.

## Source

`tools/python/embedding.py`

## Example

```bash
python tools/python/embedding.py <input.pt> <output_embedding>
```
