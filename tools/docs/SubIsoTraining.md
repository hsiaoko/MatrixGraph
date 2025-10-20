# Subgraph Matching Training: Step-by-Step Guide

This document outlines the workflow for generating ground truth data and training a model for subgraph matching.

---

## Step 1: Generate Ground Truth Data

Prepare the ground truth data that defines the correct matches for a pattern vertex in the data graph.

- **Format:** The data must be a binary array
- **Structure:** `gt = [v_1, v_2, ...]`
  - Each value  `v_i` corresponds to a candidate vertex of pattern vertex `u_0` in the data graph
---

## Step 2: Convert Graphs to PyTorch Geometric Format

Convert both the pattern and data graph from a custom CSR-like text format into a PyTorch Geometric (PyG) `.pt` file.

### Input Format Specification

The input graph file must adhere to the following structure:

- **Header Line:** `t N M`
  - `N`: Total number of vertices
  - `M`: Total number of edges
- **Vertex Lines:** `v <id> <label> <degree>`
  - All vertex lines must be listed consecutively before any edge lines
- **Edge Lines:** `e <source_id> <target_id>`

### Usage

Execute the conversion script from the command line:

```bash
python $PROJECT_ROOT_DIR/tools/python/graph_reader.py [input_path] [output_path]
```

## Step 3: Generate Graph Embeddings
Create vertex embeddings for the pattern and data graph from their PyG format files.

### Usage
Run the embedding script for each graph:

* Output: A binary array A where each element is the embedding vector for a vertex in graph G

```bash
python $PROJECT_ROOT_DIR/tools/python/data.py [input_path] [embedding_path]
```

## Step 4: Configure and Run the Training Process
Configure the experiment parameters and initiate model training.

### Configuration
Create a config.yaml file specifying the following key parameters:
* pattern_paths: List of paths to one or more pattern graph embeddings
* graph_path: Path to the data graph embedding
* gt_paths: List of ground truth paths (must correspond 1-to-1 with pattern_paths)
* output_dir: Directory to save training results and model outputs
* epochs: Number of training epochs

### Execution
Start the training process with the configured settings:

```bash
python $PROJECT_ROOT_DIR/tools/python/train_config.py --config config.yaml
```