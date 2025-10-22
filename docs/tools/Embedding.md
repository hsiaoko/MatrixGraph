# Graph Embedding Generator

A Python tool for generating C++ compatible arrays that store graph embeddings for each vertex.
It takes PyTorch Geometric format graphs as input and outputs C++ arrays with metadata.

**Binary Path**: `$PROJECT_ROOT_DIR/tools/python/embedding.py`

## Usage

```bash
python  $PROJECT_ROOT_DIR/tools/python/graph_embedding.py <input_graph_data> <output_embedding_path>
```