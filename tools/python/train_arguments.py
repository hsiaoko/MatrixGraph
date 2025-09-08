import argparse
import os
import random
import time
from ctypes import *
from typing import Optional, Tuple, List

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import add_self_loops

import algorithms
import models

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """Parse command line arguments for graph neural network training"""
    parser = argparse.ArgumentParser(description="Graph Neural Network Training")

    # Data paths
    parser.add_argument("--pattern-paths", type=str, nargs='+', required=True,
                       help="Paths to pattern graph files")
    parser.add_argument("--graph-path", type=str, required=True,
                       help="Path to main graph file")
    parser.add_argument("--gt-paths", type=str, nargs='+', required=True,
                       help="Paths to ground truth files")

    # Output paths
    parser.add_argument("--output-dir", type=str, default="./output",
                       help="Output directory for model and embeddings")
    parser.add_argument("--model-prefix", type=str, default="model",
                       help="Prefix for model files")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=10000,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                       help="Learning rate")
    parser.add_argument("--embedding-size", type=int, default=64,
                       help="Embedding size")

    return parser.parse_args()


class CustomGraphDataset(Dataset):
    """Custom graph dataset class"""

    def __init__(self, root: str, transform: Optional[callable] = None,
                 pre_transform: Optional[callable] = None):
        super().__init__(root, transform, pre_transform)

    def len(self) -> int:
        """Return the number of graphs in the dataset"""
        return 100  # Assuming there are 100 graphs in the dataset

    def get(self, idx: int):
        """Get a specific graph by index"""
        return data  # This should be implemented properly


class GATLayer(torch.nn.Module):
    """Graph Attention Network layer"""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 drop_rate: float = 0.0):
        super(GATLayer, self).__init__()
        self.gat1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels
        )
        self.gat2 = GATConv(
            in_channels=hidden_channels,
            out_channels=out_channels
        )
        self.gat3 = GATConv(
            in_channels=out_channels,
            out_channels=out_channels
        )
        self.drop = torch.nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        y = self.gat1(x, e)
        y = self.gat2(y, e)
        y = torch.nn.functional.relu(y)
        y = self.drop(y)
        return y


class GCN(torch.nn.Module):
    """Graph Convolutional Network"""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.conv1(x, edge_index)
        return x


class GraphSAGE(torch.nn.Module):
    """GraphSAGE model"""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean',
                             bias=False, root_weight=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.conv1(x, edge_index)
        return x


class IdentitySAGEConv(SAGEConv):
    """Identity SAGE convolution with custom aggregation"""

    def forward(self, aggregated: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Custom forward pass with multiple propagation steps"""
        # Original aggregation logic with multiple propagations
        aggregated = self.propagate(edge_index, x=aggregated)
        aggregated = aggregated + self.propagate(edge_index, x=aggregated)
        aggregated = aggregated + self.propagate(edge_index, x=aggregated)
        aggregated = aggregated + self.propagate(edge_index, x=aggregated)
        return aggregated


def save_tensor_for_cpp(tensor: torch.Tensor, root_path: str):
    """Save tensor in C++ readable format"""
    assert tensor.is_contiguous(), "Tensor must be contiguous"

    bin_path = os.path.join(root_path, "/embedding.bin")
    meta_path = os.path.join(root_path, "/meta.yaml")

    # Create directory if it doesn't exist
    os.makedirs(root_path, exist_ok=True)

    # Save binary data
    np_array = tensor.detach().numpy()
    np_array.tofile(bin_path)

    # Save metadata
    metadata = {
        "dtype": str(tensor.dtype).split(".")[-1],  # e.g., "float32"
        "x": tensor.shape[0],
        "y": tensor.shape[1],
        "endian": "little",  # or "big"
        "order": "row_major"  # PyTorch default row-major storage
    }

    # Save as YAML
    with open(meta_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)


def generate_embedding(input_path: str, output_path: str):
    """Generate embeddings for a graph"""
    print(f"Data path: {input_path}")
    dataset = torch.load(input_path)

    print(dataset)
    print("X: ", dataset.x)
    print("edge_index: ", dataset.edge_index)

    conv = IdentitySAGEConv(15, 15, aggr='mean')
    out2 = conv(dataset.x, dataset.edge_index)

    save_tensor_for_cpp(out2, output_path)
    print("out2: ", out2)


def load_gt(gt_path: str) -> np.ndarray:
    """Load ground truth data"""
    print(f"Load GT: {gt_path}")
    array = algorithms.read_cpp_binary_array(gt_path, 'q')
    return array


def train_gnn_model(pattern_vertices_embedding_path: str, graph_vertices_embedding_path: str,
                   gt_path: str, output_path: str = ""):
    """Train a GNN model for vertex matching"""
    graph_dataset = torch.load(graph_vertices_embedding_path)
    pattern_dataset = torch.load(pattern_vertices_embedding_path)

    print("Pattern Dataset: ", pattern_dataset)
    print("Graph Dataset: ", graph_dataset)

    conv = IdentitySAGEConv(15, 15, aggr='mean')
    pattern_vertices_embedding = conv(pattern_dataset.x, pattern_dataset.edge_index)
    graph_vertices_embedding = conv(graph_dataset.x, graph_dataset.edge_index)

    print(pattern_vertices_embedding)
    print(graph_vertices_embedding)

    gt_array = algorithms.read_cpp_binary_array(gt_path, 'q')
    generator = models.SimilarityEmbeddingGenerator(
        embedding_size=64,
        similarity_method="euclidean",
        normalize=True,
        combination_weights=[0.6, 0.2, 0.2]  # Weight cosine similarity more
    )

    x = []
    y = []

    gt_array = np.array(gt_array, dtype=np.int64)
    count = 0

    # Positive samples
    for vertex_id in gt_array:
        similarity = generator.generate(pattern_vertices_embedding[0], graph_vertices_embedding[vertex_id])
        x.append(np.array(similarity))
        y.append(1)

    # Negative samples
    for vertex_id in range(len(graph_vertices_embedding)):
        count += 1
        if count > len(gt_array):
            break
        if np.any(gt_array == vertex_id):
            continue
        else:
            similarity = generator.generate(pattern_vertices_embedding[0], graph_vertices_embedding[vertex_id])
            x.append(np.array(similarity))
            y.append(0)

    x = np.array(x)
    y = np.array(y)

    print(f"Dataset size: {len(x)}")
    input_dim = x.shape[1]
    perceptron = models.Perceptron(input_dim=input_dim)
    perceptron.train(x, y, learning_rate=0.01, epochs=10000)

    results = perceptron.predict(x)
    print(results)
    print(algorithms.calculate_metrics(y, results))


def multi_train_gnn_model(args):
    """Train GNN model with multiple patterns"""
    pattern_paths = args.pattern_paths
    graph_path = args.graph_path
    gt_paths = args.gt_paths
    output_dir = args.output_dir

    # Create output directories
    W1_path = os.path.join(output_dir, "W1/")
    W2_path = os.path.join(output_dir, "W2/")
    b1_path = os.path.join(output_dir, "b1/")
    b2_path = os.path.join(output_dir, "b2/")
    gnn_emb_path = os.path.join(output_dir, "gnn_emb/")
    pattern_emb_paths = [os.path.join(output_dir, f"p{i}_emb/") for i in range(len(pattern_paths))]

    os.makedirs(output_dir, exist_ok=True)
    for path in [W1_path, W2_path, b1_path, b2_path, gnn_emb_path] + pattern_emb_paths:
        os.makedirs(path, exist_ok=True)

    # Load datasets
    graph_dataset = torch.load(graph_path)
    pattern_datasets = [torch.load(path) for path in pattern_paths]

    print("Pattern Datasets loaded")
    print("Graph Dataset: ", graph_dataset)

    # Initialize model
    conv = IdentitySAGEConv(args.embedding_size, args.embedding_size, aggr='mean')

    # Generate embeddings
    pattern_embeddings = [conv(dataset.x, dataset.edge_index) for dataset in pattern_datasets]
    graph_embedding = conv(graph_dataset.x, graph_dataset.edge_index)

    # Load ground truth data
    gt_arrays = [algorithms.read_cpp_binary_array(path, 'q') for path in gt_paths]

    # Initialize similarity generator
    generator = models.SimilarityEmbeddingGenerator(
        embedding_size=args.embedding_size,
        similarity_method="euclidean",
        normalize=True,
        combination_weights=[0.6, 0.2, 0.2]
    )

    x = []
    y = []

    # Create training data from all patterns
    for i, (pattern_embedding, gt_array) in enumerate(zip(pattern_embeddings, gt_arrays)):
        gt_array = np.array(gt_array, dtype=np.int64)

        # Positive samples
        for vertex_id in gt_array:
            similarity = generator.generate(pattern_embedding[0], graph_embedding[vertex_id])
            x.append(np.array(similarity))
            y.append(1)

        # Negative samples
        count = 0
        for vertex_id in range(len(graph_embedding)):
            count += 1
            if count > len(gt_array):
                break
            if np.any(gt_array == vertex_id):
                continue
            else:
                similarity = generator.generate(pattern_embedding[0], graph_embedding[vertex_id])
                x.append(np.array(similarity))
                y.append(0)

    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    # Train model
    input_dim = x.shape[1]
    perceptron = models.Perceptron(input_dim=input_dim)
    perceptron.train(x_train, y_train, learning_rate=args.learning_rate, epochs=args.epochs)

    # Evaluate model
    results = perceptron.predict(x_test)
    print("Predictions:", results)
    print("Metrics:", algorithms.calculate_metrics(y_test, results))

    # Save model parameters
    algorithms.save_tensor_for_cpp(torch.tensor(perceptron.W1), W1_path)
    algorithms.save_tensor_for_cpp(torch.tensor(perceptron.b1), b1_path)
    algorithms.save_tensor_for_cpp(torch.tensor(perceptron.W2), W2_path)
    algorithms.save_tensor_for_cpp(torch.tensor(perceptron.b2), b2_path)

    # Save embeddings
    algorithms.save_tensor_for_cpp(graph_embedding, gnn_emb_path)
    for i, pattern_embedding in enumerate(pattern_embeddings):
        algorithms.save_tensor_for_cpp(pattern_embedding, pattern_emb_paths[i])

    # Test with a sample
    print("========TEST=========")
    similarity = generator.generate(pattern_embeddings[0][0], graph_embedding[590])
    print("u0 embedding: ", pattern_embeddings[0][0])
    print("v0 embedding: ", graph_embedding[590])
    print("sim: ", similarity)

    tmp_y = perceptron.predict(similarity)
    print('Prediction:', tmp_y)


if __name__ == "__main__":
    """Main entry point"""
    args = parse_arguments()

    start_time = time.time()
    multi_train_gnn_model(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time:.4f} seconds")
