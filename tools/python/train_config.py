import argparse
import os
import random
import time
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

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


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_arguments():
<<<<<<< HEAD:tools/python/train_config.py
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Graph Neural Network Training with Config")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration YAML file")
=======
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

>>>>>>> 24f77781b5df53c2eb3e6f4d2aef69947f142770:tools/python/train_arguments.py
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
        self.gat1 = GATConv(in_channels=in_channels, out_channels=hidden_channels)
        self.gat2 = GATConv(in_channels=hidden_channels, out_channels=out_channels)
        self.gat3 = GATConv(in_channels=out_channels, out_channels=out_channels)
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
        aggregated = self.propagate(edge_index, x=aggregated)
        aggregated = aggregated + self.propagate(edge_index, x=aggregated)
        aggregated = aggregated + self.propagate(edge_index, x=aggregated)
        aggregated = aggregated + self.propagate(edge_index, x=aggregated)
        return aggregated


def save_tensor_for_cpp(tensor: torch.Tensor, root_path: str):
    """Save tensor in C++ readable format"""
    assert tensor.is_contiguous(), "Tensor must be contiguous"

<<<<<<< HEAD:tools/python/train_config.py
    bin_path = os.path.join(root_path, "embedding.bin")
    meta_path = os.path.join(root_path, "meta.yaml")
=======
    bin_path = os.path.join(root_path, "/embedding.bin")
    meta_path = os.path.join(root_path, "/meta.yaml")
>>>>>>> 24f77781b5df53c2eb3e6f4d2aef69947f142770:tools/python/train_arguments.py

    # Create directory if it doesn't exist
    os.makedirs(root_path, exist_ok=True)

    # Save binary data
    np_array = tensor.detach().numpy()
    np_array.tofile(bin_path)

    # Save metadata
    metadata = {
        "dtype": str(tensor.dtype).split(".")[-1],
        "x": tensor.shape[0],
        "y": tensor.shape[1],
        "endian": "little",
        "order": "row_major"
    }

    # Save as YAML
    with open(meta_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)


def generate_embedding(input_path: str, output_path: str):
    """Generate embeddings for a graph"""
    print(f"Data path: {input_path}")
    dataset = torch.load(input_path)

    print(f"Dataset: {dataset}")
    print(f"X shape: {dataset.x.shape}")
    print(f"Edge index shape: {dataset.edge_index.shape}")

    conv = IdentitySAGEConv(15, 15, aggr='mean')
    out2 = conv(dataset.x, dataset.edge_index)

    save_tensor_for_cpp(out2, output_path)
    print(f"Output shape: {out2.shape}")


def load_gt(gt_path: str) -> np.ndarray:
    """Load ground truth data"""
    print(f"Loading GT: {gt_path}")
    array = algorithms.read_cpp_binary_array(gt_path, 'q')
    return array


<<<<<<< HEAD:tools/python/train_config.py
def train_gnn_model(pattern_path: str, graph_path: str, gt_path: str,
                   output_dir: str, config: Dict[str, Any]):
    """Train a GNN model for a single pattern"""
    graph_dataset = torch.load(graph_path)
    pattern_dataset = torch.load(pattern_path)
=======
def train_gnn_model(pattern_vertices_embedding_path: str, graph_vertices_embedding_path: str,
                   gt_path: str, output_path: str = ""):
    """Train a GNN model for vertex matching"""
    graph_dataset = torch.load(graph_vertices_embedding_path)
    pattern_dataset = torch.load(pattern_vertices_embedding_path)
>>>>>>> 24f77781b5df53c2eb3e6f4d2aef69947f142770:tools/python/train_arguments.py

    print(f"Pattern Dataset: {pattern_path}")
    print(f"Graph Dataset: {graph_path}")

    # Get model parameters from config
    in_channels = config['model']['in_channels']
    hidden_channels = config['model']['hidden_channels']
    out_channels = config['model']['out_channels']

    conv = IdentitySAGEConv(in_channels, out_channels, aggr='mean')
    pattern_embedding = conv(pattern_dataset.x, pattern_dataset.edge_index)
    graph_embedding = conv(graph_dataset.x, graph_dataset.edge_index)

    gt_array = algorithms.read_cpp_binary_array(gt_path, 'q')

    # Get generator parameters from config
    generator_config = config['similarity_generator']
    generator = models.SimilarityEmbeddingGenerator(
        embedding_size=generator_config['embedding_size'],
        similarity_method=generator_config['similarity_method'],
        normalize=generator_config['normalize'],
        combination_weights=generator_config['combination_weights']
    )

    x = []
    y = []

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

    x = np.array(x)
    y = np.array(y)

    print(f"Training samples: {len(x)}")
    input_dim = x.shape[1]

    # Get training parameters from config
    train_config = config['training']
    perceptron = models.Perceptron(input_dim=input_dim)
    perceptron.train(x, y, learning_rate=train_config['learning_rate'],
                    epochs=train_config['epochs'])

    results = perceptron.predict(x)
    metrics = algorithms.calculate_metrics(y, results)
    print(f"Metrics: {metrics}")

    return perceptron, pattern_embedding, graph_embedding


def multi_train_gnn_model(config: Dict[str, Any]):
    """Train GNN model with multiple patterns"""
<<<<<<< HEAD:tools/python/train_config.py
    # Get paths from config
    pattern_paths = config['paths']['pattern_paths']
    graph_path = config['paths']['graph_path']
    gt_paths = config['paths']['gt_paths']
    output_dir = config['paths']['output_dir']

    # Create output directories
    model_dirs = {
        'W1': os.path.join(output_dir, "W1"),
        'W2': os.path.join(output_dir, "W2"),
        'b1': os.path.join(output_dir, "b1"),
        'b2': os.path.join(output_dir, "b2"),
        'gnn_emb': os.path.join(output_dir, "gnn_emb")
    }

    # Create pattern embedding directories
    pattern_emb_dirs = []
    for i in range(len(pattern_paths)):
        pattern_emb_dirs.append(os.path.join(output_dir, f"p{i+1}_emb"))

    # Create all directories
=======
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

>>>>>>> 24f77781b5df53c2eb3e6f4d2aef69947f142770:tools/python/train_arguments.py
    os.makedirs(output_dir, exist_ok=True)
    for dir_path in list(model_dirs.values()) + pattern_emb_dirs:
        os.makedirs(dir_path, exist_ok=True)

    # Load datasets
    graph_dataset = torch.load(graph_path)
    pattern_datasets = [torch.load(path) for path in pattern_paths]

    print(f"Loaded {len(pattern_datasets)} pattern datasets")
    print(f"Graph dataset: {graph_path}")

    # Initialize model
<<<<<<< HEAD:tools/python/train_config.py
    model_config = config['model']
    conv = IdentitySAGEConv(model_config['in_channels'], model_config['out_channels'],
                           aggr='mean')
=======
    conv = IdentitySAGEConv(args.embedding_size, args.embedding_size, aggr='mean')
>>>>>>> 24f77781b5df53c2eb3e6f4d2aef69947f142770:tools/python/train_arguments.py

    # Generate embeddings
    pattern_embeddings = [conv(dataset.x, dataset.edge_index) for dataset in pattern_datasets]
    graph_embedding = conv(graph_dataset.x, graph_dataset.edge_index)

    # Load ground truth data
    gt_arrays = [algorithms.read_cpp_binary_array(path, 'q') for path in gt_paths]

    # Initialize similarity generator
    generator_config = config['similarity_generator']
    generator = models.SimilarityEmbeddingGenerator(
        embedding_size=generator_config['embedding_size'],
        similarity_method=generator_config['similarity_method'],
        normalize=generator_config['normalize'],
        combination_weights=generator_config['combination_weights']
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
    train_config = config['training']
    perceptron = models.Perceptron(input_dim=input_dim)
    perceptron.train(x_train, y_train, learning_rate=train_config['learning_rate'],
                    epochs=train_config['epochs'])

    # Evaluate model
    results = perceptron.predict(x_test)
    print("Predictions:", results[:10])  # Show first 10 predictions
    metrics = algorithms.calculate_metrics(y_test, results)
    print("Metrics:", metrics)

    # Save model parameters
    algorithms.save_tensor_for_cpp(torch.tensor(perceptron.W1), model_dirs['W1'])
    algorithms.save_tensor_for_cpp(torch.tensor(perceptron.b1), model_dirs['b1'])
    algorithms.save_tensor_for_cpp(torch.tensor(perceptron.W2), model_dirs['W2'])
    algorithms.save_tensor_for_cpp(torch.tensor(perceptron.b2), model_dirs['b2'])

    # Save embeddings
    algorithms.save_tensor_for_cpp(graph_embedding, model_dirs['gnn_emb'])
    for i, pattern_embedding in enumerate(pattern_embeddings):
        algorithms.save_tensor_for_cpp(pattern_embedding, pattern_emb_dirs[i])

    # Test with a sample
    print("========TEST=========")
    test_vertex_id = config.get('test_vertex_id', 590)
    similarity = generator.generate(pattern_embeddings[0][0], graph_embedding[test_vertex_id])
    print(f"Pattern embedding shape: {pattern_embeddings[0][0].shape}")
    print(f"Graph embedding shape: {graph_embedding[test_vertex_id].shape}")
    print(f"Similarity: {similarity}")

    prediction = perceptron.predict(similarity)
    print(f'Prediction: {prediction}')


def main():
    """Main function"""
    args = parse_arguments()
    config = load_config(args.config)

    start_time = time.time()
    multi_train_gnn_model(config)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time:.4f} seconds")


if __name__ == "__main__":
<<<<<<< HEAD:tools/python/train_config.py
    main()
=======
    """Main entry point"""
    args = parse_arguments()

    start_time = time.time()
    multi_train_gnn_model(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time:.4f} seconds")
>>>>>>> 24f77781b5df53c2eb3e6f4d2aef69947f142770:tools/python/train_arguments.py
