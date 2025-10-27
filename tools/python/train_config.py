import argparse
import os
import random
import time
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Graph Neural Network Training with Config")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configuration YAML file")
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
        return None  # This should be implemented properly


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

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super(IdentitySAGEConv, self).__init__(in_channels, out_channels, **kwargs)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Custom forward pass with multiple propagation steps"""
        # Ensure input data type matches model precision
        if hasattr(self, 'precision'):
            if self.precision == 'float16':
                x = x.half()
            elif self.precision == 'float64':
                x = x.double()

        aggregated = self.propagate(edge_index, x=x)
        aggregated = aggregated + self.propagate(edge_index, x=aggregated)
        aggregated = aggregated + self.propagate(edge_index, x=aggregated)
        aggregated = aggregated + self.propagate(edge_index, x=aggregated)
        return aggregated


def save_tensor_for_cpp(tensor: torch.Tensor, root_path: str):
    """Save tensor in C++ readable format"""
    assert tensor.is_contiguous(), "Tensor must be contiguous"

    bin_path = os.path.join(root_path, "embedding.bin")
    meta_path = os.path.join(root_path, "meta.yaml")

    # Create directory if it doesn't exist
    os.makedirs(root_path, exist_ok=True)

    # Convert to float32 for saving to ensure compatibility
    if tensor.dtype == torch.float16 or tensor.dtype == torch.float64:
        tensor = tensor.float()

    # Save binary data
    np_array = tensor.detach().cpu().numpy()
    np_array.tofile(bin_path)

    # Save metadata
    metadata = {
        "dtype": "float32",  # Always save as float32 for compatibility
        "x": tensor.shape[0],
        "y": tensor.shape[1],
        "endian": "little",
        "order": "row_major"
    }

    # Save as YAML
    with open(meta_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)


def generate_embedding(input_path: str, output_path: str, precision: str = "float32"):
    """Generate embeddings for a graph"""
    print(f"Data path: {input_path}")
    dataset = torch.load(input_path)

    # Convert data to specified precision
    if precision == "float16":
        dataset.x = dataset.x.half()
    elif precision == "float64":
        dataset.x = dataset.x.double()
    else:
        dataset.x = dataset.x.float()

    print(f"Dataset: {dataset}")
    print(f"X shape: {dataset.x.shape}, dtype: {dataset.x.dtype}")
    print(f"Edge index shape: {dataset.edge_index.shape}")

    conv = IdentitySAGEConv(15, 15, aggr='mean')

    # Set model precision
    if precision == "float16":
        conv = conv.half()
    elif precision == "float64":
        conv = conv.double()
    conv.precision = precision  # Store precision for forward pass

    out2 = conv(dataset.x, dataset.edge_index)

    save_tensor_for_cpp(out2, output_path)
    print(f"Output shape: {out2.shape}, dtype: {out2.dtype}")


def load_gt(gt_path: str) -> np.ndarray:
    """Load ground truth data"""
    print(f"Loading GT: {gt_path}")
    array = algorithms.read_cpp_binary_array(gt_path, 'q')
    return array


def calculate_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate detailed classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Calculate class distribution
    unique, counts = np.unique(y_true, return_counts=True)
    class_distribution = dict(zip(unique, counts))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "class_distribution": class_distribution,
        "total_samples": len(y_true)
    }


def calculate_similarity_numpy(emb1: np.ndarray, emb2: np.ndarray,
                               similarity_method: str = "euclidean",
                               normalize: bool = True,
                               combination_weights: List[float] = None,
                               dtype: torch.dtype = torch.float32) -> np.ndarray:
    """Calculate similarity between two numpy arrays"""
    if combination_weights is None:
        combination_weights = [0.6, 0.2, 0.2]

    # 确保输入是正确类型
    emb1 = emb1.astype(dtype)
    emb2 = emb2.astype(dtype)

    # 计算不同的相似度
    similarities = []

    # 余弦相似度
    if "cosine" in similarity_method or "all" in similarity_method:
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
        similarities.append(cos_sim)

    # 欧几里得相似度
    if "euclidean" in similarity_method or "all" in similarity_method:
        # euc_sim = torch.mean((emb1 - emb2) ** 2)
        euc_sim = (emb1 - emb2) * (emb1 - emb2)
        # euc_sim = 1 / (1 + np.linalg.norm(emb1 - emb2))
        similarities.append(euc_sim)

    # 点积相似度
    if "dot" in similarity_method or "all" in similarity_method:
        dot_sim = np.dot(emb1, emb2)
        similarities.append(dot_sim)

    # 曼哈顿相似度
    if "manhattan" in similarity_method or "all" in similarity_method:
        man_sim = 1 / (1 + np.sum(np.abs(emb1 - emb2)))
        similarities.append(man_sim)

    # 如果指定了特定的相似度方法，只使用那个
    if similarity_method in ["cosine", "euclidean", "dot", "manhattan"]:
        similarities = [similarities[0]]

    # 应用权重组合
    # if len(similarities) == len(combination_weights):
    #    combined = np.average(similarities, weights=combination_weights)
    # else:
    #    combined = np.mean(similarities)

    ## 归一化
    # if normalize:
    #    # 对于单个值，直接返回
    #    combined = dtype(combined)
    # else:
    #    combined = dtype(combined)

    # return np.array([combined], dtype=dtype)
    return similarities


def get_precision_settings(precision: str):
    """Get the corresponding torch and numpy dtypes for the given precision"""
    if precision == "float16":
        return torch.float16, np.float16
    elif precision == "float64":
        return torch.float64, np.float64
    else:  # float32 as default
        return torch.float32, np.float32


def multi_train_gnn_model(config: Dict[str, Any]):
    """Train GNN model with multiple patterns"""
    # 从配置文件中读取精度设置
    precision = config.get('precision', 'float32')
    if precision not in ['float32', 'float16', 'float64']:
        print(f"Warning: Unknown precision '{precision}', defaulting to 'float32'")
        precision = 'float32'

    print(f"Training with precision: {precision}")

    # Set torch and numpy dtypes based on precision
    torch_dtype, np_dtype = get_precision_settings(precision)

    # Get paths from config
    pattern_paths = config['paths']['pattern_paths']
    graph_path = config['paths']['graph_path']
    gt_paths = config['paths']['gt_paths']
    output_dir = config['paths']['output_dir']

    # Create output directories
    model_dirs = {
        'W1': os.path.join(output_dir, "W1/"),
        'W2': os.path.join(output_dir, "W2/"),
        'b1': os.path.join(output_dir, "b1/"),
        'b2': os.path.join(output_dir, "b2/"),
        'graph_emb': os.path.join(output_dir, "graph_emb/")
    }

    # Create pattern embedding directories
    pattern_emb_dirs = []
    for i in range(len(pattern_paths)):
        pattern_emb_dirs.append(os.path.join(output_dir, f"p{i + 1}_emb/"))

    # Create all directories
    os.makedirs(output_dir, exist_ok=True)
    for dir_path in list(model_dirs.values()) + pattern_emb_dirs:
        os.makedirs(dir_path, exist_ok=True)

    # Load datasets and convert to specified precision
    graph_dataset = torch.load(graph_path)
    pattern_datasets = [torch.load(path) for path in pattern_paths]

    # Convert data to specified precision
    if precision == "float16":
        graph_dataset.x = graph_dataset.x.half()
        for dataset in pattern_datasets:
            dataset.x = dataset.x.half()
    elif precision == "float64":
        graph_dataset.x = graph_dataset.x.double()
        for dataset in pattern_datasets:
            dataset.x = dataset.x.double()
    else:
        graph_dataset.x = graph_dataset.x.float()
        for dataset in pattern_datasets:
            dataset.x = dataset.x.float()

    print(f"Loaded {len(pattern_datasets)} pattern datasets")
    print(f"Graph dataset X dtype: {graph_dataset.x.dtype}")
    print(f"Pattern dataset X dtype: {pattern_datasets[0].x.dtype}")

    # Initialize model
    model_config = config['model']
    conv = IdentitySAGEConv(model_config['in_channels'], model_config['out_channels'],
                            aggr='mean')

    # Set model precision
    if precision == "float16":
        conv = conv.half()
    elif precision == "float64":
        conv = conv.double()
    conv.precision = precision  # Store precision for forward pass

    # Generate embeddings
    print("Generating embeddings...")
    pattern_embeddings = [conv(dataset.x, dataset.edge_index) for dataset in pattern_datasets]
    print(pattern_embeddings[0])
    graph_embedding = conv(graph_dataset.x, graph_dataset.edge_index)

    print(f"Pattern embedding dtype: {pattern_embeddings[0].dtype}")
    print(f"Graph embedding dtype: {graph_embedding.dtype}")

    # Load ground truth data
    gt_arrays = [algorithms.read_cpp_binary_array(path, 'q') for path in gt_paths]

    # 获取相似度生成器配置
    generator_config = config['similarity_generator']
    similarity_method = generator_config['similarity_method']
    normalize = generator_config['normalize']
    combination_weights = generator_config['combination_weights']

    x = []
    y = []

    # Create training data from all patterns
    for i, (pattern_embedding, gt_array) in enumerate(zip(pattern_embeddings, gt_arrays)):
        print(f"Processing pattern {i + 1} with {len(gt_array)} ground truth entries")
        gt_array = np.array(gt_array, dtype=np.int64)

        # Convert embeddings to CPU and appropriate numpy dtype for similarity calculation
        pattern_embedding_cpu = pattern_embedding.cpu().numpy().astype(np_dtype)
        graph_embedding_cpu = graph_embedding.cpu().numpy().astype(np_dtype)

        # Positive samples
        for vertex_id in gt_array:
            if vertex_id < len(graph_embedding_cpu):
                # 使用我们自己的相似度计算函数
                similarity = calculate_similarity_numpy(
                    pattern_embedding_cpu[0],
                    graph_embedding_cpu[vertex_id],
                    similarity_method=similarity_method,
                    normalize=normalize,
                    combination_weights=combination_weights,
                    dtype=np_dtype
                )
                x.append(np.array(similarity, dtype=np_dtype))
                y.append(1)

        # Negative samples
        count = 0
        max_negatives = min(len(gt_array), len(graph_embedding_cpu))
        negative_candidates = [v for v in range(len(graph_embedding_cpu)) if v not in gt_array]
        random.shuffle(negative_candidates)

        for vertex_id in negative_candidates[:max_negatives]:
            similarity = calculate_similarity_numpy(
                pattern_embedding_cpu[0],
                graph_embedding_cpu[vertex_id],
                similarity_method=similarity_method,
                normalize=normalize,
                combination_weights=combination_weights,
                dtype=np_dtype
            )
            x.append(np.array(similarity, dtype=np_dtype))
            y.append(0)
            count += 1
            if count >= max_negatives:
                break

    # Convert to numpy arrays
    x = np.array(x, dtype=np_dtype)
    y = np.array(y)

    print(x)
    print(y)

    print(f"Total training samples: {len(x)}")
    print(f"Positive samples: {np.sum(y == 1)}")
    print(f"Negative samples: {np.sum(y == 0)}")
    print(f"X dtype: {x.dtype}, shape: {x.shape}")

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=seed, stratify=y
    )

    print(f"Training set: {len(x_train)} samples")
    print(f"Test set: {len(x_test)} samples")

    # Train model
    input_dim = x.shape[2]
    train_config = config['training']

    # Initialize perceptron with appropriate precision
    # Set model precision
    print("input_dim:", input_dim)
    if precision == "float16":
        perceptron = models.Perceptron(input_dim=input_dim, hidden_dim=config['mlp_model']['hidden_channels'],
                                       dtype=torch.float16)
    elif precision == "float64":
        perceptron = models.Perceptron(input_dim=input_dim, hidden_dim=config['mlp_model']['hidden_channels'],
                                       dtype=torch.float64)
    else:
        perceptron = models.Perceptron(input_dim=input_dim, hidden_dim=config['mlp_model']['hidden_channels'],
                                       dtype=torch.float32)

    print("Starting training...")

    print(train_config)
    start_time = time.time()
    perceptron.train_model(x_train, y_train, learning_rate=train_config['learning_rate'],
                           epochs=train_config['epochs'])
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f" - Time elapsed: {elapsed_time:.4f} seconds")

    # Evaluate model on test set
    print("Evaluating on test set...")
    results = perceptron.predict(x_test)

    # print(results)
    # y_pred = (results > 0.5).int()
    # y_pred = y_pred.cpu().numpy().flatten()

    # Calculate detailed metrics
    detailed_metrics = calculate_detailed_metrics(y_test, results)

    print("\n" + "=" * 50)
    print("TEST SET PERFORMANCE METRICS:")
    print("=" * 50)
    print(f"Accuracy: {detailed_metrics['accuracy']:.4f}")
    print(f"Precision: {detailed_metrics['precision']:.4f}")
    print(f"Recall: {detailed_metrics['recall']:.4f}")
    print(f"F1-Score: {detailed_metrics['f1_score']:.4f}")
    print(f"Total test samples: {detailed_metrics['total_samples']}")
    print(f"Class distribution: {detailed_metrics['class_distribution']}")
    print("=" * 50)

    # Also calculate metrics using the existing function for comparison
    # existing_metrics = algorithms.calculate_metrics(y_test, results)
    # print("Existing metrics function results:")
    # print(existing_metrics)

    # Save model parameters (always save as float32 for compatibility)
    algorithms.save_tensor_for_cpp(torch.tensor(perceptron.fc1.weight.data, dtype=torch.float32), model_dirs['W1'])
    algorithms.save_tensor_for_cpp(torch.tensor(perceptron.fc1.bias.data, dtype=torch.float32), model_dirs['b1'])
    algorithms.save_tensor_for_cpp(torch.tensor(perceptron.fc2.weight.data, dtype=torch.float32), model_dirs['W2'])
    algorithms.save_tensor_for_cpp(torch.tensor(perceptron.fc2.bias.data, dtype=torch.float32), model_dirs['b2'])

    # Save embeddings (convert to float32 for compatibility)
    algorithms.save_tensor_for_cpp(graph_embedding.float(), model_dirs['graph_emb'])
    for i, pattern_embedding in enumerate(pattern_embeddings):
        algorithms.save_tensor_for_cpp(pattern_embedding.float(), pattern_emb_dirs[i])

    # Test with a sample
    print("\n======== SAMPLE TEST ========")
    test_vertex_id = config.get('test_vertex_id', 0)
    if test_vertex_id < len(graph_embedding):
        # 直接使用已经转换的numpy数组
        pattern_emb = pattern_embeddings[0][0].cpu().numpy().astype(np_dtype)
        graph_emb = graph_embedding[test_vertex_id].cpu().numpy().astype(np_dtype)

        similarity = calculate_similarity_numpy(
            pattern_emb,
            graph_emb,
            similarity_method=similarity_method,
            normalize=normalize,
            combination_weights=combination_weights,
            dtype=np_dtype
        )
        print(perceptron.fc1.weight.data)
        print(perceptron.fc1.bias.data)
        print(perceptron.fc2.weight.data)
        print(perceptron.fc2.bias.data)

        print(f"pattern", pattern_emb)
        print(f"graph", graph_emb)
        print(f"sim: ", similarity)

        print(f"Pattern embedding shape: {pattern_embeddings[0][0].shape}")
        print(f"Graph embedding shape: {graph_embedding[test_vertex_id].shape}")
        print(f"Similarity vector length: {len(similarity)}")

        prediction = perceptron.predict(similarity)
        print(f'Sample prediction: {prediction}')
    else:
        print(f"Test vertex ID {test_vertex_id} is out of range")

    return detailed_metrics


def main():
    """Main function"""
    # Parse command line arguments first
    args = parse_arguments()
    config = load_config(args.config)

    # 从配置文件中读取精度设置
    precision = config.get('precision', 'float32')
    print(f"Using precision from config: {precision}")

    start_time = time.time()
    metrics = multi_train_gnn_model(config)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nTraining completed with {precision} precision")
    print(f"Time elapsed: {elapsed_time:.4f} seconds")
    print(f"Final test accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
