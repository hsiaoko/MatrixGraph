import argparse
import os
import random
from ctypes import *
from typing import Optional

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
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Graph Neural Network Training Arguments")
    parser.add_argument("--data-root", type=str, required=True,
                       help="Root directory path for the dataset")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    return parser.parse_args()


class CustomGraphDataset(Dataset):
    """Custom graph dataset class"""

    def __init__(self, root: str, transform: Optional[callable] = None,
                 pre_transform: Optional[callable] = None):
        super().__init__(root, transform, pre_transform)

    def len(self) -> int:
        """Return the number of graphs in the dataset"""
        return 100  # Assuming there are 100 graphs in the dataset


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()

    # Use parameterized file paths
    data_root = args.data_root
    config_path = args.config

    # Example: Create dataset instance
    dataset = CustomGraphDataset(root=data_root)

    # Add data loading and training logic here
    print(f"Dataset path: {data_root}")
    print(f"Config file path: {config_path}")


if __name__ == "__main__":
    main()
