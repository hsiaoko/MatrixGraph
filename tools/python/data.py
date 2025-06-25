import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset
import os
import sys
from torch_geometric.nn import SAGEConv
import numpy as np
import random
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import yaml
from ctypes import *
import algorithms
import models
import numpy as np

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class CustomGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    def len(self):
        # 返回数据集中的图的数量
        return 100  # 假设数据集中有 100 个图

    def get(self, idx):
        return data


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean', bias=False, root_weight=False)
        # self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)

        # x = torch.relu(x)
        # x = self.conv2(x, edge_index)
        return x


class IdentitySAGEConv(SAGEConv):
    def forward(self, x, edge_index):
        # 原始聚合逻辑
        aggregated = self.propagate(edge_index, x=x)
        aggregated = aggregated + self.propagate(edge_index, x=aggregated)

        # 替换权重乘法为恒等操作
        # if self.root_weight:
        #    out = torch.cat([x, aggregated], dim=-1)
        # else:
        #    out = aggregated
        out = aggregated

        # 模拟 W=1 的效果（若需保持输出维度，需确保 in_channels == out_channels）
        return out


def generate_embedding(input_path, output_path):
    # dataset = CustomGraphDataset(root='./custom_data', raw_data_path=sys.argv[1])
    # dataset = CustomGraphDataset(root=sys.argv[0])

    # model = GraphSAGE(in_channels=15, hidden_channels=15, out_channels=15)

    print("Data path: ", input_path)
    dataset = torch.load(input_path)

    print(dataset)
    print("X: ", dataset.x)
    print("edge_index: ", dataset.edge_index)

    # out = model(dataset.x, dataset.edge_index)
    # print("out:", out)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    conv = IdentitySAGEConv(16, 4, aggr='mean')

    out2 = conv(dataset.x, dataset.edge_index)
    print(out2.shape)
    algorithms.save_tensor_for_cpp(out2, output_path)
    print("Save out2: ", out2)


def load_gt(gt):
    print("Load GT: ", gt)
    array = algorithms.read_cpp_binary_array(gt, 'q')
    return array


if __name__ == "__main__":
    """Example usage of the GraphReader class."""
    # if len(sys.argv) < 2:
    #    print(f"Usage: {sys.argv[0]} <graph_path> <output_path>")
    #    sys.exit(1)

    # load_gt(sys.argv[1])

    generate_embedding(sys.argv[1], sys.argv[2])
