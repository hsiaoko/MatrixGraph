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
        print("aggregate: ", aggregated)
        aggregated = aggregated + self.propagate(edge_index, x=aggregated)
        print("aggregate: ", aggregated)

        # 替换权重乘法为恒等操作
        # if self.root_weight:
        #    out = torch.cat([x, aggregated], dim=-1)
        # else:
        #    out = aggregated
        out = aggregated

        # 模拟 W=1 的效果（若需保持输出维度，需确保 in_channels == out_channels）
        return out


def save_tensor_for_cpp(tensor: torch.Tensor, root_path: str):
    """保存Tensor为C++可读格式"""
    assert tensor.is_contiguous(), "Tensor must be contiguous"

    bin_path = root_path + "embedding.bin"
    meta_path = root_path + "meta.yaml"
    # 保存二进制
    np_array = tensor.numpy()
    np_array.tofile(bin_path)

    # 保存元数据

    metadata = {
        "dtype": str(tensor.dtype).split(".")[-1],  # 如 "float32"
        "x": tensor.shape[0],
        "y": tensor.shape[1],
        "endian": "little",  # 或 "big"
        "order": "row_major"  # PyTorch默认行优先存储
    }

    # 保存为YAML
    with open(meta_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)


def main():
    """Example usage of the GraphReader class."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <graph_path> <output_path>")
        sys.exit(1)

    # dataset = CustomGraphDataset(root='./custom_data', raw_data_path=sys.argv[1])
    # dataset = CustomGraphDataset(root=sys.argv[0])

    # model = GraphSAGE(in_channels=15, hidden_channels=15, out_channels=15)

    print("Data path: ", sys.argv[1])
    dataset = torch.load(sys.argv[1])

    print(dataset)
    print("X: ", dataset.x)
    print("edge_index: ", dataset.edge_index)

    # out = model(dataset.x, dataset.edge_index)
    # print("out:", out)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    conv = IdentitySAGEConv(15, 15, aggr='mean')

    out2 = conv(dataset.x, dataset.edge_index)
    save_tensor_for_cpp(out2, sys.argv[2])
    print("out2: ", out2)


if __name__ == "__main__":
    main()
