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
from sklearn.model_selection import train_test_split

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

    conv = IdentitySAGEConv(15, 15, aggr='mean')

    out2 = conv(dataset.x, dataset.edge_index)
    save_tensor_for_cpp(out2, output_path)
    print("out2: ", out2)


def load_gt(gt):
    print("Load GT: ", gt)
    array = algorithms.read_cpp_binary_array(gt, 'q')
    return array


def train_gnn_model(pattern_vertices_embedding_path, graph_vertices_embedding_path, gt_path, output_path=""):
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
        combination_weights=[0.6, 0.2, 0.2]  # 更重视余弦相似度
    )

    x = list()
    y = list()

    print(gt_array)

    gt_array = np.array(gt_array, dtype=np.int64)
    count = 0

    for _ in gt_array:
        similarity = generator.generate(pattern_vertices_embedding[0], graph_vertices_embedding[_])
        x.append(np.array(similarity))
        y.append(1)

    for _ in range(len(graph_vertices_embedding)):
        count = count + 1
        if (count > len(gt_array)):
            break
        if np.any(gt_array == _):
            pass
        else:
            similarity = generator.generate(pattern_vertices_embedding[0], graph_vertices_embedding[_])
            x.append(np.array(similarity))
            y.append(0)

    x = np.array(x)
    y = np.array(y)

    print(len(x))
    input_dim = x.shape[1]
    perceptron = models.Perceptron(input_dim=input_dim)
    perceptron.train(x, y, learning_rate=0.01, epochs=10000)

    results = perceptron.predict(x)
    print(results)

    print(algorithms.calculate_metrics(y, results))


def multi_train_gnn_model():
    pattern_path_1 = "/data/zhuxiaoke/workspace/Torch/pt/queries/5-path.pt"
    pattern_path_2 = "/data/zhuxiaoke/workspace/Torch/pt/queries/5-star.pt"
    pattern_path_3 = "/data/zhuxiaoke/workspace/Torch/pt/queries/tree.pt"

    pattern_path_1 = "/data/zhuxiaoke/workspace/Torch/pt/queries/tree.pt"
    pattern_path_2 = "/data/zhuxiaoke/workspace/Torch/pt/queries/tree.pt"
    pattern_path_3 = "/data/zhuxiaoke/workspace/Torch/pt/queries/tree.pt"

    graph_path = "/data/zhuxiaoke/workspace/Torch/pt/yeast.pt"
    gt_path_1 = "/data/zhuxiaoke/workspace/Torch/gt/yeast/yeast_5-path.gt"
    gt_path_2 = "/data/zhuxiaoke/workspace/Torch/gt/yeast/yeast_5-star.gt"
    gt_path_3 = "/data/zhuxiaoke/workspace/Torch/gt/yeast/yeast_tree.gt"
    W1_path = "/data/zhuxiaoke/workspace/Torch/models/yeast/W1/"
    W2_path = "/data/zhuxiaoke/workspace/Torch/models/yeast/W2/"
    b1_path = "/data/zhuxiaoke/workspace/Torch/models/yeast/b1/"
    b2_path = "/data/zhuxiaoke/workspace/Torch/models/yeast/b2/"

    graph_path = "/data/zhuxiaoke/workspace/Torch/pt/dblp.pt"
    gt_path_1 = "/data/zhuxiaoke/workspace/Torch/gt/dblp/dblp_5-path.gt"
    gt_path_2 = "/data/zhuxiaoke/workspace/Torch/gt/dblp/dblp_5-star.gt"
    gt_path_3 = "/data/zhuxiaoke/workspace/Torch/gt/dblp/dblp_tree.gt"
    W1_path = "/data/zhuxiaoke/workspace/Torch/models/dblp/W1/"
    W2_path = "/data/zhuxiaoke/workspace/Torch/models/dblp/W2/"
    b1_path = "/data/zhuxiaoke/workspace/Torch/models/dblp/b1/"
    b2_path = "/data/zhuxiaoke/workspace/Torch/models/dblp/b2/"

    graph_path = "/data/zhuxiaoke/workspace/Torch/pt/livejournal.pt"
    # gt_path_1 = "/data/zhuxiaoke/workspace/Torch/gt/livejournal/livejournal_5-path.gt"
    gt_path_1 = "/data/zhuxiaoke/workspace/Torch/gt/livejournal/livejournal_tree.gt"
    gt_path_2 = "/data/zhuxiaoke/workspace/Torch/gt/livejournal/livejournal_tree.gt"
    gt_path_3 = "/data/zhuxiaoke/workspace/Torch/gt/livejournal/livejournal_tree.gt"
    W1_path = "/data/zhuxiaoke/workspace/Torch/models/livejournal/W1/"
    W2_path = "/data/zhuxiaoke/workspace/Torch/models/livejournal/W2/"
    b1_path = "/data/zhuxiaoke/workspace/Torch/models/livejournal/b1/"
    b2_path = "/data/zhuxiaoke/workspace/Torch/models/livejournal/b2/"

    # graph_path = "/data/zhuxiaoke/workspace/Torch/pt/twitter.pt"
    # gt_path_1 = "/data/zhuxiaoke/workspace/Torch/gt/twitter/twitter_5-path.gt"
    # gt_path_2 = "/data/zhuxiaoke/workspace/Torch/gt/twitter/twitter_5-star.gt"
    # gt_path_3 = "/data/zhuxiaoke/workspace/Torch/gt/twitter/twitter_tree.gt"
    # W1_path = "/data/zhuxiaoke/workspace/Torch/models/twitter/W1/"
    # W2_path = "/data/zhuxiaoke/workspace/Torch/models/twitter/W2/"
    # b1_path = "/data/zhuxiaoke/workspace/Torch/models/twitter/b1/"
    # b2_path = "/data/zhuxiaoke/workspace/Torch/models/twitter/b2/"

    graph_dataset = torch.load(graph_path)
    pattern_dataset_1 = torch.load(pattern_path_1)
    pattern_dataset_2 = torch.load(pattern_path_2)
    pattern_dataset_3 = torch.load(pattern_path_3)

    print("Pattern Dataset: ", pattern_dataset_1)
    print("Graph Dataset: ", graph_dataset)

    conv = IdentitySAGEConv(64, 64, aggr='mean')

    pattern_embedding_1 = conv(pattern_dataset_1.x, pattern_dataset_1.edge_index)
    pattern_embedding_2 = conv(pattern_dataset_2.x, pattern_dataset_2.edge_index)
    pattern_embedding_3 = conv(pattern_dataset_3.x, pattern_dataset_3.edge_index)

    graph_embedding = conv(graph_dataset.x, graph_dataset.edge_index)

    gt_array_1 = algorithms.read_cpp_binary_array(gt_path_1, 'q')
    gt_array_2 = algorithms.read_cpp_binary_array(gt_path_2, 'q')
    gt_array_3 = algorithms.read_cpp_binary_array(gt_path_3, 'q')

    generator = models.SimilarityEmbeddingGenerator(
        embedding_size=64,
        similarity_method="euclidean",
        normalize=True,
        combination_weights=[0.6, 0.2, 0.2]  # 更重视余弦相似度
    )

    x = list()
    y = list()

    count = 0

    for _ in gt_array_1:
        similarity = generator.generate(pattern_embedding_1[0], graph_embedding[_])
        x.append(np.array(similarity))
        y.append(1)

    for _ in range(len(graph_embedding)):
        count = count + 1
        if (count > len(gt_array_1)):
            break
        if np.any(gt_array_1 == _):
            pass
        else:
            similarity = generator.generate(pattern_embedding_1[0], graph_embedding[_])
            x.append(np.array(similarity))
            y.append(0)

    for _ in gt_array_2:
        similarity = generator.generate(pattern_embedding_2[0], graph_embedding[_])
        x.append(np.array(similarity))
        y.append(1)

    for _ in range(len(graph_embedding)):
        count = count + 1
        if (count > len(gt_array_2)):
            break
        if np.any(gt_array_2 == _):
            pass
        else:
            similarity = generator.generate(pattern_embedding_2[0], graph_embedding[_])
            x.append(np.array(similarity))
            y.append(0)

    for _ in gt_array_3:
        similarity = generator.generate(pattern_embedding_3[0], graph_embedding[_])
        x.append(np.array(similarity))
        y.append(1)

    for _ in range(len(graph_embedding)):
        count = count + 1
        if (count > len(gt_array_3)):
            break
        if np.any(gt_array_3 == _):
            pass
        else:
            similarity = generator.generate(pattern_embedding_3[0], graph_embedding[_])
            x.append(np.array(similarity))
            y.append(0)

    x = np.array(x)
    y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2
    )

    input_dim = x.shape[1]
    perceptron = models.Perceptron(input_dim=input_dim)
    perceptron.train(x_train, y_train, learning_rate=0.1, epochs=10)

    results = perceptron.predict(x_test)
    print("X:", x_test.shape)
    print(results)
    print(algorithms.calculate_metrics(y_test, results))

    algorithms.save_tensor_for_cpp(torch.tensor(perceptron.W1), W1_path)
    algorithms.save_tensor_for_cpp(torch.tensor(perceptron.b1), b1_path)
    algorithms.save_tensor_for_cpp(torch.tensor(perceptron.W2), W2_path)
    algorithms.save_tensor_for_cpp(torch.tensor(perceptron.b2), b2_path)

    print("========TEST=========")
    test_x = similarity = generator.generate(pattern_embedding_1[0], graph_embedding[217])
    print("u0 embedding: ", pattern_embedding_1[0])
    print("v0 embedding: ", graph_embedding[217])
    print("sim: ", similarity)
    test_y = perceptron.predict(similarity)
    print("test_y: ", test_y)

    tmp_x = np.zeros(64)
    tmp_x[0] = 1
    # tmp_x[1] = 2
    # tmp_x[2] = 3
    print("X", tmp_x)
    tmp_y = np.dot(tmp_x, perceptron.W1) + perceptron.b1

    print("tmp_y: ", tmp_y)
    tmp_y = tmp_y + perceptron.b1

    print("tmp_y: ", tmp_y)

    tmp_y = perceptron.predict(tmp_x)
    print('Y', tmp_y)


if __name__ == "__main__":
    """Example usage of the GraphReader class."""
    # if len(sys.argv) < 2:
    #    print(f"Usage: {sys.argv[0]} <graph_path> <output_path>")
    #    sys.exit(1)

    # train_gnn_model(sys.argv[1], sys.argv[2], sys.argv[3])
    multi_train_gnn_model()
