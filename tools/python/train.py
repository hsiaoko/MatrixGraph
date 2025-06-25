import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset
import os
import sys
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv
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


class GATLayer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, drop_rate=0.0):
        super(GATLayer, self).__init__()
        self.gat1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels
        )
        # self.fc1 = nn.Linear(hidden_channels, in_channels)
        self.gat2 = GATConv(
            in_channels=hidden_channels,
            out_channels=out_channels
        )
        # self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.gat3 = GATConv(
            in_channels=out_channels,
            out_channels=out_channels
        )
        self.drop = torch.nn.Dropout(drop_rate)

    def forward(self, x, e):
        y = self.gat1(x, e)
        # y = nn.functional.relu(self.fc1(y))
        y = self.gat2(y, e)
        # y = nn.functional.relu(self.fc2(y))
        # y = self.gat3(y, e)
        y = torch.nn.functional.relu(y)
        y = self.drop(y)
        return y


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, bias=False)
        # self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)

        # x = F.relu(x)
        # x = self.conv2(x, edge_index)
        return x


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
    def forward(self, aggregated, edge_index):
        # 原始聚合逻辑
        aggregated = self.propagate(edge_index, x=aggregated)
        aggregated = aggregated + self.propagate(edge_index, x=aggregated)
        aggregated = aggregated + self.propagate(edge_index, x=aggregated)
        aggregated = aggregated + self.propagate(edge_index, x=aggregated)

        # aggregated = self.propagate(edge_index, x=aggregated)
        # aggregated = aggregated + self.propagate(edge_index, x=aggregated)

        # aggregated = self.propagate(edge_index, x=aggregated)
        # aggregated = aggregated + self.propagate(edge_index, x=aggregated)

        # aggregated = self.propagate(edge_index, x=aggregated)
        # aggregated = aggregated + self.propagate(edge_index, x=aggregated)
        return aggregated


def save_tensor_for_cpp(tensor: torch.Tensor, root_path: str):
    """保存Tensor为C++可读格式"""
    assert tensor.is_contiguous(), "Tensor must be contiguous"

    bin_path = root_path + "embedding.bin"
    meta_path = root_path + "meta.yaml"
    # 保存二进制
    np_array = tensor.datch().numpy()
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

    graph_path = "/data/zhuxiaoke/workspace/Torch/pt/livejournal_1.0_fix.pt"
    gt_path_1 = "/data/zhuxiaoke/workspace/Torch/gt/livejournal/livejournal_clique.gt"
    gt_path_2 = "/data/zhuxiaoke/workspace/Torch/gt/livejournal/livejournal_tree.gt"
    gt_path_3 = "/data/zhuxiaoke/workspace/Torch/gt/livejournal/livejournal_5-star.gt"
    W1_path = "/data/zhuxiaoke/workspace/Torch/models/livejournal_npml_fix/W1/"
    W2_path = "/data/zhuxiaoke/workspace/Torch/models/livejournal_npml_fix/W2/"
    b1_path = "/data/zhuxiaoke/workspace/Torch/models/livejournal_npml_fix/b1/"
    b2_path = "/data/zhuxiaoke/workspace/Torch/models/livejournal_npml_fix/b2/"
    gnn_emb_path = "/data/zhuxiaoke/workspace/Torch/models/livejournal_npml_fix/gnn_emb/"
    p1_emb_path = "/data/zhuxiaoke/workspace/Torch/models/livejournal_npml_fix/p1_emb/"
    p2_emb_path = "/data/zhuxiaoke/workspace/Torch/models/livejournal_npml_fix/p2_emb/"
    p3_emb_path = "/data/zhuxiaoke/workspace/Torch/models/livejournal_npml_fix/p3_emb/"

    # pattern_path_1 = "/data/zhuxiaoke/workspace/Torch/pt/queries/5-path.pt"
    # pattern_path_1 = "/data/zhuxiaoke/workspace/Torch/pt/queries/tree.pt"
    # pattern_path_2 = "/data/zhuxiaoke/workspace/Torch/pt/queries/tree.pt"
    # pattern_path_3 = "/data/zhuxiaoke/workspace/Torch/pt/queries/tree.pt"
    ## pattern_path_3 = "/data/zhuxiaoke/workspace/Torch/pt/queries/5-star.pt"
    # graph_path = "/data/zhuxiaoke/workspace/Torch/pt/pokec.pt"
    ## gt_path_1 = "/data/zhuxiaoke/workspace/Torch/gt/pokec/pokec_5-path.gt"
    # gt_path_1 = "/data/zhuxiaoke/workspace/Torch/gt/pokec/pokec_tree.gt"
    # gt_path_2 = "/data/zhuxiaoke/workspace/Torch/gt/pokec/pokec_tree.gt"
    # gt_path_3 = "/data/zhuxiaoke/workspace/Torch/gt/pokec/pokec_tree.gt"
    ## gt_path_3 = "/data/zhuxiaoke/workspace/Torch/gt/pokec/pokec_5-star.gt"
    # W1_path = "/data/zhuxiaoke/workspace/Torch/models/pokec_epoch1000/W1/"
    # W2_path = "/data/zhuxiaoke/workspace/Torch/models/pokec_epoch1000/W2/"
    # b1_path = "/data/zhuxiaoke/workspace/Torch/models/pokec_epoch1000/b1/"
    # b2_path = "/data/zhuxiaoke/workspace/Torch/models/pokec_epoch1000/b2/"
    # gnn_emb_path = "/data/zhuxiaoke/workspace/Torch/models/pokec_epoch1000/gnn_emb/"
    # p1_emb_path = "/data/zhuxiaoke/workspace/Torch/models/pokec_epoch1000/p1_emb/"
    # p2_emb_path = "/data/zhuxiaoke/workspace/Torch/models/pokec_epoch1000/p2_emb/"
    # p3_emb_path = "/data/zhuxiaoke/workspace/Torch/models/pokec_epoch1000/p3_emb/"

    # graph_path = "/data/zhuxiaoke/workspace/Torch/pt/dblp.pt"
    # gt_path_1 = "/data/zhuxiaoke/workspace/Torch/gt/dblp/dblp_5-path.gt"
    # gt_path_1 = "/data/zhuxiaoke/workspace/Torch/gt/dblp/dblp_tree.gt"
    # gt_path_2 = "/data/zhuxiaoke/workspace/Torch/gt/dblp/dblp_tree.gt"
    # gt_path_3 = "/data/zhuxiaoke/workspace/Torch/gt/dblp/dblp_tree.gt"
    ## gt_path_3 = "/data/zhuxiaoke/workspace/Torch/gt/dblp/dblp_5-star.gt"
    # W1_path = "/data/zhuxiaoke/workspace/Torch/models/dblp_npml1/W1/"
    # W2_path = "/data/zhuxiaoke/workspace/Torch/models/dblp_npml1/W2/"
    # b1_path = "/data/zhuxiaoke/workspace/Torch/models/dblp_npml1/b1/"
    # b2_path = "/data/zhuxiaoke/workspace/Torch/models/dblp_npml1/b2/"
    # gnn_emb_path = "/data/zhuxiaoke/workspace/Torch/models/dblp_npml1/gnn_emb/"
    # p1_emb_path = "/data/zhuxiaoke/workspace/Torch/models/dblp_npml1/p1_emb/"
    # p2_emb_path = "/data/zhuxiaoke/workspace/Torch/models/dblp_npml1/p2_emb/"
    # p3_emb_path = "/data/zhuxiaoke/workspace/Torch/models/dblp_npml1/p3_emb/"

    # pattern_path_1 = "/data/zhuxiaoke/workspace/Torch/pt/queries/clique.pt"
    # pattern_path_2 = "/data/zhuxiaoke/workspace/Torch/pt/queries/tree.pt"
    # pattern_path_3 = "/data/zhuxiaoke/workspace/Torch/pt/queries/5-star.pt"
    # graph_path = "/data/zhuxiaoke/workspace/Torch/pt/patents.pt"
    # gt_path_3 = "/data/zhuxiaoke/workspace/Torch/gt/patents/patents_clique.gt"
    # gt_path_1 = "/data/zhuxiaoke/workspace/Torch/gt/patents/patents_tree.gt"
    # gt_path_2 = "/data/zhuxiaoke/workspace/Torch/gt/patents/patents_5-star.gt"
    # W1_path = "/data/zhuxiaoke/workspace/Torch/models/patents_epoch20000/W1/"
    # W2_path = "/data/zhuxiaoke/workspace/Torch/models/patents_epoch20000/W2/"
    # b1_path = "/data/zhuxiaoke/workspace/Torch/models/patents_epoch20000/b1/"
    # b2_path = "/data/zhuxiaoke/workspace/Torch/models/patents_epoch20000/b2/"
    # gnn_emb_path = "/data/zhuxiaoke/workspace/Torch/models/patents_epoch20000/gnn_emb/"
    # p1_emb_path = "/data/zhuxiaoke/workspace/Torch/models/patents_epoch20000/p1_emb/"
    # p2_emb_path = "/data/zhuxiaoke/workspace/Torch/models/patents_epoch20000/p2_emb/"
    # p3_emb_path = "/data/zhuxiaoke/workspace/Torch/models/patents_epoch20000/p3_emb/"

    graph_dataset = torch.load(graph_path)
    pattern_dataset_1 = torch.load(pattern_path_1)
    pattern_dataset_2 = torch.load(pattern_path_2)
    pattern_dataset_3 = torch.load(pattern_path_3)

    print("Pattern Dataset: ", pattern_dataset_1)
    print("Graph Dataset: ", graph_dataset)

    conv = IdentitySAGEConv(64, 64, aggr='mean')
    # conv = GraphSAGE(64, 64, 64)
    # conv = GATLayer(64, 64, 64)
    # conv = GCN(64, 64, 64)

    pattern_embedding_1 = conv(pattern_dataset_1.x, pattern_dataset_1.edge_index)
    pattern_embedding_2 = conv(pattern_dataset_2.x, pattern_dataset_2.edge_index)
    pattern_embedding_3 = conv(pattern_dataset_3.x, pattern_dataset_3.edge_index)

    graph_embedding = conv(graph_dataset.x, graph_dataset.edge_index)

    gt_array_1 = algorithms.read_cpp_binary_array(gt_path_1, 'q')
    gt_array_2 = algorithms.read_cpp_binary_array(gt_path_2, 'q')
    gt_array_3 = algorithms.read_cpp_binary_array(gt_path_3, 'q')
    # print("gt1:", gt_array_1)
    # print("gt2:", gt_array_2)
    # print("gt3:", gt_array_3)

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
        x, y, test_size=0.1
    )

    input_dim = x.shape[1]
    perceptron = models.Perceptron(input_dim=input_dim)
    perceptron.train(x_train, y_train, learning_rate=0.01, epochs=10000)

    results = perceptron.predict(x_test)
    print(results)
    print(algorithms.calculate_metrics(y_test, results))

    algorithms.save_tensor_for_cpp(torch.tensor(perceptron.W1), W1_path)
    algorithms.save_tensor_for_cpp(torch.tensor(perceptron.b1), b1_path)
    algorithms.save_tensor_for_cpp(torch.tensor(perceptron.W2), W2_path)
    algorithms.save_tensor_for_cpp(torch.tensor(perceptron.b2), b2_path)

    gnn_embedding = conv(graph_dataset.x, graph_dataset.edge_index)
    p1_embedding = conv(pattern_dataset_1.x, pattern_dataset_1.edge_index)
    p2_embedding = conv(pattern_dataset_3.x, pattern_dataset_2.edge_index)
    p3_embedding = conv(pattern_dataset_3.x, pattern_dataset_3.edge_index)

    algorithms.save_tensor_for_cpp(gnn_embedding, gnn_emb_path)
    algorithms.save_tensor_for_cpp(p1_embedding, p1_emb_path)
    algorithms.save_tensor_for_cpp(p2_embedding, p1_emb_path)
    algorithms.save_tensor_for_cpp(p3_embedding, p2_emb_path)

    print("========TEST=========")
    similarity = generator.generate(pattern_embedding_1[0], graph_embedding[590])
    print("u0 embedding: ", pattern_embedding_1[0])
    print("v0 embedding: ", graph_embedding[590])
    print("sim: ", similarity)

    # tmp_x = np.zeros(64)
    tmp_x = similarity
    # tmp_x[1] = 1
    # tmp_x[2] = 3
    print("X", tmp_x)
    tmp_y = np.dot(tmp_x, perceptron.W1)  # + perceptron.b1

    print("z1: ", tmp_y)
    tmp_y = tmp_y + perceptron.b1

    tmp_y = perceptron.relu(tmp_y)  # 隐藏层使用ReLU激活

    # 第二层前向传播
    z2 = np.dot(tmp_y, perceptron.W2) + perceptron.b2
    print("z2: ", z2)

    a2 = perceptron.sigmoid(z2)  # 输出层使用Sigmoid激活
    print("a2: ", a2)

    tmp_y = perceptron.predict(tmp_x)
    print('Y', tmp_y)


if __name__ == "__main__":
    """Example usage of the GraphReader class."""
    # if len(sys.argv) < 2:
    #    print(f"Usage: {sys.argv[0]} <graph_path> <output_path>")
    #    sys.exit(1)

    # train_gnn_model(sys.argv[1], sys.argv[2], sys.argv[3])
    import time

    start_time = time.time()
    multi_train_gnn_model()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"耗时: {elapsed_time:.4f} 秒")
