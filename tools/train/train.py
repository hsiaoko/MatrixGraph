import torch_geometric
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.data import InMemoryDataset, Data


class CustomGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []  # 无原始文件时返回空列表

    @property
    def processed_file_names(self):
        return ['custom_data.pt']  # 处理后的文件名

    def download(self):
        pass  # 无需下载数据

    def process(self):
        data_list = []

        # 创建3个随机图
        for i in range(3):
            # 节点特征 (4个节点，每个5维)
            x = torch.randn(4, 5)

            # 边索引 (4条边)
            edge_index = torch.tensor([
                [0, 1, 2, 3],  # 源节点
                [1, 2, 3, 0]  # 目标节点
            ], dtype=torch.long)

            # 图标签 (0或1)
            y = torch.tensor([i % 2], dtype=torch.float)

            # 创建Data对象
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

            # 保存处理后的数据
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])


edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

print(Data(edge_index=[2, 4], x=[3, 1]))

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
dataset = Planetoid(root='/tmp/Cora', name='Cora')

print(dataset)

print("num_class", dataset.num_classes)
print("num_node_feature", dataset.num_node_features)

print("dataset", dataset)

data = dataset[0]
print("data", data)
print("data.y", data.y)
print(data.train_mask.sum().item())
print(data.val_mask.sum().item())
print(data.test_mask.sum().item())

train_dataset = dataset[:540]
test_dataset = dataset[540:]

loader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch in loader:
    print(batch)


class GCN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(dataset.num_node_features, 16)

        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())

print(f'Accuracy: {acc:.4f}')
