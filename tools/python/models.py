import numpy as np
from typing import Union, List, Optional
from enum import Enum
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

# 检查GPU是否可用
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(f"使用设备: {device}")

scaler = GradScaler()


class SimilarityMethod(Enum):
    """相似度计算方法枚举"""
    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COMBINED = "combined"  # 组合多种方法


class SimilarityEmbeddingGenerator:
    """
    基于两个输入embedding生成相似度embedding的类

    参数:
        embedding_size (int): 输入embedding的维度
        similarity_method (Union[str, SimilarityMethod]): 相似度计算方法
        normalize (bool): 是否在计算前对输入embedding进行归一化
        combination_weights (Optional[List[float]]): 组合方法时的权重
    """

    def __init__(self,
                 embedding_size: int,
                 similarity_method: Union[str, SimilarityMethod] = SimilarityMethod.COSINE,
                 normalize: bool = True,
                 combination_weights: Optional[List[float]] = None):

        self.embedding_size = embedding_size
        self.normalize = normalize

        # 设置相似度计算方法
        if isinstance(similarity_method, str):
            self.similarity_method = SimilarityMethod(similarity_method.lower())
        else:
            self.similarity_method = similarity_method

        # 设置组合权重
        if self.similarity_method == SimilarityMethod.COMBINED:
            if combination_weights is None:
                self.combination_weights = [0.5, 0.3, 0.2]  # 默认权重
            else:
                assert len(combination_weights) == 3, "组合权重需要3个值"
                assert abs(sum(combination_weights) - 1.0) < 1e-6, "组合权重总和应为1"
                self.combination_weights = combination_weights
        else:
            self.combination_weights = None

    def _normalize(self, embedding: torch.Tensor) -> torch.Tensor:
        """归一化embedding"""
        norm = torch.norm(embedding)
        return embedding / (norm + 1e-8)  # 添加小常数防止除以0

    def _cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """计算余弦相似度"""
        if self.normalize:
            emb1 = self._normalize(emb1)
            emb2 = self._normalize(emb2)
        return float(torch.dot(emb1, emb2))

    def _dot_product(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """计算点积相似度"""
        return float(torch.dot(emb1, emb2))

    def _euclidean_distance(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """计算欧氏距离(转换为相似度)"""
        distance = torch.sum((emb1 - emb2) ** 2)
        return float(distance)

    def _manhattan_distance(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """计算曼哈顿距离(转换为相似度)"""
        distance = torch.sum(torch.abs(emb1 - emb2))
        return float(1 / (1 + distance))  # 将距离转换为[0,1]的相似度

    def _combined_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> List[float]:
        """组合多种相似度方法"""
        cosine = self._cosine_similarity(emb1, emb2)
        dot = self._dot_product(emb1, emb2)
        euclidean = self._euclidean_distance(emb1, emb2)

        # 归一化点积结果到[0,1]范围
        norm_dot = (dot + 1) / 2  # 假设输入已经归一化

        # 加权组合
        combined = [
            cosine * self.combination_weights[0],
            norm_dot * self.combination_weights[1],
            euclidean * self.combination_weights[2]
        ]

        return combined

    def generate(self, emb1: Union[List[float], np.ndarray, torch.Tensor],
                 emb2: Union[List[float], np.ndarray, torch.Tensor]) -> Union[float, List[float]]:
        """
        生成相似度embedding

        参数:
            emb1: 第一个embedding
            emb2: 第二个embedding

        返回:
            相似度embedding(单个值或列表)
        """
        # 转换为torch张量并移动到GPU
        if isinstance(emb1, (list, np.ndarray)):
            emb1 = torch.tensor(emb1, dtype=torch.float32, device=device)
        if isinstance(emb2, (list, np.ndarray)):
            emb2 = torch.tensor(emb2, dtype=torch.float32, device=device)

        # 如果已经是tensor但不在GPU上，移动到GPU
        if isinstance(emb1, torch.Tensor) and not emb1.is_cuda:
            emb1 = emb1.to(device)
        if isinstance(emb2, torch.Tensor) and not emb2.is_cuda:
            emb2 = emb2.to(device)

        # 验证维度
        assert emb1.shape == (self.embedding_size,), f"emb1维度应为({self.embedding_size},)"
        assert emb2.shape == (self.embedding_size,), f"emb2维度应为({self.embedding_size},)"

        # 根据选择的方法计算相似度
        if self.similarity_method == SimilarityMethod.COSINE:
            return self._cosine_similarity(emb1, emb2)
        elif self.similarity_method == SimilarityMethod.DOT:
            return self._dot_product(emb1, emb2)
        elif self.similarity_method == SimilarityMethod.EUCLIDEAN:
            return self._euclidean_distance(emb1, emb2)
        elif self.similarity_method == SimilarityMethod.MANHATTAN:
            return self._manhattan_distance(emb1, emb2)
        elif self.similarity_method == SimilarityMethod.COMBINED:
            return self._combined_similarity(emb1, emb2)
        else:
            raise ValueError(f"未知的相似度计算方法: {self.similarity_method}")

    def batch_generate(self,
                       emb_list1: List[Union[List[float], np.ndarray, torch.Tensor]],
                       emb_list2: List[Union[List[float], np.ndarray, torch.Tensor]]) -> List[
        Union[float, List[float]]]:
        """
        批量生成相似度embedding

        参数:
            emb_list1: 第一个embedding列表
            emb_list2: 第二个embedding列表

        返回:
            相似度embedding列表
        """
        assert len(emb_list1) == len(emb_list2), "两个embedding列表长度必须相同"
        return [self.generate(emb1, emb2) for emb1, emb2 in zip(emb_list1, emb_list2)]


class Perceptron(nn.Module):
    """支持GPU训练和混合精度的多层感知机"""

    def __init__(self, input_dim, hidden_dim=8, dtype=torch.float32):
        """
        初始化MLP

        参数:
            input_dim (int): 输入维度
            hidden_dim (int): 隐藏层维度
            dtype: 数据类型，支持 torch.float16, torch.float32, torch.float64
        """
        super(Perceptron, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dtype = dtype

        # 验证数据类型
        if dtype not in [torch.float16, torch.float32, torch.float64]:
            raise ValueError(f"不支持的dtype: {dtype}，支持的类型: torch.float16, torch.float32, torch.float64")

        print(f"初始化MLP，精度: {dtype}，设备: {device}")

        # 使用PyTorch的线性层，会自动初始化权重
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # 将模型移动到GPU
        self.to(device)
        # 设置数据类型
        self.to(dtype=dtype)

    def forward(self, x):
        """前向传播"""
        # 确保输入在正确的设备和数据类型上
        if not x.is_cuda:
            x = x.to(device)
        if x.dtype != self.dtype:
            x = x.to(dtype=self.dtype)

        x = torch.relu(self.fc1(x))  # 隐藏层使用ReLU激活
        x = torch.sigmoid(self.fc2(x))  # 输出层使用Sigmoid激活
        return x

    def compute_loss(self, y_pred, y_true):
        """计算二元交叉熵损失"""
        epsilon = 1e-15
        # 根据数据类型调整epsilon
        if self.dtype == torch.float16:
            epsilon = 1e-4
        elif self.dtype == torch.float64:
            epsilon = 1e-12

        loss = (torch.mean(((y_pred) - (y_true)) ** 2))
        # loss = torch.mean(torch.abs(y_pred - y_true))

        return loss
        # y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        # return torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

    def train_model(self, X, y, learning_rate=0.01, epochs=1000, batch_size=32, verbose=True):
        """训练模型"""
        # 转换为PyTorch张量并移动到GPU
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self.dtype, device=device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=self.dtype, device=device)

        # 确保y是二维的
        y = y.view(-1, 1)

        # 使用PyTorch优化器
        # optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,  # 可以与Adam相同的学习率
            betas=(0.9, 0.999),  # 动量参数
            eps=1e-8,  # 数值稳定性
            weight_decay=1e-4,  # 解耦的权重衰减，比L2更有效
            amsgrad=False  # 是否使用AMSGrad变体
        )

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0

            # 小批量训练
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                # 使用混合精度训练
                # optimizer.zero_grad()
                '''
                with autocast():
                    y_pred = self(X_batch)
                    loss = self.compute_loss(y_pred, y_batch)

                # 使用GradScaler进行梯度缩放
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                '''

                # 不使用autocast

                y_pred = self(X_batch)
                loss = self.compute_loss(y_pred, y_batch)
                # if not loss.requires_grad:
                # 如果损失不需要梯度，重新创建并设置requires_grad=True
                # loss = loss.clone().detach().requires_grad_(True)
                # print('--------')

                # 不使用GradScaler
                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            losses.append(avg_loss)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, dtype: {self.dtype}")

        return losses

    def predict(self, X, threshold=0.5):
        """预测"""
        self.eval()  # 设置为评估模式
        with torch.no_grad():
            # 转换为PyTorch张量
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=self.dtype, device=device)

            # 获取概率预测
            y_prob = self(X)
            # 转换为0/1预测
            y_pred = torch.tensor(y_prob > threshold).int()
            return y_pred.cpu().numpy().flatten()


def get_parameters(self):
    """获取模型参数"""
    return {
        'fc1_weight': self.fc1.weight.data.cpu().numpy(),
        'fc1_bias': self.fc1.bias.data.cpu().numpy(),
        'fc2_weight': self.fc2.weight.data.cpu().numpy(),
        'fc2_bias': self.fc2.bias.data.cpu().numpy(),
        'dtype': self.dtype
    }


def set_parameters(self, params):
    """设置模型参数"""
    self.fc1.weight.data = torch.tensor(params['fc1_weight'], dtype=self.dtype, device=device)
    self.fc1.bias.data = torch.tensor(params['fc1_bias'], dtype=self.dtype, device=device)
    self.fc2.weight.data = torch.tensor(params['fc2_weight'], dtype=self.dtype, device=device)
    self.fc2.bias.data = torch.tensor(params['fc2_bias'], dtype=self.dtype, device=device)


# 兼容性包装器，保持原有接口
class MLP(Perceptron):
    """MLP别名，保持向后兼容"""
    pass


if __name__ == "__main__":
    # 创建实例
    generator = SimilarityEmbeddingGenerator(
        embedding_size=4,
        similarity_method="combined",
        normalize=True,
        combination_weights=[0.6, 0.2, 0.2]  # 更重视余弦相似度
    )

    # 示例embedding
    emb1 = [0.1, 0.2, 0.3, 0.4]
    emb2 = [0.4, 0.3, 0.2, 0.1]

    # 计算相似度
    similarity = generator.generate(emb1, emb2)
    print(f"组合相似度embedding: {similarity}")

    # 批量计算
    batch_emb1 = [emb1, [0.5, 0.5, 0.5, 0.5]]
    batch_emb2 = [emb2, [0.5, 0.5, 0.5, 0.5]]
    batch_similarities = generator.batch_generate(batch_emb1, batch_emb2)
    print(f"批量相似度结果: {batch_similarities}")

    print("\n" + "=" * 50)
    print("测试不同精度的MLP (GPU训练)")
    print("=" * 50)

    # 生成一些非线性可分的数据
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)

    # 测试不同精度的MLP
    dtypes = [torch.float16, torch.float32, torch.float64]

    for dtype in dtypes:
        print(f"\n测试 {dtype} 精度:")

        # 创建MLP
        mlp = Perceptron(input_dim=2, hidden_dim=16, dtype=dtype)

        # 训练
        losses = mlp.train_model(X, y, learning_rate=0.1, epochs=500, batch_size=32, verbose=False)

        # 测试
        X_test = np.array([[0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])
        predictions = mlp.predict(X_test)
        final_loss = losses[-1] if losses else 0

        print(f"最终损失: {final_loss:.6f}")
        print(f"预测结果: {predictions}")
        print(f"参数数据类型: {mlp.fc1.weight.dtype}")
        print(f"参数所在设备: {mlp.fc1.weight.device}")
