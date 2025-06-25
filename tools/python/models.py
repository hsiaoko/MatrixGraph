import numpy as np
from typing import Union, List, Optional
from enum import Enum
import torch


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

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """归一化embedding"""
        norm = np.linalg.norm(embedding)
        return embedding / (norm + 1e-8)  # 添加小常数防止除以0

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """计算余弦相似度"""
        if self.normalize:
            emb1 = self._normalize(emb1)
            emb2 = self._normalize(emb2)
        return float(np.dot(emb1, emb2))

    def _dot_product(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """计算点积相似度"""
        return float(np.dot(emb1, emb2))

    def _euclidean_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """计算欧氏距离(转换为相似度)"""
        distance = np.square(emb1 - emb2)
        return distance

    def _manhattan_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """计算曼哈顿距离(转换为相似度)"""
        distance = np.sum(np.abs(emb1 - emb2))
        return float(1 / (1 + distance))  # 将距离转换为[0,1]的相似度

    def _combined_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> List[float]:
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

    def generate(self, emb1: Union[List[float], np.ndarray],
                 emb2: Union[List[float], np.ndarray]) -> Union[float, List[float]]:
        """
        生成相似度embedding

        参数:
            emb1: 第一个embedding
            emb2: 第二个embedding

        返回:
            相似度embedding(单个值或列表)
        """
        # 转换为numpy数组
        # emb1 = np.array(emb1, dtype=np.float32)
        # emb2 = np.array(emb2, dtype=np.float32)
        emb1 = emb1.detach().numpy().astype(np.float32)
        emb2 = emb2.detach().numpy().astype(np.float32)
        # emb1 = torch.detach().numpy(emb1, dtype=np.float32)
        # emb2 = torch.detach().numpy(emb2, dtype=np.float32)

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
                       emb_list1: List[Union[List[float], np.ndarray]],
                       emb_list2: List[Union[List[float], np.ndarray]]) -> List[Union[float, List[float]]]:
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


np


class Perceptron:

    def __init__(self, input_dim, hidden_dim=8):
        # 初始化参数
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 第一层权重和偏置 (输入层 -> 隐藏层)
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)

        # 第二层权重和偏置 (隐藏层 -> 输出层)
        self.W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.b2 = np.zeros(1)

        # def __init__(self, input_dim):
        #    # 初始化权重和偏置
        #    self.weights = np.random.randn(input_dim)
        # self.bias = np.random.randn()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)
        # 第一层前向传播
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)  # 隐藏层使用ReLU激活

        # 第二层前向传播
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)  # 输出层使用Sigmoid激活

        return self.a2 > 0.5

    def compute_loss(self, y_pred, y_true):
        # 使用二元交叉熵损失
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def forward(self, x):
        # 第一层前向传播
        self.z1 = np.dot(x, self.W1) + self.b1

        self.a1 = self.relu(self.z1)  # 隐藏层使用ReLU激活

        # 第二层前向传播
        self.z2 = np.dot(self.a1, self.W2) + self.b2

        self.a2 = self.sigmoid(self.z2)  # 输出层使用Sigmoid激活

        return self.a2

    def backward(self, x, y_true, y_pred, learning_rate):
        m = x.shape[0]  # 样本数量

        # 输出层梯度
        dz2 = y_pred - y_true.reshape(-1, 1)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0) / m

        # 隐藏层梯度
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)  # ReLU导数
        dW1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0) / m

        # 更新参数
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, learning_rate=0.01, epochs=1000, batch_size=32):
        losses = []
        for epoch in range(epochs):
            # 小批量训练
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                # 前向传播
                y_pred = self.forward(X_batch)

                # 计算损失
                loss = self.compute_loss(y_pred, y_batch)
                losses.append(loss)

                # 反向传播和参数更新
                self.backward(X_batch, y_batch, y_pred, learning_rate)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        return losses

    def predict(self, X, threshold=0.5):
        # 获取概率预测
        y_prob = self.forward(X)
        # 转换为0/1预测
        y_pred = (y_prob > threshold).astype(int)
        return y_pred.flatten()


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

    print("----perceptro----")

    # 生成一些非线性可分的数据
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)

    # 创建MLP
    mlp = MLP(input_dim=2, hidden_dim=16)

    # 训练
    losses = mlp.train(X, y, learning_rate=0.1, epochs=1000, batch_size=32)

    # 测试
    X_test = np.array([[0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])
    predictions = mlp.predict(X_test)
    print("Predictions:", predictions)
