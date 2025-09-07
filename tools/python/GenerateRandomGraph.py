import random
import os
from pathlib import Path

def generate_random_graph(num_edges, label_range, num_vertices=None):
    """
    生成随机图
    :param num_edges: 边数
    :param label_range: label范围 (min_label, max_label)
    :param num_vertices: 顶点数（如果为None，则自动计算）
    :return: 图的字符串表示
    """
    min_label, max_label = label_range

    # 如果没有指定顶点数，则根据边数估算
    if num_vertices is None:
        num_vertices = max(int(num_edges * 0.7), num_edges // 2 + 1)
        num_vertices = min(num_vertices, num_edges * 2)  # 确保顶点数合理

    # 生成顶点
    vertices = []
    degrees = [0] * num_vertices

    # 为每个顶点分配随机label
    labels = [random.randint(min_label, max_label) for _ in range(num_vertices)]

    # 生成边（避免自环和重复边）
    edges = set()
    attempts = 0
    max_attempts = num_edges * 10  # 防止无限循环

    while len(edges) < num_edges and attempts < max_attempts:
        u = random.randint(0, num_vertices - 1)
        v = random.randint(0, num_vertices - 1)

        if u != v:  # 避免自环
            edge = (min(u, v), max(u, v))  # 统一排序，避免重复
            if edge not in edges:
                edges.add(edge)
                degrees[u] += 1
                degrees[v] += 1
        attempts += 1

    # 如果无法生成足够的边，调整顶点数或边数
    if len(edges) < num_edges:
        print(f"警告: 无法生成 {num_edges} 条边，实际生成 {len(edges)} 条")
        num_edges = len(edges)

    # 构建图字符串
    graph_str = f"t {num_vertices} {num_edges}\n"

    # 添加顶点信息
    for i in range(num_vertices):
        graph_str += f"v {i} {labels[i]} {degrees[i]}\n"

    # 添加边信息
    for u, v in edges:
        graph_str += f"e {u} {v}\n"

    return graph_str

def generate_and_save_random_graphs(m, num_edges, label_range, save_path):
    """
    生成并保存多个随机图
    :param m: 生成图的数量
    :param num_edges: 每条图的边数
    :param label_range: label范围 (min_label, max_label)
    :param save_path: 保存路径
    """
    # 创建保存目录
    Path(save_path).mkdir(parents=True, exist_ok=True)

    for i in range(m):
        graph_content = generate_random_graph(num_edges, label_range)

        # 生成文件名
        filename = f"random_graph_edges{num_edges}_label{label_range[0]}-{label_range[1]}_{i+1}.txt"
        filepath = os.path.join(save_path, filename)

        # 保存文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(graph_content)

        print(f"已生成图 {i+1}/{m}: {filename}")

def main():
    # 参数设置
    m = 10           # 生成图的数量
    num_edges = 12   # 每条图的边数
    label_range = (0, 16)  # label范围
    save_path = "/data/zhuxiaoke/workspace/random_graphs"  # 保存路径

    # 生成并保存图
    generate_and_save_random_graphs(m, num_edges, label_range, save_path)
    print(f"所有图已保存到: {save_path}")

if __name__ == "__main__":
    main()
