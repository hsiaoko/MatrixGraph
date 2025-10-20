import argparse
import random
import os
from pathlib import Path


def generate_random_graph(n, m, r):
    """生成随机图"""
    max_edges = n * (n - 1) // 2
    m = min(m, max_edges)

    vertices = []
    degrees = [0] * n

    # 生成顶点
    for i in range(n):
        label = random.randint(0, r - 1)
        vertices.append((i, label))

    # 生成边
    edges = set()
    possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    selected_edges = random.sample(possible_edges, m)

    for u, v in selected_edges:
        degrees[u] += 1
        degrees[v] += 1
        edges.add((min(u, v), max(u, v)))

    # 构建文件内容
    content = [f"t {n} {m}"]
    for i, label in vertices:
        content.append(f"v {i} {label} {degrees[i]}")
    for u, v in sorted(edges):
        content.append(f"e {u} {v}")

    return "\n".join(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random graphs")
    parser.add_argument("--num-graphs", "-W", type=int, default=10, help="Number of graphs to generate")
    parser.add_argument("--vertices", "-n", type=int, default=100, help="Number of vertices per graph")
    parser.add_argument("--edges", "-m", type=int, default=200, help="Number of edges per graph")
    parser.add_argument("--label-range", "-r", type=int, default=5, help="Label range [0, r-1]")
    parser.add_argument("--output-dir", "-o", default="/root", help="Output directory")

    args = parser.parse_args()

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_graphs} graphs...")
    for i in range(args.num_graphs):
        graph_content = generate_random_graph(args.vertices, args.edges, args.label_range)

        filename = f"graph_{i}.txt"
        filepath = os.path.join(args.output_dir, filename)

        with open(filepath, 'w') as f:
            f.write(graph_content)

        print(f"Generated: {filename}")

    print(f"\nDone! Generated {args.num_graphs} graphs in {args.output_dir}")
