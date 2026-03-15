import argparse
import random
import os
from pathlib import Path


def generate_random_directed_graph(n, m, r):
    """生成随机有向图（保证从顶点0出发能访问所有顶点，避免双向边）"""
    # 确保边数足够形成连通图
    min_edges_for_connected = n - 1  # 最小生成树的边数
    max_edges = n * (n - 1)  # 有向图最大边数

    if m < min_edges_for_connected:
        print(f"警告: 边数{m}不足以形成连通图，自动调整为{n - 1}")
        m = min_edges_for_connected

    m = min(m, max_edges)

    vertices = []
    out_degrees = [0] * n
    in_degrees = [0] * n

    # 生成顶点
    for i in range(n):
        label = random.randint(0, r - 1)
        vertices.append((i, label))

    # 生成边 - 分两步：先构建生成树保证连通性，再添加剩余边
    edges = set()
    used_undirected_pairs = set()  # 记录已经使用的无向边对

    # 步骤1: 构建以顶点0为根的有向生成树（使用BFS顺序）
    print(f"构建有向生成树，确保从顶点0可达所有顶点...")
    visited = [False] * n
    visited[0] = True
    unvisited = set(range(1, n))

    # 从顶点0开始构建树
    current_frontier = [0]

    while unvisited:
        if not current_frontier:
            # 如果当前边界为空但还有未访问节点，随机连接一个
            u = random.choice(list(range(n)))
            v = random.choice(list(unvisited))
            edge = (u, v)

            # 检查是否会产生双向边
            if (v, u) not in edges:
                edges.add(edge)
                used_undirected_pairs.add(frozenset({u, v}))
                out_degrees[u] += 1
                in_degrees[v] += 1
                visited[v] = True
                unvisited.remove(v)
                current_frontier = [v]
                print(f"  添加有向边: {u} -> {v} (随机连接)")
            continue

        next_frontier = []
        for u in current_frontier:
            if not unvisited:
                break

            # 为当前节点随机连接1-3个未访问的子节点
            max_children = min(3, len(unvisited))
            num_children = random.randint(1, max_children)

            children = random.sample(list(unvisited), num_children)
            for v in children:
                edge = (u, v)

                # 检查是否会产生双向边
                if (v, u) not in edges:
                    edges.add(edge)
                    used_undirected_pairs.add(frozenset({u, v}))
                    out_degrees[u] += 1
                    in_degrees[v] += 1
                    visited[v] = True
                    unvisited.remove(v)
                    next_frontier.append(v)
                    print(f"  添加有向边: {u} -> {v}")
                else:
                    print(f"  跳过边 {u}->{v}，因为反向边 {v}->{u} 已存在")

        current_frontier = next_frontier

    # 步骤2: 添加剩余的随机有向边（避免双向边）
    print("添加剩余随机边，避免双向边...")
    possible_edges = []

    # 生成所有可能的边，排除已存在的边和会产生双向边的边
    for i in range(n):
        for j in range(n):
            if i != j:
                edge = (i, j)
                undirected_pair = frozenset({i, j})

                # 检查：边不存在 且 不会产生双向边
                if (edge not in edges and
                        undirected_pair not in used_undirected_pairs):
                    possible_edges.append(edge)

    # 随机选择剩余的边
    remaining_edges_needed = m - len(edges)
    if remaining_edges_needed > 0 and possible_edges:
        additional_edges = random.sample(possible_edges, min(remaining_edges_needed, len(possible_edges)))
        for u, v in additional_edges:
            edges.add((u, v))
            used_undirected_pairs.add(frozenset({u, v}))  # 标记这个无向对已使用
            out_degrees[u] += 1
            in_degrees[v] += 1
            print(f"  添加随机有向边: {u} -> {v}")

    # 验证从顶点0出发的可达性
    is_connected_from_0 = check_connectivity_from_vertex_0(n, edges)
    print(f"从顶点0出发的可达性验证: {'通过' if is_connected_from_0 else '失败'}")

    if not is_connected_from_0:
        print("错误: 从顶点0无法到达所有顶点！")
        # 修复连通性：添加缺失的边（避免双向边）
        missing_vertices = find_unreachable_vertices_from_0(n, edges)
        for v in missing_vertices:
            # 随机选择一个已可达的顶点连接到缺失的顶点
            attempts = 0
            max_attempts = 10  # 避免无限循环
            while attempts < max_attempts:
                u = random.choice(list(range(n)))
                edge = (u, v)
                undirected_pair = frozenset({u, v})

                # 检查是否会产生双向边
                if u != v and edge not in edges and undirected_pair not in used_undirected_pairs:
                    edges.add(edge)
                    used_undirected_pairs.add(undirected_pair)
                    out_degrees[u] += 1
                    in_degrees[v] += 1
                    print(f"  修复连通性: 添加边 {u} -> {v}")
                    break
                attempts += 1

            if attempts >= max_attempts:
                print(f"  警告: 无法为顶点 {v} 找到合适的连接边")

        # 再次验证
        is_connected_from_0 = check_connectivity_from_vertex_0(n, edges)
        print(f"修复后可达性验证: {'通过' if is_connected_from_0 else '失败'}")

    # 验证没有双向边
    has_bidirectional_edges = check_bidirectional_edges(edges)
    print(f"双向边检查: {'通过' if not has_bidirectional_edges else '失败'}")

    # 构建文件内容
    content = [f"t {n} {len(edges)}"]  # 使用实际的边数
    for i, label in vertices:
        content.append(f"v {i} {label} {out_degrees[i]}")  # 使用出度
    for u, v in sorted(edges):
        content.append(f"e {u} {v}")  # 有向边

    return "\n".join(content)


def check_bidirectional_edges(edges):
    """检查是否存在双向边"""
    edge_set = set(edges)
    for u, v in edges:
        if (v, u) in edge_set:
            print(f"发现双向边: {u}<->{v}")
            return True
    return False


def check_connectivity_from_vertex_0(n, edges):
    """检查从顶点0出发是否能访问所有顶点"""
    if n == 0:
        return True

    # 构建邻接表
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)

    # BFS检查从顶点0出发的可达性
    visited = [False] * n
    queue = [0]
    visited[0] = True
    count = 1

    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)

    return count == n


def find_unreachable_vertices_from_0(n, edges):
    """找到从顶点0不可达的顶点"""
    # 构建邻接表
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)

    # BFS找到所有可达顶点
    visited = [False] * n
    queue = [0]
    visited[0] = True

    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                queue.append(v)

    # 返回不可达的顶点
    unreachable = []
    for i in range(n):
        if not visited[i]:
            unreachable.append(i)

    return unreachable


def generate_bfs_tree_from_0(n, edges):
    """生成从顶点0出发的BFS树"""
    # 构建邻接表
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)

    # 使用BFS构建树
    visited = [False] * n
    parent = [-1] * n
    tree_edges = []

    queue = [0]
    visited[0] = True

    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                tree_edges.append((u, v))
                queue.append(v)

    return tree_edges


def generate_directed_graph_with_bfs_tree(n, m, r):
    """生成有向图并返回图和其BFS树"""
    graph_content = generate_random_directed_graph(n, m, r)

    # 从图内容中解析边
    edges = set()
    lines = graph_content.split('\n')
    for line in lines:
        if line.startswith('e '):
            parts = line.split()
            u = int(parts[1])
            v = int(parts[2])
            edges.add((u, v))

    # 生成BFS树
    tree_edges = generate_bfs_tree_from_0(n, edges)

    # 构建树内容
    tree_content = [f"# 从顶点0出发的BFS树 (边数: {len(tree_edges)})"]
    tree_content.append(f"t {n} {len(tree_edges)}")

    # 复制顶点信息
    for line in lines:
        if line.startswith('v '):
            tree_content.append(line)

    for u, v in sorted(tree_edges):
        tree_content.append(f"e {u} {v}")

    return graph_content, "\n".join(tree_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate random directed connected graphs without bidirectional edges")
    parser.add_argument("--num-graphs", "-W", type=int, default=10, help="Number of graphs to generate")
    parser.add_argument("--start-graph-id", "-i", type=int, default=1, help="Start id of graphs")
    parser.add_argument("--vertices", "-n", type=int, default=100, help="Number of vertices per graph")
    parser.add_argument("--edges", "-m", type=int, default=200, help="Number of edges per graph")
    parser.add_argument("--label-range", "-r", type=int, default=5, help="Label range [0, r-1]")
    parser.add_argument("--output-dir", "-o", default=".", help="Output directory (relative or absolute path)")
    parser.add_argument("--with-bfs-tree", action="store_true",
                        help="Also generate BFS tree representation from vertex 0")

    args = parser.parse_args()

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_graphs} directed connected graphs starting from ID {args.start_graph_id}...")
    print(f"Each graph has {args.vertices} vertices and {args.edges} edges")
    print(f"Ensuring all vertices are reachable from vertex 0")
    print(f"Ensuring no bidirectional edges exist")

    generated_count = 0
    for i in range(args.num_graphs):
        try:
            if args.with_bfs_tree:
                graph_content, tree_content = generate_directed_graph_with_bfs_tree(
                    args.vertices, args.edges, args.label_range)
            else:
                graph_content = generate_random_directed_graph(args.vertices, args.edges, args.label_range)

            if graph_content is None:
                print(f"Error generating graph {i}, skipping...")
                continue

            # 使用 start_graph_id 作为起始 ID
            graph_id = args.start_graph_id + i
            filename = f"graph_{graph_id}.txt"
            filepath = os.path.join(args.output_dir, filename)

            with open(filepath, 'w') as f:
                f.write(graph_content)

            if args.with_bfs_tree:
                tree_filename = f"bfs_tree_{graph_id}.txt"
                tree_filepath = os.path.join(args.output_dir, tree_filename)
                with open(tree_filepath, 'w') as f:
                    f.write(tree_content)
                print(f"Generated: {filename} + {tree_filename}")
            else:
                print(f"Generated: {filename}")

            generated_count += 1

        except Exception as e:
            print(f"Error generating graph {i}: {e}")
            continue

    print(
        f"\nDone! Generated {generated_count} directed connected graphs in {args.output_dir} "
        f"(IDs: {args.start_graph_id} to {args.start_graph_id + generated_count - 1})")
    print("All graphs guarantee:")
    print("  - Vertex 0 can reach every other vertex")
    print("  - No bidirectional edges exist")
