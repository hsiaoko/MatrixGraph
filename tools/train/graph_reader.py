#!/usr/bin/env python3
"""
Graph reader utility to convert graph.txt files into Python Graph objects.
Format:
t <number of vertices> <number of edges>
v <vertex-id> <vertex-label> <vertex-degree>
e <edge-id-1> <edge-id-2>
"""

from typing import Dict, List, Set, Tuple
import networkx as nx


class GraphReader:
    def __init__(self):
        self.graph = nx.Graph()
        self.vertex_labels: Dict[int, int] = {}  # vertex_id -> label
        self.vertex_degrees: Dict[int, int] = {}  # vertex_id -> degree

    def read_graph(self, filepath: str) -> nx.Graph:
        """Read a graph from the specified file path."""
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Process each line
        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if parts[0] == 't':
                # New graph header
                num_vertices = int(parts[1])
                num_edges = int(parts[2])
                self.graph = nx.Graph()
                self.vertex_labels.clear()
                self.vertex_degrees.clear()

            elif parts[0] == 'v':
                # Vertex definition
                vertex_id = int(parts[1])
                vertex_label = int(parts[2])
                vertex_degree = int(parts[3])

                self.graph.add_node(vertex_id)
                self.vertex_labels[vertex_id] = vertex_label
                self.vertex_degrees[vertex_id] = vertex_degree
                # Add vertex attributes
                self.graph.nodes[vertex_id]['label'] = vertex_label
                self.graph.nodes[vertex_id]['degree'] = vertex_degree

            elif parts[0] == 'e':
                # Edge definition
                src = int(parts[1])
                dst = int(parts[2])
                self.graph.add_edge(src, dst)

        return self.graph

    def get_vertex_labels(self) -> Dict[int, int]:
        """Get the vertex labels dictionary."""
        return self.vertex_labels

    def get_vertex_degrees(self) -> Dict[int, int]:
        """Get the vertex degrees dictionary."""
        return self.vertex_degrees

    def print_graph_info(self, k = 3):
        """Print information about the loaded graph."""
        print(f"Graph Information:")
        print(f"Number of vertices: {self.graph.number_of_nodes()}")
        print(f"Number of edges: {self.graph.number_of_edges()}")
        print("\nVertex Labels:")
        k = min(3, len(self.vertex_labels))
        c = 0
        for vertex, label in self.vertex_labels.items():
            if(c > k):break
            print(f"Vertex {vertex}: Label={label}, Degree={self.vertex_degrees[vertex]}")
            c = c + 1
        print("\nEdges:")
        k = min(3, len(self.graph.edges()))
        c = 0
        for edge in self.graph.edges():
            if(c > k):break
            print(f"Edge: {edge[0]} -> {edge[1]}")
            c = c + 1


def main():
    """Example usage of the GraphReader class."""
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <graph_file>")
        sys.exit(1)

    reader = GraphReader()
    graph = reader.read_graph(sys.argv[1])
    reader.print_graph_info()

    # Example of using the graph object
    print("\nGraph Analysis:")
    print(f"Is connected: {nx.is_connected(graph)}")
    if nx.is_connected(graph):
        print(f"Average shortest path length: {nx.average_shortest_path_length(graph)}")
    print(f"Average clustering coefficient: {nx.average_clustering(graph)}")


if __name__ == "__main__":
    main()
