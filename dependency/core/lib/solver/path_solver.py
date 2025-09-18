from collections import deque
from typing import List, Dict
import heapq
from typing import Callable, Tuple


class PathSolver:
    """
    Path finding utilities for DAG structure
    Features:
    1. find the shortest path
    2. find the weighted shortest path
    3. find the weighted longest path
    4. find all paths
    """

    def __init__(self, dag):
        """
        Initialize path solver with DAG structure
        :param dag: DAG instance to operate on
        """
        self.dag = dag

    def get_shortest_path(self, src: str, dest: str) -> List[str]:
        """
        Find the shortest path between two nodes using BFS
        :param src: Source node name
        :param dest: Destination node name
        :return: List of nodes in the shortest path
        """

        # Validate node existence
        self._validate_nodes_exist(src, dest)

        # BFS initialization
        predecessors: Dict[str, str] = {}
        queue = deque([src])
        visited = {src}

        # BFS traversal
        while queue:
            current = queue.popleft()
            if current == dest:
                break

            for child in self.dag.get_next_nodes(current):
                if child not in visited:
                    visited.add(child)
                    predecessors[child] = current
                    queue.append(child)

        # Path reconstruction
        if dest not in predecessors and src != dest:
            raise ValueError(f"No path exists from {src} to {dest}")

        path = []
        current = dest
        while current != src:
            path.append(current)
            current = predecessors[current]
        path.append(src)
        reversed_path = path[::-1]

        return reversed_path

    def get_all_paths(self, src: str, dest: str) -> List[List[str]]:
        """
        Find all possible paths between two nodes using DFS
        Note: Use with caution for large DAGs due to exponential complexity
        :return: List of all possible paths
        """
        self._validate_nodes_exist(src, dest)

        paths = []
        stack = [(src, [src])]

        while stack:
            current, path = stack.pop()
            if current == dest:
                paths.append(path)
                continue

            for child in self.dag.get_next_nodes(current):
                if child not in path:  # Prevent cycles
                    stack.append((child, path + [child]))

        if not paths:
            raise ValueError(f"No paths exist from {src} to {dest}")
        return paths

    def get_weighted_shortest_path(self, src: str, dest: str,
                                   weight_func: Callable) -> Tuple[float, List[str]]:
        """
        Find the minimum weight path between two nodes using Dijkstra's algorithm
        :param src: Source node name
        :param dest: Destination node name
        :param weight_func: Function that takes a node service and returns its weight
        :return: Tuple of (total_weight, path_nodes)
        """
        # Validate node existence
        self._validate_nodes_exist(src, dest)

        # Initialize data structures
        distances: Dict[str, float] = {node: float('inf') for node in self.dag.nodes}
        distances[src] = weight_func(self.dag.get_node(src).service)
        predecessors: Dict[str, str] = {}

        # Priority queue: (current_distance, node)
        heap = [(distances[src], src)]

        while heap:
            current_dist, current_node = heapq.heappop(heap)

            # Early exit if destination is reached
            if current_node == dest:
                break

            # Skip processed nodes with outdated distances
            if current_dist > distances[current_node]:
                continue

            # Explore neighbors
            for child in self.dag.get_next_nodes(current_node):
                # Calculate new distance
                new_dist = current_dist + weight_func(self.dag.get_node(child).service)

                # Update if found shorter path
                if new_dist < distances[child]:
                    distances[child] = new_dist
                    predecessors[child] = current_node
                    heapq.heappush(heap, (new_dist, child))

        # Path reconstruction
        if distances[dest] == float('inf'):
            raise ValueError(f"No path exists from {src} to {dest}")

        path = []
        current = dest
        while current != src:
            path.append(current)
            current = predecessors[current]
        path.append(src)

        return distances[dest], path[::-1]

    def get_weighted_longest_path(self, src: str, dest: str,
                                  weight_func: Callable) -> Tuple[float, List[str]]:
        """
        Find the maximum weight path between two nodes using Dijkstra's algorithm
        :param src: Source node name
        :param dest: Destination node name
        :param weight_func: Function that takes a node service and returns its weight
        :return: Tuple of (total_weight, path_nodes)
        """
        # Validate node existence
        self._validate_nodes_exist(src, dest)

        # Initialize data structures
        distances: Dict[str, float] = {node: float('-inf') for node in self.dag.nodes}
        distances[src] = weight_func(self.dag.get_node(src).service)
        predecessors: Dict[str, str] = {}

        # Priority queue: (current_distance, node)
        heap = [(-distances[src], src)]  # Use negative weights to simulate "maximum" path

        while heap:
            current_dist, current_node = heapq.heappop(heap)

            # Convert back the current distance
            current_dist = -current_dist

            # Early exit if destination is reached
            if current_node == dest:
                break

            # Skip processed nodes with outdated distances
            if current_dist < distances[current_node]:
                continue

            # Explore neighbors
            for child in self.dag.get_next_nodes(current_node):
                # Calculate new distance
                new_dist = current_dist + weight_func(self.dag.get_node(child).service)

                # Update if found longer path
                if new_dist > distances[child]:
                    distances[child] = new_dist
                    predecessors[child] = current_node
                    heapq.heappush(heap, (-new_dist, child))  # Push negative distance to max-heap

        # Path reconstruction
        if distances[dest] == float('-inf'):
            raise ValueError(f"No path exists from {src} to {dest}")

        path = []
        current = dest
        while current != src:
            path.append(current)
            current = predecessors[current]
        path.append(src)

        return distances[dest], path[::-1]

    def _validate_nodes_exist(self, *nodes):
        """Validate existence of nodes in DAG"""
        for node in nodes:
            self.dag.get_node(node)
