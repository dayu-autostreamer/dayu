from collections import deque
from typing import Set, Dict


class IntermediateNodeSolver:
    """
    Intermediate nodes detection utilities for DAG structure
    """

    def __init__(self, dag):
        self.dag = dag

    def get_intermediate_nodes(self, src: str, dest: str) -> Set[str]:
        """
        Find all intermediate nodes between two nodes
        :return: Set of nodes that exist on any path from src to dest
        """

        self._validate_nodes_exist(src, dest)

        # Get forward and reverse reachable nodes
        forward_reachable = self._get_forward_reachable(src)
        if dest not in forward_reachable:
            return set()

        reverse_reachable = self._get_reverse_reachable(dest)

        # Calculate intersection
        intermediates = forward_reachable & reverse_reachable
        intermediates -= {src, dest}

        return intermediates

    def _get_forward_reachable(self, node: str) -> Set[str]:
        """Get all nodes reachable from given node (forward BFS)"""
        visited = set()
        queue = deque([node])
        visited.add(node)

        while queue:
            current = queue.popleft()
            for child in self.dag.get_next_nodes(current):
                if child not in visited:
                    visited.add(child)
                    queue.append(child)

        return visited

    def _get_reverse_reachable(self, node: str) -> Set[str]:
        """Get all nodes that can reach given node (reverse BFS)"""
        visited = set()
        queue = deque([node])
        visited.add(node)

        while queue:
            current = queue.popleft()
            for parent in self.dag.get_prev_nodes(current):
                if parent not in visited:
                    visited.add(parent)
                    queue.append(parent)

        return visited

    def _validate_nodes_exist(self, *nodes):
        """Validate existence of nodes in DAG"""
        for node in nodes:
            self.dag.get_node(node)
