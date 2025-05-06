from collections import deque
from typing import Set, Dict


class LCASolver:
    """
    Solve the lowest common ancestor (LCA) of two nodes in given DAG.
    """
    def __init__(self, dag):
        self.dag = dag
        self._depth_cache: Dict[str, int] = {}
        self._ancestor_cache: Dict[str, Set[str]] = {}

        # pre-calculate depths of all nodes
        self._precompute_depths()

    def _precompute_depths(self):
        """calculate the depth (largest path length to root node) of each node"""

        visited = set()

        # find all root nodes
        roots = [n for n in self.dag.nodes if not self.dag.get_prev_nodes(n)]

        # calculate depth with BFS
        queue = deque([(root, 0) for root in roots])
        while queue:
            node, depth = queue.popleft()
            if node in self._depth_cache:
                self._depth_cache[node] = max(self._depth_cache[node], depth)
            else:
                self._depth_cache[node] = depth

            for child in self.dag.get_next_nodes(node):
                queue.append((child, depth + 1))

    def _get_ancestors(self, node: str) -> Set[str]:
        """find all ancestors of node"""
        if node not in self._ancestor_cache:
            ancestors = set()
            stack = [node]
            while stack:
                current = stack.pop()
                if current not in ancestors:
                    ancestors.add(current)
                    stack.extend(self.dag.get_prev_nodes(current))
            self._ancestor_cache[node] = ancestors
        return self._ancestor_cache[node]

    def find_lca(self, node1: str, node2: str) -> str:
        """
        find the lowest common ancestor (LCA) of two nodes
        time complexity: O(D), D represents the difference in depth between two nodes
        """
        # stage 1: check ancestor
        if node1 in self._get_ancestors(node2):
            return node1
        if node2 in self._get_ancestors(node1):
            return node2

        # stage 2: bidirectional BFS
        forward_visited = {node1: 0}  # {node: search_depth}
        backward_visited = {node2: 0}
        forward_queue = deque([(node1, 0)])
        backward_queue = deque([(node2, 0)])

        lca_candidates = []
        current_depth = 0

        while forward_queue and backward_queue:
            # forward search
            f_node, f_depth = forward_queue.popleft()
            if f_depth > current_depth:
                current_depth = f_depth
                if lca_candidates:
                    break

            for parent in self.dag.get_prev_nodes(f_node):
                if parent in backward_visited:
                    lca_candidates.append(parent)
                if parent not in forward_visited:
                    forward_visited[parent] = f_depth + 1
                    forward_queue.append((parent, f_depth + 1))

            # backward search
            b_node, b_depth = backward_queue.popleft()
            for parent in self.dag.get_prev_nodes(b_node):
                if parent in forward_visited:
                    lca_candidates.append(parent)
                if parent not in backward_visited:
                    backward_visited[parent] = b_depth + 1
                    backward_queue.append((parent, b_depth + 1))

        # stage 3：verify candidate node
        if not lca_candidates:
            raise ValueError(f"No LCA between {node1} and {node2}")

        # choose the max depth node as LCA
        return max(lca_candidates, key=lambda x: self._depth_cache[x])