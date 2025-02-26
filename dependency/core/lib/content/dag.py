import json
from typing import List

from .service import Service


class Node:
    def __init__(self, service: Service, prev_nodes=None, next_nodes=None):
        self.service = service
        self.prev_nodes = prev_nodes if prev_nodes else []
        self.next_nodes = next_nodes if next_nodes else []

    def get_prev_nodes(self):
        return self.prev_nodes

    def set_prev_nodes(self, prev_nodes):
        self.prev_nodes = prev_nodes

    def get_next_nodes(self):
        return self.next_nodes

    def set_next_nodes(self, next_nodes):
        self.next_nodes = next_nodes

    def add_next_node(self, next_node: Service):
        self.next_nodes.append(next_node.get_service_name())

    def add_prev_node(self, prev_node: Service):
        self.prev_nodes.append(prev_node.get_service_name())

    def to_dict(self):
        return {
            "service": Service.to_dict(self.service),
            "prev_nodes": self.prev_nodes,
            "next_nodes": self.next_nodes,
        }

    @classmethod
    def from_dict(cls, data: dict):
        node = cls(Service.from_dict(data['service']))
        node.set_prev_nodes(data['prev_nodes']) if 'prev_nodes' in data else None
        node.set_next_nodes(data['next_nodes']) if 'next_nodes' in data else None

        return node

    def serialize(self):
        return json.dumps(self.to_dict())

    @classmethod
    def deserialize(cls, data):
        return cls.from_dict(json.loads(data))

    def __repr__(self):
        return f"{self.service}"


class DAG:
    def __init__(self):
        self.nodes = {}

    def get_node(self, service_name: str) -> Node:
        if service_name not in self.nodes:
            raise KeyError(f"Service {service_name} does not exist in DAG")

        return self.nodes[service_name]

    def set_node_service(self, service_name: str, service: Service):
        self.get_node(service_name).service = service

    def get_next_nodes(self, service_name: str) -> List[Service]:
        if service_name not in self.nodes:
            raise KeyError(f"Service {service_name} does not exist in DAG")

        return self.nodes[service_name].next_nodes

    def get_prev_nodes(self, service_name: str) -> List[Service]:
        if service_name not in self.nodes:
            raise KeyError(f"Service {service_name} does not exist in DAG")

        return self.nodes[service_name].prev_nodes

    def add_node(self, in_node):
        if isinstance(in_node, Node):
            service_name = in_node.service.get_service_name()
            if service_name in self.nodes:
                raise ValueError(f'Node "{in_node}" already exists in DAG')
            self.nodes[service_name] = in_node
        elif isinstance(in_node, Service):
            service_name = in_node.get_service_name()
            if service_name in self.nodes:
                raise ValueError(f'Node "{in_node}" already exists in DAG')
            self.nodes[service_name] = Node(in_node)
        else:
            raise TypeError(f'Expected input type Service or Node, got {type(in_node)}')

    def add_edge(self, from_node: Service, to_node: Service):
        from_node_name = from_node.get_service_name()
        to_node_name = to_node.get_service_name()

        if from_node_name not in self.nodes:
            self.add_node(from_node)
        if to_node_name not in self.nodes:
            self.add_node(to_node)
        self.nodes[from_node_name].add_next_node(to_node)
        self.nodes[to_node_name].add_prev_node(from_node)

    def add_start_node(self, start_node: Service):
        start_node_name = start_node.get_service_name()
        if start_node_name in self.nodes:
            raise ValueError(f'Start Node "{start_node}" already exists in DAG')

        self.add_node(start_node)

        # Collect all child service names
        child_names = set()
        for node in self.nodes.values():
            child_names.update(node.next_nodes)

        # Find nodes without parents (excluding start_node)
        nodes_without_parents = [
            node for node in self.nodes.values()
            if node.service.get_service_name() not in child_names
               and node.service.get_service_name() != start_node_name
        ]

        # Connect start node to parentless nodes
        for node in nodes_without_parents:
            self.add_edge(start_node, node.service)

    def add_end_node(self, end_node: Service):
        end_node_name = end_node.get_service_name()
        if end_node_name in self.nodes:
            raise ValueError(f'End node "{end_node_name}" already exists in DAG')

        self.add_node(end_node)

        # Connect nodes without children to end node
        nodes_without_children = [
            node for node in self.nodes.values()
            if not node.next_nodes and node.service.get_service_name() != end_node_name
        ]

        for node in nodes_without_children:
            self.add_edge(node.service, end_node)

    def validate_dag(self):
        self._check_edge_consistency()
        self._check_duplicate_edges()
        self._check_cycles()
        self._check_connectivity()

    def _check_duplicate_edges(self):
        """check if dag has duplicate edges"""
        edge_set = set()
        for node in self.nodes.values():
            for child_name in node.next_nodes:
                edge = (node.service.get_service_name(), child_name)
                if edge in edge_set:
                    raise ValueError(f"Duplicate edge in the DAG: {edge[0]} -> {edge[1]}")
                edge_set.add(edge)

    def _check_cycles(self):
        """check if dag has cycles"""
        visited = set()
        recursion_stack = set()

        def dfs(cur_service_name):
            if cur_service_name in recursion_stack:
                return True
            if cur_service_name in visited:
                return False
            visited.add(cur_service_name)
            recursion_stack.add(cur_service_name)
            for child_name in self.nodes[cur_service_name].next_nodes:
                if dfs(child_name):
                    return True
            recursion_stack.remove(cur_service_name)
            return False

        for service_name in self.nodes:
            if service_name not in visited:
                if dfs(service_name):
                    raise ValueError(f"Cycle detected in DAG")

    def _check_connectivity(self):
        """check if dag has multiple disconnected components"""
        if not self.nodes:
            return

        visited = set()
        start_name = next(iter(self.nodes.keys()))
        stack = [start_name]
        visited.add(start_name)

        while stack:
            current_name = stack.pop()
            current_node = self.nodes[current_name]
            # Get all neighbors (children and parents)
            neighbors = current_node.next_nodes + current_node.prev_nodes
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        if len(visited) != len(self.nodes):
            raise ValueError("DAG contains multiple disconnected components")

    def _check_edge_consistency(self):
        """
        check if prev_nodes and next_nodes in Node is consistent
        1. Auto-repair missing references
        2. Throw error for conflicting references
        """
        # First pass: Check forward references (next_nodes -> prev_nodes)
        for service_name, node in self.nodes.items():
            # Check next_nodes consistency
            for child_name in node.next_nodes:
                child_node = self.nodes.get(child_name)
                if not child_node:
                    raise ValueError(f"Invalid reference: {service_name} -> {child_name} "
                                     f"(child node doesn't exist)")
                # Repair missing parent reference
                if service_name not in child_node.prev_nodes:
                    child_node.prev_nodes.append(service_name)
            # Check prev_nodes consistency
            for parent_name in node.prev_nodes:
                parent_node = self.nodes.get(parent_name)
                if not parent_node:
                    raise ValueError(f"Invalid reference: {parent_name} <- {service_name} "
                                     f"(parent node doesn't exist)")
                # Repair missing child reference
                if service_name not in parent_node.next_nodes:
                    parent_node.next_nodes.append(service_name)

        # Second pass: Verify full consistency after repairs
        for service_name, node in self.nodes.items():
            # Verify next_nodes -> prev_nodes consistency
            for child_name in node.next_nodes:
                if service_name not in self.nodes[child_name].prev_nodes:
                    raise ValueError(
                        f"Edge mismatch: {service_name} lists {child_name} as child, "
                        f"but {child_name} doesn't list {service_name} as parent"
                    )
            # Verify prev_nodes -> next_nodes consistency
            for parent_name in node.prev_nodes:
                if service_name not in self.nodes[parent_name].next_nodes:
                    raise ValueError(
                        f"Edge mismatch: {service_name} lists {parent_name} as parent, "
                        f"but {parent_name} doesn't list {service_name} as child"
                    )

    def to_dict(self):
        dag_dict = {}
        for service_name, node in self.nodes.items():
            dag_dict[service_name] = node.to_dict()

        return dag_dict

    @classmethod
    def from_dict(cls, dag_dict: dict):
        dag = cls()
        for node_name, node_data in dag_dict.items():
            node = Node.from_dict(node_data)
            dag.add_node(node)
        return dag

    def serialize(self):
        return json.dumps(self.to_dict())

    @classmethod
    def deserialize(cls, data: str):
        dag_dict = json.loads(data)
        return cls.from_dict(dag_dict)

    def __repr__(self):
        return "\n".join(
            [f"{node} -> {[self.get_node(child).service for child in node.next_nodes]}"
             for node in self.nodes.values()]
        )
