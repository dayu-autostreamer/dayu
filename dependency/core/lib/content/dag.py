from typing import List

from .service import Service


class Node:
    def __init__(self, service: Service):
        self.service = service
        self.prev_nodes = []
        self.next_nodes = []

    def add_next_node(self, next_node: Service):
        self.next_nodes.append(next_node.get_service_name())

    def add_prev_node(self, prev_node: Service):
        self.prev_nodes.append(prev_node.get_service_name())

    @staticmethod
    def serialize(node: 'Node'):
        return {
            "service": Service.serialize(node.service),
            "prev_nodes": node.prev_nodes,
            "next_nodes": node.next_nodes,
        }

    def __repr__(self):
        return f"{self.service}"


class DAG:
    def __init__(self):
        self.nodes = {}

    def get_node(self, service_name: str) -> Node:
        if service_name not in self.nodes:
            raise KeyError(f"Service {service_name} does not exist in DAG")

        return self.nodes[service_name]

    def get_next_nodes(self, service_name: str) -> List[Service]:
        if service_name not in self.nodes:
            raise KeyError(f"Service {service_name} does not exist in DAG")

        return self.nodes[service_name].next_nodes

    def get_prev_nodes(self, service_name: str) -> List[Service]:
        if service_name not in self.nodes:
            raise KeyError(f"Service {service_name} does not exist in DAG")

        return self.nodes[service_name].prev_nodes

    def add_node(self, service: Service):
        assert isinstance(service, Service), f"Type error: Expected Service, got {type(service)}"

        service_name = service.get_service_name()
        if service_name in self.nodes:
            raise ValueError(f'Node "{service}" already exists in DAG')
        self.nodes[service_name] = Node(service)

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
        self._check_duplicate_edges()
        self._check_cycles()
        self._check_connectivity()

    def _check_duplicate_edges(self):
        """check if dag has duplicate edges"""
        edge_set = set()
        for node in self.nodes.values():
            for child_name in node.next_nodes:
                child = self.get_node(child_name)
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

    @staticmethod
    def serialize(dag: 'DAG'):
        dag_dict = {}
        for service_name, node in dag.nodes.items():
            dag_dict[service_name] = Node.serialize(node)
        return dag_dict

    @staticmethod
    def deserialize(dag_dict: dict):
        dag = DAG()
        for node_name, node_data in dag_dict.items():
            service = Service.deserialize(node_data["service"])
            dag.add_node(service)

        for node_name, node_data in dag_dict.items():
            for child_name in node_data["next_nodes"]:
                node_service = dag.get_node(node_name).service
                child_service = dag.get_node(child_name).service
                dag.add_edge(node_service, child_service)
        dag.validate_dag()
        return dag

    def __repr__(self):
        return "\n".join(
            [f"{node} -> {[self.get_node(child).service for child in node.next_nodes]}"
             for node in self.nodes.values()]
        )
