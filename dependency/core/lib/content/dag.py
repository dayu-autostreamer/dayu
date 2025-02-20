from .service import Service


class Node:
    def __init__(self, service):
        self.service = service
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    @staticmethod
    def serialize(node: 'Node'):
        return {
            "service": Service.serialize(node.service),
            "children": [Service.serialize(child.service) for child in node.children]
        }

    def __repr__(self):
        return f"{self.service}"


class DAG:
    def __init__(self):
        self.nodes = {}

    def add_node(self, content: Service):
        assert isinstance(content, Service), f"Type of Node error: Expected Service, got {type(content)}"

        if content in self.nodes:
            raise ValueError(f'Node "{content}" already exists in DAG "{DAG.serialize(self)}"')
        self.nodes[content] = Node(content)

    def add_edge(self, from_node: Service, to_node: Service):
        if from_node not in self.nodes:
            self.add_node(from_node)
        if to_node not in self.nodes:
            self.add_node(to_node)
        self.nodes[from_node].add_child(self.nodes[to_node])

    def add_start_node(self, start_node: Service):
        assert start_node not in self.nodes, f'DAG has invalid service name same as the start node: "{start_node}"'

        self.add_node(start_node)

        # set an edge between start node and nodes without parents
        child_nodes = set()
        for node in self.nodes.values():
            for child in node.children:
                child_nodes.add(child.service)

        nodes_without_parents = [
            node for node in self.nodes.values()
            if node.service not in child_nodes and node.service != start_node
        ]

        for node in nodes_without_parents:
            self.add_edge(start_node, node.service)

    def add_end_node(self, end_node: Service):
        assert end_node not in self.nodes, f'DAG has invalid service name same as the end node: "{end_node}"'

        self.add_node(end_node)

        # set an edge between nodes without children and end node
        nodes_without_children = [
            node for node in self.nodes.values()
            if not node.children and node.service != end_node
        ]

        for node in nodes_without_children:
            self.add_edge(node.service, end_node)

    def validate_dag(self):
        self._check_duplicate_edges()
        self._check_cycles()

    def _check_duplicate_edges(self):
        """check if dag has duplicate edges"""
        edge_set = set()
        for node in self.nodes.values():
            for child in node.children:
                edge = (node.service, child.service)
                if edge in edge_set:
                    raise ValueError(f"There is a duplicate edge in the DAG: {edge[0]} -> {edge[1]}")
                edge_set.add(edge)

    def _check_cycles(self):
        """check if dag has cycles"""
        visited = set()
        recursion_stack = set()

        def dfs(current_service):

            if current_service in recursion_stack:
                return True
            if current_service in visited:
                return False

            visited.add(current_service)
            recursion_stack.add(current_service)

            for child in self.nodes[current_service].children:
                if dfs(child.service):
                    return True

            recursion_stack.remove(current_service)
            return False

        for service, node in self.nodes.items():
            if dfs(node.service):
                raise ValueError(f"There is a cycle in the DAG, involving nodes: {service}")

    @staticmethod
    def serialize(dag: 'DAG'):
        dag_dict = {}
        for node_name, node in dag.nodes.items():
            dag_dict[node_name] = Node.serialize(node)
        return dag_dict

    @staticmethod
    def deserialize(dag_dict: dict):
        dag = DAG()
        for node_name, node_data in dag_dict.items():
            service = Service.deserialize(node_data["service"])
            dag.add_node(service)
            for child_data in node_data["children"]:
                child_service = Service(child_data["service"])
                dag.add_edge(service, child_service)
        dag.validate_dag()
        return dag

    def __repr__(self):
        return "\n".join([f"{node} -> {[child.service for child in node.children]}" for node in self.nodes.values()])
