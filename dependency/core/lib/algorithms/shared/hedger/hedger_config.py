from typing import List, Tuple
from dataclasses import dataclass, fields, replace

from core.lib.network import NodeInfo
from core.lib.content import DAG


def from_partial_dict(cls, data: dict):
    allowed = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in allowed}
    return replace(cls(), **filtered)


@dataclass
class OffloadingConstraintCfg:
    allow_stay: bool = True  # Allow execution at the same location of the past one
    forbid_return: bool = True  # Forbidden to return to any historical node
    cloud_sticky: bool = True  # Stay on the cloud after offloading to the cloud
    use_monotone_metric: bool = False  # Enable monotonicity constraint with physical node hops
    metric_non_decreasing: bool = True  # True: metric[node_t] >= metric[last], work only with use_monotone_metric=True
    penalty_switch: float = 0.0  # Node switching penalty coefficient
    penalty_relax: float = 0.0  # Penalty coefficient forced to relax for no feasible action


@dataclass
class DeploymentConstraintCfg:
    enforce_capacity: bool = True  # Enforce memory capacity
    min_edge_replicas: int = 0  # Enforce each service deployed on at least how many edge nodes
    penalty_capacity_relax: float = 1.0  # Penalty for relax the memory capacity


class PhysicalTopology:
    def __init__(self, edge_nodes: list, source_device: str):
        if source_device not in edge_nodes:
            raise ValueError(f"Source device {source_device} is not in edge nodes list {edge_nodes}")
        edge_nodes = edge_nodes.copy()
        edge_nodes.remove(source_device)
        self.nodes = [source_device] + edge_nodes + [NodeInfo.get_cloud_node()]
        self.source_idx = 0
        self.cloud_idx = len(self.nodes) - 1

    def __getitem__(self, item):
        return self.nodes[item]

    def __len__(self):
        return len(self.nodes)

    def index(self, name: str) -> int:
        return self.nodes.index(name)

    @property
    def node_num(self) -> int:
        return len(self.nodes)

    @property
    def links(self) -> List[Tuple[int, int]]:
        edge_nodes = self.nodes[:-1]
        edge_edge_links = []
        edge_cloud_links = []
        for i in range(len(edge_nodes)):
            for j in range(i + 1, len(edge_nodes)):
                edge_edge_links.append((i, j))
                edge_edge_links.append((j, i))
            edge_edge_links.append((i, self.cloud_idx))
        return edge_edge_links + edge_cloud_links


class LogicalTopology:
    def __init__(self, dag: DAG):
        self.dag = dag

        self.service_list = list(self.dag.nodes.keys())
        self.service_list.remove('start')
        self.service_list.remove('end')

    def __len__(self):
        return len(self.dag)

    def __getitem__(self, item):
        return self.service_list[item]

    def index(self, name: str) -> int:
        return self.service_list.index(name)

    @property
    def node_num(self) -> int:
        return len(self.dag)

    @property
    def links(self):
        service_links = []
        for service in self.service_list:
            succ_services = self.dag.get_next_nodes(service)
            for succ in succ_services:
                if succ in self.service_list:
                    service_links.append((self.index(service), self.index(succ)))

        return service_links


