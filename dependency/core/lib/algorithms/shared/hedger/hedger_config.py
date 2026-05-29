from typing import List, Optional, Tuple
from dataclasses import dataclass, fields, replace

from core.lib.network import NodeInfo
from core.lib.content import DAG
from core.lib.common import TaskConstant


def from_partial_dict(cls, data: dict):
    data = data or {}
    allowed = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in allowed}
    return replace(cls(), **filtered)


@dataclass
class DeploymentConstraintCfg:
    penalty_capacity_relax: float = 1.0  # Penalty coefficient for deployment capacity corrections
    # Uniform cap for the number of edge replicas placed on one edge node.
    # None disables the cap. This is intentionally not per-service or per-node.
    max_edge_replicas_per_device: Optional[int] = None
    # Fraction of computed available edge memory used as deployment budget.
    # Values below 1.0 reserve memory for the OS/runtime and rollout overhead.
    edge_memory_budget_ratio: float = 1.0
    # Pair-wise Bernoulli matrix policy. The actor proposes the service-device
    # edge replica matrix directly; deterministic inference uses the Bernoulli
    # mode (logit > 0), and projection only corrects memory/count infeasibility.
    # Unknown/stale runtime evidence is diagnostic only; state quality falls
    # back to static suitability when fresh runtime evidence is missing.
    queue_normalizer: float = 8.0
    negative_queue_threshold: float = 0.65
    negative_hotspot_threshold: float = 0.08
    negative_runtime_risk_threshold: float = 0.50
    negative_unknown_threshold: float = 0.50
    negative_stale_threshold: float = 0.85
    positive_quality_threshold: float = 0.20


class PhysicalTopology:
    def __init__(self, edge_nodes: list, source_device: str):
        cloud_node = NodeInfo.get_cloud_node()
        edge_nodes = list(dict.fromkeys(edge_nodes or []))
        if not edge_nodes:
            if source_device == cloud_node:
                self.nodes = [cloud_node]
            else:
                self.nodes = [source_device, cloud_node]
            self.source_idx = 0
            self.cloud_idx = len(self.nodes) - 1
            return

        self.nodes = edge_nodes + [cloud_node]
        self.source_idx = edge_nodes.index(source_device) if source_device in edge_nodes else 0
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
        if TaskConstant.START.value in self.service_list:
            self.service_list.remove(TaskConstant.START.value)
        if TaskConstant.END.value in self.service_list:
            self.service_list.remove(TaskConstant.END.value)

    def __len__(self):
        return len(self.service_list)

    def __getitem__(self, item):
        return self.service_list[item]

    def index(self, name: str) -> int:
        return self.service_list.index(name)

    @property
    def node_num(self) -> int:
        return len(self.service_list)

    @property
    def links(self):
        service_links = []
        for service in self.service_list:
            succ_services = self.dag.get_next_nodes(service)
            for succ in succ_services:
                if succ in self.service_list:
                    service_links.append((self.index(service), self.index(succ)))

        return service_links
