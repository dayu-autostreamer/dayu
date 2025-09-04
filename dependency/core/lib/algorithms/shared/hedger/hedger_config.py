from typing import List, Tuple
from dataclasses import dataclass, fields, replace


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
        pass

    @property
    def node_num(self) -> int:
        pass

    def edges(self) -> List[Tuple]:
        pass


class LogicalTopology:
    def __init__(self, dag: dict):
        pass

    @property
    def node_num(self) -> int:
        pass
