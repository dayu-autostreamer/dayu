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
    edge_memory_budget_ratio: float = 0.65
    # Retention bonus applied only during capacity projection when removing a
    # selected edge replica would leave that service without any edge replica.
    # The model logit remains the main score; this only breaks small-margin ties
    # toward preserving offloading choice space.
    last_edge_preserve_bonus: float = 0.08
    # Direct matrix deployment policy. The actor scores service-device edge
    # candidates and deterministic inference selects edge pairs whose learned
    # logit is above zero. Projection only repairs hard feasibility.
    queue_normalizer: float = 8.0
    negative_queue_threshold: float = 0.65
    negative_hotspot_threshold: float = 0.08
    negative_runtime_risk_threshold: float = 0.50
    untrusted_unknown_threshold: float = 0.50
    untrusted_stale_threshold: float = 0.85
    positive_quality_threshold: float = 0.30
    # Effective-option labels used by deployment offline calibration.  They
    # define which feasible edge pairs are trained as useful alternatives, not
    # an inference-time rule that adds replicas.
    option_quality_ratio: float = 0.60
    option_quality_tolerance: float = 0.16
    option_pressure_floor: float = 0.25
    # Soft offline targets for the direct Bernoulli matrix. These targets are
    # only used during deployment offline/online learning and debug recording;
    # deterministic inference still uses the learned logit boundary p=0.5.
    soft_target_temperature: float = 0.18
    soft_target_pressure_tolerance: float = 0.30
    soft_target_min: float = 0.04
    soft_target_max: float = 0.92
    # Non-positive labels must stay below the deterministic p=0.5 boundary.
    # This keeps offline soft labels from asking the actor to hold weak pairs
    # near threshold while selected-non-soft losses try to suppress them.
    soft_target_negative_ceiling: float = 0.40
    soft_target_untrusted_weight_floor: float = 0.25
    soft_target_risk_penalty: float = 0.55
    trusted_runtime_confidence_threshold: float = 0.25
    # QK remains an actor feature, but should not teach the offline label by
    # default.  Non-zero values are only for controlled ablations.
    label_qk_prior_weight: float = 0.0
    arch_prior_compute_weight: float = 0.45
    arch_prior_memory_weight: float = 0.25
    arch_prior_memory_fit_weight: float = 0.30
    untrusted_arch_prior_floor: float = 0.18
    untrusted_label_confidence_floor: float = 0.22
    # Historical observed quality still gives weak supervision when runtime
    # evidence is stale/unknown; uncertainty lowers weight and soft target,
    # but does not turn the pair into a hard negative label.
    untrusted_history_quality_weight: float = 0.30
    exploration_quality_threshold: float = 0.35
    exploration_target: float = 0.58
    executed_effective_target_floor: float = 0.72
    executed_effective_weight_bonus: float = 0.75
    service_need_bias_scale: float = 1.0
    # Service-level need should only lift service-device pairs that are plausible
    # options. Otherwise one hot service can push weak/unknown pairs above the
    # Bernoulli p=0.5 boundary together with the good pairs.
    service_need_pair_gate_enabled: bool = True
    service_need_gate_temperature: float = 0.12
    service_need_gate_quality_center: Optional[float] = None
    service_need_gate_min: float = 0.05
    service_need_untrusted_gate_penalty: float = 0.65
    service_need_runtime_risk_gate_weight: float = 0.75
    service_need_memory_gate_weight: float = 0.35
    service_need_pair_bias_max: float = 1.0
    service_mass_temperature: float = 1.0
    service_target_mass_pressure_scale: float = 1.0
    # Budget-aware logit calibration.  This is not a decoder rule: it lets the
    # actor logits see the memory pressure that appears after deterministic
    # p>0.5 matrix selection.
    budget_logit_scale: float = 0.22
    budget_temperature: float = 0.12


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
