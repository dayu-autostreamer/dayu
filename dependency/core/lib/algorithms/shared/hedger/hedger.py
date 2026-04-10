import copy
from typing import Dict, List, Optional
import threading
import random
import math
from dataclasses import dataclass
import torch
import time
import os
import glob

from core.lib.common import LOGGER, FileOps, Context, Recorder

from .topology_encoder import TopologyEncoders
from .ppo_agent import HedgerOffloadingPPO, HedgerDeploymentPPO
from .hedger_config import from_partial_dict, OffloadingConstraintCfg, DeploymentConstraintCfg, LogicalTopology, \
    PhysicalTopology
from .state_buffer import StateBuffer, BufferWaitCfg

__all__ = ('Hedger',)


@dataclass(frozen=True)
class HedgerStateCfg:
    offloading_seq_len: int
    deployment_seq_len: int
    min_dynamic_len: int
    wait_timeout_s: Optional[float]
    require_full_seq: bool
    latency_slo: Optional[float]
    deployment_reward_window: int


@dataclass(frozen=True)
class HedgerCheckpointCfg:
    load_encoder: bool = True
    load_deployment_agent: bool = True
    load_offloading_agent: bool = True
    load_optimizer: bool = True
    reset_steps_on_load: bool = False


class Hedger:
    def __init__(self, network_params: dict, hyper_params: dict, agent_params: dict):
        self.encoder_network_params = network_params['topology_encoder']

        self.mode = hyper_params['mode']
        self.device = torch.device(hyper_params['device'])
        self.seed = int(hyper_params.get('seed', 0))
        self.deployment_interval = float(hyper_params['deployment_interval'])
        self.offloading_interval = float(hyper_params['offloading_interval'])
        self.update_epochs = max(1, int(hyper_params.get('update_epochs', 1)))
        self.total_steps = max(0, int(hyper_params.get('total_steps', 0)))
        self.model_dir = Context.get_file_path(hyper_params['model_dir'])
        self.save_interval = max(1, int(hyper_params.get('save_interval', 1)))

        self.checkpoint_cfg = self._build_checkpoint_cfg(hyper_params)

        self.train_deployment_flag = bool(hyper_params.get("train_deployment", True))
        self.train_offloading_flag = bool(hyper_params.get("train_offloading", True))

        self.offloading_rollout_len = max(1, int(hyper_params.get("offloading_rollout_len", 32)))
        self.deployment_rollout_len = max(1, int(hyper_params.get("deployment_rollout_len", 8)))
        self.offloading_batch_size = max(1, int(hyper_params.get("offloading_batch_size", 16)))
        self.deployment_batch_size = max(1, int(hyper_params.get("deployment_batch_size", 4)))
        self.max_state_buffer_size = max(1, int(hyper_params.get("max_state_buffer_size", 1000)))
        self.state_cfg = self._build_state_cfg(hyper_params)

        self.offloading_agent_params = agent_params['offloading_agent']
        self.deployment_agent_params = agent_params['deployment_agent']

        self.deployment_thread_stop_event = threading.Event()
        self.offloading_thread_stop_event = threading.Event()

        self.physical_topology = None
        self.logical_topology = None

        self.shared_topology_encoder = None
        self.deployment_agent = None
        self.offloading_agent = None

        self.state_buffer = None

        self._deployment_update_steps = 0
        self._offloading_update_steps = 0
        # Global training epoch counter, incremented after PPO updates.
        self._epoch = 0

        self.register_topology_encoder()
        self.register_deployment_agent()
        self.register_offloading_agent()

        self._data_lock = threading.Lock()

        FileOps.create_directory(self.model_dir)
        if bool(hyper_params.get('load_model', False)):
            self.load_checkpoint(epoch=hyper_params.get('load_epoch'))

        self.initial_deployment_plan = None
        self.deployment_plan = None
        self.offloading_plan = None

        self.deployment_transitions: List[dict] = []
        self.offloading_transitions: List[dict] = []

        self.dep_recorder = None
        self.off_recorder = None

        self.cur_deploy_mask = None

        threading.Thread(target=self.run, daemon=True).start()

    def set_seed(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def _build_checkpoint_cfg(self, hyper_params: dict) -> HedgerCheckpointCfg:
        return HedgerCheckpointCfg(
            load_encoder=bool(hyper_params.get("load_encoder", True)),
            load_deployment_agent=bool(hyper_params.get("load_deployment_agent", True)),
            load_offloading_agent=bool(hyper_params.get("load_offloading_agent", True)),
            load_optimizer=bool(hyper_params.get("load_optimizer", True)),
            reset_steps_on_load=bool(hyper_params.get("reset_steps_on_load", False)),
        )

    def _build_state_cfg(self, hyper_params: dict) -> HedgerStateCfg:
        default_seq_len = max(1, int(hyper_params.get("state_seq_len", 8)))
        reward_window_default = max(
            1,
            int(math.ceil(self.deployment_interval / max(self.offloading_interval, 1e-6))),
        )
        latency_slo = hyper_params.get("latency_slo", hyper_params.get("slo_threshold"))
        latency_slo = None if latency_slo is None else float(latency_slo)
        wait_timeout_s = hyper_params.get("state_wait_timeout_s", 1.0)
        wait_timeout_s = None if wait_timeout_s is None else max(0.0, float(wait_timeout_s))
        return HedgerStateCfg(
            offloading_seq_len=max(1, int(hyper_params.get("offloading_state_seq_len", default_seq_len))),
            deployment_seq_len=max(1, int(hyper_params.get("deployment_state_seq_len", default_seq_len))),
            min_dynamic_len=max(0, int(hyper_params.get("state_min_dynamic_len", 1))),
            wait_timeout_s=wait_timeout_s,
            require_full_seq=bool(hyper_params.get("state_require_full_seq", False)),
            latency_slo=latency_slo,
            deployment_reward_window=max(1, int(hyper_params.get("deployment_reward_window", reward_window_default))),
        )

    def _sync_agent_topology_bindings(self):
        """Synchronize agent-side source/cloud indices with the registered physical topology."""
        if self.physical_topology is None:
            return
        if self.deployment_agent is not None:
            self.deployment_agent.cloud_idx = self.physical_topology.cloud_idx
        if self.offloading_agent is not None:
            self.offloading_agent.source = self.physical_topology.source_idx
            self.offloading_agent.cloud_idx = self.physical_topology.cloud_idx

    @staticmethod
    def _format_log_value(value, digits: int = 4) -> str:
        if value is None:
            return "none"
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, int):
            return str(value)
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            return str(value)
        if math.isfinite(value_float):
            return f"{value_float:.{digits}f}"
        return str(value_float)

    def _summarize_runtime_config(self) -> str:
        return (
            f"mode={self.mode}, device={self.device}, seed={self.seed}, "
            f"intervals(dep/off)={self._format_log_value(self.deployment_interval, 2)}/"
            f"{self._format_log_value(self.offloading_interval, 2)}s, "
            f"rollout(dep/off)={self.deployment_rollout_len}/{self.offloading_rollout_len}, "
            f"batch(dep/off)={self.deployment_batch_size}/{self.offloading_batch_size}, "
            f"state_seq(dep/off)={self.state_cfg.deployment_seq_len}/{self.state_cfg.offloading_seq_len}, "
            f"total_steps={self.total_steps}, save_interval={self.save_interval}"
        )

    def _summarize_topology(self) -> str:
        logical_services = len(self.logical_topology) if self.logical_topology is not None else 0
        logical_edges = len(self.logical_topology.links) if self.logical_topology is not None else 0
        physical_nodes = len(self.physical_topology) if self.physical_topology is not None else 0
        physical_edges = len(self.physical_topology.links) if self.physical_topology is not None else 0
        source_name = (
            self.physical_topology[self.physical_topology.source_idx]
            if self.physical_topology is not None and physical_nodes > 0
            else "unregistered"
        )
        cloud_name = (
            self.physical_topology[self.physical_topology.cloud_idx]
            if self.physical_topology is not None and physical_nodes > 0
            else "unregistered"
        )
        return (
            f"logical_services={logical_services}, logical_edges={logical_edges}, "
            f"physical_nodes={physical_nodes}, physical_edges={physical_edges}, "
            f"source={source_name}, cloud={cloud_name}"
        )

    def _summarize_metrics(self, metrics: Optional[Dict[str, float]]) -> str:
        if not metrics:
            return "none"
        return ", ".join(
            f"{key}={self._format_log_value(value)}"
            for key, value in sorted(metrics.items())
        )

    def _summarize_deploy_mask(self, deploy_mask: Optional[torch.Tensor]) -> str:
        if deploy_mask is None:
            return "services=0, edge_replicas=0, cloud_replicas=0, cloud_only=0"

        mask = deploy_mask.detach().clone().cpu().bool()
        if mask.numel() == 0 or self.physical_topology is None:
            return "services=0, edge_replicas=0, cloud_replicas=0, cloud_only=0"

        cloud_idx = self.physical_topology.cloud_idx
        edge_mask = mask[:, :cloud_idx]
        edge_replicas = int(edge_mask.sum().item()) if edge_mask.numel() else 0
        cloud_replicas = int(mask[:, cloud_idx].sum().item())
        cloud_only = int((~edge_mask.any(dim=1)).sum().item()) if edge_mask.numel() else int(mask.size(0))
        return (
            f"services={mask.size(0)}, edge_replicas={edge_replicas}, "
            f"cloud_replicas={cloud_replicas}, cloud_only={cloud_only}"
        )

    def _summarize_deployment_plan(self, deployment_plan: Optional[dict], sample_size: int = 3) -> str:
        if not deployment_plan:
            return "services=0, replicas=0, edge_replicas=0, sample=[]"

        cloud_name = None
        if self.physical_topology is not None and len(self.physical_topology) > 0:
            cloud_name = self.physical_topology[self.physical_topology.cloud_idx]

        total_replicas = 0
        edge_replicas = 0
        sample_parts = []
        for idx, (service_name, device_names) in enumerate(deployment_plan.items()):
            if isinstance(device_names, (list, tuple, set)):
                devices = list(device_names)
            else:
                devices = [device_names]
            total_replicas += len(devices)
            if cloud_name is not None:
                edge_replicas += sum(1 for device_name in devices if device_name != cloud_name)
            if idx < sample_size:
                sample_parts.append(f"{service_name}->[{','.join(map(str, devices))}]")
        sample_text = "; ".join(sample_parts) if sample_parts else "[]"
        return (
            f"services={len(deployment_plan)}, replicas={total_replicas}, "
            f"edge_replicas={edge_replicas}, sample={sample_text}"
        )

    def _summarize_offloading_plan(self, offloading_plan: Optional[dict], sample_size: int = 3) -> str:
        if not offloading_plan:
            return "services=0, cloud=0, unique_targets=0, sample=[]"

        cloud_name = None
        if self.physical_topology is not None and len(self.physical_topology) > 0:
            cloud_name = self.physical_topology[self.physical_topology.cloud_idx]

        cloud_assignments = 0
        unique_targets = set()
        sample_parts = []
        for idx, (service_name, device_name) in enumerate(offloading_plan.items()):
            if device_name == cloud_name:
                cloud_assignments += 1
            unique_targets.add(device_name)
            if idx < sample_size:
                sample_parts.append(f"{service_name}->{device_name}")
        sample_text = "; ".join(sample_parts) if sample_parts else "[]"
        return (
            f"services={len(offloading_plan)}, cloud={cloud_assignments}, "
            f"unique_targets={len(unique_targets)}, sample={sample_text}"
        )

    def _summarize_state_snapshot(
            self,
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        service_count = int(logic_feats["model_flops"].numel()) if "model_flops" in logic_feats else 0
        device_count = int(phys_feats["gpu_flops"].numel()) if "gpu_flops" in phys_feats else 0

        latest_complexity = 0.0
        if "task_complexity_seq" in logic_feats and logic_feats["task_complexity_seq"].numel():
            latest_complexity = float(logic_feats["task_complexity_seq"][:, -1].mean().item())

        latest_latency = 0.0
        if "hist_latency_seq" in logic_feats and logic_feats["hist_latency_seq"].numel():
            latest_latency = float(logic_feats["hist_latency_seq"][:, -1].mean().item())

        latest_gpu_util = 0.0
        if "gpu_util_seq" in phys_feats and phys_feats["gpu_util_seq"].numel():
            latest_gpu_util = float(phys_feats["gpu_util_seq"][:, -1].mean().item())

        latest_mem_util = 0.0
        if "mem_util_seq" in phys_feats and phys_feats["mem_util_seq"].numel():
            latest_mem_util = float(phys_feats["mem_util_seq"][:, -1].mean().item())

        latest_cloud_bw = 0.0
        latest_edge_bw = 0.0
        if "bandwidth_seq" in phys_feats and phys_feats["bandwidth_seq"].numel():
            bandwidth_seq = phys_feats["bandwidth_seq"]
            cloud_idx = self.physical_topology.cloud_idx if self.physical_topology is not None else bandwidth_seq.size(0) - 1
            latest_cloud_bw = float(bandwidth_seq[cloud_idx, -1].item())
            edge_bandwidth = bandwidth_seq[:, -1]
            if edge_bandwidth.numel() > 1:
                edge_bandwidth = torch.cat([edge_bandwidth[:cloud_idx], edge_bandwidth[cloud_idx + 1:]])
            else:
                edge_bandwidth = torch.empty(0, dtype=edge_bandwidth.dtype)
            latest_edge_bw = float(edge_bandwidth.mean().item()) if edge_bandwidth.numel() else 0.0

        base = (
            f"services={service_count}, devices={device_count}, "
            f"latest_complexity={self._format_log_value(latest_complexity)}, "
            f"latest_latency={self._format_log_value(latest_latency)}, "
            f"latest_edge_bw={self._format_log_value(latest_edge_bw)}, "
            f"latest_cloud_bw={self._format_log_value(latest_cloud_bw)}, "
            f"latest_gpu_util={self._format_log_value(latest_gpu_util)}, "
            f"latest_mem_util={self._format_log_value(latest_mem_util)}"
        )
        if metrics:
            base += f", metrics=({self._summarize_metrics(metrics)})"
        return base

    def register_topology_encoder(self):
        if self.shared_topology_encoder:
            return

        self.shared_topology_encoder = TopologyEncoders(
            d_model=self.encoder_network_params['encoder_emb_dim'],
            heads=self.encoder_network_params['logic_gat_heads'],
            num_roles=self.encoder_network_params['phys_role_num'],
            role_emb_dim=self.encoder_network_params['phys_role_emb_dim'],
            dropout=self.encoder_network_params['encoder_dropout'],
        ).to(self.device)

    def register_deployment_agent(self):
        if self.deployment_agent:
            return

        assert self.shared_topology_encoder, 'Shared topology encoder must be registered before deployment agent.'

        self.deployment_agent = HedgerDeploymentPPO(
            encoder=self.shared_topology_encoder,
            d_model=self.encoder_network_params['encoder_emb_dim'],
            actor_lr=self.deployment_agent_params['actor_lr'],
            critic_lr=self.deployment_agent_params['critic_lr'],
            gamma=self.deployment_agent_params['gamma'],
            lamda=self.deployment_agent_params['lamda'],
            clip_eps=self.deployment_agent_params['clip_eps'],
            update_encoder=self.deployment_agent_params['update_encoder'],
            cloud_node_idx=self.physical_topology.cloud_idx if self.physical_topology is not None else -1,
            constraint_cfg=from_partial_dict(DeploymentConstraintCfg, self.deployment_agent_params),
        ).to(self.device)
        self._sync_agent_topology_bindings()

    def register_offloading_agent(self):
        if self.offloading_agent:
            return

        assert self.shared_topology_encoder, 'Shared topology encoder must be registered before offloading agent.'

        self.offloading_agent = HedgerOffloadingPPO(
            encoder=self.shared_topology_encoder,
            d_model=self.encoder_network_params['encoder_emb_dim'],
            actor_lr=self.offloading_agent_params['actor_lr'],
            critic_lr=self.offloading_agent_params['critic_lr'],
            gamma=self.offloading_agent_params['gamma'],
            lamda=self.offloading_agent_params['lamda'],
            clip_eps=self.offloading_agent_params['clip_eps'],
            update_encoder=self.offloading_agent_params['update_encoder'],
            source_node_idx=self.physical_topology.source_idx if self.physical_topology is not None else 0,
            cloud_node_idx=self.physical_topology.cloud_idx if self.physical_topology is not None else -1,
            constraint_cfg=from_partial_dict(OffloadingConstraintCfg, self.offloading_agent_params),
        ).to(self.device)
        self._sync_agent_topology_bindings()

    def register_physical_topology(self, edge_nodes, source_device):
        if self.physical_topology:
            return

        self.physical_topology = PhysicalTopology(edge_nodes, source_device)
        self._sync_agent_topology_bindings()

        LOGGER.info(f"[Hedger][Topology] Registered physical topology: {self._summarize_topology()}")
        LOGGER.debug(
            f"[Hedger][Topology] Physical nodes={self.physical_topology.nodes}, "
            f"links={self.physical_topology.links}"
        )

    def register_logical_topology(self, dag):
        if self.logical_topology:
            return

        self.logical_topology = LogicalTopology(dag)

        LOGGER.info(f"[Hedger][Topology] Registered logical topology: {self._summarize_topology()}")
        LOGGER.debug(
            f"[Hedger][Topology] Logical services={self.logical_topology.service_list}, "
            f"links={self.logical_topology.links}"
        )

    def register_state_buffer(self):
        if self.state_buffer:
            return

        assert self.logical_topology, "Logical topology must be registered before registering state buffer."
        assert self.physical_topology, "Physical topology must be registered before registering state buffer."

        self.state_buffer = StateBuffer(self.max_state_buffer_size,
                                        logical_topology=self.logical_topology,
                                        physical_topology=self.physical_topology)
        LOGGER.info(
            f"[Hedger][StateBuffer] Registered state buffer: capacity={self.max_state_buffer_size}, "
            f"fixed_lan_bandwidth={self.state_buffer.fixed_lan_bandwidth_mbps:.2f}Mbps"
        )

    def register_initial_deployment(self, deployment_plan):
        assert self.logical_topology, "Logical topology must be registered before registering initial deployment."
        assert self.physical_topology, "Physical topology must be registered before registering initial deployment."

        if self.initial_deployment_plan is not None:
            return
        self.initial_deployment_plan = copy.deepcopy(deployment_plan)
        self.deployment_plan = copy.deepcopy(deployment_plan)

        # Cache the current deployment mask.
        self.cur_deploy_mask = self._map_deployment_plan_to_deployment_mask(deployment_plan or {})
        LOGGER.info(
            f"[Hedger][Deployment] Registered initial deployment: "
            f"{self._summarize_deployment_plan(self.initial_deployment_plan)}"
        )
        LOGGER.debug(f"[Hedger][Deployment] Initial deployment detail: {self.initial_deployment_plan}")

    def get_offloading_plan(self):
        return copy.deepcopy(self.offloading_plan)

    def get_initial_deployment_plan(self):
        return copy.deepcopy(self.initial_deployment_plan)

    def get_redeployment_plan(self):
        self.cur_deploy_mask = self._map_deployment_plan_to_deployment_mask(self.deployment_plan)
        return copy.deepcopy(self.deployment_plan)

    def _build_state_wait_cfg(self) -> BufferWaitCfg:
        return BufferWaitCfg(
            min_dynamic_len=self.state_cfg.min_dynamic_len,
            require_full_seq=self.state_cfg.require_full_seq,
            timeout_s=self.state_cfg.wait_timeout_s,
        )

    def _current_deploy_mask(self) -> torch.Tensor:
        if self.cur_deploy_mask is not None:
            return self.cur_deploy_mask.detach().clone().cpu()

        deploy_mask = torch.zeros(
            (len(self.logical_topology), len(self.physical_topology)),
            dtype=torch.bool,
        )
        deploy_mask[:, self.physical_topology.cloud_idx] = True
        return deploy_mask

    def _build_edge_index(self, links) -> torch.Tensor:
        """Build a graph edge-index tensor and keep empty graphs well-formed."""
        if not links:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
        return torch.tensor(links, dtype=torch.long, device=self.device).t().contiguous()

    def _sleep_until_next_tick(self, last_tick: float, interval_s: float) -> float:
        """Sleep long enough to preserve the target loop cadence."""
        interval_s = max(0.0, float(interval_s))
        if interval_s <= 0.0:
            return time.time()

        now = time.time()
        sleep_s = interval_s if last_tick <= 0 else max(0.0, interval_s - (now - last_tick))
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        return time.time()

    def _collect_graph_state(self, seq_len: int):
        assert self.state_buffer is not None, "State buffer must be registered before collecting state."
        logic_feats_raw, phys_feats_raw = self.state_buffer.get_state(
            seq_len=seq_len,
            wait_cfg=self._build_state_wait_cfg(),
            pad_mode="edge",
        )

        logic_feats = {
            "model_flops": torch.tensor(logic_feats_raw["model_flops"], dtype=torch.float32),
            "model_mem": torch.tensor(logic_feats_raw["model_mem"], dtype=torch.float32),
            "task_complexity_seq": torch.tensor(logic_feats_raw["task_complexity_seq"], dtype=torch.float32),
            "hist_latency_seq": torch.tensor(logic_feats_raw["hist_latency_seq"], dtype=torch.float32),
        }
        phys_feats = {
            "gpu_flops": torch.tensor(phys_feats_raw["gpu_flops"], dtype=torch.float32),
            "role_id": torch.tensor(phys_feats_raw["role_id"], dtype=torch.long),
            "mem_capacity": torch.tensor(phys_feats_raw["mem_capacity"], dtype=torch.float32),
            "bandwidth_seq": torch.tensor(phys_feats_raw["bandwidth_seq"], dtype=torch.float32),
            "gpu_util_seq": torch.tensor(phys_feats_raw["gpu_util_seq"], dtype=torch.float32),
            "mem_util_seq": torch.tensor(phys_feats_raw["mem_util_seq"], dtype=torch.float32),
        }
        return logic_feats, phys_feats

    def _compute_slo_violation(self, latest_latency: torch.Tensor) -> float:
        if self.state_cfg.latency_slo is None:
            return 0.0
        if latest_latency.numel() == 0:
            return 0.0
        threshold = float(self.state_cfg.latency_slo)
        return float((latest_latency > threshold).float().mean().item())

    def _compute_cloud_fraction(self) -> float:
        num_services = len(self.logical_topology)
        if num_services == 0:
            return 0.0

        if not self.offloading_plan:
            return 1.0

        cloud_name = self.physical_topology[self.physical_topology.cloud_idx]
        cloud_count = sum(1 for service_name in self.logical_topology.service_list
                          if self.offloading_plan.get(service_name) == cloud_name)
        return float(cloud_count / num_services)

    def _compute_deploy_change_cost(self, prev_deploy_mask: Optional[torch.Tensor]) -> float:
        if prev_deploy_mask is None:
            return 0.0

        current_mask = self._current_deploy_mask().bool()
        prev_mask = prev_deploy_mask.detach().clone().cpu().bool()
        if current_mask.shape != prev_mask.shape:
            return 0.0

        cloud_idx = self.physical_topology.cloud_idx
        current_edge = current_mask[:, :cloud_idx]
        prev_edge = prev_mask[:, :cloud_idx]
        return float(torch.logical_xor(current_edge, prev_edge).sum().item())

    def _collect_deployment_state(self, prev_deploy_mask: Optional[torch.Tensor] = None):
        logic_feats, phys_feats = self._collect_graph_state(self.state_cfg.deployment_seq_len)
        reward_stats = self.state_buffer.get_offloading_reward_stats(last_k=self.state_cfg.deployment_reward_window)
        metrics = {
            "avg_offloading_reward": float(reward_stats["mean"]),
            "deploy_change_cost": self._compute_deploy_change_cost(prev_deploy_mask),
        }
        done = False
        return logic_feats, phys_feats, metrics, done

    def _collect_offloading_state(self):
        logic_feats, phys_feats = self._collect_graph_state(self.state_cfg.offloading_seq_len)
        latest_latency = logic_feats["hist_latency_seq"][:, -1] if logic_feats["hist_latency_seq"].numel() else \
            torch.empty(0, dtype=torch.float32)
        metrics = {
            "latency": float(latest_latency.mean().item()) if latest_latency.numel() else 0.0,
            "slo_violation": self._compute_slo_violation(latest_latency),
            "cloud_fraction": self._compute_cloud_fraction(),
        }
        done = False
        return logic_feats, phys_feats, metrics, done

    def inference_hedger(self):
        assert self.logical_topology is not None, "Logical topology must be registered before inference."
        assert self.physical_topology is not None, "Physical topology must be registered before inference."

        LOGGER.info(f"[Hedger][Inference] Start: {self._summarize_runtime_config()}, {self._summarize_topology()}")
        self.set_seed()

        self.shared_topology_encoder.eval()
        self.deployment_agent.eval()
        self.offloading_agent.eval()

        logic_links = self._build_edge_index(self.logical_topology.links)
        phys_links = self._build_edge_index(self.physical_topology.links)
        LOGGER.debug(
            f"[Hedger][Inference] Graph summary: logical_edges={logic_links.size(1)}, "
            f"physical_edges={phys_links.size(1)}"
        )

        self.deployment_thread_stop_event.clear()
        self.offloading_thread_stop_event.clear()

        if self.cur_deploy_mask is None:
            if self.deployment_plan is not None:
                self.cur_deploy_mask = self._map_deployment_plan_to_deployment_mask(self.deployment_plan).detach().cpu()
            elif self.initial_deployment_plan is not None:
                self.deployment_plan = copy.deepcopy(self.initial_deployment_plan)
                self.cur_deploy_mask = self._map_deployment_plan_to_deployment_mask(
                    self.initial_deployment_plan
                ).detach().cpu()
            else:
                LOGGER.warning('[Hedger][Inference] No previous deployment state found, initialize to pure cloud deployment.')
                self.cur_deploy_mask = torch.zeros(
                    (len(self.logical_topology), len(self.physical_topology)),
                    dtype=torch.bool,
                )
                self.cur_deploy_mask[:, self.physical_topology.cloud_idx] = True
                self.deployment_plan = self._map_deployment_mask_to_deployment_plan(self.cur_deploy_mask)
        elif self.deployment_plan is None:
            self.deployment_plan = self._map_deployment_mask_to_deployment_plan(self._current_deploy_mask())

        LOGGER.info(
            f"[Hedger][Inference] Initial deployment state: "
            f"{self._summarize_deploy_mask(self.cur_deploy_mask)}, "
            f"{self._summarize_deployment_plan(self.deployment_plan)}"
        )
        LOGGER.info(
            f"[Hedger][Inference] Initial offloading state: "
            f"{self._summarize_offloading_plan(self.offloading_plan)}"
        )

        deployment_thread = threading.Thread(target=self.inference_deployment_agent, daemon=True)
        offloading_thread = threading.Thread(target=self.inference_offloading_agent, daemon=True)
        deployment_thread.start()
        offloading_thread.start()

        while not (self.deployment_thread_stop_event.is_set() or self.offloading_thread_stop_event.is_set()):
            if not deployment_thread.is_alive():
                LOGGER.warning('[Hedger][Inference] Deployment worker stopped unexpectedly.')
                self.offloading_thread_stop_event.set()
                break
            if not offloading_thread.is_alive():
                LOGGER.warning('[Hedger][Inference] Offloading worker stopped unexpectedly.')
                self.deployment_thread_stop_event.set()
                break
            time.sleep(0.5)

        self.deployment_thread_stop_event.set()
        self.offloading_thread_stop_event.set()
        LOGGER.info(
            f"[Hedger][Inference] Finished: "
            f"{self._summarize_deployment_plan(self.deployment_plan)}, "
            f"{self._summarize_offloading_plan(self.offloading_plan)}"
        )

    def inference_deployment_agent(self):
        LOGGER.info(
            f"[Hedger][Inference][Deployment] Worker started: "
            f"interval={self._format_log_value(self.deployment_interval, 2)}s"
        )

        assert self.logical_topology is not None and self.physical_topology is not None, \
            "Topologies must be registered before starting deployment inference."

        logic_edge_index = self._build_edge_index(self.logical_topology.links)
        phys_edge_index = self._build_edge_index(self.physical_topology.links)

        deployment_time_ticket = 0
        prev_deploy_mask = self._current_deploy_mask()
        logic_feats, phys_feats, _, _ = self._collect_deployment_state(prev_deploy_mask=prev_deploy_mask)

        step = 0
        while not self.deployment_thread_stop_event.is_set():
            try:
                logic_feats_dev = {k: v.to(self.device) for k, v in logic_feats.items()}
                phys_feats_dev = {k: v.to(self.device) for k, v in phys_feats.items()}
                prev_deploy_mask_dev = prev_deploy_mask.to(self.device) if prev_deploy_mask is not None else None

                with torch.inference_mode():
                    deploy_mask, _, _, _, aux = self.deployment_agent.policy(
                        logic_edge_index=logic_edge_index,
                        logic_feats=logic_feats_dev,
                        phys_edge_index=phys_edge_index,
                        phys_feats=phys_feats_dev,
                        topo_order=None,
                        prev_deploy_mask=prev_deploy_mask_dev,
                    )

                self.cur_deploy_mask = deploy_mask.detach().cpu()
                self.deployment_plan = self._map_deployment_mask_to_deployment_plan(deploy_mask)
                deployment_time_ticket = self._sleep_until_next_tick(
                    deployment_time_ticket,
                    self.deployment_interval,
                )

                prev_deploy_mask = self._current_deploy_mask()
                logic_feats, phys_feats, metrics, _ = self._collect_deployment_state(prev_deploy_mask=prev_deploy_mask)
                LOGGER.debug(
                    f"[Hedger][Inference][Deployment] step={step}, "
                    f"{self._summarize_deploy_mask(self.cur_deploy_mask)}, "
                    f"{self._summarize_deployment_plan(self.deployment_plan)}, "
                    f"{self._summarize_state_snapshot(logic_feats, phys_feats, metrics)}, "
                    f"capacity_relax_cnt={aux['capacity_relax_cnt']}"
                )
                step += 1
            except Exception as e:
                LOGGER.exception(f"[Hedger][Inference][Deployment] Worker loop error: {e}")
                time.sleep(0.5)

    def inference_offloading_agent(self):
        LOGGER.info(
            f"[Hedger][Inference][Offloading] Worker started: "
            f"interval={self._format_log_value(self.offloading_interval, 2)}s"
        )

        assert self.logical_topology is not None and self.physical_topology is not None, \
            "Topologies must be registered before starting offloading inference."

        logic_edge_index = self._build_edge_index(self.logical_topology.links)
        phys_edge_index = self._build_edge_index(self.physical_topology.links)

        offloading_time_ticket = 0
        logic_feats, phys_feats, _, _ = self._collect_offloading_state()

        step = 0
        while not self.offloading_thread_stop_event.is_set():
            try:
                logic_feats_dev = {k: v.to(self.device) for k, v in logic_feats.items()}
                phys_feats_dev = {k: v.to(self.device) for k, v in phys_feats.items()}
                static_mask = self._current_deploy_mask()
                static_mask_dev = static_mask.to(self.device)

                with torch.inference_mode():
                    actions, _, _, _, aux = self.offloading_agent.policy(
                        logic_edge_index=logic_edge_index,
                        logic_feats=logic_feats_dev,
                        phys_edge_index=phys_edge_index,
                        phys_feats=phys_feats_dev,
                        static_mask=static_mask_dev,
                        topo_order=None,
                    )

                self.offloading_plan = self._map_offloading_mask_to_offloading_plan(actions)
                offloading_time_ticket = self._sleep_until_next_tick(
                    offloading_time_ticket,
                    self.offloading_interval,
                )

                logic_feats, phys_feats, metrics, _ = self._collect_offloading_state()
                LOGGER.debug(
                    f"[Hedger][Inference][Offloading] step={step}, "
                    f"{self._summarize_offloading_plan(self.offloading_plan)}, "
                    f"{self._summarize_state_snapshot(logic_feats, phys_feats, metrics)}, "
                    f"switches={aux['switches']}, correction_cnt={aux['correction_cnt']}, "
                    f"aux_cost={self._format_log_value(aux['aux_cost'])}"
                )
                step += 1
            except Exception as e:
                LOGGER.exception(f"[Hedger][Inference][Offloading] Worker loop error: {e}")
                time.sleep(0.5)

    def train_hedger(self):
        assert self.logical_topology is not None, "Logical topology must be registered before training."
        assert self.physical_topology is not None, "Physical topology must be registered before training."

        LOGGER.info(f"[Hedger][Train] Start: {self._summarize_runtime_config()}, {self._summarize_topology()}")
        self.set_seed()
        self.shared_topology_encoder.train()
        self.deployment_agent.train()
        self.offloading_agent.train()
        self.deployment_thread_stop_event.clear()
        self.offloading_thread_stop_event.clear()

        logic_links = self._build_edge_index(self.logical_topology.links)
        phys_links = self._build_edge_index(self.physical_topology.links)

        LOGGER.debug(
            f"[Hedger][Train] Graph summary: logical_edges={logic_links.size(1)}, "
            f"physical_edges={phys_links.size(1)}"
        )

        if self.cur_deploy_mask is None:
            LOGGER.warning('[Hedger][Train] No previous deployment state found, initialize to pure cloud deployment.')
            self.cur_deploy_mask = torch.zeros((len(self.logical_topology), len(self.physical_topology)),
                                               dtype=torch.bool, device=self.device)
            self.cur_deploy_mask[:, self.physical_topology.cloud_idx] = True

        LOGGER.info(
            f"[Hedger][Train] Initial deployment state: {self._summarize_deploy_mask(self.cur_deploy_mask)}"
        )

        if not self.train_deployment_flag and not self.train_offloading_flag:
            LOGGER.warning('[Hedger][Train] Both training workers are disabled, skip training run.')
            return

        deployment_thread = None
        offloading_thread = None
        if self.train_deployment_flag:
            deployment_thread = threading.Thread(target=self.train_deployment_agent, daemon=True)
            deployment_thread.start()
        if self.train_offloading_flag:
            offloading_thread = threading.Thread(target=self.train_offloading_agent, daemon=True)
            offloading_thread.start()

        while True:
            try:
                if deployment_thread is not None and not deployment_thread.is_alive():
                    LOGGER.warning('[Hedger][Train] Deployment worker stopped unexpectedly.')
                    self.offloading_thread_stop_event.set()
                    break
                if offloading_thread is not None and not offloading_thread.is_alive():
                    LOGGER.warning('[Hedger][Train] Offloading worker stopped unexpectedly.')
                    self.deployment_thread_stop_event.set()
                    break

                updates_in_tick = 0

                # Run a PPO update for the offloading agent.
                if self.train_offloading_flag and len(self.offloading_transitions) >= self.offloading_rollout_len:
                    with self._data_lock:
                        off_transitions = self.offloading_transitions[:self.offloading_rollout_len]
                        del self.offloading_transitions[:self.offloading_rollout_len]
                        off_remaining = len(self.offloading_transitions)

                    self.offloading_agent.ppo_update(off_transitions,
                                                     epochs=self.update_epochs,
                                                     batch_size=self.offloading_batch_size)
                    self._offloading_update_steps += 1
                    updates_in_tick += 1
                    LOGGER.info(
                        f"[Hedger][Train][Offloading] PPO update={self._offloading_update_steps}, "
                        f"used={len(off_transitions)}, remaining={off_remaining}"
                    )

                # Run a PPO update for the deployment agent.
                if self.train_deployment_flag and len(self.deployment_transitions) >= self.deployment_rollout_len:
                    with self._data_lock:
                        dep_transitions = self.deployment_transitions[:self.deployment_rollout_len]
                        del self.deployment_transitions[:self.deployment_rollout_len]
                        dep_remaining = len(self.deployment_transitions)

                    self.deployment_agent.ppo_update(dep_transitions,
                                                     epochs=self.update_epochs,
                                                     batch_size=self.deployment_batch_size)
                    self._deployment_update_steps += 1
                    updates_in_tick += 1
                    LOGGER.info(
                        f"[Hedger][Train][Deployment] PPO update={self._deployment_update_steps}, "
                        f"used={len(dep_transitions)}, remaining={dep_remaining}"
                    )

                # Save a checkpoint.
                if updates_in_tick > 0:
                    prev_epoch = self._epoch
                    self._epoch += updates_in_tick
                    # Save once whenever `_epoch` crosses a multiple of `save_interval`.
                    if (prev_epoch // self.save_interval) != (self._epoch // self.save_interval):
                        try:
                            self.save_checkpoint(epoch=self._epoch)
                        except Exception as e:
                            LOGGER.exception(
                                f"[Hedger][Train] Failed to save checkpoint at epoch={self._epoch}: {e}"
                            )

                if self._epoch > self.total_steps:
                    LOGGER.info(f"[Hedger][Train] Reached training step budget: epoch={self._epoch}, limit={self.total_steps}")
                    break

                time.sleep(0.5)
            except Exception as e:
                LOGGER.exception(f"[Hedger][Train] Coordinator loop error: {e}")
                continue

        self.deployment_thread_stop_event.set()
        self.offloading_thread_stop_event.set()
        LOGGER.info(
            f"[Hedger][Train] Finished: epoch={self._epoch}, "
            f"dep_updates={self._deployment_update_steps}, off_updates={self._offloading_update_steps}, "
            f"{self._summarize_deployment_plan(self.deployment_plan)}, "
            f"{self._summarize_offloading_plan(self.offloading_plan)}"
        )

    def train_deployment_agent(self):
        """
        Sampling loop for the deployment agent:
            - Collect the current topology and system state periodically
            - Sample a new deployment mask via `deployment_agent.policy`
            - Let the environment run for one deployment interval, then collect metrics and reward
            - Push the transition into `self.deployment_transitions` for PPO updates
        """
        if not self.train_deployment_flag:
            LOGGER.info("[Hedger][Train][Deployment] Worker disabled by config, skip startup.")
            return

        assert self.logical_topology is not None and self.physical_topology is not None, \
            "Topologies must be registered before starting deployment training."

        LOGGER.info(
            f"[Hedger][Train][Deployment] Worker started: "
            f"interval={self._format_log_value(self.deployment_interval, 2)}s, "
            f"rollout={self.deployment_rollout_len}, batch={self.deployment_batch_size}"
        )

        self.dep_recorder = Recorder(
            "deployment_train.csv",
            fmt="csv",
            fieldnames=["step", "epoch", "dep_updates", "dep_reward", "avg_off_reward",
                        "dep_change_cost", "cap_relax_cnt", "dep_offload_weight",
                        "dep_change_weight", "cap_relax_weight"],
            overwrite=True,
            flush_every=1,
        )

        # Static graph edge indices can be reused across iterations.
        logic_edge_index = self._build_edge_index(self.logical_topology.links)
        phys_edge_index = self._build_edge_index(self.physical_topology.links)

        step = 0
        deployment_time_ticket = 0

        prev_deploy_mask = copy.deepcopy(self.cur_deploy_mask)
        logic_feats, phys_feats, _, _ = self._collect_deployment_state(prev_deploy_mask=prev_deploy_mask)

        while not self.deployment_thread_stop_event.is_set():
            try:
                # Move features onto the active device.
                logic_feats_dev = {k: v.to(self.device) for k, v in logic_feats.items()}
                phys_feats_dev = {k: v.to(self.device) for k, v in phys_feats.items()}
                prev_deploy_mask_dev = prev_deploy_mask.to(self.device) if prev_deploy_mask is not None else None

                with torch.no_grad():
                    # Sample a new deployment strategy.
                    deploy_mask, logp, ent, value, aux = self.deployment_agent.policy(
                        logic_edge_index=logic_edge_index,
                        logic_feats=logic_feats_dev,
                        phys_edge_index=phys_edge_index,
                        phys_feats=phys_feats_dev,
                        topo_order=None,  # Derived internally from the logical graph.
                        prev_deploy_mask=prev_deploy_mask_dev
                    )
                deploy_plan = self._map_deployment_mask_to_deployment_plan(deploy_mask)
                self.deployment_plan = deploy_plan
                self.cur_deploy_mask = deploy_mask.detach().cpu()

                deployment_time_ticket = self._sleep_until_next_tick(
                    deployment_time_ticket,
                    self.deployment_interval,
                )

                new_logic_feats, new_phys_feats, metrics, done = self._collect_deployment_state(
                    prev_deploy_mask=prev_deploy_mask,)

                # Compute the reward from environment metrics and policy-side auxiliaries.
                reward = self._compute_deployment_reward(metrics, aux)

                # Keep transition payloads on CPU to avoid cross-thread device issues.
                tr = {
                    "logic_edge_index": logic_edge_index.cpu(),
                    "logic_feats": {k: v.cpu() for k, v in logic_feats_dev.items()},
                    "phys_edge_index": phys_edge_index.cpu(),
                    "phys_feats": {k: v.cpu() for k, v in phys_feats_dev.items()},
                    "deploy_mask": deploy_mask.cpu(),
                    "topo_order": None,  # Recomputed during evaluation if needed.
                    "prev_deploy_mask": prev_deploy_mask.cpu() if prev_deploy_mask is not None else None,
                    "logp": logp.detach().cpu(),
                    "value": value.detach().cpu(),
                    "reward": float(reward),
                    "done": bool(done),
                }

                with self._data_lock:
                    self.deployment_transitions.append(tr)
                    transition_count = len(self.deployment_transitions)

                logic_feats = new_logic_feats
                phys_feats = new_phys_feats
                prev_deploy_mask = deploy_mask.detach().cpu()

                self.dep_recorder.log(
                    step=step,
                    epoch=self._epoch,
                    dep_updates=self._deployment_update_steps,
                    dep_reward=reward,
                    avg_off_reward=metrics["avg_offloading_reward"],
                    dep_change_cost=metrics["deploy_change_cost"],
                    cap_relax_cnt=aux["capacity_relax_cnt"],
                    dep_offload_weight=self.deployment_agent_params["reward_dep_offload_weight"],
                    dep_change_weight=self.deployment_agent_params["reward_dep_change_weight"],
                    cap_relax_weight=self.deployment_agent_params["penalty_capacity_relax"],
                )
                LOGGER.debug(
                    f"[Hedger][Train][Deployment] step={step}, reward={self._format_log_value(reward)}, "
                    f"{self._summarize_deploy_mask(self.cur_deploy_mask)}, "
                    f"{self._summarize_deployment_plan(self.deployment_plan)}, "
                    f"{self._summarize_state_snapshot(logic_feats, phys_feats, metrics)}, "
                    f"capacity_relax_cnt={aux['capacity_relax_cnt']}, transitions={transition_count}"
                )

                step += 1
            except Exception as e:
                LOGGER.exception(f"[Hedger][Train][Deployment] Worker loop error: {e}")
                time.sleep(0.5)

        self.dep_recorder.close()
        LOGGER.info("[Hedger][Train][Deployment] Worker stopped.")

    def train_offloading_agent(self):
        """
        Sampling loop for the offloading agent:
            - Collect topology and system state at a shorter cadence
            - Select the execution device for each service via `offloading_agent.policy`
            - Let the environment execute the policy and report latency/SLO/cloud metrics
            - Push the transition into `self.offloading_transitions`
        """
        if not self.train_offloading_flag:
            LOGGER.info("[Hedger][Train][Offloading] Worker disabled by config, skip startup.")
            return

        assert self.logical_topology is not None and self.physical_topology is not None, \
            "Topologies must be registered before starting offloading training."

        LOGGER.info(
            f"[Hedger][Train][Offloading] Worker started: "
            f"interval={self._format_log_value(self.offloading_interval, 2)}s, "
            f"rollout={self.offloading_rollout_len}, batch={self.offloading_batch_size}"
        )

        self.off_recorder = Recorder(
            "offloading_train.csv",
            fmt="csv",
            fieldnames=["step", "epoch", "off_updates", "off_reward", "latency",
                        "slo_violation", "cloud_fraction", "aux_cost", "off_latency_weight",
                        "off_slo_weight", "off_cloud_weight", "switch_cnt", "correction_cnt",
                        "switch_weight", "correction_weight"],
            overwrite=True,
            flush_every=10,
        )

        logic_edge_index = self._build_edge_index(self.logical_topology.links)
        phys_edge_index = self._build_edge_index(self.physical_topology.links)

        step = 0
        offloading_time_ticket = 0
        logic_feats, phys_feats, _, _ = self._collect_offloading_state()
        while not self.offloading_thread_stop_event.is_set():
            try:
                logic_feats_dev = {k: v.to(self.device) for k, v in logic_feats.items()}
                phys_feats_dev = {k: v.to(self.device) for k, v in phys_feats.items()}
                static_mask = self._current_deploy_mask()
                static_mask_dev = static_mask.to(self.device)

                with torch.no_grad():
                    actions, logp, ent, value, aux = self.offloading_agent.policy(
                        logic_edge_index=logic_edge_index,
                        logic_feats=logic_feats_dev,
                        phys_edge_index=phys_edge_index,
                        phys_feats=phys_feats_dev,
                        static_mask=static_mask_dev,
                        topo_order=None,
                    )
                    offloading_plan = self._map_offloading_mask_to_offloading_plan(actions)
                    self.offloading_plan = offloading_plan

                offloading_time_ticket = self._sleep_until_next_tick(
                    offloading_time_ticket,
                    self.offloading_interval,
                )

                new_logic_feats, new_phys_feats, metrics, done = self._collect_offloading_state()

                reward = self._compute_offloading_reward(metrics, aux)
                if self.state_buffer is not None:
                    self.state_buffer.add_offloading_reward(reward)

                tr = {
                    "logic_edge_index": logic_edge_index.cpu(),
                    "logic_feats": {k: v.cpu() for k, v in logic_feats_dev.items()},
                    "phys_edge_index": phys_edge_index.cpu(),
                    "phys_feats": {k: v.cpu() for k, v in phys_feats_dev.items()},
                    "actions": actions.cpu(),
                    "static_mask": static_mask_dev.cpu(),
                    "topo_order": None,
                    "logp": logp.detach().cpu(),
                    "value": value.detach().cpu(),
                    "reward": float(reward),
                    "done": bool(done),
                }

                with self._data_lock:
                    self.offloading_transitions.append(tr)
                    transition_count = len(self.offloading_transitions)

                logic_feats = new_logic_feats
                phys_feats = new_phys_feats

                self.off_recorder.log(
                    step=step,
                    epoch=self._epoch,
                    off_updates=self._offloading_update_steps,
                    off_reward=reward,
                    latency=metrics["latency"],
                    slo_violation=metrics["slo_violation"],
                    cloud_fraction=metrics["cloud_fraction"],
                    aux_cost=aux["aux_cost"],
                    off_latency_weight=self.offloading_agent_params["reward_off_latency_weight"],
                    off_slo_weight=self.offloading_agent_params["reward_off_slo_weight"],
                    off_cloud_weight=self.offloading_agent_params["reward_off_cloud_weight"],
                    switch_cnt=aux["switches"],
                    correction_cnt=aux["correction_cnt"],
                    switch_weight=self.offloading_agent_params["penalty_switch"],
                    correction_weight=self.offloading_agent_params["penalty_relax"],
                )
                LOGGER.debug(
                    f"[Hedger][Train][Offloading] step={step}, reward={self._format_log_value(reward)}, "
                    f"{self._summarize_offloading_plan(self.offloading_plan)}, "
                    f"{self._summarize_state_snapshot(logic_feats, phys_feats, metrics)}, "
                    f"switches={aux['switches']}, correction_cnt={aux['correction_cnt']}, "
                    f"aux_cost={self._format_log_value(aux['aux_cost'])}, transitions={transition_count}"
                )

                step += 1
            except Exception as e:
                LOGGER.exception(f"[Hedger][Train][Offloading] Worker loop error: {e}")
                time.sleep(0.5)

        self.off_recorder.close()
        LOGGER.info("[Hedger][Train][Offloading] Worker stopped.")

    def _compute_offloading_reward(self, metrics, aux) -> float:
        """
        Compute the offloading-agent reward.

        The reward is defined as negative performance cost plus policy-side
        penalties. `metrics` contains environment indicators, while `aux`
        contains switch/correction statistics gathered inside the policy.
        """
        metrics = metrics or {}

        # Extract the core metrics.
        latency = float(metrics["latency"])
        slo_v = float(metrics["slo_violation"])
        cloud_frac = float(metrics["cloud_fraction"])

        # Reward weights from hyper-parameters.
        w_lat = float(self.offloading_agent_params["reward_off_latency_weight"])
        w_slo = float(self.offloading_agent_params["reward_off_slo_weight"])
        w_cloud = float(self.offloading_agent_params["reward_off_cloud_weight"])

        # Base reward: lower latency, fewer SLO violations, and lower cloud usage.
        reward = 0.0
        reward -= w_lat * latency
        reward -= w_slo * slo_v
        reward -= w_cloud * cloud_frac

        # Additional constraint cost: device switches and post-sampling corrections.
        aux_cost = float(aux["aux_cost"])
        reward -= aux_cost

        return reward

    def _compute_deployment_reward(self, metrics, aux) -> float:
        """
        Compute the deployment-agent reward.

        The reward combines:
            - aggregated feedback from offloading (`avg_offloading_reward`)
            - deployment change cost
            - capacity-correction penalty
        """
        metrics = metrics or {}

        avg_off_r = float(metrics["avg_offloading_reward"])
        deploy_change_cost = float(metrics["deploy_change_cost"])

        w_change = float(self.deployment_agent_params["reward_dep_change_weight"])
        w_off = float(self.deployment_agent_params["reward_dep_offload_weight"])

        reward = 0.0
        reward -= w_change * deploy_change_cost

        # Treat the lower-level offloading reward as bottom-up feedback.
        reward += w_off * avg_off_r

        # Penalize post-sampling capacity corrections.
        cap_relax_cnt = float(aux["capacity_relax_cnt"])
        penalty_capacity_relax = float(self.deployment_agent_params["penalty_capacity_relax"])
        reward -= penalty_capacity_relax * cap_relax_cnt

        return reward

    def _epoch_checkpoint_path(self, epoch: int) -> str:
        return os.path.join(self.model_dir, f'hedger_ckpt_epoch_{epoch}.pt')

    def _latest_epoch_checkpoint_path(self) -> Optional[str]:
        pattern = os.path.join(self.model_dir, 'hedger_ckpt_epoch_*.pt')
        files = glob.glob(pattern)
        if not files:
            return None

        # Parse the epoch number from the checkpoint filename.
        def parse_epoch(p):
            try:
                base = os.path.basename(p)
                # Expected pattern: `hedger_ckpt_epoch_{n}.pt`
                num = base.replace('hedger_ckpt_epoch_', '').replace('.pt', '')
                return int(num)
            except Exception:
                return -1

        files = sorted(files, key=parse_epoch, reverse=True)
        return files[0] if files else None

    def save_checkpoint(self, epoch: Optional[int] = None):
        """Save the encoder, both agents, and their optimizers into one checkpoint file."""
        if epoch is None:
            epoch = self._epoch
        path = self._epoch_checkpoint_path(epoch)
        ckpt = {
            'encoder': self.shared_topology_encoder.state_dict(),
            'deployment_agent': self.deployment_agent.state_dict(),
            'offloading_agent': self.offloading_agent.state_dict(),
            'deployment_actor_opt': self.deployment_agent.actor_opt.state_dict(),
            'deployment_critic_opt': self.deployment_agent.critic_opt.state_dict(),
            'offloading_actor_opt': self.offloading_agent.actor_opt.state_dict(),
            'offloading_critic_opt': self.offloading_agent.critic_opt.state_dict(),
            'meta': {
                'time': time.time(),
                'seed': self.seed,
                'deployment_updates': self._deployment_update_steps,
                'offloading_updates': self._offloading_update_steps,
                'device': str(self.device),
                'epoch': epoch,
            }
        }
        torch.save(ckpt, path)
        LOGGER.info(f"[Hedger][Checkpoint] Saved epoch={epoch} to {path}")

    def load_checkpoint(self, epoch: int = None):
        """
        Load encoder, agent, and optimizer state from a checkpoint.

        The behavior is controlled by hyper-parameter flags:
            - whether to load the encoder
            - whether to load each agent
            - whether to restore optimizer state
            - whether to reset epoch/update counters
        """
        if epoch is not None:
            ep_path = self._epoch_checkpoint_path(int(epoch))
            if os.path.exists(ep_path):
                target_path = ep_path
            else:
                LOGGER.warning(
                    f"[Hedger][Checkpoint] Requested epoch checkpoint not found at {ep_path}; "
                    f"fall back to latest checkpoint."
                )
                target_path = self._latest_epoch_checkpoint_path()
        else:
            target_path = self._latest_epoch_checkpoint_path()

        if not target_path or not os.path.exists(target_path):
            LOGGER.warning(f"[Hedger][Checkpoint] No checkpoint found in {self.model_dir}.")
            return

        ckpt = torch.load(target_path, map_location=self.device)
        LOGGER.info(f"[Hedger][Checkpoint] Loading from {target_path}")

        # Load encoder weights.
        if self.checkpoint_cfg.load_encoder:
            enc_state = ckpt.get('encoder', None)
            if enc_state is not None:
                self.shared_topology_encoder.load_state_dict(enc_state)
                LOGGER.info('[Hedger][Checkpoint] Loaded encoder state.')
            else:
                LOGGER.warning('[Hedger][Checkpoint] Missing encoder state in checkpoint.')
        else:
            LOGGER.info('[Hedger][Checkpoint] Skip encoder loading per config (load_encoder=False).')

        # Load the two agent modules.
        # Deployment agent.
        if self.checkpoint_cfg.load_deployment_agent:
            dep_state = ckpt.get('deployment_agent', None)
            if dep_state is not None:
                self.deployment_agent.load_state_dict(dep_state, strict=False)
                LOGGER.info('[Hedger][Checkpoint] Loaded deployment agent state.')
            else:
                LOGGER.warning('[Hedger][Checkpoint] Missing deployment_agent state in checkpoint.')
        else:
            LOGGER.info(
                '[Hedger][Checkpoint] Skip deployment agent loading per config '
                '(load_deployment_agent=False).'
            )

        # Offloading agent.
        if self.checkpoint_cfg.load_offloading_agent:
            off_state = ckpt.get('offloading_agent', None)
            if off_state is not None:
                self.offloading_agent.load_state_dict(off_state, strict=False)
                LOGGER.info('[Hedger][Checkpoint] Loaded offloading agent state.')
            else:
                LOGGER.warning('[Hedger][Checkpoint] Missing offloading_agent state in checkpoint.')
        else:
            LOGGER.info(
                '[Hedger][Checkpoint] Skip offloading agent loading per config '
                '(load_offloading_agent=False).'
            )

        # Load optimizer state.
        if self.checkpoint_cfg.load_optimizer:
            # Restore only the optimizers that correspond to loaded agents.
            if self.checkpoint_cfg.load_deployment_agent and 'deployment_actor_opt' in ckpt:
                self.deployment_agent.actor_opt.load_state_dict(ckpt['deployment_actor_opt'])
                self._move_optimizer_state(self.deployment_agent.actor_opt, self.device)
                LOGGER.info('[Hedger][Checkpoint] Loaded deployment actor optimizer state.')

            if self.checkpoint_cfg.load_deployment_agent and 'deployment_critic_opt' in ckpt:
                self.deployment_agent.critic_opt.load_state_dict(ckpt['deployment_critic_opt'])
                self._move_optimizer_state(self.deployment_agent.critic_opt, self.device)
                LOGGER.info('[Hedger][Checkpoint] Loaded deployment critic optimizer state.')

            if self.checkpoint_cfg.load_offloading_agent and 'offloading_actor_opt' in ckpt:
                self.offloading_agent.actor_opt.load_state_dict(ckpt['offloading_actor_opt'])
                self._move_optimizer_state(self.offloading_agent.actor_opt, self.device)
                LOGGER.info('[Hedger][Checkpoint] Loaded offloading actor optimizer state.')

            if self.checkpoint_cfg.load_offloading_agent and 'offloading_critic_opt' in ckpt:
                self.offloading_agent.critic_opt.load_state_dict(ckpt['offloading_critic_opt'])
                self._move_optimizer_state(self.offloading_agent.critic_opt, self.device)
                LOGGER.info('[Hedger][Checkpoint] Loaded offloading critic optimizer state.')
        else:
            LOGGER.info('[Hedger][Checkpoint] Skip optimizer state loading per config (load_optimizer=False).')

        # Restore or reset training counters.
        meta = ckpt.get('meta', {})

        if self.checkpoint_cfg.reset_steps_on_load:
            # Restart counters from zero for subsequent training phases.
            self._deployment_update_steps = 0
            self._offloading_update_steps = 0
            self._epoch = 0
            LOGGER.info('[Hedger][Checkpoint] Reset update counters and epoch to 0 per config (reset_steps_on_load=True).')
        else:
            self._deployment_update_steps = meta.get('deployment_updates', 0)
            self._offloading_update_steps = meta.get('offloading_updates', 0)
            self._epoch = int(meta.get('epoch', self._epoch))
            LOGGER.info(
                f"[Hedger][Checkpoint] Restored counters: "
                f"dep_updates={self._deployment_update_steps}, "
                f"off_updates={self._offloading_update_steps}, "
                f"epoch={self._epoch}"
            )

        LOGGER.info(f"[Hedger][Checkpoint] Loaded successfully from {target_path}")

    @staticmethod
    def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device):
        """Ensure all optimizer state tensors live on the same device as the model."""
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def _map_deployment_plan_to_deployment_mask(self, deployment_plan: dict):
        """
        Convert a deployment-plan dictionary into a deployment-mask tensor.

        `deployment_plan`: `service_name -> [device_name, ...]`

        Returns:
            `deploy_mask`: boolean tensor of shape `(num_services, num_devices)`.
            The cloud replica is always kept enabled for every service.
        """
        num_services = len(self.logical_topology)
        num_devices = len(self.physical_topology)
        deploy_mask = torch.zeros((num_services, num_devices), dtype=torch.bool, device=self.device)

        for service_name, device_names in (deployment_plan or {}).items():
            try:
                s_idx = self.logical_topology.index(service_name)
            except ValueError:
                LOGGER.debug(f"[Hedger][Deployment] Ignore unknown service in plan: {service_name}")
                continue
            if isinstance(device_names, (list, tuple, set)):
                iterable = device_names
            else:
                iterable = [device_names]
            for device_name in iterable:
                try:
                    d_idx = self.physical_topology.index(device_name)
                except ValueError:
                    LOGGER.debug(f"[Hedger][Deployment] Ignore unknown device in plan: {device_name}")
                    continue
                deploy_mask[s_idx, d_idx] = True

        deploy_mask[:, self.physical_topology.cloud_idx] = True

        return deploy_mask

    def _map_deployment_mask_to_deployment_plan(self, deploy_mask: torch.Tensor):
        """
        Convert a deployment-mask tensor into a deployment-plan dictionary.

        `deploy_mask`: boolean tensor of shape `(num_services, num_devices)`

        Returns:
            `deployment_plan`: `service_name -> [device_name, ...]`
        """
        deployment_plan = {}
        num_services = deploy_mask.size(0)
        num_devices = deploy_mask.size(1)

        for s_idx in range(num_services):
            for d_idx in range(num_devices):
                if deploy_mask[s_idx, d_idx]:
                    service_name = self.logical_topology[s_idx]
                    device_name = self.physical_topology[d_idx]
                    if service_name not in deployment_plan:
                        deployment_plan[service_name] = [device_name]
                    else:
                        deployment_plan[service_name].append(device_name)

        return deployment_plan

    def _map_offloading_mask_to_offloading_plan(self, offloading_mask: torch.Tensor):
        """
        Convert an offloading-action tensor into an offloading-plan dictionary.

        `offloading_mask`: integer tensor of shape `(num_services,)`, where
        each value is a device index.

        Returns:
            `offloading_plan`: `service_name -> device_name`
        """
        offloading_plan = {}
        num_services = offloading_mask.size(0)

        for s_idx in range(num_services):
            d_idx = offloading_mask[s_idx].item()
            service_name = self.logical_topology[s_idx]
            device_name = self.physical_topology[d_idx]
            offloading_plan[service_name] = device_name

        return offloading_plan

    @property
    def _ready_for_run(self):
        return self.physical_topology and self.logical_topology

    def run(self):
        wait_logged = False
        while not self._ready_for_run:
            if not wait_logged:
                LOGGER.debug(
                    f"[Hedger][Lifecycle] Waiting for topology registration: "
                    f"logical_ready={self.logical_topology is not None}, "
                    f"physical_ready={self.physical_topology is not None}"
                )
                wait_logged = True
            time.sleep(0.5)

        if self.mode == 'train':
            self.train_hedger()
        elif self.mode == 'inference':
            self.inference_hedger()
        else:
            raise ValueError(f'Unsupported mode {self.mode} for Hedger, only "train" and "inference" are supported.')
