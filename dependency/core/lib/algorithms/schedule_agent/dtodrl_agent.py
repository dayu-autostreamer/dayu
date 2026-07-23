import abc
import copy
import os
import threading

from core.lib.common import ClassFactory, ClassType, Context, LOGGER
from core.lib.estimation import OverheadEstimator

from .base_agent import BaseAgent
from .full_plan_support import (
    apply_full_plan,
    deployment_from_snapshot,
    load_profiled_latency_model,
    service_names,
    topological_order,
    validate_profile_coverage,
    visible_replica_loads,
)

__all__ = ("DTODRLAgent",)


@ClassFactory.register(ClassType.SCH_AGENT, alias="dtodrl")
class DTODRLAgent(BaseAgent, abc.ABC):
    """GAT/PPO dependent-task offloading baseline adapted from DTODRL."""

    def __init__(
        self,
        system,
        agent_id: int,
        configuration=None,
        latency_profile=None,
        latency_slo_s=3.0,
        mode="inference",
        checkpoint_path="dtodrl-agent-{agent_id}.pt",
        load_checkpoint=False,
        hidden_dim=64,
        learning_rate=3e-4,
        batch_size=16,
        ppo_clip=0.2,
        entropy_weight=0.01,
        ppo_epochs=4,
        save_interval=1,
        profile_quantile=0.5,
        queue_state_max_age_s=1.5,
        random_seed=0,
    ):
        super().__init__(system, agent_id)
        self.system = system
        self.agent_id = agent_id
        self.configuration, self.latency_model = load_profiled_latency_model(
            configuration,
            latency_profile,
        )
        self.latency_slo_s = max(1e-6, float(latency_slo_s))
        self.mode = str(mode).strip().lower()
        if self.mode not in ("train", "inference"):
            raise ValueError("DTODRL mode must be 'train' or 'inference'")
        checkpoint_ref = str(checkpoint_path).format(agent_id=agent_id)
        self.checkpoint_path = (
            checkpoint_ref
            if os.path.isabs(checkpoint_ref)
            else Context.get_file_path(checkpoint_ref)
        )
        self.load_checkpoint = bool(load_checkpoint)
        self.hidden_dim = max(8, int(hidden_dim))
        self.learning_rate = float(learning_rate)
        self.batch_size = max(1, int(batch_size))
        self.ppo_clip = float(ppo_clip)
        self.entropy_weight = float(entropy_weight)
        self.ppo_epochs = max(1, int(ppo_epochs))
        self.save_interval = max(1, int(save_interval))
        self.profile_quantile = min(1.0, max(0.0, float(profile_quantile)))
        self.queue_state_max_age_s = max(0.0, float(queue_state_max_age_s))
        self.random_seed = int(random_seed) + int(agent_id)
        self.overhead_estimator = OverheadEstimator(
            "DTODRL",
            "scheduler/fragsplice",
            agent_id=agent_id,
        )
        self.policy = None
        self.policy_signature = None
        self.pending = {}
        self.completed = []
        self.last_decision = None
        self.last_training_metrics = None
        self._lock = threading.RLock()

    def _signature(self, dag, deployment, order):
        order_set = set(order)
        return {
            "state_schema": 1,
            "configuration": copy.deepcopy(self.configuration),
            "latency_slo_s": self.latency_slo_s,
            "profile_quantile": self.profile_quantile,
            "queue_state_max_age_s": self.queue_state_max_age_s,
            "services": list(order),
            "candidates": {
                service: list(deployment[service])
                for service in order
            },
            "edges": sorted([
                [predecessor, service]
                for service in order
                for predecessor in dag[service].get("prev_nodes", [])
                if predecessor in order_set
            ]),
        }

    def _ensure_policy(self, signature):
        if self.policy is not None:
            if signature != self.policy_signature:
                raise ValueError(
                    "DTODRL requires one fixed DAG and deployment per experiment"
                )
            return
        try:
            from .dtodrl.policy import DTODRLPolicy
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise RuntimeError(
                    "DTODRL requires PyTorch in the Scheduler image"
                ) from exc
            raise
        self.policy_signature = copy.deepcopy(signature)
        self.policy = DTODRLPolicy(
            signature=signature,
            checkpoint_path=self.checkpoint_path,
            mode=self.mode,
            hidden_dim=self.hidden_dim,
            learning_rate=self.learning_rate,
            ppo_clip=self.ppo_clip,
            entropy_weight=self.entropy_weight,
            ppo_epochs=self.ppo_epochs,
            random_seed=self.random_seed,
            load_checkpoint=self.load_checkpoint,
        )

    def _build_state(self, snapshot, dag, deployment):
        order = [
            service
            for service in topological_order(dag)
            if service in service_names(dag)
        ]
        loads = visible_replica_loads(
            snapshot,
            self.latency_model,
            dag,
            deployment,
            self.profile_quantile,
            self.queue_state_max_age_s,
        )
        candidates = {service: list(deployment[service]) for service in order}
        max_candidates = max(len(value) for value in candidates.values())
        max_duration = max(
            loads[(service, device)]["demand"]
            for service in order
            for device in candidates[service]
        )
        max_duration = max(1e-6, float(max_duration))

        index = {service: position for position, service in enumerate(order)}
        adjacency = [[False for _ in order] for _ in order]
        depth = {}
        for service in order:
            predecessors = [
                predecessor
                for predecessor in dag[service].get("prev_nodes", [])
                if predecessor in index
            ]
            depth[service] = 1 + max(
                (depth[predecessor] for predecessor in predecessors),
                default=-1,
            )
            for predecessor in predecessors:
                adjacency[index[service]][index[predecessor]] = True
        max_depth = max(depth.values(), default=1) or 1
        degree_scale = max(1, len(order) - 1)

        node_features = []
        candidate_features = []
        candidate_mask = []
        for service in order:
            indegree = len([
                item
                for item in dag[service].get("prev_nodes", [])
                if item in index
            ])
            outdegree = len([
                item
                for item in dag[service].get("next_nodes", [])
                if item in index
            ])
            durations = [
                loads[(service, device)]["demand"]
                for device in candidates[service]
            ]
            node_features.append([
                indegree / degree_scale,
                outdegree / degree_scale,
                depth[service] / max_depth,
                min(durations) / max_duration,
                max(durations) / max_duration,
            ])
            minimum = max(1e-6, min(durations))
            row = []
            mask = []
            for position in range(max_candidates):
                if position >= len(candidates[service]):
                    row.append([0.0] * 5)
                    mask.append(False)
                    continue
                device = candidates[service][position]
                replica = loads[(service, device)]
                row.append([
                    replica["demand"] / max_duration,
                    min(4.0, replica["demand"] / minimum) / 4.0,
                    min(4.0, replica["workload"] / self.latency_slo_s) / 4.0,
                    min(8, replica["waiting_count"]) / 8.0,
                    1.0 if replica["busy"] else 0.0,
                ])
                mask.append(True)
            candidate_features.append(row)
            candidate_mask.append(mask)

        state = {
            "node_features": node_features,
            "adjacency": adjacency,
            "candidate_features": candidate_features,
            "candidate_mask": candidate_mask,
        }
        return state, candidates, order

    @staticmethod
    def _metadata_slo(metadata, fallback):
        metadata = metadata if isinstance(metadata, dict) else {}
        for key in ("slo_seconds", "slo", "latency_slo_s", "deadline_seconds"):
            try:
                value = float(metadata.get(key))
            except (TypeError, ValueError):
                continue
            if value > 0.0:
                return value
        return float(fallback)

    def get_schedule_plan(self, info):
        with self.overhead_estimator, self._lock:
            dag = info["dag"]
            snapshot = self.system.get_scheduling_snapshot()
            deployment = deployment_from_snapshot(self.system, snapshot)
            validate_profile_coverage(
                self.latency_model,
                self.configuration,
                dag,
                deployment,
                "DTODRL",
            )
            state, candidates, order = self._build_state(
                snapshot,
                dag,
                deployment,
            )
            signature = self._signature(dag, deployment, order)
            self._ensure_policy(signature)
            actions, log_probability, value = self.policy.select(
                state,
                deterministic=self.mode == "inference",
            )
            plan = {
                service: candidates[service][actions[index]]
                for index, service in enumerate(order)
            }
            root_uuid = str(
                (info.get("task_context") or {}).get("root_uuid") or ""
            )
            if self.mode == "train":
                if not root_uuid:
                    raise ValueError(
                        "DTODRL training requires task_context.root_uuid"
                    )
                self.pending[root_uuid] = {
                    "state": state,
                    "actions": actions,
                    "old_log_probability": log_probability,
                    "old_value": value,
                    "slo": self._metadata_slo(
                        info.get("meta_data"),
                        self.latency_slo_s,
                    ),
                }
            self.last_decision = {
                "mode": self.mode,
                "root_uuid": root_uuid,
                "plan": copy.deepcopy(plan),
                "value": value,
            }
            LOGGER.info(
                "[DTODRL] source=%s mode=%s root=%s value=%.4f plan=%s",
                info.get("source_id"),
                self.mode,
                root_uuid or "-",
                value,
                plan,
            )
            return apply_full_plan(
                self.configuration,
                dag,
                plan,
                info.get("source_device"),
                self.cloud_device,
            )

    def update_task(self, task):
        if self.mode != "train":
            return
        root_getter = getattr(task, "get_root_uuid", None)
        root_uuid = str(root_getter() if callable(root_getter) else "")
        with self._lock:
            transition = self.pending.pop(root_uuid, None)
            if transition is None:
                return
            latency_getter = getattr(task, "get_real_end_to_end_time", None)
            try:
                latency = float(latency_getter())
            except (TypeError, ValueError):
                LOGGER.warning(
                    "[DTODRL] Ignore task %s without valid end-to-end latency",
                    root_uuid,
                )
                return
            metadata_getter = getattr(task, "get_metadata", None)
            metadata = metadata_getter() if callable(metadata_getter) else {}
            slo = self._metadata_slo(metadata, transition["slo"])
            ratio = max(0.0, latency) / max(1e-6, slo)
            transition["reward"] = -ratio - max(0.0, ratio - 1.0)
            self.completed.append(transition)
            if len(self.completed) < self.batch_size:
                return
            batch = self.completed[:self.batch_size]
            del self.completed[:self.batch_size]
            self.last_training_metrics = self.policy.update(batch)
            if self.policy.update_count % self.save_interval == 0:
                self.policy.save()
            LOGGER.info(
                "[DTODRL] update=%s batch=%s metrics=%s",
                self.policy.update_count,
                len(batch),
                {
                    key: round(value, 6)
                    for key, value in self.last_training_metrics.items()
                },
            )

    def update_scenario(self, scenario):
        pass

    def update_resource(self, device, resource):
        pass

    def update_policy(self, policy):
        pass

    def run(self):
        pass

    def get_schedule_overhead(self):
        return self.overhead_estimator.get_latest_overhead()
