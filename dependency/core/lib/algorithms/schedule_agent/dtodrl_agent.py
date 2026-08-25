import abc
import copy
import json
import math
import os
import threading

from core.lib.common import (
    ClassFactory,
    ClassType,
    ConfigLoader,
    Context,
    LOGGER,
)
from core.lib.estimation import OverheadEstimator
from core.lib.scheduling import (
    SchedulingSnapshotScope,
    deployment_from_snapshot,
    materialize_offloading_plan,
    service_names,
    snapshot_queue_states,
    topological_order,
)

from .base_agent import BaseAgent

__all__ = ("DTODRLAgent",)


_PROFILE_VERSION = 5
_PROFILE_METRIC = "real_execute_time_seconds"
_PROFILE_HISTORY_SIZE = 128


def _load_mapping(value, label):
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if isinstance(value, str):
        loaded = ConfigLoader.load(Context.get_file_path(value))
        if isinstance(loaded, dict):
            return loaded
    raise TypeError(f"DTODRL {label} must be a mapping or mounted file path")


def _samples(value):
    raw = value.get("samples", []) if isinstance(value, dict) else []
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        try:
            item = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(item) and item > 0.0:
            result.append(item)
    return result[-_PROFILE_HISTORY_SIZE:]


def _quantile(values, quantile, default=0.0):
    if not values:
        return max(0.0, float(default))
    ordered = sorted(float(item) for item in values)
    q = min(1.0, max(0.0, float(quantile)))
    return max(0.0, ordered[int(round(q * (len(ordered) - 1)))])


class _DTODRLLatencyProfile:
    """DTODRL-owned immutable view of its offline timing profile."""

    def __init__(self, configuration, profile):
        self.profile = copy.deepcopy(profile)
        self.context = {}
        if self.profile:
            if self.profile.get("version") != _PROFILE_VERSION:
                raise ValueError(
                    "DTODRL latency profile version "
                    f"{self.profile.get('version')!r} is incompatible with "
                    f"version {_PROFILE_VERSION}"
                )
            if self.profile.get("metric") != _PROFILE_METRIC:
                raise ValueError(
                    "DTODRL latency profile metric must be "
                    f"{_PROFILE_METRIC!r}"
                )
            if not isinstance(self.profile.get("pairs", {}), dict):
                raise TypeError("DTODRL profile pairs must be a mapping")
            self.context = self._normalize_context(
                self.profile.get("context")
            )
            expected = self._normalize_configuration(configuration)
            if self.context["configuration"] != expected:
                raise ValueError(
                    "DTODRL latency profile context mismatch for configuration"
                )
            self._validate_profile_pairs()

    @staticmethod
    def _normalize_configuration(configuration):
        if not isinstance(configuration, dict):
            raise TypeError("DTODRL profile configuration must be a mapping")
        try:
            return json.loads(json.dumps(
                configuration,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            ))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "DTODRL profile configuration must be JSON serializable"
            ) from exc

    @staticmethod
    def _normalize_deployment(deployment):
        if not isinstance(deployment, dict):
            raise TypeError("DTODRL profile deployment must be a mapping")
        normalized = {}
        for service, raw_devices in sorted(
            deployment.items(), key=lambda item: str(item[0])
        ):
            if isinstance(raw_devices, str):
                raw_devices = [raw_devices]
            if not isinstance(raw_devices, (list, tuple, set)):
                raise TypeError(
                    "DTODRL profile deployment for "
                    f"{service!r} must be a node list"
                )
            normalized[str(service)] = sorted({
                str(device).strip()
                for device in raw_devices
                if str(device).strip()
            })
        return normalized

    @staticmethod
    def _normalize_dag(dag):
        if not isinstance(dag, dict):
            raise TypeError("DTODRL profile DAG must be a mapping")
        normalized = {}
        for raw_name, raw_node in sorted(
            dag.items(), key=lambda item: str(item[0])
        ):
            name = str(raw_name)
            node = raw_node if isinstance(raw_node, dict) else {}
            service = node.get("service")
            service = service if isinstance(service, dict) else {}
            normalized[name] = {
                "service_name": str(node.get(
                    "service_name",
                    service.get("service_name", name),
                )),
                "prev_nodes": sorted(
                    str(item) for item in node.get("prev_nodes", [])
                ),
                "next_nodes": sorted(
                    str(item) for item in node.get("next_nodes", [])
                ),
            }
        return normalized

    @classmethod
    def _normalize_context(cls, context):
        if not isinstance(context, dict):
            raise ValueError("DTODRL latency profile has no valid context")
        missing = [
            field for field in ("configuration", "deployment", "dag")
            if field not in context
        ]
        if missing:
            raise ValueError(
                "DTODRL latency profile context is incomplete: "
                + ", ".join(missing)
            )
        return {
            "configuration": cls._normalize_configuration(
                context["configuration"]
            ),
            "deployment": cls._normalize_deployment(context["deployment"]),
            "dag": cls._normalize_dag(context["dag"]),
        }

    def _validate_profile_pairs(self):
        deployment = self.context["deployment"]
        for key in ("pairs", "handoff_pairs", "pair_log_drift"):
            store = self.profile.get(key) or {}
            if not isinstance(store, dict):
                continue
            for service, devices in store.items():
                if not isinstance(devices, dict):
                    continue
                for device in devices:
                    if str(device) not in deployment.get(str(service), []):
                        raise ValueError(
                            "DTODRL latency profile contains data outside its "
                            f"deployment context: {service}@{device}"
                        )

    def ensure_context(self, configuration, deployment, dag):
        expected = {
            "configuration": self._normalize_configuration(configuration),
            "deployment": self._normalize_deployment(deployment),
            "dag": self._normalize_dag(dag),
        }
        if self.context and self.context != expected:
            mismatches = [
                field for field in expected
                if self.context.get(field) != expected[field]
            ]
            raise ValueError(
                "DTODRL latency profile context mismatch for "
                + ", ".join(mismatches)
            )
        self.context = expected
        missing = []
        for service in service_names(dag):
            devices = deployment.get(service, [])
            if not devices:
                raise ValueError(
                    f"DTODRL fixed deployment has no replica for {service}"
                )
            for device in devices:
                if not self._pair_values(service, device):
                    missing.append(f"{service}@{device}")
        if missing:
            raise ValueError(
                "DTODRL latency profile does not cover the active fixed "
                "deployment: " + ", ".join(sorted(missing))
            )

    def _pair_values(self, service, device):
        value = (
            (self.profile.get("pairs") or {})
            .get(str(service), {})
            .get(str(device))
        )
        return _samples(value)

    def estimate(self, service, device, quantile):
        pairs = self.profile.get("pairs") or {}
        values = self._pair_values(service, device)
        if not values:
            values = [
                item
                for value in (pairs.get(str(service)) or {}).values()
                for item in _samples(value)
            ]
        if not values:
            values = [
                item
                for devices in pairs.values()
                if isinstance(devices, dict)
                for value in devices.values()
                for item in _samples(value)
            ]
        drift_store = self.profile.get("pair_log_drift") or {}
        drift_store = drift_store if isinstance(drift_store, dict) else {}
        service_drift = drift_store.get(str(service)) or {}
        service_drift = service_drift if isinstance(service_drift, dict) else {}
        drift = service_drift.get(str(device), 0.0)
        try:
            drift = float(drift)
        except (TypeError, ValueError):
            drift = 0.0
        if not math.isfinite(drift):
            drift = 0.0
        return max(1e-6, _quantile(values, quantile, 0.1) * math.exp(drift))

    def estimate_handoff(self, service, device, quantile):
        handoffs = self.profile.get("handoff_pairs") or {}
        handoffs = handoffs if isinstance(handoffs, dict) else {}
        value = (
            handoffs.get(str(service), {})
            .get(str(device))
        )
        return _quantile(_samples(value), quantile)


def _replica_load(profile, service, device, state, quantile):
    processing = profile.estimate(service, device, quantile)
    handoff = profile.estimate_handoff(service, device, quantile)
    demand = processing + handoff
    state = state if isinstance(state, dict) else {}
    try:
        waiting_count = max(0, int(state.get("waiting_count") or 0))
    except (TypeError, ValueError):
        waiting_count = 0
    busy = bool(state.get("busy"))
    remaining = 0.0
    if busy:
        phase = str(state.get("running_phase") or "processing").lower()
        try:
            elapsed = float(
                state.get("phase_elapsed_s", state.get("running_elapsed_s"))
                or 0.0
            ) + float(state.get("_age_s") or 0.0)
        except (TypeError, ValueError):
            elapsed = 0.0
        elapsed = max(0.0, elapsed)
        if phase == "processing":
            processing_remaining = processing - elapsed
            if processing_remaining <= 1e-6:
                processing_remaining = processing
            remaining = processing_remaining + handoff
        elif phase in ("handoff", "sending", "returning"):
            remaining = handoff - elapsed
            if remaining <= 1e-6:
                remaining = max(1e-3, handoff)
        else:
            remaining = demand
    return {
        "workload": max(0.0, remaining + waiting_count * demand),
        "demand": max(1e-6, demand),
        "waiting_count": waiting_count,
        "busy": busy,
    }


def _visible_replica_loads(
    snapshot,
    profile,
    dag,
    deployment,
    quantile,
    max_age_s,
):
    states = snapshot_queue_states(snapshot, max_age_s=max_age_s)
    return {
        (service, device): _replica_load(
            profile,
            service,
            device,
            states.get((service, device)),
            quantile,
        )
        for service in service_names(dag)
        for device in deployment.get(service, [])
    }


@ClassFactory.register(ClassType.SCH_AGENT, alias="dtodrl")
class DTODRLAgent(BaseAgent, abc.ABC):
    """GAT/PPO dependent-task offloading baseline adapted from DTODRL."""

    def __init__(
        self,
        system,
        agent_id: int,
        configuration=None,
        latency_profile=None,
        latency_slo_s=2.5,
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
        decision_log_interval=100,
    ):
        super().__init__(system, agent_id)
        self.configuration = _load_mapping(configuration, "configuration")
        self.latency_model = _DTODRLLatencyProfile(
            self.configuration,
            _load_mapping(latency_profile, "latency_profile"),
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
            "scheduler/dtodrl",
            agent_id=agent_id,
            write_file=False,
            log_each=False,
        )
        self.decision_log_interval = max(1, int(decision_log_interval))
        self._decision_count = 0
        self.policy = None
        self.policy_signature = None
        self.pending = {}
        self.completed = []
        self.last_training_metrics = None
        self._lock = threading.RLock()

    def _signature(self, dag, deployment, order):
        order_set = set(order)
        return {
            "state_schema": 1,
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
        loads = _visible_replica_loads(
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
            snapshot = self.system.get_scheduling_snapshot(
                scope=SchedulingSnapshotScope.LIVE,
            )
            deployment = deployment_from_snapshot(snapshot)
            self.latency_model.ensure_context(
                self.configuration,
                deployment,
                dag,
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
            self._decision_count += 1
            if (
                self._decision_count == 1
                or self._decision_count % self.decision_log_interval == 0
            ):
                LOGGER.info(
                    "[DTODRL] count=%s source=%s mode=%s root=%s "
                    "value=%.4f plan=%s",
                    self._decision_count,
                    info.get("source_id"),
                    self.mode,
                    root_uuid or "-",
                    value,
                    plan,
                )
            return materialize_offloading_plan(
                self.configuration,
                dag,
                plan,
                info.get("source_device"),
                self.cloud_device,
            )

    def update_task(self, task):
        if self.mode != "train":
            return
        root_uuid = str(task.get_root_uuid() or "")
        with self._lock:
            transition = self.pending.pop(root_uuid, None)
            if transition is None:
                return
            try:
                latency = float(task.get_real_end_to_end_time())
            except (TypeError, ValueError):
                LOGGER.warning(
                    "[DTODRL] Ignore task %s without valid end-to-end latency",
                    root_uuid,
                )
                return
            slo = self._metadata_slo(
                task.get_metadata(), transition["slo"]
            )
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
