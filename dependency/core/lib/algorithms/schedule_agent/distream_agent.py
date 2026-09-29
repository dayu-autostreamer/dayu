import abc
import copy
import json
import math
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
)

from .base_agent import BaseAgent

__all__ = ("DistreamAgent",)


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
    raise TypeError(f"Distream {label} must be a mapping or mounted file path")


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


class _DistreamLatencyProfile:
    """Distream-owned immutable view of its offline timing profile."""

    def __init__(self, configuration, profile):
        self.profile = copy.deepcopy(profile)
        self.context = {}
        if self.profile:
            if self.profile.get("version") != _PROFILE_VERSION:
                raise ValueError(
                    "Distream latency profile version "
                    f"{self.profile.get('version')!r} is incompatible with "
                    f"version {_PROFILE_VERSION}"
                )
            if self.profile.get("metric") != _PROFILE_METRIC:
                raise ValueError(
                    "Distream latency profile metric must be "
                    f"{_PROFILE_METRIC!r}"
                )
            if not isinstance(self.profile.get("pairs", {}), dict):
                raise TypeError("Distream profile pairs must be a mapping")
            self.context = self._normalize_context(
                self.profile.get("context")
            )
            expected = self._normalize_configuration(configuration)
            if self.context["configuration"] != expected:
                raise ValueError(
                    "Distream latency profile context mismatch for "
                    "configuration"
                )
            self._validate_profile_pairs()

    @staticmethod
    def _normalize_configuration(configuration):
        if not isinstance(configuration, dict):
            raise TypeError("Distream profile configuration must be a mapping")
        try:
            return json.loads(json.dumps(
                configuration,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            ))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Distream profile configuration must be JSON serializable"
            ) from exc

    @staticmethod
    def _normalize_deployment(deployment):
        if not isinstance(deployment, dict):
            raise TypeError("Distream profile deployment must be a mapping")
        normalized = {}
        for service, raw_devices in sorted(
            deployment.items(), key=lambda item: str(item[0])
        ):
            if isinstance(raw_devices, str):
                raw_devices = [raw_devices]
            if not isinstance(raw_devices, (list, tuple, set)):
                raise TypeError(
                    "Distream profile deployment for "
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
            raise TypeError("Distream profile DAG must be a mapping")
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
            raise ValueError("Distream latency profile has no valid context")
        missing = [
            field for field in ("configuration", "deployment", "dag")
            if field not in context
        ]
        if missing:
            raise ValueError(
                "Distream latency profile context is incomplete: "
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
                            "Distream latency profile contains data outside "
                            "its deployment context: "
                            f"{service}@{device}"
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
                "Distream latency profile context mismatch for "
                + ", ".join(mismatches)
            )
        self.context = expected
        missing = []
        for service in service_names(dag):
            devices = deployment.get(service, [])
            if not devices:
                raise ValueError(
                    f"Distream fixed deployment has no replica for {service}"
                )
            for device in devices:
                if not self._pair_values(service, device):
                    missing.append(f"{service}@{device}")
        if missing:
            raise ValueError(
                "Distream latency profile does not cover the active fixed "
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


@ClassFactory.register(ClassType.SCH_AGENT, alias="distream")
class DistreamAgent(BaseAgent, abc.ABC):
    """Reactive workload-balancing baseline adapted from Distream.

    Dayu cannot migrate an already committed invocation or repartition a
    pipeline.  The compatible Distream mechanism is therefore applied at task
    admission: each service is assigned to the replica with the smallest
    currently visible projected workload, and the resulting full DAG mapping
    is committed once.
    """

    def __init__(
        self,
        system,
        agent_id: int,
        configuration=None,
        latency_profile=None,
        profile_quantile=0.5,
        queue_state_max_age_s=1.5,
        decision_log_interval=100,
    ):
        super().__init__(system, agent_id)
        self.configuration = _load_mapping(configuration, "configuration")
        self.latency_model = _DistreamLatencyProfile(
            self.configuration,
            _load_mapping(latency_profile, "latency_profile"),
        )
        self.profile_quantile = min(1.0, max(0.0, float(profile_quantile)))
        self.queue_state_max_age_s = max(0.0, float(queue_state_max_age_s))
        self.overhead_estimator = OverheadEstimator(
            "Distream",
            "scheduler/distream",
            agent_id=agent_id,
            write_file=False,
            log_each=False,
        )
        self.decision_log_interval = max(1, int(decision_log_interval))
        self._decision_count = 0
        self._lock = threading.RLock()

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
            loads = _visible_replica_loads(
                snapshot,
                self.latency_model,
                dag,
                deployment,
                self.profile_quantile,
                self.queue_state_max_age_s,
            )
            plan = {}
            projected = {}
            for service in service_names(dag):
                device = min(
                    deployment[service],
                    key=lambda candidate: (
                        loads[(service, candidate)]["workload"]
                        + loads[(service, candidate)]["demand"],
                        loads[(service, candidate)]["waiting_count"],
                        str(candidate),
                    ),
                )
                plan[service] = str(device)
                projected[service] = (
                    loads[(service, device)]["workload"]
                    + loads[(service, device)]["demand"]
                )
            self._decision_count += 1
            if (
                self._decision_count == 1
                or self._decision_count % self.decision_log_interval == 0
            ):
                LOGGER.info(
                    "[Distream] count=%s source=%s "
                    "projected_workload=%s plan=%s",
                    self._decision_count,
                    info.get("source_id"),
                    {key: round(value, 4) for key, value in projected.items()},
                    plan,
                )
            return materialize_offloading_plan(
                self.configuration,
                dag,
                plan,
                info.get("source_device"),
                self.cloud_device,
            )

    def update_scenario(self, scenario):
        pass

    def update_resource(self, device, resource):
        pass

    def update_policy(self, policy):
        pass

    def update_task(self, task):
        pass

    def run(self):
        pass

    def get_schedule_overhead(self):
        return self.overhead_estimator.get_latest_overhead()
