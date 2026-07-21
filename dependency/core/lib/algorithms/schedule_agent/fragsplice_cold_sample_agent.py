import abc
import copy
import os
import threading

from core.lib.common import (
    ClassFactory,
    ClassType,
    ConfigLoader,
    Context,
    GlobalInstanceManager,
    LOGGER,
    TaskConstant,
)
from core.lib.estimation import OverheadEstimator

from .base_agent import BaseAgent
from .fragsplice.latency_model import FragSpliceLatencyModel
from .fragsplice_agent import _load_mapping

__all__ = ("FragSpliceColdSampleAgent",)


class _FragSpliceColdState:
    def __init__(self, history_size, profile=None, seen=None):
        self.model = FragSpliceLatencyModel(
            profile=profile,
            history_size=history_size,
        )
        self.seen = dict(seen or {})
        self.decision_index = 0
        self.deployment = {}
        self.lock = threading.RLock()


@ClassFactory.register(ClassType.SCH_AGENT, alias="fragsplice_cold_sample")
class FragSpliceColdSampleAgent(BaseAgent, abc.ABC):
    """Balanced cold-start sampler for the active fixed deployment only."""

    def __init__(
        self,
        system,
        agent_id: int,
        configuration=None,
        profile_path="fragsplice-profile.json",
        warmup_samples=2,
        samples_per_pair=30,
    ):
        super().__init__(system, agent_id)
        self.system = system
        self.agent_id = agent_id
        self.configuration = _load_mapping(configuration, "configuration")
        self.profile_path = Context.get_file_path(str(profile_path))
        self.warmup_samples = max(0, int(warmup_samples))
        self.samples_per_pair = max(1, int(samples_per_pair))
        profile = {}
        if os.path.exists(self.profile_path):
            try:
                loaded = ConfigLoader.load(self.profile_path)
                profile = loaded if isinstance(loaded, dict) else {}
            except (OSError, ValueError, TypeError) as exc:
                LOGGER.warning(
                    "[FragSpliceColdSample] Ignore unreadable existing profile %s: %s",
                    self.profile_path,
                    exc,
                )
        profile_context = FragSpliceLatencyModel.validate_profile_context(
            profile,
            self.configuration,
        )
        seen = self._load_progress(profile.get("cold_progress"), profile)
        revision_getter = getattr(system, "runtime_directory_revision", None)
        revision = revision_getter() if callable(revision_getter) else 0
        self._state = GlobalInstanceManager.get_instance(
            _FragSpliceColdState,
            ("fragsplice-cold", id(system), revision),
            history_size=max(64, self.samples_per_pair + self.warmup_samples),
            profile=profile,
            seen=seen,
        )
        if profile_context is not None:
            self._state.model.ensure_profile_context(
                **profile_context,
                require_complete=True,
            )
        else:
            self._state.model.ensure_profile_context(
                configuration=self.configuration,
            )
        self.overhead_estimator = OverheadEstimator(
            "FragSpliceColdSample", "scheduler/fragsplice", agent_id=agent_id
        )
        LOGGER.info(
            "[FragSpliceColdSample] profile=%s warmup=%s samples_per_pair=%s",
            self.profile_path,
            self.warmup_samples,
            self.samples_per_pair,
        )

    def _load_progress(self, progress, profile):
        seen = {}
        raw_seen = progress.get("seen") if isinstance(progress, dict) else None
        if isinstance(raw_seen, dict):
            for service, devices in raw_seen.items():
                if not isinstance(devices, dict):
                    continue
                for device, count in devices.items():
                    try:
                        count = max(0, int(count))
                    except (TypeError, ValueError):
                        continue
                    seen[(str(service), str(device))] = count
        pairs = profile.get("pairs") if isinstance(profile, dict) else None
        if isinstance(pairs, dict):
            for service, devices in pairs.items():
                if not isinstance(devices, dict):
                    continue
                for device, value in devices.items():
                    samples = FragSpliceLatencyModel._pair_samples(value)
                    pair = (str(service), str(device))
                    seen.setdefault(pair, self.warmup_samples + len(samples))
        return seen

    def _progress(self):
        nested = {}
        for (service, device), count in sorted(self._state.seen.items()):
            nested.setdefault(service, {})[device] = int(count)
        return {
            "warmup_samples": self.warmup_samples,
            "samples_per_pair": self.samples_per_pair,
            "seen": nested,
        }

    @staticmethod
    def _normalize_deployment(deployment):
        result = {}
        for service, devices in (deployment or {}).items():
            if isinstance(devices, str):
                devices = [devices]
            if not isinstance(devices, (list, tuple, set)):
                continue
            normalized = sorted({str(item) for item in devices if str(item or "")})
            if normalized:
                result[str(service)] = normalized
        return result

    def _refresh_deployment(self, deployment=None):
        if deployment is None:
            deployment = self.system.runtime_service_nodes()
        current = self._normalize_deployment(deployment)
        if current:
            self._state.model.ensure_profile_context(deployment=current)
        if current != self._state.deployment:
            self._state.deployment = current
            for service, devices in current.items():
                for device in devices:
                    self._state.seen.setdefault(
                        (service, device),
                        self.warmup_samples
                        + self._state.model.sample_count(service, device),
                    )

    def _target_reached(self, service, device):
        return self._state.seen.get((service, device), 0) >= self.warmup_samples + self.samples_per_pair

    def is_complete(self):
        with self._state.lock:
            self._refresh_deployment()
            pairs = [
                (service, device)
                for service, devices in self._state.deployment.items()
                for device in devices
            ]
            return bool(pairs) and all(self._target_reached(*pair) for pair in pairs)

    @staticmethod
    def _planned_pair_counts(snapshot):
        counts = {}
        records = list(snapshot.get("reservations", [])) + list(snapshot.get("commitments", []))
        for record in records:
            if not isinstance(record, dict):
                continue
            dag = record.get("dag")
            if not isinstance(dag, dict):
                plan = record.get("plan")
                dag = plan.get("dag") if isinstance(plan, dict) else None
            if not isinstance(dag, dict):
                continue
            for service, node in dag.items():
                spec = node.get("service", {}) if isinstance(node, dict) else {}
                device = str(spec.get("execute_device") or "")
                if device:
                    pair = (str(service), device)
                    counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _choose_device(self, service, planned):
        devices = self._state.deployment.get(service, [])
        if not devices:
            raise ValueError(
                f"FragSplice cold profile has no deployed replica for service {service}"
            )
        return min(
            devices,
            key=lambda device: (
                self._state.seen.get((service, device), 0)
                + planned.get((service, device), 0),
                (devices.index(device) - self._state.decision_index) % len(devices),
            ),
        )

    def get_schedule_plan(self, info):
        with self.overhead_estimator, self._state.lock:
            snapshot_getter = getattr(self.system, "get_scheduling_snapshot", None)
            snapshot = snapshot_getter() if callable(snapshot_getter) else {}
            deployment = snapshot.get("deployment") if isinstance(snapshot, dict) else None
            self._refresh_deployment(
                deployment if isinstance(deployment, dict) else None
            )
            planned = self._planned_pair_counts(snapshot)
            dag = copy.deepcopy(info["dag"])
            self._state.model.ensure_profile_context(
                configuration=self.configuration,
                deployment=self._state.deployment,
                dag=dag,
                require_complete=True,
            )
            for service in dag:
                if service == TaskConstant.START.value:
                    device = str(info.get("source_device") or "")
                elif service == TaskConstant.END.value:
                    device = self.cloud_device
                else:
                    device = self._choose_device(service, planned)
                dag[service]["service"]["execute_device"] = device
            self._state.decision_index += 1
            policy = copy.deepcopy(self.configuration)
            policy["dag"] = dag
            return policy

    def update_task(self, task):
        with self._state.lock:
            deployment_getter = getattr(task, "get_deployment", None)
            deployment = deployment_getter() if callable(deployment_getter) else None
            self._refresh_deployment(
                deployment if isinstance(deployment, dict) and deployment else None
            )
            self._state.model.ensure_profile_context(
                configuration=self.configuration,
                deployment=self._state.deployment,
                dag=task.get_dag(),
                require_complete=True,
            )
            recorded = 0
            progressed = 0
            for service_name in task.get_dag().nodes:
                if service_name in (TaskConstant.START.value, TaskConstant.END.value):
                    continue
                service = task.get_service(service_name)
                device = str(service.get_execute_device() or "")
                pair = (str(service_name), device)
                if pair not in self._state.seen or self._target_reached(*pair):
                    continue
                duration = service.get_real_execute_time()
                try:
                    valid = float(duration) > 0.0
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    continue
                self._state.seen[pair] += 1
                progressed += 1
                if self._state.seen[pair] > self.warmup_samples:
                    self._state.model.record_service_sample(
                        service_name, device, service
                    )
                    recorded += 1
            if progressed:
                observed_pairs = [
                    (str(name), str(task.get_service(name).get_execute_device() or ""))
                    for name in task.get_dag().nodes
                    if name not in (TaskConstant.START.value, TaskConstant.END.value)
                ]
                if recorded and all(
                    self._state.model.sample_count(service, device) > 0
                    for service, device in observed_pairs
                ):
                    self._state.model.record_task_residual(task)
                self._state.model.save(
                    self.profile_path,
                    deployment=self._state.deployment,
                    cold_progress=self._progress(),
                )
            LOGGER.info(
                "[FragSpliceColdSample] source=%s task=%s recorded=%s progress=%s",
                task.get_source_id(),
                task.get_task_id(),
                recorded,
                {
                    f"{service}@{device}": min(
                        self.samples_per_pair,
                        max(0, count - self.warmup_samples),
                    )
                    for (service, device), count in sorted(self._state.seen.items())
                },
            )

    def should_generate(self, info):
        complete = self.is_complete()
        return {
            "generate": not complete,
            "reason": "fragsplice_profile_complete" if complete else "fragsplice_profile_collecting",
        }

    def get_profile(self):
        with self._state.lock:
            return self._state.model.to_profile(
                deployment=self._state.deployment,
                cold_progress=self._progress(),
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
