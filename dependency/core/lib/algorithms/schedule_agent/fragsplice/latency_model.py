import copy
import json
import math
import os
import tempfile
import threading
from collections import defaultdict, deque
from statistics import median

from core.lib.common import TaskConstant


def _positive(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0.0 else None


def _non_negative(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0.0 else None


def _handoff_duration(service):
    """Return Processor-finish to Controller-ack time when it was recorded."""

    getter = getattr(service, "get_tmp_data", None)
    timing = getter() if callable(getter) else {}
    if not isinstance(timing, dict):
        return None
    execute_end = _non_negative(timing.get("execute_end"))
    real_execute_end = _non_negative(timing.get("real_execute_end"))
    if execute_end is None or real_execute_end is None:
        return None
    return max(0.0, execute_end - real_execute_end)


def _dispatch_duration(service):
    """Return Controller-dispatch to Processor-start time when recorded.

    Cold profiling runs with one in-flight root, so this interval represents
    serialization, local delivery, and Processor admission rather than FIFO
    queueing. Online traces are deliberately not used for this component.
    """

    getter = getattr(service, "get_tmp_data", None)
    timing = getter() if callable(getter) else {}
    if not isinstance(timing, dict):
        return None
    execute_start = _non_negative(timing.get("execute_start"))
    real_execute_start = _non_negative(timing.get("real_execute_start"))
    if execute_start is None or real_execute_start is None:
        return None
    return max(0.0, real_execute_start - execute_start)


def _timestamp(service, name):
    getter = getattr(service, "get_tmp_data", None)
    timing = getter() if callable(getter) else {}
    if not isinstance(timing, dict):
        return None
    return _non_negative(timing.get(name))


class FragSpliceLatencyModel:
    """Distribution Profiler for service demand and non-Processor overheads.

    The model deliberately predicts processor demand (``real_execute_time``),
    never queue-inclusive service span. A stable service-device baseline is
    multiplied by one jointly resampled task residual vector. Pair samples are
    not sampled again after a joint residual is available, which avoids
    counting the same content variation twice.

    Non-Processor overhead distributions are collected only by the
    single-in-flight cold profiler. Online traces may contain queueing,
    temporary routing failures, or recovery stalls that the future-state
    simulator must not learn as an intrinsic dispatch/control/handoff cost.
    """

    PROFILE_VERSION = 5
    PROFILE_METRIC = "real_execute_time_seconds"
    PROFILE_CONTEXT_FIELDS = ("configuration", "deployment", "dag")
    _SAVE_LOCK = threading.Lock()

    def __init__(
        self,
        profile=None,
        history_size=128,
        drift_alpha=0.15,
        residual_half_life_tasks=8.0,
    ):
        self.history_size = max(16, int(history_size))
        self.drift_alpha = min(1.0, max(0.01, float(drift_alpha)))
        self.residual_half_life_tasks = max(
            1.0, float(residual_half_life_tasks)
        )
        self._samples = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.history_size)))
        self._handoff_samples = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.history_size))
        )
        self._transfer_samples = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.history_size))
        )
        self._dispatch_samples = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.history_size))
        )
        self._control_samples = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.history_size))
        )
        self._completion_samples = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )
        self._pair_log_drift = defaultdict(dict)
        self._task_residuals = defaultdict(lambda: deque(maxlen=self.history_size))
        self._profile_context = None
        self._lock = threading.RLock()
        if profile:
            self.load(profile)

    @staticmethod
    def _normalize_configuration(configuration):
        if not isinstance(configuration, dict):
            raise TypeError("FragSplice profile configuration must be a mapping")
        try:
            return json.loads(json.dumps(
                configuration,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            ))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "FragSplice profile configuration must be JSON serializable"
            ) from exc

    @staticmethod
    def _normalize_deployment(deployment):
        if not isinstance(deployment, dict):
            raise TypeError("FragSplice profile deployment must be a mapping")
        normalized = {}
        for service, raw_devices in sorted(deployment.items(), key=lambda item: str(item[0])):
            if isinstance(raw_devices, str):
                raw_devices = [raw_devices]
            if not isinstance(raw_devices, (list, tuple, set)):
                raise TypeError(
                    f"FragSplice profile deployment for {service!r} must be a node list"
                )
            normalized[str(service)] = sorted({
                str(device).strip()
                for device in raw_devices
                if str(device).strip()
            })
        return normalized

    @staticmethod
    def _normalize_dag(dag):
        converter = getattr(dag, "to_dict", None)
        if callable(converter):
            dag = converter()
        if not isinstance(dag, dict):
            raise TypeError("FragSplice profile DAG must be a mapping")
        normalized = {}
        for raw_name, raw_node in sorted(dag.items(), key=lambda item: str(item[0])):
            name = str(raw_name)
            node = raw_node if isinstance(raw_node, dict) else {}
            service = node.get("service")
            service = service if isinstance(service, dict) else {}
            service_name = node.get(
                "service_name",
                service.get("service_name", name),
            )
            normalized[name] = {
                "service_name": str(service_name),
                "prev_nodes": sorted(str(item) for item in node.get("prev_nodes", [])),
                "next_nodes": sorted(str(item) for item in node.get("next_nodes", [])),
            }
        return normalized

    @classmethod
    def build_profile_context(cls, configuration, deployment, dag):
        return {
            "configuration": cls._normalize_configuration(configuration),
            "deployment": cls._normalize_deployment(deployment),
            "dag": cls._normalize_dag(dag),
        }

    @classmethod
    def _normalize_profile_context(cls, context):
        if not isinstance(context, dict):
            raise ValueError("FragSplice latency profile has no valid context")
        missing = [field for field in cls.PROFILE_CONTEXT_FIELDS if field not in context]
        if missing:
            raise ValueError(
                "FragSplice latency profile context is incomplete: "
                + ", ".join(missing)
            )
        return cls.build_profile_context(
            context["configuration"],
            context["deployment"],
            context["dag"],
        )

    @classmethod
    def validate_profile_context(cls, profile, configuration):
        """Reject persisted profiles that predate or mismatch strict context."""
        if not profile:
            return None
        if profile.get("version") != cls.PROFILE_VERSION:
            raise ValueError(
                f"FragSplice latency profile version {profile.get('version')!r} is not "
                f"compatible with strict context version {cls.PROFILE_VERSION}; "
                "collect a new cold profile"
            )
        if profile.get("metric") != cls.PROFILE_METRIC:
            raise ValueError(
                "FragSplice latency profile metric must be "
                f"{cls.PROFILE_METRIC!r}"
            )
        context = cls._normalize_profile_context(profile.get("context"))
        expected = cls._normalize_configuration(configuration)
        if context["configuration"] != expected:
            raise ValueError(
                "FragSplice latency profile context mismatch for configuration"
            )
        return context

    def ensure_profile_context(
        self,
        configuration=None,
        deployment=None,
        dag=None,
        require_complete=False,
    ):
        supplied = {}
        if configuration is not None:
            supplied["configuration"] = self._normalize_configuration(configuration)
        if deployment is not None:
            supplied["deployment"] = self._normalize_deployment(deployment)
        if dag is not None:
            supplied["dag"] = self._normalize_dag(dag)

        with self._lock:
            if self._profile_context is None:
                self._profile_context = {}
            for field, expected in supplied.items():
                actual = self._profile_context.get(field)
                if actual is not None and actual != expected:
                    raise ValueError(
                        f"FragSplice latency profile context mismatch for {field}"
                    )
                self._profile_context[field] = copy.deepcopy(expected)
            if require_complete:
                missing = [
                    field for field in self.PROFILE_CONTEXT_FIELDS
                    if field not in self._profile_context
                ]
                if missing:
                    raise ValueError(
                        "FragSplice latency profile context is not initialized: "
                        + ", ".join(missing)
                    )
            return copy.deepcopy(self._profile_context)

    def _ensure_pair_in_context(self, service, device):
        if self._profile_context is None:
            return
        deployment = self._profile_context.get("deployment")
        if deployment is None:
            return
        service = str(service)
        device = str(device)
        if device not in deployment.get(service, []):
            raise ValueError(
                "FragSplice latency profile contains a sample outside its "
                f"deployment context: {service}@{device}"
            )

    @staticmethod
    def _pair_samples(value):
        if isinstance(value, (int, float)):
            parsed = _positive(value)
            return [parsed] if parsed is not None else []
        if isinstance(value, list):
            raw = value
        elif isinstance(value, dict):
            raw = value.get("samples") or value.get("values") or []
            if not raw:
                for key in ("median", "p50", "mean"):
                    parsed = _positive(value.get(key))
                    if parsed is not None:
                        raw = [parsed]
                        break
        else:
            raw = []
        parsed = [_positive(item) for item in raw]
        return [item for item in parsed if item is not None]

    @staticmethod
    def _overhead_samples(value):
        if isinstance(value, (int, float)):
            parsed = _non_negative(value)
            return [parsed] if parsed is not None else []
        if isinstance(value, list):
            raw = value
        elif isinstance(value, dict):
            raw = value.get("samples") or value.get("values") or []
            if not raw:
                for key in ("median", "p50", "mean"):
                    parsed = _non_negative(value.get(key))
                    if parsed is not None:
                        raw = [parsed]
                        break
        else:
            raw = []
        parsed = [_non_negative(item) for item in raw]
        return [item for item in parsed if item is not None]

    def load(self, profile):
        if not isinstance(profile, dict):
            raise TypeError("FragSplice latency profile must be a mapping")
        pairs = profile.get("pairs", profile)
        if not isinstance(pairs, dict):
            raise TypeError("FragSplice profile pairs must be a mapping")
        with self._lock:
            if profile.get("context") is not None:
                self._profile_context = self._normalize_profile_context(
                    profile["context"]
                )
            for service, devices in pairs.items():
                if not isinstance(devices, dict):
                    continue
                for device, value in devices.items():
                    self._ensure_pair_in_context(service, device)
                    for sample in self._pair_samples(value):
                        self._samples[str(service)][str(device)].append(sample)
            handoffs = profile.get("handoff_pairs", {})
            if isinstance(handoffs, dict):
                for service, devices in handoffs.items():
                    if not isinstance(devices, dict):
                        continue
                    for device, value in devices.items():
                        self._ensure_pair_in_context(service, device)
                        for sample in self._pair_samples(value):
                            self._handoff_samples[str(service)][str(device)].append(sample)
            for profile_key, target in (
                ("transfer_pairs", self._transfer_samples),
                ("dispatch_pairs", self._dispatch_samples),
                ("control_pairs", self._control_samples),
            ):
                values_by_service = profile.get(profile_key, {})
                if not isinstance(values_by_service, dict):
                    continue
                for service, devices in values_by_service.items():
                    if not isinstance(devices, dict):
                        continue
                    for device, value in devices.items():
                        for sample in self._overhead_samples(value):
                            target[str(service)][str(device)].append(sample)
            completions = profile.get("completion_overhead", {})
            if isinstance(completions, dict):
                for source, value in completions.items():
                    for sample in self._overhead_samples(value):
                        self._completion_samples[str(source)].append(sample)
            drifts = profile.get("pair_log_drift", {})
            if isinstance(drifts, dict):
                for service, devices in drifts.items():
                    if not isinstance(devices, dict):
                        continue
                    for device, value in devices.items():
                        self._ensure_pair_in_context(service, device)
                        try:
                            value = float(value)
                        except (TypeError, ValueError):
                            continue
                        if math.isfinite(value):
                            self._pair_log_drift[str(service)][str(device)] = value
            histories = profile.get("task_residuals", {})
            if isinstance(histories, dict):
                for source, records in histories.items():
                    if not isinstance(records, list):
                        continue
                    for record in records[-self.history_size:]:
                        if isinstance(record, dict):
                            self._task_residuals[str(source)].append(copy.deepcopy(record))

    def has_samples(self):
        with self._lock:
            return any(samples for devices in self._samples.values() for samples in devices.values())

    def sample_count(self, service, device):
        with self._lock:
            return len(self._samples.get(str(service), {}).get(str(device), ()))

    def record_sample(self, service, device, duration, handoff=None):
        duration = _positive(duration)
        if duration is None:
            return False
        with self._lock:
            self._ensure_pair_in_context(service, device)
            self._samples[str(service)][str(device)].append(duration)
            handoff = _non_negative(handoff)
            if handoff is not None:
                self._handoff_samples[str(service)][str(device)].append(handoff)
        return True

    def record_service_sample(self, service_name, device, service):
        return self.record_sample(
            service_name,
            device,
            service.get_real_execute_time(),
            handoff=_handoff_duration(service),
        )

    def pair_values(self, service, device):
        with self._lock:
            return list(self._samples.get(str(service), {}).get(str(device), ()))

    def handoff_values(self, service, device):
        with self._lock:
            return list(
                self._handoff_samples.get(str(service), {}).get(str(device), ())
            )

    @staticmethod
    def _context_values(store, service, device):
        service = str(service)
        device = str(device)
        values = list(store.get(service, {}).get(device, ()))
        if not values:
            values = [
                item
                for samples in store.get(service, {}).values()
                for item in samples
            ]
        if not values:
            values = [
                item
                for devices in store.values()
                for samples in devices.values()
                for item in samples
            ]
        return values

    def transfer_values(self, service, device):
        with self._lock:
            return self._context_values(
                self._transfer_samples, service, device
            )

    def dispatch_values(self, service, device):
        with self._lock:
            return self._context_values(
                self._dispatch_samples, service, device
            )

    def control_values(self, service, device):
        with self._lock:
            return self._context_values(
                self._control_samples, service, device
            )

    def completion_values(self, source_id):
        source_key = str(source_id)
        with self._lock:
            values = list(self._completion_samples.get(source_key, ()))
            if not values:
                values = [
                    item
                    for samples in self._completion_samples.values()
                    for item in samples
                ]
            return values

    def _pair_drift(self, service, device):
        return float(
            self._pair_log_drift.get(str(service), {}).get(str(device), 0.0)
        )

    def _base_estimate(self, service, device, quantile=0.5):
        service = str(service)
        device = str(device)
        values = list(self._samples.get(service, {}).get(device, ()))
        if not values:
            values = [
                item
                for samples in self._samples.get(service, {}).values()
                for item in samples
            ]
        if not values:
            values = [
                item
                for devices in self._samples.values()
                for samples in devices.values()
                for item in samples
            ]
        if not values:
            return 0.1
        ordered = sorted(values)
        q = min(1.0, max(0.0, float(quantile)))
        index = int(round(q * (len(ordered) - 1)))
        return max(1e-6, float(ordered[index]))

    def estimate(self, service, device, quantile=0.5):
        with self._lock:
            base = self._base_estimate(service, device, quantile)
            drift = self._pair_drift(service, device)
        return max(1e-6, base * math.exp(drift))

    def lower_bound(self, service, device):
        with self._lock:
            values = list(self._samples.get(str(service), {}).get(str(device), ()))
            drift = self._pair_drift(service, device)
        if not values:
            return self.estimate(service, device, 0.5)
        return max(1e-6, min(values) * math.exp(drift))

    @staticmethod
    def _task_residual(record, service):
        raw = (
            record.get(str(service), record.get("__shared__", 0.0))
            if isinstance(record, dict) else 0.0
        )
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) else 0.0

    def sample_lower_bound(self, source_id, service, device):
        """Return the infimum of ``sample_task`` for this source and pair.

        ``lower_bound`` describes the cold pair samples after current pair
        drift. Once task residual history exists, however, ``sample_task``
        draws from the residual-conditioned median rather than directly from
        those pair samples. Pair drift and task residuals may use different
        gauges while still cancelling in the sampled duration, so the pair
        minimum is not necessarily a lower bound on scenario samples.
        """
        source_key = str(source_id)
        with self._lock:
            histories = list(self._task_residuals.get(source_key, ()))
            drift = self._pair_drift(service, device)
            base = (
                self._base_estimate(service, device, 0.5)
                * math.exp(drift)
            )
            pair_values = list(
                self._samples.get(str(service), {}).get(str(device), ())
            )
        if histories:
            residual = min(
                self._task_residual(record, service)
                for record in histories
            )
            return max(1e-6, base * math.exp(residual))
        if pair_values:
            return max(
                1e-6,
                min(pair_values) * math.exp(drift),
            )
        return max(1e-6, base)

    def estimate_handoff(self, service, device, quantile=0.5):
        values = self.handoff_values(service, device)
        if not values:
            return 0.0
        ordered = sorted(values)
        q = min(1.0, max(0.0, float(quantile)))
        index = int(round(q * (len(ordered) - 1)))
        return max(0.0, float(ordered[index]))

    def lower_bound_handoff(self, service, device):
        values = self.handoff_values(service, device)
        return max(0.0, min(values)) if values else 0.0

    @staticmethod
    def _quantile(values, quantile, default=0.0):
        if not values:
            return max(0.0, float(default))
        ordered = sorted(float(item) for item in values)
        q = min(1.0, max(0.0, float(quantile)))
        index = int(round(q * (len(ordered) - 1)))
        return max(0.0, ordered[index])

    def estimate_transfer(self, service, device, quantile=0.5):
        return self._quantile(
            self.transfer_values(service, device), quantile
        )

    def estimate_dispatch(self, service, device, quantile=0.5):
        return self._quantile(
            self.dispatch_values(service, device), quantile
        )

    def estimate_control(self, service, device, quantile=0.5):
        return self._quantile(
            self.control_values(service, device), quantile
        )

    def estimate_completion(self, source_id, quantile=0.5):
        return self._quantile(
            self.completion_values(source_id), quantile
        )

    def sample_task(self, source_id, plan, rng):
        """Draw one correlated service-time vector for a full plan."""
        source_key = str(source_id)
        with self._lock:
            histories = list(self._task_residuals.get(source_key, ()))
        # Video complexity is locally correlated but can change abruptly.  An
        # exponential recency kernel gives the most recent completed tasks a
        # well-defined influence horizon while retaining older modes as
        # low-probability tail scenarios.
        weights = [
            2.0 ** (
                -(len(histories) - 1 - index)
                / self.residual_half_life_tasks
            )
            for index in range(len(histories))
        ]
        residual = (
            rng.choices(histories, weights=weights, k=1)[0]
            if histories else {}
        )
        sampled = {}
        for service in sorted(plan):
            device = str(plan[service])
            base = self.estimate(service, device, 0.5)
            if histories:
                shared = self._task_residual(residual, service)
                sampled[service] = max(1e-6, base * math.exp(shared))
            else:
                values = self.pair_values(service, device)
                if values:
                    raw = rng.choice(values)
                    sampled[service] = max(
                        1e-6, raw * math.exp(self._pair_drift(service, device))
                    )
                else:
                    sampled[service] = base
        return sampled

    def sample_handoffs(self, plan, rng):
        sampled = {}
        for service in sorted(plan):
            device = str(plan[service])
            values = self.handoff_values(service, device)
            sampled[service] = max(0.0, rng.choice(values)) if values else 0.0
        return sampled

    def sample_stage_overheads(self, source_id, dag, plan, rng):
        """Sample non-Processor timing components for one full DAG scenario."""

        transfer = {}
        dispatch = {}
        control = {}
        for service in sorted(dag):
            if service == TaskConstant.START.value:
                continue
            node = dag.get(service, {}) if isinstance(dag, dict) else {}
            spec = node.get("service", {}) if isinstance(node, dict) else {}
            device = str(plan.get(service) or spec.get("execute_device") or "")
            transfer_values = self.transfer_values(service, device)
            control_values = self.control_values(service, device)
            transfer[service] = (
                max(0.0, rng.choice(transfer_values))
                if transfer_values else 0.0
            )
            control[service] = (
                max(0.0, rng.choice(control_values))
                if control_values else 0.0
            )
            if service != TaskConstant.END.value:
                dispatch_values = self.dispatch_values(service, device)
                dispatch[service] = (
                    max(0.0, rng.choice(dispatch_values))
                    if dispatch_values else 0.0
                )
        completion_values = self.completion_values(source_id)
        completion = (
            max(0.0, rng.choice(completion_values))
            if completion_values else 0.0
        )
        return {
            "transfer": transfer,
            "dispatch": dispatch,
            "control": control,
            "completion": completion,
        }

    @staticmethod
    def _task_dag_dict(task):
        dag_getter = getattr(task, "get_dag", None)
        dag = dag_getter() if callable(dag_getter) else None
        converter = getattr(dag, "to_dict", None)
        if callable(converter):
            dag = converter()
        return dag if isinstance(dag, dict) else None

    @staticmethod
    def _task_slo_time(task, name):
        getter = getattr(task, name, None)
        if not callable(getter):
            return None
        try:
            return _positive(getter())
        except (TypeError, ValueError):
            return None

    def record_task_overheads(self, task, include_dispatch=False):
        """Record cold-profile non-Processor intervals from one task.

        This method is intentionally called only by the single-in-flight cold
        sampler. Online ``execute_start -> real_execute_start`` and
        predecessor-to-release intervals may contain queueing or infrastructure
        recovery stalls and would otherwise poison the intrinsic overhead
        distributions used by every later scenario.
        """

        dag = self._task_dag_dict(task)
        if dag is None:
            return False
        source_getter = getattr(task, "get_source_id", None)
        source_id = source_getter() if callable(source_getter) else ""
        slo_start = self._task_slo_time(task, "get_slo_start_time")
        slo_end = self._task_slo_time(task, "get_slo_end_time")
        observed = False
        with self._lock:
            for service_name, node in dag.items():
                if service_name == TaskConstant.START.value:
                    continue
                service_getter = getattr(task, "get_service", None)
                if not callable(service_getter):
                    continue
                try:
                    service = service_getter(service_name)
                except (KeyError, AssertionError):
                    continue
                device_getter = getattr(service, "get_execute_device", None)
                device = str(device_getter() if callable(device_getter) else "")
                if not device:
                    continue

                transfer_getter = getattr(service, "get_transmit_time", None)
                transfer = _non_negative(
                    transfer_getter() if callable(transfer_getter) else None
                )
                # A zero value means the predecessor and target Controller are
                # co-located. The simulator already assigns zero to that case;
                # retaining it in the remote-transfer distribution would bias
                # cross-device plans downward.
                if transfer is not None and transfer > 0.0:
                    self._transfer_samples[str(service_name)][device].append(transfer)
                    observed = True

                predecessors = node.get("prev_nodes", []) if isinstance(node, dict) else []
                predecessor_ends = []
                for predecessor in predecessors:
                    if predecessor == TaskConstant.START.value:
                        if slo_start is not None:
                            predecessor_ends.append(slo_start)
                        continue
                    try:
                        predecessor_service = service_getter(predecessor)
                    except (KeyError, AssertionError):
                        continue
                    predecessor_end = _timestamp(
                        predecessor_service, "execute_end"
                    )
                    if predecessor_end is not None:
                        predecessor_ends.append(predecessor_end)

                release = _timestamp(service, "transmit_start")
                if transfer is None or transfer <= 0.0 or release is None:
                    release = _timestamp(service, "execute_start")
                if service_name == TaskConstant.END.value:
                    release = _timestamp(service, "transmit_start") or release
                if release is not None and predecessor_ends:
                    control = max(0.0, release - max(predecessor_ends))
                    self._control_samples[str(service_name)][device].append(control)
                    observed = True

                if include_dispatch and service_name != TaskConstant.END.value:
                    dispatch = _dispatch_duration(service)
                    if dispatch is not None:
                        self._dispatch_samples[str(service_name)][device].append(dispatch)
                        observed = True

            if slo_end is not None:
                try:
                    end_service = task.get_service(TaskConstant.END.value)
                except (AttributeError, KeyError, AssertionError):
                    end_service = None
                transfer_end = (
                    _timestamp(end_service, "transmit_end")
                    if end_service is not None else None
                )
                if transfer_end is not None:
                    completion = max(0.0, slo_end - transfer_end)
                    self._completion_samples[str(source_id)].append(completion)
                    observed = True
        return observed

    def update_task(self, task):
        source_key = str(task.get_source_id())
        observations = []
        with self._lock:
            for service_name in task.get_dag().nodes:
                if service_name in (TaskConstant.START.value, TaskConstant.END.value):
                    continue
                service = task.get_service(service_name)
                duration = _positive(service.get_real_execute_time())
                device = str(service.get_execute_device() or "")
                if duration is None or not device:
                    continue
                self._ensure_pair_in_context(service_name, device)
                if not self._samples[str(service_name)][device]:
                    self._samples[str(service_name)][device].append(duration)
                base = self._base_estimate(service_name, device, 0.5)
                drift = self._pair_drift(service_name, device)
                observations.append({
                    "service": str(service_name),
                    "device": device,
                    "duration": duration,
                    "base": base,
                    "drift": drift,
                })

            if not observations:
                return False

            # Separate a task-wide content component from pair-specific drift.
            # This keeps recent video complexity in one joint residual vector
            # instead of folding it into every service-device baseline.
            content_components = [
                math.log(item["duration"] / max(item["base"], 1e-9))
                - item["drift"]
                for item in observations
            ]
            shared = median(content_components)
            residual = {"__shared__": shared}
            for item in observations:
                service = item["service"]
                device = item["device"]
                raw = math.log(item["duration"] / max(item["base"], 1e-9))
                target_drift = raw - shared
                old_drift = item["drift"]
                new_drift = (
                    (1.0 - self.drift_alpha) * old_drift
                    + self.drift_alpha * target_drift
                )
                self._pair_log_drift[service][device] = new_drift
                residual[service] = raw - new_drift
            self._task_residuals[source_key].append(residual)
        return True

    def record_task_residual(self, task):
        """Record one joint residual vector without changing pair samples."""
        source_key = str(task.get_source_id())
        residual = {}
        with self._lock:
            for service_name in task.get_dag().nodes:
                if service_name in (TaskConstant.START.value, TaskConstant.END.value):
                    continue
                service = task.get_service(service_name)
                duration = _positive(service.get_real_execute_time())
                device = str(service.get_execute_device() or "")
                if duration is None or not device:
                    continue
                self._ensure_pair_in_context(service_name, device)
                baseline = self.estimate(service_name, device, 0.5)
                residual[str(service_name)] = math.log(
                    duration / max(baseline, 1e-9)
                )
            if not residual:
                return False
            residual["__shared__"] = median(residual.values())
            self._task_residuals[source_key].append(residual)
        return True

    def to_profile(self, deployment=None, cold_progress=None):
        context = self.ensure_profile_context(
            deployment=deployment,
            require_complete=True,
        )
        with self._lock:
            pairs = {}
            for service, devices in sorted(self._samples.items()):
                pairs[service] = {}
                for device, values in sorted(devices.items()):
                    values = list(values)
                    if not values:
                        continue
                    ordered = sorted(values)
                    pairs[service][device] = {
                        "samples": values,
                        "p50": ordered[int(round(0.50 * (len(ordered) - 1)))],
                        "p90": ordered[int(round(0.90 * (len(ordered) - 1)))],
                        "p95": ordered[int(round(0.95 * (len(ordered) - 1)))],
                    }
            handoffs = {}
            for service, devices in sorted(self._handoff_samples.items()):
                for device, values in sorted(devices.items()):
                    values = list(values)
                    if not values:
                        continue
                    ordered = sorted(values)
                    handoffs.setdefault(service, {})[device] = {
                        "samples": values,
                        "p50": ordered[int(round(0.50 * (len(ordered) - 1)))],
                        "p90": ordered[int(round(0.90 * (len(ordered) - 1)))],
                    }

            def serialize_nested(store):
                result = {}
                for service, devices in sorted(store.items()):
                    for device, raw_values in sorted(devices.items()):
                        values = list(raw_values)
                        if not values:
                            continue
                        ordered = sorted(values)
                        result.setdefault(service, {})[device] = {
                            "samples": values,
                            "p50": ordered[int(round(0.50 * (len(ordered) - 1)))],
                            "p90": ordered[int(round(0.90 * (len(ordered) - 1)))],
                            "p95": ordered[int(round(0.95 * (len(ordered) - 1)))],
                        }
                return result

            transfers = serialize_nested(self._transfer_samples)
            dispatches = serialize_nested(self._dispatch_samples)
            controls = serialize_nested(self._control_samples)
            completions = {}
            for source, raw_values in sorted(self._completion_samples.items()):
                values = list(raw_values)
                if not values:
                    continue
                ordered = sorted(values)
                completions[source] = {
                    "samples": values,
                    "p50": ordered[int(round(0.50 * (len(ordered) - 1)))],
                    "p90": ordered[int(round(0.90 * (len(ordered) - 1)))],
                    "p95": ordered[int(round(0.95 * (len(ordered) - 1)))],
                }
            drifts = {
                service: {
                    device: value
                    for device, value in sorted(devices.items())
                }
                for service, devices in sorted(self._pair_log_drift.items())
                if devices
            }
            residuals = {
                source: list(records)
                for source, records in self._task_residuals.items()
                if records
            }
        payload = {
            "version": self.PROFILE_VERSION,
            "metric": self.PROFILE_METRIC,
            "context": context,
            "deployment": copy.deepcopy(context["deployment"]),
            "pairs": pairs,
            "handoff_pairs": handoffs,
            "transfer_pairs": transfers,
            "dispatch_pairs": dispatches,
            "control_pairs": controls,
            "completion_overhead": completions,
            "pair_log_drift": drifts,
            "task_residuals": residuals,
        }
        if isinstance(cold_progress, dict):
            payload["cold_progress"] = copy.deepcopy(cold_progress)
        return payload

    def save(self, path, deployment=None, cold_progress=None):
        path = os.path.abspath(os.fspath(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Source agents share one profile path. Serialize the complete
        # snapshot-and-replace operation so an older snapshot cannot finish
        # after and overwrite a newer update.
        with self._SAVE_LOCK:
            payload = self.to_profile(
                deployment=deployment,
                cold_progress=cold_progress,
            )
            descriptor, temporary = tempfile.mkstemp(
                prefix=".fragsplice-", suffix=".json", dir=os.path.dirname(path)
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
                    stream.write("\n")
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)


class FragSpliceStaticLatencyModel:
    """Deterministic cold-profile view used by the profiler ablation.

    Every scenario receives the same P50 value from the validated cold
    profile. Task residuals, pair drift, and online feedback are deliberately
    excluded while the execution-state estimator and plan search remain
    unchanged.
    """

    def __init__(self, source):
        self.source = source

    def estimate(self, service, device, quantile=0.5):
        del quantile
        values = self.source.pair_values(service, device)
        if values:
            return max(
                1e-6,
                self.source._quantile(values, 0.5, default=0.1),
            )
        return max(
            1e-6,
            self.source._base_estimate(service, device, 0.5),
        )

    def lower_bound(self, service, device):
        return self.estimate(service, device)

    def sample_lower_bound(self, source_id, service, device):
        del source_id
        return self.estimate(service, device)

    def estimate_handoff(self, service, device, quantile=0.5):
        del quantile
        return self.source._quantile(
            self.source.handoff_values(service, device), 0.5
        )

    def lower_bound_handoff(self, service, device):
        return self.estimate_handoff(service, device)

    def sample_task(self, source_id, plan, rng):
        del source_id, rng
        return {
            service: self.estimate(service, device)
            for service, device in sorted(plan.items())
        }

    def sample_handoffs(self, plan, rng):
        del rng
        return {
            service: self.estimate_handoff(service, device)
            for service, device in sorted(plan.items())
        }

    def sample_stage_overheads(self, source_id, dag, plan, rng):
        del rng
        transfer = {}
        dispatch = {}
        control = {}
        for service in sorted(dag):
            if service == TaskConstant.START.value:
                continue
            node = dag.get(service, {}) if isinstance(dag, dict) else {}
            spec = node.get("service", {}) if isinstance(node, dict) else {}
            device = str(plan.get(service) or spec.get("execute_device") or "")
            transfer[service] = self.source._quantile(
                self.source.transfer_values(service, device), 0.5
            )
            control[service] = self.source._quantile(
                self.source.control_values(service, device), 0.5
            )
            if service != TaskConstant.END.value:
                dispatch[service] = self.source._quantile(
                    self.source.dispatch_values(service, device), 0.5
                )
        return {
            "transfer": transfer,
            "dispatch": dispatch,
            "control": control,
            "completion": self.source._quantile(
                self.source.completion_values(source_id), 0.5
            ),
        }


__all__ = ("FragSpliceLatencyModel", "FragSpliceStaticLatencyModel")
