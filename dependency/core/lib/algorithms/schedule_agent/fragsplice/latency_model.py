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


class FragSpliceLatencyModel:
    """Factorized empirical service-time model with online correction.

    The model deliberately predicts processor demand (``real_execute_time``),
    never queue-inclusive service span. A stable service-device baseline is
    multiplied by one jointly resampled task residual vector. Pair samples are
    not sampled again after a joint residual is available, which avoids
    counting the same content variation twice.
    """

    PROFILE_VERSION = 2
    _SAVE_LOCK = threading.Lock()

    def __init__(self, profile=None, history_size=128, drift_alpha=0.15):
        self.history_size = max(16, int(history_size))
        self.drift_alpha = min(1.0, max(0.01, float(drift_alpha)))
        self._samples = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.history_size)))
        self._handoff_samples = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.history_size))
        )
        self._pair_log_drift = defaultdict(dict)
        self._task_residuals = defaultdict(lambda: deque(maxlen=self.history_size))
        self._lock = threading.RLock()
        if profile:
            self.load(profile)

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

    def load(self, profile):
        if not isinstance(profile, dict):
            raise TypeError("FragSplice latency profile must be a mapping")
        pairs = profile.get("pairs", profile)
        if not isinstance(pairs, dict):
            raise TypeError("FragSplice profile pairs must be a mapping")
        with self._lock:
            for service, devices in pairs.items():
                if not isinstance(devices, dict):
                    continue
                for device, value in devices.items():
                    for sample in self._pair_samples(value):
                        self._samples[str(service)][str(device)].append(sample)
            handoffs = profile.get("handoff_pairs", {})
            if isinstance(handoffs, dict):
                for service, devices in handoffs.items():
                    if not isinstance(devices, dict):
                        continue
                    for device, value in devices.items():
                        for sample in self._pair_samples(value):
                            self._handoff_samples[str(service)][str(device)].append(sample)
            drifts = profile.get("pair_log_drift", {})
            if isinstance(drifts, dict):
                for service, devices in drifts.items():
                    if not isinstance(devices, dict):
                        continue
                    for device, value in devices.items():
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

    def sample_task(self, source_id, plan, rng):
        """Draw one correlated service-time vector for a full plan."""
        source_key = str(source_id)
        with self._lock:
            histories = list(self._task_residuals.get(source_key, ()))
        residual = (
            rng.choices(histories, weights=range(1, len(histories) + 1), k=1)[0]
            if histories else {}
        )
        sampled = {}
        for service in sorted(plan):
            device = str(plan[service])
            base = self.estimate(service, device, 0.5)
            if histories:
                shared = residual.get(
                    str(service), residual.get("__shared__", 0.0)
                )
                try:
                    shared = float(shared)
                except (TypeError, ValueError):
                    shared = 0.0
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
                    "handoff": _handoff_duration(service),
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
                if item["handoff"] is not None:
                    self._handoff_samples[service][device].append(item["handoff"])
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
            "metric": "real_execute_time_seconds",
            "deployment": copy.deepcopy(deployment or {}),
            "pairs": pairs,
            "handoff_pairs": handoffs,
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


__all__ = ("FragSpliceLatencyModel",)
