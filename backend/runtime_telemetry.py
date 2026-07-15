"""Backend-owned last-known-good runtime telemetry snapshots."""

import copy
import threading
import time

from core.lib.common import LOGGER
from core.lib.network import NetworkAPIMethod, NetworkAPIPath


class RuntimeTelemetryCache:
    """Sample Scheduler and Kubernetes telemetry with one background worker.

    The active RuntimeDirectory is converted to exact Pod name/UID references
    at bind time.  Sampling never re-reads lifecycle state, and generation
    checks prevent a response from an old install or rollout from being
    published after rebind/uninstall.
    """

    def __init__(
        self,
        request,
        runtime_metrics,
        interval_seconds=2.0,
        metrics_interval_seconds=10.0,
        scheduler_request_timeout_seconds=3.0,
        kubernetes_request_timeout_seconds=5.0,
        clock=time.monotonic,
    ):
        self._request = request
        self._runtime_metrics_request = runtime_metrics
        self._interval_seconds = max(0.1, float(interval_seconds))
        self._metrics_interval_seconds = max(
            self._interval_seconds, float(metrics_interval_seconds),
        )
        self._scheduler_request_timeout_seconds = max(
            0.1, float(scheduler_request_timeout_seconds),
        )
        self._kubernetes_request_timeout_seconds = max(
            0.1, float(kubernetes_request_timeout_seconds),
        )
        self._clock = clock
        self._lock = threading.Lock()
        self._sample_lock = threading.Lock()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread = None

        self._resource_url = None
        self._binding_key = None
        self._pod_refs = ()
        self._pod_context = {}
        self._generation = 0
        self._metrics_due_at = None

        self._resource = None
        self._scheduling_overhead = None
        self._runtime_metrics = {}
        self._scheduler_sampled_at = None
        self._runtime_metrics_sampled_at = None

    @staticmethod
    def _directory_binding(directory):
        if directory is None:
            return None, (), {}

        refs = []
        context = {}
        for unit in directory.routes:
            if unit.slot.component != "processor":
                continue
            pod_name = str(unit.pod_name or "")
            pod_uid = str(unit.pod_uid or "")
            if not pod_name or not pod_uid:
                # An active directory should always contain these identities.
                # Telemetry fails closed for a malformed route without making
                # the lifecycle transaction itself depend on Metrics Server.
                LOGGER.warning(
                    f"Skip runtime telemetry for {unit.runtime_id}: "
                    "active processor route has no exact Pod name/UID"
                )
                continue
            previous = context.get(pod_name)
            if previous is not None and previous["pod_uid"] != pod_uid:
                raise ValueError(
                    f"RuntimeDirectory contains conflicting UIDs for Pod {pod_name!r}"
                )
            refs.append({"name": pod_name, "uid": pod_uid})
            context[pod_name] = {
                "pod_uid": pod_uid,
                "runtime_id": unit.runtime_id,
                "logical_service": unit.slot.logical_service,
                "target_node": unit.slot.target_node,
            }

        refs.sort(key=lambda item: (item["name"], item["uid"]))
        binding_key = (
            str(directory.install_id),
            int(directory.revision),
            tuple(
                (
                    ref["name"],
                    ref["uid"],
                    context[ref["name"]]["runtime_id"],
                    context[ref["name"]]["logical_service"],
                    context[ref["name"]]["target_node"],
                )
                for ref in refs
            ),
        )
        return binding_key, tuple(refs), context

    @staticmethod
    def _placeholder_metrics(pod_context):
        """Represent committed routes before their first bounded K8s sample."""
        return {
            pod_name: {
                "name": pod_name,
                "uid": context["pod_uid"],
                "node": context["target_node"],
                "node_info": {},
                "phase": "",
                "ready": None,
                "pod_ip": "",
                "created_at": "",
                "resources": {},
                "usage": {},
                "runtime_id": context["runtime_id"],
                "logical_service": context["logical_service"],
            }
            for pod_name, context in pod_context.items()
        }

    def bind(self, resource_url, directory):
        """Bind one committed directory and discard another generation's data."""
        normalized_url = str(resource_url or "").strip() or None
        binding_key, pod_refs, pod_context = self._directory_binding(directory)
        if (normalized_url is None) != (binding_key is None):
            raise ValueError("resource_url and RuntimeDirectory must be bound together")

        with self._lock:
            if normalized_url == self._resource_url and binding_key == self._binding_key:
                return
            retain_scheduler_snapshot = (
                normalized_url is not None
                and normalized_url == self._resource_url
                and binding_key is not None
                and self._binding_key is not None
                and binding_key[0] == self._binding_key[0]
            )
            self._resource_url = normalized_url
            self._binding_key = binding_key
            self._pod_refs = pod_refs
            self._pod_context = pod_context
            self._generation += 1
            self._metrics_due_at = 0.0 if binding_key is not None else None
            self._runtime_metrics = self._placeholder_metrics(pod_context)
            self._runtime_metrics_sampled_at = None
            if not retain_scheduler_snapshot:
                self._resource = None
                self._scheduling_overhead = None
                self._scheduler_sampled_at = None
        self._wake_event.set()

    def unbind(self):
        self.bind(None, None)

    def start(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self._wake_event.set()
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="dayu-runtime-telemetry",
                daemon=True,
            )
            thread = self._thread
        thread.start()

    def close(self, join_timeout=1.0):
        self._stop_event.set()
        self._wake_event.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(join_timeout)))

    def snapshot(self, logical_service=""):
        """Return an isolated in-memory snapshot without network or lifecycle I/O."""
        logical_service = str(logical_service or "")
        with self._lock:
            metrics = self._runtime_metrics
            if logical_service:
                metrics = {
                    pod_name: metric
                    for pod_name, metric in metrics.items()
                    if metric.get("logical_service") == logical_service
                }
            install_id = self._binding_key[0] if self._binding_key else None
            directory_revision = self._binding_key[1] if self._binding_key else None
            return {
                "install_id": install_id,
                "directory_revision": directory_revision,
                "resource": copy.deepcopy(self._resource),
                "scheduling_overhead": copy.deepcopy(self._scheduling_overhead),
                "runtime_metrics": copy.deepcopy(metrics),
                "scheduler_sampled_at": self._scheduler_sampled_at,
                "runtime_metrics_sampled_at": self._runtime_metrics_sampled_at,
            }

    def _sample_scheduler(self, resource_url):
        scheduler_base = resource_url.rsplit(NetworkAPIPath.SCHEDULER_GET_RESOURCE, 1)[0]
        resource = None
        scheduling_overhead = None
        try:
            resource = self._request(
                resource_url,
                method=NetworkAPIMethod.SCHEDULER_GET_RESOURCE,
                timeout=self._scheduler_request_timeout_seconds,
            )
        except Exception as exc:
            LOGGER.warning(f"Failed to sample scheduler resource telemetry: {exc}")
            LOGGER.exception(exc)
        try:
            scheduling_overhead = self._request(
                f"{scheduler_base}{NetworkAPIPath.SCHEDULER_OVERHEAD}",
                method=NetworkAPIMethod.SCHEDULER_OVERHEAD,
                timeout=self._scheduler_request_timeout_seconds,
            )
        except Exception as exc:
            LOGGER.warning(f"Failed to sample scheduler overhead telemetry: {exc}")
            LOGGER.exception(exc)
        return resource, scheduling_overhead

    def _sample_runtime_metrics(self, pod_refs, pod_context):
        if not pod_refs:
            return {}
        sampled = self._runtime_metrics_request(
            [dict(ref) for ref in pod_refs],
            request_timeout_seconds=self._kubernetes_request_timeout_seconds,
        )
        if not isinstance(sampled, dict):
            return None

        result = self._placeholder_metrics(pod_context)
        for pod_name, metric in sampled.items():
            pod_name = str(pod_name)
            bound_context = pod_context.get(pod_name)
            if bound_context is None or not isinstance(metric, dict):
                continue
            # ClusterClient already joins by exact UID. Retain the UID check at
            # this publication boundary so a replacement Pod can never inherit
            # the previous incarnation's management telemetry.
            if str(metric.get("uid") or "") != bound_context["pod_uid"]:
                continue
            result[pod_name] = {
                **result[pod_name],
                **copy.deepcopy(metric),
                "runtime_id": bound_context["runtime_id"],
                "logical_service": bound_context["logical_service"],
            }
        return result

    def _sample_once(self):
        # The daemon is single-threaded; this guard also makes direct/manual
        # sample triggers single-flight instead of creating overlap.
        if not self._sample_lock.acquire(blocking=False):
            return False
        try:
            with self._lock:
                resource_url = self._resource_url
                binding_key = self._binding_key
                pod_refs = self._pod_refs
                pod_context = copy.deepcopy(self._pod_context)
                generation = self._generation
                metrics_due = (
                    self._metrics_due_at is not None
                    and self._clock() >= self._metrics_due_at
                )
            if resource_url is None or binding_key is None:
                return False

            resource, scheduling_overhead = self._sample_scheduler(resource_url)
            runtime_metrics = None
            if metrics_due:
                try:
                    runtime_metrics = self._sample_runtime_metrics(pod_refs, pod_context)
                except Exception as exc:
                    LOGGER.warning(f"Failed to sample Kubernetes runtime telemetry: {exc}")
                    LOGGER.exception(exc)

            resource_valid = isinstance(resource, dict)
            overhead_valid = scheduling_overhead is not None
            runtime_metrics_valid = isinstance(runtime_metrics, dict)
            sampled_at = time.time()
            with self._lock:
                if (
                    generation != self._generation
                    or resource_url != self._resource_url
                    or binding_key != self._binding_key
                ):
                    return False
                if metrics_due:
                    # Settle-then-schedule: success and failure both wait for
                    # the configured period, so an outage cannot create a hot
                    # retry loop against kube-apiserver or Metrics Server.
                    self._metrics_due_at = self._clock() + self._metrics_interval_seconds
                if resource_valid:
                    self._resource = copy.deepcopy(resource)
                if overhead_valid:
                    self._scheduling_overhead = copy.deepcopy(scheduling_overhead)
                if resource_valid or overhead_valid:
                    self._scheduler_sampled_at = sampled_at
                if runtime_metrics_valid:
                    self._runtime_metrics = copy.deepcopy(runtime_metrics)
                    self._runtime_metrics_sampled_at = sampled_at
            return resource_valid or overhead_valid or runtime_metrics_valid
        finally:
            self._sample_lock.release()

    def _run(self):
        while not self._stop_event.is_set():
            self._wake_event.clear()
            try:
                self._sample_once()
            except Exception as exc:
                # Preserve every last-known-good field independently and retry
                # only after the previous cycle has fully settled.
                LOGGER.warning(f"Failed to sample runtime telemetry: {exc}")
                LOGGER.exception(exc)

            with self._lock:
                is_bound = self._binding_key is not None
                metrics_due_at = self._metrics_due_at
            wait_seconds = None
            if is_bound:
                metrics_wait = max(0.0, metrics_due_at - self._clock())
                wait_seconds = min(self._interval_seconds, metrics_wait)
            self._wake_event.wait(wait_seconds)
