"""Transactional orchestration for Dayu's managed RuntimeService data plane.

The backend is the only Python process that talks to Kubernetes.  Runtime
workers receive a small immutable bootstrap document and exact per-task routes;
they never discover Pods, Services, Nodes, or ports themselves.
"""

from __future__ import annotations

import copy
import json
import math
import os
import threading
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from core.lib.common import LOGGER, TaskConstant
from core.lib.network import (
    HTTPClientError,
    NetworkAPIMethod,
    NetworkAPIPath,
    http_request_or_raise,
)
from core.lib.scheduling.source_selection import (
    ALL_EDGE_NODES,
    SOURCE_CANDIDATE_NODES_FIELD,
    SOURCE_SELECTION_SCOPE_FIELD,
    selection_scope_from_template,
)

from cluster_client import ClusterClient, kubernetes_resource_ref
from runtime_model import (
    RuntimeCleanupResource,
    RuntimeCleanupRef,
    RuntimeDirectory,
    RuntimeEndpoint,
    RuntimeRetirement,
    RuntimeSession,
    RuntimeSlot,
    RuntimeUninstallProgress,
    RuntimeUnit,
)
from runtime_service_client import RuntimeServiceCancelled, RuntimeServiceClient
from runtime_session_store import RuntimeSessionConflict, RuntimeSessionStore, StoredRuntimeSession


class RuntimeOrchestrationError(RuntimeError):
    """A managed-runtime transaction could not be completed safely."""


class RuntimePreflightError(RuntimeOrchestrationError):
    """Cluster state cannot satisfy the managed RuntimeService contract."""


class RuntimePublicationError(RuntimeOrchestrationError):
    """The scheduler could not atomically publish an exact directory."""


class RuntimeOperationCancelled(RuntimeOrchestrationError):
    """A managed-runtime operation yielded to lifecycle cancellation."""


class RuntimeRetirementPending(RuntimeOrchestrationError):
    """A newer rollout is deferred while the previous revision retires."""


class SchedulerRequestError(RuntimeOrchestrationError):
    """A Scheduler management request failed with actionable context."""

    def __init__(self, endpoint: str, error: HTTPClientError):
        self.endpoint = str(endpoint or "")
        self.status_code = error.status_code
        self.detail = error.detail
        if self.status_code is None:
            message = f"Scheduler {self.endpoint} request failed"
        else:
            message = (
                f"Scheduler {self.endpoint} rejected the request "
                f"(HTTP {self.status_code})"
            )
        if self.detail:
            message = f"{message}: {self.detail}"
        super().__init__(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_code(exc: Exception) -> Optional[int]:
    value = getattr(exc, "status", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _endpoint_url(endpoint: RuntimeEndpoint, path: str = "") -> str:
    base = f"http://{endpoint.url_authority}"
    return f"{base}/{str(path).lstrip('/')}" if path else base


def _source_id(source_info: Mapping[str, Any]) -> str:
    source = source_info.get("source") or {}
    value = source.get("id")
    if value is None or str(value) == "":
        raise RuntimeOrchestrationError("every source requires a non-empty source.id")
    return str(value)


class RuntimeOrchestrator:
    """Own install, rollout, directory publication and bounded retirement.

    Kubernetes clients are created lazily and share one ``ApiClient``.  A
    process lock plus ConfigMap compare-and-swap prevents two backend requests
    from publishing different runtime directories for the same installation.
    """

    INSTALL_LABEL = "dayu.io/install-id"
    MANAGED_LABEL_SELECTOR = "app.kubernetes.io/managed-by=dayu-backend"
    SUPPORTED_JETPACK_MAJORS = frozenset({4, 5, 6})
    _PUBLICATION_PHASES = frozenset({"publishing", "publishing-rollout"})
    _INITIAL_ACTIVATION_PHASES = frozenset({
        "activating-scheduler", "activating-runtime",
    })
    _RECOVERY_PHASES = (
        _INITIAL_ACTIVATION_PHASES
        | _PUBLICATION_PHASES
        | frozenset({"activating-rollout"})
    )

    def __init__(
        self,
        template_helper,
        namespace: str,
        cluster_client: Optional[ClusterClient] = None,
        runtime_client: Optional[RuntimeServiceClient] = None,
        session_store: Optional[RuntimeSessionStore] = None,
        request=http_request_or_raise,
        clock=time.monotonic,
        wall_clock=time.time,
    ):
        self.template_helper = template_helper
        self.namespace = str(namespace or "").strip()
        if not self.namespace:
            raise ValueError("namespace must be non-empty")
        self._cluster = cluster_client
        self._runtime = runtime_client
        self._sessions = session_store
        self._request = request
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.RLock()
        # ``None`` is a valid, durable snapshot (there is no active session),
        # so it cannot also mean "not loaded". Keep that state explicit: the
        # management read path performs at most one lazy ConfigMap read and is
        # then a pure in-process lookup until a lifecycle transaction reloads.
        self._snapshot_load_lock = threading.Lock()
        self._snapshot_loaded = False
        self._stored: Optional[StoredRuntimeSession] = None
        self._inventory_lock = threading.RLock()
        self._inventory_cache: Dict[str, Dict[str, Any]] = {}
        self._inventory_cached_at: Optional[float] = None

        base_info = self.template_helper.load_base_info()
        default_cloud_processor_backup = base_info.get(
            "default-cloud-processor-backup", False,
        )
        if not isinstance(default_cloud_processor_backup, bool):
            raise ValueError("default-cloud-processor-backup must be a boolean")
        self.default_cloud_processor_backup = default_cloud_processor_backup
        runtime_config = base_info.get("runtime") or {}
        self.activation_timeout = float(runtime_config.get("activation-timeout-seconds", 300))
        self.operation_timeout = float(runtime_config.get("operation-timeout-seconds", 900))
        self.scheduler_request_timeout = float(
            runtime_config.get("scheduler-request-timeout-seconds", 30)
        )
        self.retirement_grace = float(
            runtime_config.get("retirement-grace-seconds", 180)
        )
        self.lease_ttl = float(runtime_config.get("lease-ttl-seconds", 3600))
        self.inventory_ttl = max(1.0, float(runtime_config.get("inventory-ttl-seconds", 30)))
        timeouts = (
            self.activation_timeout,
            self.operation_timeout,
            self.scheduler_request_timeout,
            self.retirement_grace,
            self.lease_ttl,
        )
        if any(not math.isfinite(value) or value <= 0 for value in timeouts):
            raise ValueError(
                "runtime activation, operation, scheduler request, retirement, and lease "
                "timeouts must be positive"
            )

    def _ensure_clients(self) -> None:
        request_timeout = max(1.0, min(30.0, self.operation_timeout))
        if self._cluster is None:
            self._cluster = ClusterClient(
                self.namespace,
                request_timeout_seconds=min(10.0, request_timeout),
            )
        if self._runtime is None:
            self._runtime = RuntimeServiceClient(
                self.namespace, api=self._cluster.custom,
                request_timeout_seconds=request_timeout,
            )
        if self._sessions is None:
            self._sessions = RuntimeSessionStore(
                self.namespace, api=self._cluster.core,
                request_timeout_seconds=request_timeout,
            )

    @property
    def cluster(self) -> ClusterClient:
        self._ensure_clients()
        return self._cluster

    @property
    def runtime(self) -> RuntimeServiceClient:
        self._ensure_clients()
        return self._runtime

    @property
    def sessions(self) -> RuntimeSessionStore:
        self._ensure_clients()
        return self._sessions

    def _load(self) -> Optional[StoredRuntimeSession]:
        if self._snapshot_loaded:
            return self._stored
        # Serialize the one lazy load with transaction-boundary reloads. The
        # fast path above intentionally takes no lifecycle lock, so
        # /install_state never waits behind Scheduler/Kubernetes operations.
        with self._snapshot_load_lock:
            if not self._snapshot_loaded:
                self._stored = self.sessions.load()
                self._snapshot_loaded = True
        return self._stored

    def _reload_for_transaction(self) -> Optional[StoredRuntimeSession]:
        """Read the CAS record at a lifecycle transaction boundary."""
        with self._snapshot_load_lock:
            self._stored = self.sessions.load()
            self._snapshot_loaded = True
            return self._stored

    def _save(self, session: RuntimeSession) -> RuntimeSession:
        with self._snapshot_load_lock:
            expected = self._stored.resource_version if self._stored is not None else None
            try:
                self._stored = self.sessions.compare_and_swap(session, expected)
                self._snapshot_loaded = True
            except RuntimeSessionConflict:
                self._stored = self.sessions.load()
                self._snapshot_loaded = True
                raise
            except Exception as write_error:
                # A timeout/EOF can arrive after the API server committed the
                # CAS. Calibrate that one uncertain outcome with a bounded GET
                # and accept only an exact desired value; never swallow a
                # genuinely different writer's state.
                try:
                    observed = self.sessions.load()
                except Exception as read_error:
                    # The next management read must not trust the stale local
                    # snapshot after two ambiguous control-plane failures.
                    self._snapshot_loaded = False
                    raise write_error from read_error
                self._stored = observed
                self._snapshot_loaded = True
                if observed is not None and observed.session == session:
                    return observed.session
                raise
            return self._stored.session

    @classmethod
    def requires_recovery(cls, session: Optional[RuntimeSession]) -> bool:
        return session is not None and session.phase in cls._RECOVERY_PHASES

    def _mark_snapshot_deleted(self) -> None:
        """Publish the durable absence of a session as one snapshot update."""
        with self._snapshot_load_lock:
            self._stored = None
            self._snapshot_loaded = True

    def current_session(self) -> Optional[RuntimeSession]:
        """Return the process-owned CAS snapshot without a caller refresh switch."""
        stored = self._load()
        return stored.session if stored else None

    def active_directory(self) -> Optional[RuntimeDirectory]:
        """Return the committed immutable directory without lifecycle I/O.

        Publication recovery belongs to :meth:`recover` and transaction
        boundaries. A management read must never wait behind a long old-route
        retirement after the new directory has already been committed.
        """
        session = self.current_session()
        if session is None or session.phase != "active" or session.active_directory_revision < 1:
            return None
        return session.directory

    def _remaining_timeout(self, deadline: float, cap: Optional[float] = None) -> float:
        remaining = float(deadline) - self._clock()
        if remaining <= 0:
            raise RuntimeOrchestrationError("managed runtime operation exceeded its deadline")
        return min(remaining, float(cap)) if cap is not None else remaining

    @staticmethod
    def _raise_if_cancelled(cancel_event) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeOperationCancelled(
                "managed runtime operation was cancelled by a lifecycle operation"
            )

    def _refresh_inventory(
        self, request_timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Dict[str, Any]]:
        with self._inventory_lock:
            if request_timeout_seconds is None:
                inventory = self.cluster.node_inventory()
            else:
                inventory = self.cluster.node_inventory(
                    request_timeout_seconds=request_timeout_seconds,
                )
            self._inventory_cache = copy.deepcopy(inventory)
            self._inventory_cached_at = self._clock()
            return inventory

    def node_inventory(
        self, request_timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Return one backend-owned topology snapshot with no caller refresh API."""
        with self._inventory_lock:
            if (
                self._inventory_cached_at is None
                or self._clock() - self._inventory_cached_at >= self.inventory_ttl
            ):
                self._refresh_inventory(
                    request_timeout_seconds=request_timeout_seconds,
                )
            return copy.deepcopy(self._inventory_cache)

    @staticmethod
    def _cloud_node(inventory: Mapping[str, Mapping[str, Any]]) -> str:
        configured = str(os.getenv("CLOUD_NODE_NAME") or "").strip()
        if configured:
            record = inventory.get(configured)
            if record is None:
                raise RuntimePreflightError(f"configured cloud node {configured!r} does not exist")
            if not record.get("ready"):
                raise RuntimePreflightError(f"configured cloud node {configured!r} is not Ready")
            return configured
        candidates = sorted(
            name for name, record in inventory.items()
            if record.get("role") == "cloud" and record.get("ready")
        )
        if len(candidates) != 1:
            raise RuntimePreflightError(
                "exactly one Ready cloud node is required; set CLOUD_NODE_NAME when the cluster has "
                f"multiple control-plane nodes (candidates={candidates})"
            )
        return candidates[0]

    @staticmethod
    def _selected_nodes(source_deploy: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
        return tuple(sorted({
            str(node)
            for source_info in source_deploy or ()
            for node in (source_info.get("node_set") or ())
            if str(node)
        }))

    @staticmethod
    def _source_candidate_nodes(
        source_deploy: Sequence[Mapping[str, Any]],
    ) -> Tuple[str, ...]:
        return tuple(sorted({
            str(node)
            for source_info in source_deploy or ()
            for node in (source_info.get(SOURCE_CANDIDATE_NODES_FIELD) or ())
            if str(node)
        }))

    @staticmethod
    def _validate_inventory_targets(
        inventory: Mapping[str, Mapping[str, Any]], target_nodes: Iterable[str],
    ) -> Tuple[str, ...]:
        targets = tuple(sorted({str(node) for node in target_nodes if str(node)}))
        missing = [node for node in targets if node not in inventory]
        not_ready = [node for node in targets if node in inventory and not inventory[node].get("ready")]
        if missing or not_ready:
            raise RuntimePreflightError(
                f"runtime target validation failed: missing={missing}, not_ready={not_ready}"
            )
        return targets

    @staticmethod
    def _ensure_managed_agent_targets(
        report: Mapping[str, Any], target_nodes: Iterable[str],
    ) -> None:
        targets = {str(node) for node in target_nodes if str(node)}
        details = []
        agents = report.get("agents") or {}
        for name in ("sedna_lc", "edgemesh_agent"):
            state = agents.get(name) or {}
            missing = sorted(targets & set(state.get("missing_nodes") or ()))
            not_ready = sorted(targets & set(state.get("not_ready_nodes") or ()))
            if missing or not_ready:
                details.append(f"{name}(missing={missing}, not_ready={not_ready})")
        if len(agents) < 2:
            details.append("managed-agent report is incomplete")
        if details:
            raise RuntimePreflightError(
                "managed RuntimeService prerequisites are not Ready on every target node: "
                + "; ".join(details)
            )

    @staticmethod
    def _managed_agent_nodes(report: Mapping[str, Any]) -> set:
        agents = report.get("agents") or {}
        ready_sets = []
        for name in ("sedna_lc", "edgemesh_agent"):
            state = agents.get(name)
            if not isinstance(state, Mapping):
                raise RuntimePreflightError(f"managed-agent report omitted {name!r}")
            ready_sets.append({str(node) for node in (state.get("ready_nodes") or ()) if str(node)})
        return set.intersection(*ready_sets) if ready_sets else set()

    def _authorize_source_candidates(
        self,
        inventory: Mapping[str, Mapping[str, Any]],
        source_deploy: Sequence[Mapping[str, Any]],
        scope: str,
        cloud_node: str,
    ) -> list:
        processor_candidates = set(self._selected_nodes(source_deploy))
        if not processor_candidates:
            raise RuntimePreflightError("at least one processor candidate node is required")
        required_targets = processor_candidates | {cloud_node}
        self._validate_inventory_targets(inventory, required_targets)

        non_edge = sorted(
            node for node in processor_candidates
            if (inventory.get(node) or {}).get("role") != "edge"
        )
        if non_edge:
            raise RuntimePreflightError(
                f"generator/processor edge candidates must be edge nodes: {non_edge}"
            )

        ready_edges = {
            name for name, record in inventory.items()
            if record.get("role") == "edge" and record.get("ready")
        }
        probe_targets = required_targets | (ready_edges if scope == ALL_EDGE_NODES else set())
        report = self.cluster.validate_managed_agents(probe_targets)
        self._ensure_managed_agent_targets(report, required_targets)
        managed_nodes = self._managed_agent_nodes(report)

        shared_source_candidates = sorted(ready_edges & managed_nodes)
        authorized = copy.deepcopy(list(source_deploy))
        for source_info in authorized:
            if scope == ALL_EDGE_NODES:
                candidates = list(shared_source_candidates)
            else:
                candidates = list(source_info.get("node_set") or ())
            if not candidates:
                raise RuntimePreflightError(
                    f"source {_source_id(source_info)!r} has no Ready managed-agent-covered source candidates"
                )
            source_info[SOURCE_CANDIDATE_NODES_FIELD] = candidates
            source_info[SOURCE_SELECTION_SCOPE_FIELD] = scope
        return authorized

    def _preflight_nodes(
        self,
        inventory: Mapping[str, Mapping[str, Any]],
        target_nodes: Iterable[str],
        validate_agents: bool = True,
    ) -> None:
        targets = self._validate_inventory_targets(inventory, target_nodes)
        if not validate_agents:
            return
        report = self.cluster.validate_managed_agents(targets)
        self._ensure_managed_agent_targets(report, targets)

    @staticmethod
    def _compact_inventory(
        inventory: Mapping[str, Mapping[str, Any]], nodes: Iterable[str],
    ) -> Dict[str, Dict[str, Any]]:
        result = {}
        for name in sorted(set(nodes)):
            record = inventory[name]
            result[name] = {
                "role": record.get("role", "worker"),
                "address": record.get("address", ""),
                "ready": bool(record.get("ready")),
            }
        return result

    def _support_endpoint(self, component: str, port: int, target_node: str) -> Dict[str, Any]:
        return {
            "component": component,
            "target_node": target_node,
            "runtime_id": component,
            "fqdn": f"{component}-cloud.{self.namespace}.svc.cluster.local",
            "port": int(port),
        }

    def _bootstrap(
        self,
        install_id: str,
        local_node: str,
        cloud_node: str,
        inventory: Mapping[str, Mapping[str, Any]],
        selected_nodes: Iterable[str],
        endpoint_units: Iterable[RuntimeUnit],
        directory_revision: int = 0,
    ) -> str:
        endpoints = []
        for unit in endpoint_units:
            if unit.endpoint is None:
                continue
            endpoints.append({
                **unit.slot.to_dict(),
                "runtime_id": unit.runtime_id,
                "fqdn": unit.endpoint.dns_name,
                "port": unit.endpoint.port,
                "deployment_revision": unit.runtime_revision,
                "runtime_service_uid": unit.endpoint.runtime_service_uid,
                "service_uid": unit.endpoint.service_uid,
                "endpoint_pod_uid": unit.endpoint.pod_uid,
                "install_id": install_id,
            })
        endpoints.extend((
            self._support_endpoint("backend", 8000, cloud_node),
            self._support_endpoint("redis", 6379, cloud_node),
            {
                "component": "datasource",
                "target_node": str(self.template_helper.load_base_info().get("datasource", {}).get("node") or ""),
                "runtime_id": "datasource",
                "fqdn": f"datasource-edge.{self.namespace}.svc.cluster.local",
                "port": 8000,
            },
        ))
        value = {
            "mode": "runtime-service",
            "namespace": self.namespace,
            "install_id": install_id,
            "local_node": local_node,
            "cloud_node": cloud_node,
            "runtime_directory_revision": int(directory_revision),
            "lease_ttl_seconds": self.lease_ttl,
            "nodes": self._compact_inventory(inventory, selected_nodes),
            "endpoints": endpoints,
        }
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _ensure_created(
        self,
        manifest: Mapping[str, Any],
        request_timeout_seconds: Optional[float] = None,
    ) -> Mapping[str, Any]:
        try:
            return self.runtime.create(
                manifest,
                request_timeout_seconds=request_timeout_seconds,
            )
        except Exception as exc:
            if _status_code(exc) != 409:
                raise
        name = str((manifest.get("metadata") or {}).get("name") or "")
        existing = self.runtime.get(
            name,
            request_timeout_seconds=request_timeout_seconds,
        )
        if (existing.get("spec") or {}) != (manifest.get("spec") or {}):
            raise RuntimeOrchestrationError(
                f"RuntimeService {name!r} already exists with a different immutable spec"
            )
        return existing

    def _activate(
        self,
        rendered: Sequence[Any],
        timeout_seconds: Optional[float] = None,
        cancel_event=None,
    ) -> Tuple[RuntimeUnit, ...]:
        self._raise_if_cancelled(cancel_event)
        if not rendered:
            return ()
        timeout_seconds = min(
            self.activation_timeout,
            float(timeout_seconds) if timeout_seconds is not None else self.activation_timeout,
        )
        if timeout_seconds <= 0:
            raise RuntimeOrchestrationError("RuntimeService activation deadline was exhausted")
        deadline = self._clock() + timeout_seconds
        install_ids = {
            str(((item.manifest.get("metadata") or {}).get("labels") or {}).get(
                self.INSTALL_LABEL,
            ) or "")
            for item in rendered
        }
        if len(install_ids) != 1 or not next(iter(install_ids)):
            raise RuntimeOrchestrationError(
                "one activation batch must contain exactly one install identity"
            )
        install_id = next(iter(install_ids))
        for item in rendered:
            self._raise_if_cancelled(cancel_event)
            try:
                self._ensure_created(
                    item.manifest,
                    request_timeout_seconds=self._remaining_timeout(deadline),
                )
            except Exception:
                # A synchronous Kubernetes request cannot be pre-empted, but
                # cancellation wins over a transport error observed after the
                # lifecycle token was set.
                self._raise_if_cancelled(cancel_event)
                raise
            self._raise_if_cancelled(cancel_event)
        expectations = {item.unit.runtime_id: item.unit for item in rendered}
        try:
            observed = self.runtime.wait_for_conditions(
                expectations,
                condition_types=("Ready", "Activated"),
                timeout_seconds=self._remaining_timeout(deadline),
                label_selector=(
                    f"{self.MANAGED_LABEL_SELECTOR},"
                    f"{self.INSTALL_LABEL}={install_id}"
                ),
                cancel_event=cancel_event,
            )
        except RuntimeServiceCancelled:
            self._raise_if_cancelled(cancel_event)
            raise
        self._raise_if_cancelled(cancel_event)
        return tuple(
            self.runtime.bind_observed_unit(item.unit, observed[item.unit.runtime_id])
            for item in rendered
        )

    def _scheduler_call(
        self,
        scheduler: RuntimeUnit,
        path: str,
        method: str,
        payload: Any = None,
        params: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        cancel_event=None,
    ):
        self._raise_if_cancelled(cancel_event)
        if scheduler.endpoint is None:
            raise RuntimePublicationError("scheduler RuntimeService has no endpoint")
        kwargs = {}
        if payload is not None:
            kwargs["data"] = {"data": json.dumps(payload, ensure_ascii=False)}
        if params:
            kwargs["params"] = dict(params)
        total_timeout = min(
            float(timeout) if timeout is not None else self.operation_timeout,
            self.scheduler_request_timeout,
        )
        if total_timeout <= 0:
            raise RuntimeOrchestrationError("scheduler request deadline was exhausted")
        # ``http_request.timeout`` applies to each attempt. Divide the caller's
        # total budget across retries so a three-attempt request cannot silently
        # consume three operation timeouts (plus backoff).
        attempts = 3 if total_timeout >= 1.0 else 1
        backoff_budget = 0.6 if attempts == 3 else 0.0
        per_attempt_timeout = max(0.001, (total_timeout - backoff_budget) / attempts)
        try:
            response = self._request(
                _endpoint_url(scheduler.endpoint, path),
                method=method,
                timeout=per_attempt_timeout,
                retry=attempts,
                retry_interval=0.2,
                retry_backoff=2,
                cancel_event=cancel_event,
                **kwargs,
            )
        except HTTPClientError as exc:
            self._raise_if_cancelled(cancel_event)
            raise SchedulerRequestError(path, exc) from exc
        except Exception:
            self._raise_if_cancelled(cancel_event)
            raise
        self._raise_if_cancelled(cancel_event)
        return response

    def _decision(
        self,
        scheduler: RuntimeUnit,
        path: str,
        method: str,
        source_deploy: Sequence[Mapping[str, Any]],
        timeout: Optional[float] = None,
        cancel_event=None,
    ) -> Mapping[str, Any]:
        response = self._scheduler_call(
            scheduler,
            path,
            method,
            source_deploy,
            timeout=timeout,
            cancel_event=cancel_event,
        )
        if not isinstance(response, Mapping) or not isinstance(response.get("plan"), Mapping):
            raise RuntimeOrchestrationError(f"scheduler {path} returned no valid plan")
        return response["plan"]

    @staticmethod
    def _source_selection(
        raw_plan: Mapping[str, Any], source_deploy: Sequence[Mapping[str, Any]],
    ) -> Dict[str, str]:
        selected = {}
        for source_info in source_deploy:
            source_id = _source_id(source_info)
            value = raw_plan.get(source_id)
            if value is None:
                try:
                    value = raw_plan.get(int(source_id))
                except ValueError:
                    pass
            if not value:
                raise RuntimeOrchestrationError(
                    f"source selection plan omitted source {source_id!r}; implicit placement is forbidden"
                )
            value = str(value)
            candidates = {
                str(node)
                for node in (source_info.get(SOURCE_CANDIDATE_NODES_FIELD) or ())
            }
            if not candidates:
                raise RuntimeOrchestrationError(
                    f"source {_source_id(source_info)!r} has no authorized source candidate set"
                )
            if value not in candidates:
                raise RuntimeOrchestrationError(
                    f"source {source_id!r} selected unexpected node {value!r}; candidates={sorted(candidates)}"
                )
            selected[source_id] = value
        return selected

    @staticmethod
    def _service_names(source_deploy: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
        names = {
            str(service)
            for source_info in source_deploy
            for service in (source_info.get("dag") or {})
            if str(service) not in {TaskConstant.START.value, TaskConstant.END.value, "start", "end"}
        }
        return tuple(sorted(names))

    def _deployment(
        self,
        raw_plan: Mapping[str, Any],
        source_deploy: Sequence[Mapping[str, Any]],
        cloud_node: str,
    ) -> Dict[str, Tuple[str, ...]]:
        services = set(self._service_names(source_deploy))
        allowed_nodes = set(self._selected_nodes(source_deploy)) | {cloud_node}
        normalized: Dict[str, set] = {service: set() for service in services}
        for raw_service, raw_nodes in raw_plan.items():
            service = str(raw_service)
            if service not in services:
                raise RuntimeOrchestrationError(
                    f"deployment selected unknown logical service {service!r}"
                )
            if not isinstance(raw_nodes, list):
                raise RuntimeOrchestrationError(
                    f"deployment for service {service!r} must be a JSON list of node names"
                )
            for raw_node in raw_nodes:
                node = str(raw_node)
                if node not in allowed_nodes:
                    raise RuntimeOrchestrationError(
                        f"deployment selected unexpected node {node!r} for service {service!r}"
                    )
                normalized[service].add(node)
        missing = sorted(service for service, nodes in normalized.items() if not nodes)
        if missing:
            raise RuntimeOrchestrationError(
                f"deployment plan omitted services {missing}; every service requires an explicit "
                "policy placement"
            )
        # Cloud backup is an operational replica composed by Backend after the
        # Scheduler plan has passed the complete service/node contract. It must
        # never repair an incomplete or otherwise invalid policy result.
        if self.default_cloud_processor_backup:
            for nodes in normalized.values():
                nodes.add(cloud_node)
        return {service: tuple(sorted(nodes)) for service, nodes in sorted(normalized.items())}

    @staticmethod
    def _logical_templates(template_helper, policy, source_deploy):
        templates = template_helper.load_policy_apply_yaml(policy)
        normalized_sources, service_dict = template_helper.normalize_source_deploy(source_deploy)
        templates["processor"] = template_helper.load_application_apply_yaml(service_dict)
        return templates, normalized_sources

    def _specialize_template_for_node(
        self, logical_template: Mapping[str, Any], node: str, inventory,
    ):
        """Select a node-compatible image without performing another cluster read.

        Monitor and processor images share the same JetPack build matrix.  The
        inventory used for lifecycle planning already contains the node labels,
        so image selection must happen here before the RuntimeService is
        rendered instead of being rediscovered by a worker.
        """
        template = copy.deepcopy(logical_template)
        if inventory[node].get("role") != "edge":
            return template, {}
        labels = inventory[node].get("labels") or {}
        raw_major = labels.get("jetson.nvidia.com/jetpack.major")
        if raw_major is None:
            return template, {}
        try:
            major = int(raw_major)
        except (TypeError, ValueError):
            raise RuntimeOrchestrationError(
                f"node {node!r} has invalid JetPack major label {raw_major!r}"
            )
        if major not in self.SUPPORTED_JETPACK_MAJORS:
            raise RuntimeOrchestrationError(
                f"node {node!r} requires unsupported JetPack major {major}; "
                f"published image variants are {sorted(self.SUPPORTED_JETPACK_MAJORS)}"
            )
        container = template.setdefault("pod-template", {})
        image = container.get("image")
        if image:
            full_image = self.template_helper.process_image(image)
            container["image"] = self.template_helper.specify_jetpack_image(full_image, major)
        return template, {"JETPACK": major}

    def _render_initial(
        self,
        renderer,
        templates,
        source_deploy,
        source_selection,
        deployment,
        inventory,
        cloud_node,
        revision,
        scheduler_unit,
    ):
        processor_nodes = {node for nodes in deployment.values() for node in nodes}
        source_nodes = set(source_selection.values())
        # Controller/monitor routes are topology infrastructure, not placement
        # by-products. A later scheduler decision may move a processor to any
        # immutable processor candidate, so every such node must be routable in
        # the initial atomic directory even when revision 1 does not use it.
        candidate_nodes = set(self._selected_nodes(source_deploy))
        runtime_nodes = sorted(processor_nodes | source_nodes | candidate_nodes | {cloud_node})

        # Distributor endpoint is deterministic before creation and becomes
        # identity-bound only after the exact Ready/Activated observation.
        distributor_preview = renderer.render(
            templates["distributor"],
            RuntimeSlot("distributor", cloud_node, "cloud"), revision,
        ).unit
        cloud_monitor_preview = renderer.render(
            templates["monitor"],
            RuntimeSlot("monitor", cloud_node, "cloud"), revision,
        ).unit
        static_units = (scheduler_unit, distributor_preview, cloud_monitor_preview)

        rendered = [renderer.render(
            templates["distributor"],
            RuntimeSlot("distributor", cloud_node, "cloud"), revision,
            extra_env={
                "DAYU_RUNTIME_BOOTSTRAP": self._bootstrap(
                    renderer.install_id, cloud_node, cloud_node, inventory, runtime_nodes,
                    static_units,
                ),
            },
        )]
        for node in runtime_nodes:
            position = "cloud" if node == cloud_node else "edge"
            bootstrap = self._bootstrap(
                renderer.install_id, node, cloud_node, inventory, runtime_nodes, static_units,
            )
            rendered.append(renderer.render(
                templates["controller"], RuntimeSlot("controller", node, position), revision,
                extra_env={"DAYU_RUNTIME_BOOTSTRAP": bootstrap},
            ))
            monitor_template, device_env = self._specialize_template_for_node(
                templates["monitor"], node, inventory,
            )
            rendered.append(renderer.render(
                monitor_template, RuntimeSlot("monitor", node, position), revision,
                extra_env={"DAYU_RUNTIME_BOOTSTRAP": bootstrap, **device_env},
            ))

        enriched_sources = copy.deepcopy(source_deploy)
        for source_info in enriched_sources:
            selected = source_selection[_source_id(source_info)]
            source_info["source_device"] = selected
            source_info.setdefault("source", {})["source_device"] = selected
        rendered.extend(renderer.render_generator_sources(
            templates["generator"], enriched_sources, revision,
            selected_nodes=source_selection,
            common_env={
                # The renderer overwrites local_node for each generator through
                # NODE_NAME; RuntimeContext uses bootstrap.local_node first, so
                # generators receive a per-source bootstrap below.
                "CLOUD_NODE_NAME": cloud_node,
            },
        ))
        # Generator rendering is source-oriented; replace its common bootstrap
        # with the correct selected node after rendering.
        for index, item in enumerate(rendered):
            if item.unit.slot.component != "generator":
                continue
            node = item.unit.slot.target_node
            rerendered = renderer.render(
                templates["generator"], item.unit.slot, revision,
                extra_env={
                    **{
                        env["name"]: env.get("value", "")
                        for env in item.manifest["spec"]["podTemplate"]["spec"]["containers"][0].get("env", [])
                        if env.get("name") not in {"DAYU_RUNTIME_BOOTSTRAP"}
                    },
                    "DAYU_RUNTIME_BOOTSTRAP": self._bootstrap(
                        renderer.install_id, node, cloud_node, inventory, runtime_nodes, static_units,
                    ),
                },
            )
            rendered[index] = rerendered

        by_service = {
            str(info["service_name"]): info
            for info in templates["processor"].values()
        }
        for service, nodes in deployment.items():
            service_info = by_service.get(service)
            if service_info is None:
                raise RuntimeOrchestrationError(f"no processor template exists for service {service!r}")
            for node in nodes:
                position = "cloud" if node == cloud_node else "edge"
                logical_template, device_env = self._specialize_template_for_node(
                    service_info["service"], node, inventory,
                )
                rendered.append(renderer.render(
                    logical_template,
                    RuntimeSlot("processor", node, position, logical_service=service),
                    revision,
                    extra_env={
                        "PROCESSOR_SERVICE_NAME": f"processor-{service}",
                        "DAYU_RUNTIME_BOOTSTRAP": self._bootstrap(
                            renderer.install_id, node, cloud_node, inventory, runtime_nodes, static_units,
                        ),
                        **device_env,
                    },
                ))
        return rendered, enriched_sources

    @staticmethod
    def _scheduler_unit(units: Iterable[RuntimeUnit]) -> RuntimeUnit:
        matches = [unit for unit in units if unit.slot.component == "scheduler"]
        if len(matches) != 1:
            raise RuntimeOrchestrationError(f"expected exactly one scheduler RuntimeService, found {len(matches)}")
        return matches[0]

    @staticmethod
    def _directory_matches(value: Any, directory: RuntimeDirectory) -> bool:
        if not isinstance(value, Mapping) or value.get("hash") != directory.content_hash:
            return False
        try:
            observed = RuntimeDirectory.from_dict(value)
        except (TypeError, ValueError):
            return False
        return (
            observed.install_id == directory.install_id
            and observed.revision == directory.revision
            and observed.content_hash == directory.content_hash
        )

    def _publication_readback(
        self,
        scheduler: RuntimeUnit,
        directory: RuntimeDirectory,
        deadline: Optional[float] = None,
        cancel_event=None,
    ) -> bool:
        self._raise_if_cancelled(cancel_event)
        timeout = self._remaining_timeout(deadline, self.operation_timeout) if deadline else None
        readback = self._scheduler_call(
            scheduler,
            NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY,
            NetworkAPIMethod.SCHEDULER_GET_RUNTIME_DIRECTORY,
            timeout=timeout,
            cancel_event=cancel_event,
        )
        self._raise_if_cancelled(cancel_event)
        return self._directory_matches(readback, directory)

    def _publish_initial(
        self,
        scheduler: RuntimeUnit,
        directory: RuntimeDirectory,
        deadline: Optional[float] = None,
        cancel_event=None,
    ) -> None:
        payload = {"expected_revision": 0, "directory": directory.to_dict()}
        publication_error = None
        try:
            self._raise_if_cancelled(cancel_event)
            timeout = self._remaining_timeout(deadline, self.operation_timeout) if deadline else None
            response = self._scheduler_call(
                scheduler,
                NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY,
                NetworkAPIMethod.SCHEDULER_PUT_RUNTIME_DIRECTORY,
                payload,
                timeout=timeout,
                cancel_event=cancel_event,
            )
            if isinstance(response, Mapping) and response.get("hash") == directory.content_hash:
                self._raise_if_cancelled(cancel_event)
                return
        except RuntimeOperationCancelled:
            raise
        except Exception as exc:
            # A transport failure may occur after Scheduler committed the CAS.
            publication_error = exc
        try:
            if self._publication_readback(
                scheduler,
                directory,
                deadline=deadline,
                cancel_event=cancel_event,
            ):
                return
        except RuntimeOperationCancelled:
            raise
        except Exception as exc:
            publication_error = publication_error or exc
        raise RuntimePublicationError(
            "initial RuntimeDirectory publication was not durably observable"
        ) from publication_error

    def _publish_rollout(
        self,
        scheduler: RuntimeUnit,
        base_revision: int,
        directory: RuntimeDirectory,
        deadline: Optional[float] = None,
        cancel_event=None,
    ) -> Mapping[str, Any]:
        proposal_id = str(uuid.uuid4())
        publication_error = None
        try:
            self._raise_if_cancelled(cancel_event)
            timeout = self._remaining_timeout(deadline, self.operation_timeout) if deadline else None
            proposal = self._scheduler_call(
                scheduler,
                NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY_PROPOSALS,
                NetworkAPIMethod.SCHEDULER_PROPOSE_RUNTIME_DIRECTORY,
                {
                    "proposal_id": proposal_id,
                    "base_revision": int(base_revision),
                    "directory": directory.to_dict(),
                    "ttl_seconds": max(60, self.operation_timeout),
                },
                timeout=timeout,
                cancel_event=cancel_event,
            )
            if not isinstance(proposal, Mapping) or proposal.get("proposal_id") != proposal_id:
                raise RuntimePublicationError(
                    "scheduler did not persist the RuntimeDirectory proposal"
                )
            self._raise_if_cancelled(cancel_event)
            timeout = self._remaining_timeout(deadline, self.operation_timeout) if deadline else None
            response = self._scheduler_call(
                scheduler,
                NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY_PROPOSAL_COMMIT.format(proposal_id=proposal_id),
                NetworkAPIMethod.SCHEDULER_COMMIT_RUNTIME_DIRECTORY,
                {
                    "expected_revision": int(base_revision),
                    "retirement_grace_seconds": self.retirement_grace,
                },
                timeout=timeout,
                cancel_event=cancel_event,
            )
            if isinstance(response, Mapping) and response.get("hash") == directory.content_hash:
                self._raise_if_cancelled(cancel_event)
                return self._validate_retirement_status(
                    response.get("retirement"),
                    base_revision,
                )
        except RuntimeOperationCancelled:
            raise
        except Exception as exc:
            # Proposal/commit are idempotently recoverable through the exact
            # directory hash. This covers a process or transport failure after
            # Scheduler's commit but before backend session CAS.
            publication_error = exc
        try:
            if self._publication_readback(
                scheduler,
                directory,
                deadline=deadline,
                cancel_event=cancel_event,
            ):
                return self._retirement_status(
                    scheduler,
                    base_revision,
                    operation_deadline=deadline,
                    cancel_event=cancel_event,
                )
        except RuntimeOperationCancelled:
            raise
        except Exception as exc:
            publication_error = publication_error or exc
        raise RuntimePublicationError(
            "RuntimeDirectory commit was not durably observable"
        ) from publication_error

    @staticmethod
    def _publication_candidate(session: RuntimeSession) -> RuntimeDirectory:
        if session.phase == "publishing":
            routes = session.pending
            revision = 1
        elif session.phase == "publishing-rollout":
            retired_keys = {
                unit.logical_key for unit in (
                    session.retirement.units if session.retirement else ()
                )
            }
            routes = tuple(
                unit for unit in session.active if unit.logical_key not in retired_keys
            ) + tuple(session.pending)
            revision = session.active_directory_revision + 1
        else:
            raise RuntimeOrchestrationError(
                f"session phase {session.phase!r} is not recoverable publication state"
            )
        if not routes:
            raise RuntimeOrchestrationError("recoverable publication contains no runtime routes")
        incomplete = sorted(
            unit.runtime_id for unit in routes
            if not unit.runtime_service_uid or not unit.pod_name or not unit.pod_uid
        )
        if incomplete:
            raise RuntimeOrchestrationError(
                f"publication contains RuntimeServices without observed workload identity: {incomplete}"
            )
        return RuntimeDirectory(
            install_id=session.install_id,
            revision=revision,
            routes=routes,
        )

    def _finalize_publication(
        self,
        session: RuntimeSession,
        directory: RuntimeDirectory,
        retirement_status: Optional[Mapping[str, Any]] = None,
    ) -> RuntimeSession:
        revision_increment = 1
        if session.phase == "publishing":
            revision_increment = max(0, 2 - session.next_runtime_revision)
        retirement = session.retirement
        if session.phase == "publishing-rollout":
            if retirement is None:
                raise RuntimePublicationError(
                    "rollout publication has no persisted retirement ownership"
                )
            status = self._validate_retirement_status(
                retirement_status,
                retirement.revision,
                maximum_deadline=retirement.deadline,
            )
            # Scheduler owns the clock and creates this deadline in the same
            # atomic commit that exposes N+1. Persist that exact value; a
            # Backend crash or delayed response can recover it via status GET.
            retirement = replace(
                retirement,
                deadline=status["deadline"],
                fenced=status["retired"],
                forced_count=status["revoked_count"],
            )
        finalized = replace(
            session,
            phase="active",
            next_runtime_revision=session.next_runtime_revision + revision_increment,
            active_directory_revision=directory.revision,
            active=directory.routes,
            pending=(),
            retirement=retirement,
            last_error="",
            updated_at=_utc_now(),
        )
        try:
            return self._save(finalized)
        except RuntimeSessionConflict:
            current = self._stored.session if self._stored is not None else None
            if (
                current is not None
                and current.phase == "active"
                and current.directory.content_hash == directory.content_hash):
                return current
            raise

    def _recover_publication(
        self,
        session: RuntimeSession,
        cancel_event=None,
    ) -> RuntimeSession:
        if session.phase not in self._PUBLICATION_PHASES:
            return session
        directory = self._publication_candidate(session)
        scheduler_units = session.pending if session.phase == "publishing" else session.active
        scheduler = self._scheduler_unit(scheduler_units)
        deadline = self._clock() + self.operation_timeout
        if session.phase == "publishing":
            self._publish_initial(
                scheduler,
                directory,
                deadline=deadline,
                cancel_event=cancel_event,
            )
            retirement_status = None
        else:
            retirement_status = self._publish_rollout(
                scheduler,
                session.active_directory_revision,
                directory,
                deadline=deadline,
                cancel_event=cancel_event,
            )
        self._raise_if_cancelled(cancel_event)
        return self._finalize_publication(
            session,
            directory,
            retirement_status=retirement_status,
        )

    def recover(self) -> Optional[RuntimeSession]:
        """Reconcile the only crash-sensitive Scheduler/ConfigMap boundary.

        Candidate resources and their immutable UIDs were persisted before the
        directory CAS, so recovery needs no Kubernetes discovery or per-Pod
        probing. A committed old revision remains in ``retirement`` and is
        reconciled independently from publication.
        """

        with self._lock:
            stored = self._reload_for_transaction()
            if stored is None:
                return None
            session = stored.session
            if session.phase in self._INITIAL_ACTIVATION_PHASES:
                # Activation is a watch-driven operation and cannot be resumed
                # safely after losing its deadline and in-memory watch state.
                # Preserve every exact identity already observed so the
                # target-bound uninstall path remains authoritative.
                session = self._save(replace(
                    session,
                    phase="failed",
                    last_error=(
                        f"backend restarted during {session.phase}; "
                        "uninstall this installation before retrying"
                    ),
                    updated_at=_utc_now(),
                ))
            if session.phase == "activating-rollout":
                session = self._restore_active_after_rollout_failure(
                    session,
                    "backend restarted before processor rollout publication",
                )
            if session.phase in self._PUBLICATION_PHASES:
                try:
                    session = self._recover_publication(session)
                except Exception as exc:
                    # Keep the publication phase recoverable and expose the
                    # failed attempt. Background recovery retries with bounded
                    # exponential backoff; a later finalization clears it.
                    try:
                        current = (
                            self._stored.session
                            if self._stored is not None else None
                        )
                        if (
                                current is not None
                                and current.install_id == session.install_id
                                and current.operation_id == session.operation_id
                                and current.phase in self._PUBLICATION_PHASES
                                and current.last_error != str(exc)):
                            self._save(replace(
                                current,
                                last_error=str(exc),
                                updated_at=_utc_now(),
                            ))
                    except Exception:
                        LOGGER.exception(
                            "failed to persist publication recovery error"
                        )
                    raise
            return session

    def install(
        self,
        policy: Mapping[str, Any],
        source_deploy: Sequence[Mapping[str, Any]],
        source_label: str = "",
        install_id: str = "",
        cancel_event=None,
    ) -> RuntimeDirectory:
        install_id = str(install_id or "").strip()
        try:
            canonical_install_id = str(uuid.UUID(install_id))
        except (ValueError, AttributeError, TypeError):
            raise ValueError("install_id must be a canonical UUID") from None
        if install_id != canonical_install_id:
            raise ValueError("install_id must be a canonical UUID")
        self._raise_if_cancelled(cancel_event)
        with self._lock:
            self._raise_if_cancelled(cancel_event)
            stored = self._reload_for_transaction()
            if stored is not None and stored.session.phase in self._PUBLICATION_PHASES:
                self._recover_publication(
                    stored.session,
                    cancel_event=cancel_event,
                )
                stored = self._stored
            self._raise_if_cancelled(cancel_event)
            if stored is not None:
                raise RuntimeOrchestrationError("a runtime session already exists; uninstall it before installing")
            operation_deadline = self._clock() + self.operation_timeout
            templates, normalized_sources = self._logical_templates(
                self.template_helper, policy, source_deploy,
            )
            self._raise_if_cancelled(cancel_event)
            source_scope = selection_scope_from_template(templates["scheduler"])
            inventory = self._refresh_inventory()
            self._raise_if_cancelled(cancel_event)
            cloud_node = self._cloud_node(inventory)
            normalized_sources = self._authorize_source_candidates(
                inventory, normalized_sources, source_scope, cloud_node,
            )
            self._raise_if_cancelled(cancel_event)
            processor_candidates = set(self._selected_nodes(normalized_sources))
            source_candidates = set(self._source_candidate_nodes(normalized_sources))
            permitted_runtime_nodes = processor_candidates | source_candidates | {cloud_node}

            operation_id = str(uuid.uuid4())
            revision = 1
            renderer = self.template_helper.create_runtime_renderer(install_id)
            scheduler_preview = renderer.render(
                templates["scheduler"], RuntimeSlot("scheduler", cloud_node, "cloud"), revision,
            )
            bootstrap = self._bootstrap(
                install_id, cloud_node, cloud_node, inventory,
                permitted_runtime_nodes, (scheduler_preview.unit,),
            )
            scheduler_rendered = renderer.render(
                templates["scheduler"], RuntimeSlot("scheduler", cloud_node, "cloud"), revision,
                extra_env={"DAYU_RUNTIME_BOOTSTRAP": bootstrap},
            )
            session = RuntimeSession(
                install_id=install_id,
                operation_id=operation_id,
                phase="activating-scheduler",
                next_runtime_revision=revision,
                pending=(scheduler_rendered.unit,),
                source_label=source_label,
                policy_id=str(policy.get("id") or ""),
                source_deploy=normalized_sources,
                updated_at=_utc_now(),
            )
            self._save(session)
            try:
                scheduler_unit = self._activate(
                    (scheduler_rendered,),
                    timeout_seconds=self._remaining_timeout(
                        operation_deadline, self.activation_timeout,
                    ),
                    cancel_event=cancel_event,
                )[0]
                # Persist the exact observed RuntimeService/Pod identity before
                # invoking Scheduler.  If stop cancels either planning call,
                # uninstall can retire the scheduler by immutable UID without
                # rediscovering or guessing ownership.
                session = replace(
                    session,
                    pending=(scheduler_unit,),
                    updated_at=_utc_now(),
                )
                self._save(session)
                self._raise_if_cancelled(cancel_event)
                source_plan = self._decision(
                    scheduler_unit,
                    NetworkAPIPath.SCHEDULER_SELECT_SOURCE_NODES,
                    NetworkAPIMethod.SCHEDULER_SELECT_SOURCE_NODES,
                    normalized_sources,
                    timeout=self._remaining_timeout(operation_deadline, self.operation_timeout),
                    cancel_event=cancel_event,
                )
                self._raise_if_cancelled(cancel_event)
                source_selection = self._source_selection(source_plan, normalized_sources)
                enriched = copy.deepcopy(normalized_sources)
                for source_info in enriched:
                    node = source_selection[_source_id(source_info)]
                    source_info["source_device"] = node
                    source_info.setdefault("source", {})["source_device"] = node
                deployment_plan = self._decision(
                    scheduler_unit,
                    NetworkAPIPath.SCHEDULER_INITIAL_DEPLOYMENT,
                    NetworkAPIMethod.SCHEDULER_INITIAL_DEPLOYMENT,
                    enriched,
                    timeout=self._remaining_timeout(operation_deadline, self.operation_timeout),
                    cancel_event=cancel_event,
                )
                self._raise_if_cancelled(cancel_event)
                deployment = self._deployment(deployment_plan, enriched, cloud_node)
                target_nodes = set(source_selection.values()) | {
                    node for nodes in deployment.values() for node in nodes
                } | {cloud_node}
                # The first preflight covered every source candidate plus the
                # cloud node.  ``_deployment`` rejects any node outside that
                # exact set, so checking managed agents again would only issue
                # two more cluster-wide Pod lists on every install.
                if not target_nodes.issubset(permitted_runtime_nodes):
                    raise RuntimePreflightError(
                        f"scheduler selected targets outside the preflight snapshot: "
                        f"{sorted(target_nodes - permitted_runtime_nodes)}"
                    )
                rendered, enriched = self._render_initial(
                    renderer, templates, enriched, source_selection, deployment,
                    inventory, cloud_node, revision, scheduler_unit,
                )
                pending = (scheduler_unit,) + tuple(item.unit for item in rendered)
                session = replace(
                    session,
                    phase="activating-runtime",
                    pending=pending,
                    source_deploy=enriched,
                    updated_at=_utc_now(),
                )
                self._save(session)
                active = (scheduler_unit,) + self._activate(
                    rendered,
                    timeout_seconds=self._remaining_timeout(
                        operation_deadline, self.activation_timeout,
                    ),
                    cancel_event=cancel_event,
                )
                directory = RuntimeDirectory(install_id=install_id, revision=1, routes=active)
                session = replace(
                    session,
                    phase="publishing",
                    pending=active,
                    updated_at=_utc_now(),
                )
                self._save(session)
                self._publish_initial(
                    scheduler_unit, directory, deadline=operation_deadline,
                    cancel_event=cancel_event,
                )
                self._raise_if_cancelled(cancel_event)
                session = self._finalize_publication(session, directory)
                return directory
            except RuntimeOperationCancelled:
                # Cancellation is lifecycle control flow.  Keep the last exact
                # CAS boundary (including observed UIDs where available) for
                # the uninstall transaction that follows, and never rewrite it
                # as a generic failed installation.
                LOGGER.info("managed RuntimeService install yielded to lifecycle cancellation")
                raise
            except Exception as exc:
                LOGGER.exception("managed RuntimeService install failed")
                try:
                    current = self._stored.session if self._stored is not None else None
                    if (
                        current is not None
                        and current.operation_id == session.operation_id
                        and current.phase != "active"):
                        phase = (
                            current.phase
                            if current.phase in self._PUBLICATION_PHASES
                            else "failed"
                        )
                        self._save(replace(
                            current,
                            phase=phase,
                            last_error=str(exc),
                            updated_at=_utc_now(),
                        ))
                except Exception:
                    LOGGER.exception("failed to persist managed-runtime failure state")
                raise

    def _render_processor_candidates(
        self,
        session: RuntimeSession,
        deployment: Mapping[str, Sequence[str]],
        inventory: Mapping[str, Mapping[str, Any]],
        cloud_node: str,
        templates: Mapping[str, Any],
    ):
        active_by_key = {unit.logical_key: unit for unit in session.active}
        desired_slots = [
            RuntimeSlot(
                "processor", node, "cloud" if node == cloud_node else "edge",
                logical_service=service,
            )
            for service, nodes in deployment.items()
            for node in nodes
        ]
        desired_keys = {slot.logical_key for slot in desired_slots}
        renderer = self.template_helper.create_runtime_renderer(session.install_id)
        scheduler = self._scheduler_unit(session.active)
        distributor = next(unit for unit in session.active if unit.slot.component == "distributor")
        runtime_nodes = {unit.slot.target_node for unit in session.active} | {
            slot.target_node for slot in desired_slots
        }
        by_service = {
            str(info["service_name"]): info
            for info in templates["processor"].values()
        }

        rendered = []
        replacement_keys = set()
        for slot in desired_slots:
            service_info = by_service.get(slot.logical_service)
            if service_info is None:
                raise RuntimeOrchestrationError(
                    f"no processor template exists for service {slot.logical_service!r}"
                )
            logical_template, device_env = self._specialize_template_for_node(
                service_info["service"], slot.target_node, inventory,
            )
            candidate = renderer.render(
                logical_template, slot, session.next_runtime_revision,
                extra_env={
                    "PROCESSOR_SERVICE_NAME": f"processor-{slot.logical_service}",
                    "DAYU_RUNTIME_BOOTSTRAP": self._bootstrap(
                        session.install_id, slot.target_node, cloud_node, inventory,
                        runtime_nodes, (scheduler, distributor),
                        session.active_directory_revision,
                    ),
                    **device_env,
                },
            )
            current = active_by_key.get(slot.logical_key)
            if current is not None and current.rollout_hash == candidate.unit.rollout_hash:
                continue
            rendered.append(candidate)
            replacement_keys.add(slot.logical_key)

        kept = tuple(
            unit for unit in session.active
            if unit.slot.component != "processor"
            or (unit.logical_key in desired_keys and unit.logical_key not in replacement_keys)
        )
        retired = tuple(
            unit for unit in session.active
            if unit.slot.component == "processor"
            and (unit.logical_key not in desired_keys or unit.logical_key in replacement_keys)
        )
        return tuple(rendered), kept, retired

    @staticmethod
    def _merge_units(*groups: Iterable[RuntimeUnit]) -> Tuple[RuntimeUnit, ...]:
        """Merge exact runtime ownership records by immutable resource name."""

        units = {}
        for group in groups:
            for unit in group:
                existing = units.get(unit.runtime_id)
                if (
                    existing is not None
                    and existing.runtime_service_uid
                    and unit.runtime_service_uid
                    and existing.runtime_service_uid != unit.runtime_service_uid
                ):
                    raise RuntimeOrchestrationError(
                        f"conflicting RuntimeService UIDs for {unit.runtime_id!r}"
                    )
                if existing is None or (
                    not existing.runtime_service_uid and unit.runtime_service_uid
                ):
                    units[unit.runtime_id] = unit
        return tuple(units[name] for name in sorted(units))

    @staticmethod
    def _merge_cleanup(*groups: Iterable[Any]) -> Tuple[RuntimeCleanupRef, ...]:
        """Compact exact deletion ownership without retaining route payloads."""

        refs = {}
        for group in groups:
            for value in group:
                ref = (
                    RuntimeCleanupRef.from_unit(value)
                    if isinstance(value, RuntimeUnit)
                    else value
                )
                if not isinstance(ref, RuntimeCleanupRef):
                    ref = RuntimeCleanupRef.from_dict(ref)
                existing = refs.get(ref.runtime_id)
                if (
                    existing is not None
                    and existing.runtime_service_uid
                    and ref.runtime_service_uid
                    and existing.runtime_service_uid != ref.runtime_service_uid
                ):
                    raise RuntimeOrchestrationError(
                        f"conflicting cleanup UIDs for {ref.runtime_id!r}"
                    )
                if existing is None or (
                    not existing.runtime_service_uid and ref.runtime_service_uid
                ):
                    refs[ref.runtime_id] = ref
        return tuple(refs[name] for name in sorted(refs))

    def _restore_active_after_rollout_failure(
        self,
        session: RuntimeSession,
        error: str,
    ) -> RuntimeSession:
        """Keep the committed old directory active and defer candidate cleanup."""

        if session.phase != "activating-rollout":
            return session
        next_runtime_revision = max(
            session.next_runtime_revision,
            max(
                (unit.runtime_revision + 1 for unit in session.pending),
                default=session.next_runtime_revision,
            ),
        )
        return self._save(replace(
            session,
            phase="active",
            next_runtime_revision=next_runtime_revision,
            pending=(),
            retirement=None,
            cleanup=self._merge_cleanup(session.cleanup, session.pending),
            last_error=str(error or "processor rollout was interrupted before publication"),
            updated_at=_utc_now(),
        ))

    def redeploy(self, policy: Mapping[str, Any], cancel_event=None) -> bool:
        with self._lock:
            self._raise_if_cancelled(cancel_event)
            stored = self._reload_for_transaction()
            if stored is not None and stored.session.phase in self._PUBLICATION_PHASES:
                self._recover_publication(
                    stored.session,
                    cancel_event=cancel_event,
                )
                stored = self._stored
            if stored is None or stored.session.phase != "active":
                raise RuntimeOrchestrationError("processor rollout requires an active runtime session")
            session = stored.session
            if session.retirement is not None:
                if self._wall_clock() < session.retirement.deadline:
                    raise RuntimeRetirementPending(
                        "previous RuntimeDirectory revision is still retiring"
                    )
                # The durable deadline, not Scheduler availability or a stuck
                # finalizer, is the upper bound on rollout serialization. Exact
                # old identities remain owned by the asynchronous cleanup set.
                LOGGER.warning(
                    f"[Runtime Retirement] revision={session.retirement.revision} "
                    "reached its deadline; release rollout gate and continue cleanup"
                )
                session = self._save(replace(
                    session,
                    retirement=None,
                    cleanup=self._merge_cleanup(
                        session.cleanup,
                        session.retirement.units,
                    ),
                    updated_at=_utc_now(),
                ))
            scheduler = self._scheduler_unit(session.active)
            operation_deadline = self._clock() + self.operation_timeout
            self._raise_if_cancelled(cancel_event)
            inventory = self.node_inventory()
            cloud_node = self._cloud_node(inventory)
            raw_plan = self._decision(
                scheduler,
                NetworkAPIPath.SCHEDULER_REDEPLOYMENT,
                NetworkAPIMethod.SCHEDULER_REDEPLOYMENT,
                session.source_deploy,
                timeout=self._remaining_timeout(operation_deadline, self.operation_timeout),
                cancel_event=cancel_event,
            )
            self._raise_if_cancelled(cancel_event)
            deployment = self._deployment(raw_plan, session.source_deploy, cloud_node)
            target_nodes = {node for nodes in deployment.values() for node in nodes}
            # Install already validated Sedna LC and EdgeMesh coverage on every
            # immutable source candidate plus cloud. Repeating two cluster-wide
            # Pod lists on every automatic redeploy would add no new placement
            # guarantee; RuntimeService activation remains the exact live gate.
            self._preflight_nodes(inventory, target_nodes, validate_agents=False)
            templates, normalized_sources = self._logical_templates(
                self.template_helper, policy, session.source_deploy,
            )
            if normalized_sources != list(session.source_deploy):
                raise RuntimeOrchestrationError("persisted normalized source deployment changed during rollout")
            rendered, kept, retired_units = self._render_processor_candidates(
                session, deployment, inventory, cloud_node, templates,
            )
            if not rendered and not retired_units:
                return False

            pending_units = tuple(item.unit for item in rendered)
            session = replace(
                session,
                operation_id=str(uuid.uuid4()),
                phase="activating-rollout",
                pending=pending_units,
                updated_at=_utc_now(),
            )
            self._save(session)
            try:
                activated = self._activate(
                    rendered,
                    timeout_seconds=self._remaining_timeout(
                        operation_deadline, self.activation_timeout,
                    ),
                    cancel_event=cancel_event,
                )
                candidate_units = tuple(kept) + tuple(activated)
                directory = RuntimeDirectory(
                    install_id=session.install_id,
                    revision=session.active_directory_revision + 1,
                    routes=candidate_units,
                )
                # Persist old-resource ownership immediately before the
                # crash-sensitive directory CAS. The deadline is armed only
                # after publication is durably observed, so proposal latency
                # cannot consume task grace.
                retirement = RuntimeRetirement(
                    revision=session.active_directory_revision,
                    units=retired_units,
                    deadline=None,
                    started_at=_utc_now(),
                )
                session = replace(
                    session,
                    phase="publishing-rollout",
                    pending=activated,
                    retirement=retirement,
                    updated_at=_utc_now(),
                )
                self._save(session)
                retirement_status = self._publish_rollout(
                    scheduler,
                    session.active_directory_revision,
                    directory,
                    deadline=operation_deadline,
                    cancel_event=cancel_event,
                )
                self._raise_if_cancelled(cancel_event)
                self._finalize_publication(
                    session,
                    directory,
                    retirement_status=retirement_status,
                )
                return True
            except RuntimeOperationCancelled:
                # Keep the last durable transaction boundary exactly as-is.
                # Before publication this retains candidate ownership for
                # uninstall; during publication it retains the recoverable
                # ambiguous-CAS state; after publication it retains retirement
                # ownership for exact UID cleanup. Do not record cancellation as
                # a runtime fault.
                LOGGER.info("managed processor rollout yielded to lifecycle cancellation")
                raise
            except Exception as exc:
                LOGGER.exception("managed processor rollout failed")
                try:
                    current = self._stored.session if self._stored is not None else None
                    if (
                        current is not None
                        and current.operation_id == session.operation_id
                        and current.phase != "active"):
                        if current.phase == "activating-rollout":
                            self._restore_active_after_rollout_failure(
                                current,
                                str(exc),
                            )
                        else:
                            self._save(replace(
                                current,
                                last_error=str(exc),
                                updated_at=_utc_now(),
                            ))
                except Exception:
                    LOGGER.exception("failed to persist rollout failure state")
                raise

    @staticmethod
    def _validate_retirement_status(
        response: Any,
        revision: int,
        maximum_deadline: Optional[float] = None,
    ) -> Mapping[str, Any]:
        if not isinstance(response, Mapping):
            raise RuntimeOrchestrationError(
                "scheduler retirement status is unavailable"
            )
        try:
            observed_revision = int(response.get("revision"))
            count = int(response.get("count"))
            observed_deadline = float(response.get("deadline"))
            revoked_count = int(response.get("revoked_count", 0))
        except (TypeError, ValueError) as exc:
            raise RuntimeOrchestrationError(
                "scheduler retirement status is invalid"
            ) from exc
        if (
            observed_revision != int(revision)
            or count < 0
            or revoked_count < 0
            or not math.isfinite(observed_deadline)
            or observed_deadline <= 0
        ):
            raise RuntimeOrchestrationError(
                "scheduler retirement identity is invalid"
            )
        if (
            maximum_deadline is not None
            and observed_deadline > float(maximum_deadline)
        ):
            raise RuntimeOrchestrationError(
                "scheduler extended an immutable retirement deadline"
            )
        return {
            "revision": observed_revision,
            "count": count,
            "deadline": observed_deadline,
            "retired": bool(response.get("retired")),
            "revoked_count": revoked_count,
        }

    def _retirement_status(
        self,
        scheduler: RuntimeUnit,
        revision: int,
        maximum_deadline: Optional[float] = None,
        operation_deadline: Optional[float] = None,
        cancel_event=None,
    ) -> Mapping[str, Any]:
        timeout = min(5, self.operation_timeout)
        if operation_deadline is not None:
            timeout = self._remaining_timeout(operation_deadline, timeout)
        response = self._scheduler_call(
            scheduler,
            NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY_TASK_LEASES,
            NetworkAPIMethod.SCHEDULER_COUNT_TASK_LEASES,
            params={"revision": int(revision)},
            timeout=timeout,
            cancel_event=cancel_event,
        )
        return self._validate_retirement_status(
            response,
            revision,
            maximum_deadline=maximum_deadline,
        )

    def _retire_revision(
        self,
        scheduler: RuntimeUnit,
        revision: int,
        deadline: float,
        cancel_event=None,
    ) -> Mapping[str, Any]:
        """Start or reconcile one immutable revision retirement."""

        response = self._scheduler_call(
            scheduler,
            NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY_TASK_LEASES,
            NetworkAPIMethod.SCHEDULER_RETIRE_TASK_LEASES,
            {"revision": int(revision), "deadline": float(deadline)},
            timeout=min(5, self.operation_timeout),
            cancel_event=cancel_event,
        )
        return self._validate_retirement_status(
            response,
            revision,
            maximum_deadline=deadline,
        )

    def _reconcile_retirement_locked(
        self,
        session: RuntimeSession,
        cancel_event=None,
    ) -> bool:
        retirement = session.retirement
        if retirement is None:
            return False
        scheduler = self._scheduler_unit(session.active)
        status = None
        status_error = None
        try:
            status = self._retirement_status(
                scheduler,
                retirement.revision,
                maximum_deadline=retirement.deadline,
                cancel_event=cancel_event,
            )
        except RuntimeOperationCancelled:
            raise
        except Exception as exc:
            status_error = exc

        if status is not None and status["deadline"] < retirement.deadline:
            retirement = replace(retirement, deadline=status["deadline"])
        deadline_reached = self._wall_clock() >= retirement.deadline
        ready = bool(status and (status["count"] == 0 or status["retired"]))
        if not ready and not deadline_reached:
            error = str(status_error or "")
            if error and session.last_error != error:
                self._save(replace(
                    session,
                    retirement=retirement,
                    last_error=error,
                    updated_at=_utc_now(),
                ))
            elif status is not None and session.last_error and not session.cleanup:
                self._save(replace(
                    session,
                    retirement=retirement,
                    last_error="",
                    updated_at=_utc_now(),
                ))
            return False

        retirement = replace(
            retirement,
            fenced=bool(deadline_reached or (status and status["retired"])),
            forced_count=(
                status["revoked_count"] if status is not None
                else retirement.forced_count
            ),
        )
        try:
            self._delete_units(
                retirement.units,
                session.install_id,
                cancel_event=cancel_event,
                timeout_seconds=30,
            )
        except RuntimeOperationCancelled:
            raise
        except Exception as exc:
            # Resource finalizers are garbage-collection work, not a rollout
            # lock. Keep exact ownership durable and release the one retirement
            # slot once lease protection ended or its deadline elapsed.
            self._save(replace(
                session,
                retirement=None,
                cleanup=self._merge_cleanup(session.cleanup, retirement.units),
                last_error=str(exc),
                updated_at=_utc_now(),
            ))
            LOGGER.warning(
                f"[Runtime Cleanup] deferred revision={retirement.revision}: {exc}"
            )
            return True

        self._save(replace(
            session,
            retirement=None,
            last_error="" if not session.cleanup else session.last_error,
            updated_at=_utc_now(),
        ))
        LOGGER.info(
            f"[Runtime Retirement] revision={retirement.revision} "
            f"forced={retirement.fenced} revoked={retirement.forced_count} "
            f"scheduler_acknowledged={status is not None}"
        )
        return True

    def _reconcile_cleanup_locked(
        self,
        session: RuntimeSession,
        cancel_event=None,
    ) -> bool:
        if not session.cleanup:
            return False
        try:
            self._delete_units(
                session.cleanup,
                session.install_id,
                cancel_event=cancel_event,
                timeout_seconds=30,
            )
        except RuntimeOperationCancelled:
            raise
        except Exception as exc:
            if session.last_error != str(exc):
                self._save(replace(
                    session,
                    last_error=str(exc),
                    updated_at=_utc_now(),
                ))
            LOGGER.warning(f"[Runtime Cleanup] exact-UID cleanup remains pending: {exc}")
            return False
        self._save(replace(
            session,
            cleanup=(),
            last_error="",
            updated_at=_utc_now(),
        ))
        return True

    def reconcile_retirement(self, cancel_event=None) -> bool:
        """Perform one bounded reconciliation tick without polling or sleeping."""

        with self._lock:
            self._raise_if_cancelled(cancel_event)
            # Reconciliation is driven by the single backend-owned worker.
            # Reuse the process snapshot instead of turning every one-second
            # tick into a Kubernetes ConfigMap GET; CAS writes reload on a real
            # conflict, and process recovery performs one deliberate reload.
            stored = self._load()
            if stored is None:
                return False
            session = stored.session
            if session.phase in self._PUBLICATION_PHASES:
                session = self._recover_publication(
                    session,
                    cancel_event=cancel_event,
                )
            if session.phase != "active":
                return False
            changed = False
            if session.retirement is not None:
                changed = self._reconcile_retirement_locked(
                    session,
                    cancel_event=cancel_event,
                )
                # Retirement and garbage collection are independent lanes.
                # Always reload the latest in-process CAS result: retirement
                # may just have moved its exact identities into ``cleanup``.
                if self._stored is None:
                    return changed
                session = self._stored.session
                if session.phase != "active":
                    return changed
            if session.cleanup:
                changed = self._reconcile_cleanup_locked(
                    session,
                    cancel_event=cancel_event,
                ) or changed
            return changed

    def _clear_runtime_directory(self, scheduler: RuntimeUnit, install_id: str) -> None:
        # This is an in-cluster metadata operation. Keep DELETE plus its
        # ambiguity readback inside one short budget so Scheduler trouble does
        # not consume the public uninstall window.
        deadline = self._clock() + min(20, self.operation_timeout)
        clear_error = None
        try:
            response = self._scheduler_call(
                scheduler,
                NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY,
                NetworkAPIMethod.SCHEDULER_CLEAR_RUNTIME_DIRECTORY,
                {"install_id": str(install_id)},
                timeout=self._remaining_timeout(deadline),
            )
            if (
                isinstance(response, Mapping)
                and response.get("cleared") is True
                and response.get("install_id") == str(install_id)):
                return
        except Exception as exc:
            clear_error = exc
        # The DELETE may have committed before its response was lost. An empty
        # readback is the durable, idempotent success condition.
        try:
            readback = self._scheduler_call(
                scheduler,
                NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY,
                NetworkAPIMethod.SCHEDULER_GET_RUNTIME_DIRECTORY,
                timeout=self._remaining_timeout(deadline),
            )
            if isinstance(readback, Mapping) and (
                int(readback.get("directory_revision", readback.get("revision", -1))) == 0
                and not (readback.get("routes") or readback.get("entries"))):
                return
        except Exception as exc:
            clear_error = clear_error or exc
        raise RuntimeOrchestrationError(
            "scheduler RuntimeDirectory clear was not durably observable"
        ) from clear_error

    def _delete_units(
        self,
        units: Iterable[Any],
        install_id: str,
        cancel_event=None,
        timeout_seconds: Optional[float] = None,
        allow_active: bool = False,
        propagation_policy: str = "Background",
    ) -> None:
        self._raise_if_cancelled(cancel_event)
        units = tuple(units)
        if not allow_active and self._stored is not None:
            active_ids = {
                unit.runtime_id for unit in self._stored.session.active
            }
            overlap = sorted(
                unit.runtime_id for unit in units
                if unit.runtime_id in active_ids
            )
            if overlap:
                raise RuntimeOrchestrationError(
                    f"refuse to garbage-collect active RuntimeServices: {overlap}"
                )
        total_timeout = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else min(self.activation_timeout, 120)
        )
        if not math.isfinite(total_timeout) or total_timeout <= 0:
            raise RuntimeOrchestrationError("RuntimeService deletion deadline was exhausted")
        deadline = self._clock() + total_timeout
        identities = {}
        unresolved = set()
        for unit in units:
            endpoint = getattr(unit, "endpoint", None)
            uid = unit.runtime_service_uid or (
                endpoint.runtime_service_uid if endpoint else None
            )
            if unit.runtime_id in identities and identities[unit.runtime_id] != uid:
                raise RuntimeOrchestrationError(
                    f"conflicting RuntimeService UIDs for deletion of {unit.runtime_id!r}"
                )
            if uid:
                identities[unit.runtime_id] = uid
            else:
                unresolved.add(unit.runtime_id)
        if unresolved:
            self._raise_if_cancelled(cancel_event)
            try:
                response = self.runtime.list(
                    request_timeout_seconds=self._remaining_timeout(deadline),
                )
            except Exception:
                self._raise_if_cancelled(cancel_event)
                raise
            self._raise_if_cancelled(cancel_event)
            for obj in response.get("items") or ():
                metadata = obj.get("metadata") or {}
                name = str(metadata.get("name") or "")
                if name not in unresolved:
                    continue
                labels = metadata.get("labels") or {}
                uid = str(metadata.get("uid") or "")
                spec_install_id = str((obj.get("spec") or {}).get("installID") or "")
                if not uid or (
                    labels.get(self.INSTALL_LABEL) != str(install_id)
                    and spec_install_id != str(install_id)
                ):
                    raise RuntimeOrchestrationError(
                        f"RuntimeService {name!r} has no exact install ownership identity"
                    )
                identities[name] = uid
            # Missing names are already absent; never downgrade to an
            # unguarded delete when activation crashed before UID persistence.
        try:
            self.runtime.delete_many(
                identities,
                timeout_seconds=self._remaining_timeout(deadline),
                cancel_event=cancel_event,
                propagation_policy=propagation_policy,
            )
        except RuntimeServiceCancelled:
            self._raise_if_cancelled(cancel_event)
            raise
        self._raise_if_cancelled(cancel_event)

    @staticmethod
    def _owned_units(session: RuntimeSession) -> Tuple[RuntimeUnit, ...]:
        retirement_units = session.retirement.units if session.retirement else ()
        return RuntimeOrchestrator._merge_units(
            session.active,
            session.pending,
            retirement_units,
        )

    def _uninstall_timestamp(self) -> str:
        return datetime.fromtimestamp(
            float(self._wall_clock()), timezone.utc,
        ).isoformat()

    def _new_uninstall_progress(self) -> RuntimeUninstallProgress:
        timestamp = self._uninstall_timestamp()
        return RuntimeUninstallProgress(
            started_at=timestamp,
            last_progress_at=timestamp,
        )

    def _observe_uninstall_resources(
        self,
        session: RuntimeSession,
    ) -> Tuple[RuntimeCleanupResource, ...]:
        """Return one complete, fail-closed snapshot of the cleanup barrier."""

        install_id = session.install_id
        owned_units = self._owned_units(session)
        runtime_names = {unit.runtime_id for unit in owned_units} | {
            unit.runtime_id for unit in session.cleanup
        }
        runtime_service_uids = {
            uid
            for unit in tuple(owned_units) + tuple(session.cleanup)
            for uid in (
                getattr(unit, "runtime_service_uid", "")
                or (
                    getattr(getattr(unit, "endpoint", None), "runtime_service_uid", "")
                ),
            )
            if uid
        }
        service_uids = {
            unit.endpoint.service_uid
            for unit in owned_units
            if unit.endpoint is not None and unit.endpoint.service_uid
        }
        pod_names = {
            unit.pod_name for unit in owned_units if unit.pod_name
        }
        pod_uids = {
            unit.pod_uid or (
                unit.endpoint.pod_uid if unit.endpoint is not None else ""
            )
            for unit in owned_units
        } - {""}
        deadline = self._clock() + min(30.0, self.operation_timeout)
        resources = [
            RuntimeCleanupResource.from_dict(item)
            for item in self.cluster.runtime_cleanup_resources(
                install_id,
                ownership={
                    "runtime_names": runtime_names,
                    "runtime_service_uids": runtime_service_uids,
                    "service_uids": service_uids,
                    "pod_names": pod_names,
                    "pod_uids": pod_uids,
                },
                request_timeout_seconds=self._remaining_timeout(deadline),
            )
        ]
        response = self.runtime.list(
            request_timeout_seconds=self._remaining_timeout(deadline),
        )
        for item in response.get("items") or ():
            metadata = item.get("metadata") or {}
            labels = metadata.get("labels") or {}
            name = str(metadata.get("name") or "")
            uid = str(metadata.get("uid") or "")
            if not (
                str((item.get("spec") or {}).get("installID") or "") == install_id
                or labels.get(self.INSTALL_LABEL) == install_id
                or name in runtime_names
                or uid in runtime_service_uids
            ):
                continue
            resources.append(RuntimeCleanupResource.from_dict(
                kubernetes_resource_ref("RuntimeService", item),
            ))
        return tuple(sorted(
            resources,
            key=lambda resource: resource.identity,
        ))

    def begin_uninstall(self, expected_install_id: str = "") -> Optional[RuntimeSession]:
        """Persist administrative stop intent without waiting for teardown."""

        with self._lock:
            stored = self._reload_for_transaction()
            if stored is None:
                return None
            session = stored.session
            if expected_install_id and session.install_id != expected_install_id:
                return None
            if session.phase not in {"uninstalling", "finalizing-uninstall"}:
                session = self._save(replace(
                    session,
                    operation_id=str(uuid.uuid4()),
                    phase="uninstalling",
                    uninstall=self._new_uninstall_progress(),
                    last_error="",
                    updated_at=_utc_now(),
                ))
            elif session.uninstall is None:
                # Upgrade/recovery path for a Session written before the
                # cleanup barrier became part of the durable schema.
                session = self._save(replace(
                    session,
                    uninstall=self._new_uninstall_progress(),
                    updated_at=_utc_now(),
                ))
            return session

    def uninstall(self, expected_install_id: str = "") -> bool:
        """Stop the installation without allowing task leases to veto teardown."""

        with self._lock:
            stored = self._reload_for_transaction()
            if stored is None:
                return True
            session = stored.session
            if expected_install_id and session.install_id != expected_install_id:
                return True
            all_units = self._owned_units(session)
            schedulers = tuple(
                unit for unit in all_units if unit.slot.component == "scheduler"
            )
            if not schedulers:
                raise RuntimeOrchestrationError(
                    "runtime session contains no scheduler RuntimeService"
                )
            active_schedulers = tuple(
                unit for unit in session.active if unit.slot.component == "scheduler"
            )
            scheduler = active_schedulers[0] if len(active_schedulers) == 1 else schedulers[0]
            generators = tuple(
                unit for unit in all_units if unit.slot.component == "generator"
            )
            workers = tuple(
                unit for unit in all_units
                if unit.slot.component not in {"generator", "scheduler"}
            )

            directory_may_be_published = (
                session.active_directory_revision > 0
                or session.phase in self._PUBLICATION_PHASES
            )
            revisions = set()
            if session.active_directory_revision > 0:
                revisions.add(session.active_directory_revision)
            if session.retirement is not None:
                revisions.add(session.retirement.revision)
            if session.phase in self._PUBLICATION_PHASES:
                revisions.add(session.active_directory_revision + 1)

            made_progress = False
            if session.phase not in {"uninstalling", "finalizing-uninstall"}:
                session = self._save(replace(
                    session,
                    operation_id=str(uuid.uuid4()),
                    phase="uninstalling",
                    uninstall=self._new_uninstall_progress(),
                    last_error="",
                    updated_at=_utc_now(),
                ))
                made_progress = True
            elif session.uninstall is None:
                session = self._save(replace(
                    session,
                    uninstall=self._new_uninstall_progress(),
                    updated_at=_utc_now(),
                ))
                made_progress = True

            try:
                if session.phase != "finalizing-uninstall":
                    # Stop task admission first. Uninstall intentionally does not
                    # preserve in-flight work: this matches the public stop
                    # contract and prevents stale leases from delaying teardown.
                    self._delete_units(
                        generators,
                        session.install_id,
                        timeout_seconds=30,
                        allow_active=True,
                        propagation_policy="Foreground",
                    )
                    if directory_may_be_published:
                        fence_deadline = self._wall_clock()
                        for revision in sorted(revisions):
                            try:
                                self._retire_revision(
                                    scheduler,
                                    revision,
                                    fence_deadline,
                                )
                            except Exception as exc:
                                LOGGER.warning(
                                    f"[Runtime Uninstall] Could not fence revision {revision}; "
                                    f"continue full teardown: {exc}"
                                )
                        try:
                            self._clear_runtime_directory(
                                scheduler,
                                session.install_id,
                            )
                        except Exception as exc:
                            # Full installation deletion is itself the final
                            # fence. Scheduler unavailability must not turn task
                            # state into an uninstall lock.
                            LOGGER.warning(
                                f"[Runtime Uninstall] Could not clear RuntimeDirectory; "
                                f"continue exact UID teardown: {exc}"
                            )
                    session = self._save(replace(
                        session,
                        phase="finalizing-uninstall",
                        uninstall=replace(
                            session.uninstall,
                            last_progress_at=self._uninstall_timestamp(),
                        ),
                        last_error="",
                        updated_at=_utc_now(),
                    ))
                    made_progress = True

                progress = session.uninstall
                if not progress.deletion_submitted:
                    # Scheduler deletion is the definitive admission fence when
                    # its directory clear could not be acknowledged. Submit it
                    # before workers and persist that all exact deletes crossed
                    # the API boundary, so later reconciles only observe GC.
                    self._delete_units(
                        schedulers,
                        session.install_id,
                        timeout_seconds=60,
                        allow_active=True,
                        propagation_policy="Foreground",
                    )
                    self._delete_units(
                        tuple(workers) + tuple(session.cleanup),
                        session.install_id,
                        timeout_seconds=60,
                        allow_active=True,
                        propagation_policy="Foreground",
                    )
                    progress = replace(
                        progress,
                        deletion_submitted=True,
                        last_progress_at=self._uninstall_timestamp(),
                    )
                    session = self._save(replace(
                        session,
                        uninstall=progress,
                        last_error="",
                        updated_at=_utc_now(),
                    ))
                    made_progress = True

                remaining = self._observe_uninstall_resources(session)
                if remaining:
                    previous_identities = progress.identities
                    next_identities = frozenset(
                        resource.identity for resource in remaining
                    )
                    last_progress_at = progress.last_progress_at
                    if previous_identities and next_identities < previous_identities:
                        last_progress_at = self._uninstall_timestamp()
                        made_progress = True
                    next_progress = replace(
                        progress,
                        last_progress_at=last_progress_at,
                        remaining=remaining,
                    )
                    changed = next_progress != progress or bool(session.last_error)
                    if changed:
                        self._save(replace(
                            session,
                            uninstall=next_progress,
                            last_error="",
                            updated_at=_utc_now(),
                        ))
                    return made_progress

                expected = self._stored.resource_version if self._stored else None
                self.sessions.delete(expected_resource_version=expected)
                self._mark_snapshot_deleted()
                return True
            except Exception as exc:
                LOGGER.exception("managed RuntimeService uninstall failed")
                try:
                    stored = self._reload_for_transaction()
                except Exception:
                    LOGGER.exception("could not reload RuntimeSession after uninstall failure")
                    raise exc
                if stored is None or stored.session.install_id != session.install_id:
                    # The CAS DELETE may have committed even if its response
                    # was lost. Absence (or a newer installation) is
                    # authoritative; never recreate an old session merely to
                    # persist an uninstall error.
                    return True
                current = stored.session
                try:
                    if current.last_error != str(exc):
                        self._save(replace(
                            current,
                            phase=(
                                "finalizing-uninstall"
                                if current.phase == "finalizing-uninstall" else "uninstalling"
                            ),
                            last_error=str(exc),
                            updated_at=_utc_now(),
                        ))
                except RuntimeSessionConflict:
                    if self._stored is None:
                        return True
                    raise
                raise

    def sample_runtime_metrics(
        self,
        pod_refs: Sequence[Mapping[str, Any]],
        request_timeout_seconds: float,
    ) -> Dict[str, Dict[str, Any]]:
        """Sample immutable Pod refs using the independently cached inventory.

        The caller owns the active-directory generation. This method deliberately
        does not read lifecycle state, so a rollout cannot silently substitute a
        newer set of Pod identities while an older sample is in flight.
        """
        request_timeout_seconds = float(request_timeout_seconds)
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        refs = [dict(ref) for ref in pod_refs or ()]
        if not refs:
            return {}
        return self.cluster.runtime_metrics(
            refs,
            node_inventory=self.node_inventory(
                request_timeout_seconds=request_timeout_seconds,
            ),
            request_timeout_seconds=request_timeout_seconds,
        )
