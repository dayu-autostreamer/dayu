"""Transactional orchestration for Dayu's managed RuntimeService data plane.

The backend is the only Python process that talks to Kubernetes.  Runtime
workers receive a small immutable bootstrap document and exact per-task routes;
they never discover Pods, Services, Nodes, or ports themselves.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from core.lib.common import LOGGER, TaskConstant
from core.lib.network import NetworkAPIMethod, NetworkAPIPath, http_request

from cluster_client import ClusterClient
from runtime_model import RuntimeDirectory, RuntimeEndpoint, RuntimeSession, RuntimeSlot, RuntimeUnit
from runtime_service_client import RuntimeServiceClient
from runtime_session_store import RuntimeSessionConflict, RuntimeSessionStore, StoredRuntimeSession


class RuntimeOrchestrationError(RuntimeError):
    """A managed-runtime transaction could not be completed safely."""


class RuntimePreflightError(RuntimeOrchestrationError):
    """Cluster state cannot satisfy the managed RuntimeService contract."""


class RuntimePublicationError(RuntimeOrchestrationError):
    """The scheduler could not atomically publish an exact directory."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_code(exc: Exception) -> Optional[int]:
    value = getattr(exc, "status", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _endpoint_url(endpoint: RuntimeEndpoint, path: str = "") -> str:
    base = f"http://{endpoint.dns_name}:{endpoint.port}"
    return f"{base}/{str(path).lstrip('/')}" if path else base


def _source_id(source_info: Mapping[str, Any]) -> str:
    source = source_info.get("source") or {}
    value = source.get("id")
    if value is None or str(value) == "":
        raise RuntimeOrchestrationError("every source requires a non-empty source.id")
    return str(value)


class RuntimeOrchestrator:
    """Own install, rollout, directory publication, drain and retirement.

    Kubernetes clients are created lazily and share one ``ApiClient``.  A
    process lock plus ConfigMap compare-and-swap prevents two backend requests
    from publishing different runtime directories for the same installation.
    """

    INSTALL_LABEL = "dayu.io/install-id"
    MANAGED_LABEL_SELECTOR = "app.kubernetes.io/managed-by=dayu-backend"
    _PUBLICATION_PHASES = frozenset({"publishing", "publishing-rollout"})

    def __init__(
        self,
        template_helper,
        namespace: str,
        cluster_client: Optional[ClusterClient] = None,
        runtime_client: Optional[RuntimeServiceClient] = None,
        session_store: Optional[RuntimeSessionStore] = None,
        request=http_request,
        clock=time.monotonic,
        sleeper=time.sleep,
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
        self._sleep = sleeper
        self._lock = threading.RLock()
        self._stored: Optional[StoredRuntimeSession] = None
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
        self.drain_timeout = float(runtime_config.get("drain-timeout-seconds", 3600))
        self.drain_quiet_window = float(runtime_config.get("drain-quiet-window-seconds", 10))
        self.lease_ttl = float(runtime_config.get("lease-ttl-seconds", 3600))
        self.inventory_ttl = max(1.0, float(runtime_config.get("inventory-ttl-seconds", 30)))
        if min(self.activation_timeout, self.operation_timeout, self.drain_timeout, self.lease_ttl) <= 0:
            raise ValueError("runtime activation, operation, drain, and lease timeouts must be positive")
        if self.drain_quiet_window < 0:
            raise ValueError("runtime drain quiet window must not be negative")
        if self.drain_timeout <= self.lease_ttl + self.drain_quiet_window:
            raise ValueError(
                "runtime drain timeout must exceed lease TTL plus the quiet window"
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
        if self._stored is None:
            self._stored = self.sessions.load()
        return self._stored

    def _reload_for_transaction(self) -> Optional[StoredRuntimeSession]:
        """Read the CAS record at a lifecycle transaction boundary."""
        self._stored = self.sessions.load()
        return self._stored

    def _save(self, session: RuntimeSession) -> RuntimeSession:
        expected = self._stored.resource_version if self._stored is not None else None
        try:
            self._stored = self.sessions.compare_and_swap(session, expected)
        except RuntimeSessionConflict:
            self._stored = self.sessions.load()
            raise
        return self._stored.session

    def current_session(self) -> Optional[RuntimeSession]:
        """Return the process-owned CAS snapshot without a caller refresh switch."""
        stored = self._load()
        return stored.session if stored else None

    def active_directory(self) -> Optional[RuntimeDirectory]:
        with self._lock:
            session = self.current_session()
            if session is not None and session.phase in self._PUBLICATION_PHASES:
                # This is a one-off recovery path after a backend restart or an
                # ambiguous HTTP/CAS result.  It talks only to the Scheduler;
                # steady-state UI reads remain pure ConfigMap-backed state.
                session = self._recover_publication(session)
            if session is None or session.phase != "active" or session.active_directory_revision < 1:
                return None
            return session.directory

    def _remaining_timeout(self, deadline: float, cap: Optional[float] = None) -> float:
        remaining = float(deadline) - self._clock()
        if remaining <= 0:
            raise RuntimeOrchestrationError("managed runtime operation exceeded its deadline")
        return min(remaining, float(cap)) if cap is not None else remaining

    def _refresh_inventory(self) -> Dict[str, Dict[str, Any]]:
        inventory = self.cluster.node_inventory()
        self._inventory_cache = copy.deepcopy(inventory)
        self._inventory_cached_at = self._clock()
        return inventory

    def node_inventory(self) -> Dict[str, Dict[str, Any]]:
        """Return one backend-owned topology snapshot with no caller refresh API."""
        with self._lock:
            if (
                self._inventory_cached_at is None
                or self._clock() - self._inventory_cached_at >= self.inventory_ttl
            ):
                self._refresh_inventory()
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

    def _preflight_nodes(
        self,
        inventory: Mapping[str, Mapping[str, Any]],
        target_nodes: Iterable[str],
        validate_agents: bool = True,
    ) -> None:
        targets = tuple(sorted({str(node) for node in target_nodes if str(node)}))
        missing = [node for node in targets if node not in inventory]
        not_ready = [node for node in targets if node in inventory and not inventory[node].get("ready")]
        if missing or not_ready:
            raise RuntimePreflightError(
                f"runtime target validation failed: missing={missing}, not_ready={not_ready}"
            )
        if not validate_agents:
            return
        report = self.cluster.validate_managed_agents(targets)
        if not report.get("ok"):
            details = []
            for name, state in (report.get("agents") or {}).items():
                if state.get("missing_nodes") or state.get("not_ready_nodes"):
                    details.append(
                        f"{name}(missing={state.get('missing_nodes')}, not_ready={state.get('not_ready_nodes')})"
                    )
            raise RuntimePreflightError(
                "managed RuntimeService prerequisites are not Ready on every target node: "
                + "; ".join(details)
            )

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

    def _ensure_created(self, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            return self.runtime.create(manifest)
        except Exception as exc:
            if _status_code(exc) != 409:
                raise
        name = str((manifest.get("metadata") or {}).get("name") or "")
        existing = self.runtime.get(name)
        if (existing.get("spec") or {}) != (manifest.get("spec") or {}):
            raise RuntimeOrchestrationError(
                f"RuntimeService {name!r} already exists with a different immutable spec"
            )
        return existing

    def _activate(
        self,
        rendered: Sequence[Any],
        timeout_seconds: Optional[float] = None,
    ) -> Tuple[RuntimeUnit, ...]:
        if not rendered:
            return ()
        timeout_seconds = min(
            self.activation_timeout,
            float(timeout_seconds) if timeout_seconds is not None else self.activation_timeout,
        )
        if timeout_seconds <= 0:
            raise RuntimeOrchestrationError("RuntimeService activation deadline was exhausted")
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
            self._ensure_created(item.manifest)
        expectations = {item.unit.runtime_id: item.unit for item in rendered}
        observed = self.runtime.wait_for_conditions(
            expectations,
            condition_types=("Ready", "Activated"),
            timeout_seconds=timeout_seconds,
            label_selector=(
                f"{self.MANAGED_LABEL_SELECTOR},"
                f"{self.INSTALL_LABEL}={install_id}"
            ),
        )
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
    ):
        if scheduler.endpoint is None:
            raise RuntimePublicationError("scheduler RuntimeService has no endpoint")
        kwargs = {}
        if payload is not None:
            kwargs["data"] = {"data": json.dumps(payload, ensure_ascii=False)}
        if params:
            kwargs["params"] = dict(params)
        total_timeout = float(timeout) if timeout is not None else self.operation_timeout
        if total_timeout <= 0:
            raise RuntimeOrchestrationError("scheduler request deadline was exhausted")
        # ``http_request.timeout`` applies to each attempt. Divide the caller's
        # total budget across retries so a three-attempt request cannot silently
        # consume three operation timeouts (plus backoff).
        attempts = 3 if total_timeout >= 1.0 else 1
        backoff_budget = 0.6 if attempts == 3 else 0.0
        per_attempt_timeout = max(0.001, (total_timeout - backoff_budget) / attempts)
        return self._request(
            _endpoint_url(scheduler.endpoint, path),
            method=method,
            timeout=per_attempt_timeout,
            retry=attempts,
            retry_interval=0.2,
            retry_backoff=2,
            **kwargs,
        )

    def _decision(
        self,
        scheduler: RuntimeUnit,
        path: str,
        method: str,
        source_deploy: Sequence[Mapping[str, Any]],
        timeout: Optional[float] = None,
    ) -> Mapping[str, Any]:
        response = self._scheduler_call(
            scheduler, path, method, source_deploy, timeout=timeout,
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
            candidates = {str(node) for node in (source_info.get("node_set") or ())}
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

    def _processor_template(self, service_info: Mapping[str, Any], node: str, inventory):
        template = copy.deepcopy(service_info["service"])
        if inventory[node].get("role") != "edge":
            return template, {}
        labels = inventory[node].get("labels") or {}
        raw_major = labels.get("jetson.nvidia.com/jetpack.major")
        try:
            major = int(raw_major)
        except (TypeError, ValueError):
            major = -1
        if major < 0:
            return template, {}
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
        # preflighted source candidate, so every candidate must be routable in
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
            rendered.append(renderer.render(
                templates["monitor"], RuntimeSlot("monitor", node, position), revision,
                extra_env={"DAYU_RUNTIME_BOOTSTRAP": bootstrap},
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
                logical_template, device_env = self._processor_template(service_info, node, inventory)
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
    ) -> bool:
        timeout = self._remaining_timeout(deadline, self.operation_timeout) if deadline else None
        readback = self._scheduler_call(
            scheduler,
            NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY,
            NetworkAPIMethod.SCHEDULER_GET_RUNTIME_DIRECTORY,
            timeout=timeout,
        )
        return self._directory_matches(readback, directory)

    def _publish_initial(
        self,
        scheduler: RuntimeUnit,
        directory: RuntimeDirectory,
        deadline: Optional[float] = None,
    ) -> None:
        payload = {"expected_revision": 0, "directory": directory.to_dict()}
        publication_error = None
        try:
            timeout = self._remaining_timeout(deadline, self.operation_timeout) if deadline else None
            response = self._scheduler_call(
                scheduler,
                NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY,
                NetworkAPIMethod.SCHEDULER_PUT_RUNTIME_DIRECTORY,
                payload,
                timeout=timeout,
            )
            if isinstance(response, Mapping) and response.get("hash") == directory.content_hash:
                return
        except Exception as exc:
            # A transport failure may occur after Scheduler committed the CAS.
            publication_error = exc
        try:
            if self._publication_readback(scheduler, directory, deadline=deadline):
                return
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
    ) -> None:
        proposal_id = str(uuid.uuid4())
        publication_error = None
        try:
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
            )
            if not isinstance(proposal, Mapping) or proposal.get("proposal_id") != proposal_id:
                raise RuntimePublicationError(
                    "scheduler did not persist the RuntimeDirectory proposal"
                )
            timeout = self._remaining_timeout(deadline, self.operation_timeout) if deadline else None
            response = self._scheduler_call(
                scheduler,
                NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY_PROPOSAL_COMMIT.format(proposal_id=proposal_id),
                NetworkAPIMethod.SCHEDULER_COMMIT_RUNTIME_DIRECTORY,
                {"expected_revision": int(base_revision)},
                timeout=timeout,
            )
            if isinstance(response, Mapping) and response.get("hash") == directory.content_hash:
                return
        except Exception as exc:
            # Proposal/commit are idempotently recoverable through the exact
            # directory hash. This covers a process or transport failure after
            # Scheduler's commit but before backend session CAS.
            publication_error = exc
        try:
            if self._publication_readback(scheduler, directory, deadline=deadline):
                return
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
            retired_keys = {unit.logical_key for unit in session.retired}
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
    ) -> RuntimeSession:
        revision_increment = 1
        if session.phase == "publishing":
            revision_increment = max(0, 2 - session.next_runtime_revision)
        finalized = replace(
            session,
            phase="active",
            next_runtime_revision=session.next_runtime_revision + revision_increment,
            active_directory_revision=directory.revision,
            active=directory.routes,
            pending=(),
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

    def _recover_publication(self, session: RuntimeSession) -> RuntimeSession:
        if session.phase not in self._PUBLICATION_PHASES:
            return session
        directory = self._publication_candidate(session)
        scheduler_units = session.pending if session.phase == "publishing" else session.active
        scheduler = self._scheduler_unit(scheduler_units)
        deadline = self._clock() + self.operation_timeout
        if session.phase == "publishing":
            self._publish_initial(scheduler, directory, deadline=deadline)
        else:
            self._publish_rollout(
                scheduler,
                session.active_directory_revision,
                directory,
                deadline=deadline,
            )
        return self._finalize_publication(session, directory)

    def recover(self) -> Optional[RuntimeSession]:
        """Reconcile the only crash-sensitive Scheduler/ConfigMap boundary.

        Candidate resources and their immutable UIDs were persisted before the
        directory CAS, so recovery needs no Kubernetes discovery or per-Pod
        probing. Old rollout resources remain in ``retired`` until the normal
        drain path can prove their directory revision has no task leases.
        """

        with self._lock:
            stored = self._reload_for_transaction()
            if stored is None:
                return None
            session = stored.session
            if session.phase in self._PUBLICATION_PHASES:
                session = self._recover_publication(session)
            return session

    def install(
        self,
        policy: Mapping[str, Any],
        source_deploy: Sequence[Mapping[str, Any]],
        source_label: str = "",
    ) -> RuntimeDirectory:
        with self._lock:
            stored = self._reload_for_transaction()
            if stored is not None and stored.session.phase in self._PUBLICATION_PHASES:
                self._recover_publication(stored.session)
                stored = self._stored
            if stored is not None:
                raise RuntimeOrchestrationError("a runtime session already exists; uninstall it before installing")
            operation_deadline = self._clock() + self.operation_timeout
            inventory = self._refresh_inventory()
            cloud_node = self._cloud_node(inventory)
            selected_nodes = self._selected_nodes(source_deploy)
            if not selected_nodes:
                raise RuntimePreflightError("at least one source candidate node is required")
            non_edge_sources = sorted(
                node for node in selected_nodes
                if (inventory.get(node) or {}).get("role") != "edge"
            )
            if non_edge_sources:
                raise RuntimePreflightError(
                    f"generator source candidates must be edge nodes: {non_edge_sources}"
                )
            self._preflight_nodes(inventory, set(selected_nodes) | {cloud_node})

            install_id = str(uuid.uuid4())
            operation_id = str(uuid.uuid4())
            revision = 1
            templates, normalized_sources = self._logical_templates(
                self.template_helper, policy, source_deploy,
            )
            renderer = self.template_helper.create_runtime_renderer(install_id)
            scheduler_preview = renderer.render(
                templates["scheduler"], RuntimeSlot("scheduler", cloud_node, "cloud"), revision,
            )
            bootstrap = self._bootstrap(
                install_id, cloud_node, cloud_node, inventory,
                set(selected_nodes) | {cloud_node}, (scheduler_preview.unit,),
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
                )[0]
                source_plan = self._decision(
                    scheduler_unit,
                    NetworkAPIPath.SCHEDULER_SELECT_SOURCE_NODES,
                    NetworkAPIMethod.SCHEDULER_SELECT_SOURCE_NODES,
                    normalized_sources,
                    timeout=self._remaining_timeout(operation_deadline, self.operation_timeout),
                )
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
                )
                deployment = self._deployment(deployment_plan, enriched, cloud_node)
                target_nodes = set(source_selection.values()) | {
                    node for nodes in deployment.values() for node in nodes
                } | {cloud_node}
                # The first preflight covered every source candidate plus the
                # cloud node.  ``_deployment`` rejects any node outside that
                # exact set, so checking managed agents again would only issue
                # two more cluster-wide Pod lists on every install.
                if not target_nodes.issubset(set(selected_nodes) | {cloud_node}):
                    raise RuntimePreflightError(
                        f"scheduler selected targets outside the preflight snapshot: "
                        f"{sorted(target_nodes - (set(selected_nodes) | {cloud_node}))}"
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
                )
                session = self._finalize_publication(session, directory)
                return directory
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
            logical_template, device_env = self._processor_template(
                service_info, slot.target_node, inventory,
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

    def redeploy(self, policy: Mapping[str, Any]) -> bool:
        with self._lock:
            stored = self._reload_for_transaction()
            if stored is not None and stored.session.phase in self._PUBLICATION_PHASES:
                self._recover_publication(stored.session)
                stored = self._stored
            if stored is None or stored.session.phase != "active":
                raise RuntimeOrchestrationError("processor rollout requires an active runtime session")
            session = stored.session
            scheduler = self._scheduler_unit(session.active)
            if session.retired:
                try:
                    self._drain(scheduler, session.active_directory_revision - 1)
                    self._delete_units(session.retired, session.install_id)
                    session = replace(
                        session, retired=(), last_error="", updated_at=_utc_now(),
                    )
                    self._save(session)
                except Exception as exc:
                    self._save(replace(
                        session,
                        phase="active",
                        last_error=f"retirement pending: {exc}",
                        updated_at=_utc_now(),
                    ))
                    raise RuntimeOrchestrationError(
                        "previous RuntimeDirectory retirement is still pending"
                    ) from exc
            operation_deadline = self._clock() + self.operation_timeout
            inventory = self.node_inventory()
            cloud_node = self._cloud_node(inventory)
            raw_plan = self._decision(
                scheduler,
                NetworkAPIPath.SCHEDULER_REDEPLOYMENT,
                NetworkAPIMethod.SCHEDULER_REDEPLOYMENT,
                session.source_deploy,
                timeout=self._remaining_timeout(operation_deadline, self.operation_timeout),
            )
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
            rendered, kept, retired = self._render_processor_candidates(
                session, deployment, inventory, cloud_node, templates,
            )
            if not rendered and not retired:
                return False

            pending_units = tuple(item.unit for item in rendered)
            session = replace(
                session,
                operation_id=str(uuid.uuid4()),
                phase="activating-rollout",
                pending=pending_units,
                retired=retired,
                updated_at=_utc_now(),
            )
            self._save(session)
            try:
                activated = self._activate(
                    rendered,
                    timeout_seconds=self._remaining_timeout(
                        operation_deadline, self.activation_timeout,
                    ),
                )
                candidate_units = tuple(kept) + tuple(activated)
                directory = RuntimeDirectory(
                    install_id=session.install_id,
                    revision=session.active_directory_revision + 1,
                    routes=candidate_units,
                )
                session = replace(
                    session, phase="publishing-rollout", pending=activated, updated_at=_utc_now(),
                )
                self._save(session)
                self._publish_rollout(
                    scheduler,
                    session.active_directory_revision,
                    directory,
                    deadline=operation_deadline,
                )
                old_revision = session.active_directory_revision
                session = self._finalize_publication(session, directory)
                self._drain(scheduler, old_revision)
                self._delete_units(retired, session.install_id)
                self._save(replace(session, retired=(), updated_at=_utc_now()))
                return True
            except Exception as exc:
                LOGGER.exception("managed processor rollout failed")
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
                    elif current is not None and current.phase == "active":
                        # Publication was committed and session CAS succeeded;
                        # only old-revision drain/deletion is pending.
                        self._save(replace(
                            current,
                            phase="active",
                            last_error=str(exc),
                            updated_at=_utc_now(),
                        ))
                except Exception:
                    LOGGER.exception("failed to persist rollout failure state")
                raise

    def _lease_count(self, scheduler: RuntimeUnit, revision: int) -> int:
        response = self._scheduler_call(
            scheduler,
            NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY_TASK_LEASES,
            NetworkAPIMethod.SCHEDULER_COUNT_TASK_LEASES,
            None,
            params={"revision": int(revision)},
            timeout=min(30, self.operation_timeout),
        )
        if not isinstance(response, Mapping):
            raise RuntimeOrchestrationError("scheduler task-lease count is unavailable")
        try:
            return int(response.get("count"))
        except (TypeError, ValueError):
            raise RuntimeOrchestrationError("scheduler task-lease count is invalid")

    def _clear_runtime_directory(self, scheduler: RuntimeUnit, install_id: str) -> None:
        clear_error = None
        try:
            response = self._scheduler_call(
                scheduler,
                NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY,
                NetworkAPIMethod.SCHEDULER_CLEAR_RUNTIME_DIRECTORY,
                {"install_id": str(install_id)},
                timeout=min(30, self.operation_timeout),
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
                timeout=min(30, self.operation_timeout),
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

    def _drain(self, scheduler: RuntimeUnit, revision: int) -> None:
        deadline = self._clock() + self.drain_timeout
        quiet_since = None
        while self._clock() < deadline:
            count = self._lease_count(scheduler, revision)
            if count == 0:
                quiet_since = quiet_since or self._clock()
                if self._clock() - quiet_since >= self.drain_quiet_window:
                    return
            else:
                quiet_since = None
            self._sleep(min(1.0, max(0.05, self.drain_quiet_window / 4)))
        raise RuntimeOrchestrationError(
            f"timed out draining tasks pinned to RuntimeDirectory revision {revision}"
        )

    def _delete_units(self, units: Iterable[RuntimeUnit], install_id: str) -> None:
        identities = {}
        unresolved = set()
        for unit in units:
            uid = unit.runtime_service_uid or (
                unit.endpoint.runtime_service_uid if unit.endpoint else None
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
            selector = (
                f"{self.MANAGED_LABEL_SELECTOR},"
                f"{self.INSTALL_LABEL}={str(install_id)}"
            )
            response = self.runtime.list(label_selector=selector)
            for obj in response.get("items") or ():
                metadata = obj.get("metadata") or {}
                name = str(metadata.get("name") or "")
                if name not in unresolved:
                    continue
                labels = metadata.get("labels") or {}
                uid = str(metadata.get("uid") or "")
                spec_install_id = str((obj.get("spec") or {}).get("installID") or "")
                if (
                    labels.get(self.INSTALL_LABEL) != str(install_id)
                    or spec_install_id != str(install_id)
                    or not uid):
                    raise RuntimeOrchestrationError(
                        f"RuntimeService {name!r} has no exact install ownership identity"
                    )
                identities[name] = uid
            # Missing names are already absent; never downgrade to an
            # unguarded delete when activation crashed before UID persistence.
        self.runtime.delete_many(
            identities,
            timeout_seconds=min(self.activation_timeout, 120),
            label_selector=(
                f"{self.MANAGED_LABEL_SELECTOR},"
                f"{self.INSTALL_LABEL}={str(install_id)}"
            ),
        )

    def uninstall(self) -> None:
        with self._lock:
            stored = self._reload_for_transaction()
            if stored is None:
                return
            session = stored.session
            if session.phase in {"clearing-directory", "finalizing-uninstall"}:
                finalizing_units = tuple({
                                             unit.runtime_id: unit
                                             for unit in (*session.active, *session.pending, *session.retired)
                                         }.values())
                schedulers = tuple(
                    unit for unit in finalizing_units
                    if unit.slot.component == "scheduler"
                )
                if not schedulers:
                    raise RuntimeOrchestrationError(
                        "finalizing uninstall contains no scheduler RuntimeService identity"
                    )
                try:
                    if session.phase == "clearing-directory":
                        self._clear_runtime_directory(schedulers[0], session.install_id)
                        self._delete_units(
                            (
                                unit for unit in finalizing_units
                                if unit.slot.component != "scheduler"
                            ),
                            session.install_id,
                        )
                        session = replace(
                            session,
                            phase="finalizing-uninstall",
                            active=schedulers,
                            pending=(),
                            retired=(),
                            last_error="",
                            updated_at=_utc_now(),
                        )
                        self._save(session)
                    self._delete_units(schedulers, session.install_id)
                    expected = self._stored.resource_version if self._stored else None
                    self.sessions.delete(expected_resource_version=expected)
                    self._stored = None
                    return
                except Exception as exc:
                    LOGGER.exception("managed RuntimeService uninstall finalization failed")
                    self._save(replace(
                        session,
                        phase=session.phase,
                        last_error=str(exc),
                        updated_at=_utc_now(),
                    ))
                    raise
            drain_revisions = set()
            if session.active_directory_revision > 0:
                drain_revisions.add(session.active_directory_revision)
            # A committed rollout can fail after the directory CAS but before
            # old resources drain. Both the current and immediately previous
            # revisions may still own live task leases in that state.
            if session.retired and session.active_directory_revision > 1:
                drain_revisions.add(session.active_directory_revision - 1)
            if session.phase in self._PUBLICATION_PHASES:
                drain_revisions.add(session.active_directory_revision + 1)
            elif session.pending and (
                session.active
                or any(unit.slot.component == "generator" for unit in session.pending)
            ):
                # A pre-publication failed transaction is ambiguous but safe to
                # drain: an unpublished revision has a zero lease count.
                drain_revisions.add(session.active_directory_revision + 1)
            session = replace(
                session,
                operation_id=str(uuid.uuid4()),
                phase="uninstalling",
                updated_at=_utc_now(),
            )
            self._save(session)
            # Failed transactions can contain the same unit in active,
            # pending, and retired.  Build one exact deletion set so cleanup is
            # idempotent and never deletes the scheduler before a pending
            # generator merely because the active directory is still empty.
            all_units_by_id = {}
            for unit in (*session.active, *session.pending, *session.retired):
                existing = all_units_by_id.get(unit.runtime_id)
                if existing is None or (
                    existing.endpoint is None and unit.endpoint is not None):
                    all_units_by_id[unit.runtime_id] = unit
            all_units = tuple(all_units_by_id[name] for name in sorted(all_units_by_id))
            schedulers = tuple(unit for unit in all_units if unit.slot.component == "scheduler")
            if not schedulers:
                raise RuntimeOrchestrationError("runtime session contains no scheduler RuntimeService")
            active_schedulers = tuple(
                unit for unit in session.active if unit.slot.component == "scheduler"
            )
            scheduler = active_schedulers[0] if len(active_schedulers) == 1 else schedulers[0]
            generators = tuple(unit for unit in all_units if unit.slot.component == "generator")
            remaining = tuple(
                unit for unit in all_units
                if unit.slot.component not in {"generator", "scheduler"}
            )
            try:
                # Stop producing new leases before waiting for existing tasks.
                self._delete_units(generators, session.install_id)
                for revision in sorted(drain_revisions):
                    self._drain(scheduler, revision)
                # Persist each irreversible boundary separately.  While the
                # directory is being cleared the Scheduler must remain live;
                # once finalizing-uninstall is stored, retries use Kubernetes
                # deletion only and never call a Scheduler that may be gone.
                session = replace(
                    session,
                    phase="clearing-directory",
                    # Keep every not-yet-deleted UID in the CAS record. A crash
                    # after directory clear can then finish Kubernetes cleanup
                    # without discovery or stale routability.
                    active=tuple(schedulers) + tuple(remaining),
                    pending=(),
                    retired=(),
                    last_error="",
                    updated_at=_utc_now(),
                )
                self._save(session)
                # Clear before deleting any route target. The Scheduler remains
                # live for the clear/readback transaction and is deleted last.
                self._clear_runtime_directory(scheduler, session.install_id)
                self._delete_units(remaining, session.install_id)
                session = replace(
                    session,
                    phase="finalizing-uninstall",
                    active=schedulers,
                    last_error="",
                    updated_at=_utc_now(),
                )
                self._save(session)
                self._delete_units(schedulers, session.install_id)
                expected = self._stored.resource_version if self._stored else None
                self.sessions.delete(expected_resource_version=expected)
                self._stored = None
            except Exception as exc:
                LOGGER.exception("managed RuntimeService uninstall failed")
                self._save(replace(
                    session,
                    phase=(
                        session.phase
                        if session.phase in {"clearing-directory", "finalizing-uninstall"}
                        else "failed"
                    ),
                    last_error=str(exc),
                    updated_at=_utc_now(),
                ))
                raise

    def runtime_metrics(self, logical_service: str = "") -> Dict[str, Dict[str, Any]]:
        directory = self.active_directory()
        if directory is None:
            return {}
        refs = []
        pod_context = {}
        for unit in directory.routes:
            if logical_service and (
                unit.slot.component != "processor"
                or unit.slot.logical_service != str(logical_service)
            ):
                continue
            if unit.pod_name and unit.pod_uid:
                refs.append({"name": unit.pod_name, "uid": unit.pod_uid})
                pod_context[unit.pod_name] = {
                    "runtime_id": unit.runtime_id,
                    "logical_service": unit.slot.logical_service,
                }
        metrics = self.cluster.runtime_metrics(
            refs, node_inventory=self.node_inventory(),
        ) if refs else {}
        for pod_name, metric in metrics.items():
            metric.update(pod_context.get(pod_name, {}))
        return metrics
