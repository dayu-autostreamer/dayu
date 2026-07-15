"""Backend-only Kubernetes access shared by runtime orchestration operations."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from kubernetes import client, config


def _get(value: Any, *path: str, default=None):
    current = value
    for key in path:
        if isinstance(current, Mapping):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
        if current is None:
            return default
    return current


def _items(response: Any):
    return _get(response, "items", default=[]) or []


def _pod_ready(pod: Any) -> bool:
    if _get(pod, "metadata", "deletion_timestamp") or _get(pod, "metadata", "deletionTimestamp"):
        return False
    if _get(pod, "status", "phase") != "Running":
        return False
    conditions = _get(pod, "status", "conditions", default=[]) or []
    ready_condition = any(
        _get(condition, "type") == "Ready" and str(_get(condition, "status")).lower() == "true"
        for condition in conditions
    )
    statuses = _get(pod, "status", "container_statuses", default=None)
    if statuses is None:
        statuses = _get(pod, "status", "containerStatuses", default=[]) or []
    return ready_condition and bool(statuses) and all(bool(_get(status, "ready")) for status in statuses)


def _node_record(node: Any) -> Dict[str, Any]:
    name = str(_get(node, "metadata", "name", default="") or "")
    labels = dict(_get(node, "metadata", "labels", default={}) or {})
    role = "worker"
    if (
        "node-role.kubernetes.io/control-plane" in labels
        or "node-role.kubernetes.io/master" in labels):
        role = "cloud"
    elif "node-role.kubernetes.io/edge" in labels:
        role = "edge"
    address = ""
    for item in _get(node, "status", "addresses", default=[]) or []:
        if _get(item, "type") == "InternalIP":
            address = str(_get(item, "address", default="") or "")
            break
    ready = False
    for condition in _get(node, "status", "conditions", default=[]) or []:
        if _get(condition, "type") == "Ready":
            ready = str(_get(condition, "status")).lower() == "true"
            break
    return {
        "name": name,
        "role": role,
        "address": address,
        "labels": labels,
        "ready": ready,
        "capacity": dict(_get(node, "status", "capacity", default={}) or {}),
        "allocatable": dict(_get(node, "status", "allocatable", default={}) or {}),
    }


class ClusterClient:
    """One ApiClient with reusable typed clients for backend control-plane work."""

    def __init__(
        self,
        namespace: str,
        api_client=None,
        core_api=None,
        custom_api=None,
        load_config: bool = True,
        sedna_lc_selector: str = "sedna=lc",
        edgemesh_selector: str = "k8s-app=kubeedge,kubeedge=edgemesh-agent",
        runtime_selector: str = "dayu.io/mesh-managed=true",
        request_timeout_seconds: float = 10,
    ):
        self.namespace = str(namespace or "").strip()
        if not self.namespace:
            raise ValueError("namespace must be non-empty")
        if api_client is None and (core_api is None or custom_api is None):
            if load_config:
                config.load_incluster_config()
            api_client = client.ApiClient()
        self.api_client = api_client
        self.core = core_api or client.CoreV1Api(api_client)
        self.custom = custom_api or client.CustomObjectsApi(api_client)
        self.sedna_lc_selector = sedna_lc_selector
        self.edgemesh_selector = edgemesh_selector
        self.runtime_selector = str(runtime_selector or "").strip()
        if not self.runtime_selector:
            raise ValueError("runtime_selector must be non-empty")
        self.request_timeout = float(request_timeout_seconds)
        if self.request_timeout <= 0:
            raise ValueError("request_timeout_seconds must be positive")

    def _bounded_timeout(self, request_timeout_seconds: Optional[float] = None) -> float:
        timeout = self.request_timeout if request_timeout_seconds is None else float(
            request_timeout_seconds
        )
        if timeout <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        return min(self.request_timeout, timeout)

    def node_inventory(
        self, request_timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Return a complete node inventory from exactly one list call."""

        request_timeout = self._bounded_timeout(request_timeout_seconds)
        result = {}
        for node in _items(self.core.list_node(_request_timeout=request_timeout)):
            record = _node_record(node)
            if record["name"]:
                result[record["name"]] = record
        return result

    def _agent_pods(self, selector: str):
        return _items(self.core.list_pod_for_all_namespaces(
            label_selector=selector,
            _request_timeout=self.request_timeout,
        ))

    def validate_managed_agents(self, target_nodes: Iterable[str]) -> Dict[str, Any]:
        """Check that a Ready Sedna LC and EdgeMesh agent cover every target node."""

        target_nodes = tuple(sorted({str(node) for node in target_nodes if str(node)}))
        if not target_nodes:
            raise ValueError("target_nodes must be non-empty")

        report = {"target_nodes": list(target_nodes), "agents": {}, "ok": True}
        for agent_name, selector in (
                ("sedna_lc", self.sedna_lc_selector),
                ("edgemesh_agent", self.edgemesh_selector)):
            present = set()
            ready = set()
            pod_names: Dict[str, list] = {}
            for pod in self._agent_pods(selector):
                node_name = str(_get(pod, "spec", "node_name", default="") or _get(
                    pod, "spec", "nodeName", default="") or "")
                if not node_name:
                    continue
                present.add(node_name)
                pod_names.setdefault(node_name, []).append(str(_get(pod, "metadata", "name", default="") or ""))
                if _pod_ready(pod):
                    ready.add(node_name)
            missing = sorted(set(target_nodes) - present)
            not_ready = sorted((set(target_nodes) & present) - ready)
            report["agents"][agent_name] = {
                "selector": selector,
                "missing_nodes": missing,
                "not_ready_nodes": not_ready,
                "ready_nodes": sorted(set(target_nodes) & ready),
                "pods": {node: sorted(names) for node, names in sorted(pod_names.items()) if node in target_nodes},
            }
            if missing or not_ready:
                report["ok"] = False
        return report

    def runtime_metrics(
        self,
        pod_refs: Sequence[Mapping[str, Any]],
        namespace: Optional[str] = None,
        node_inventory: Optional[Mapping[str, Mapping[str, Any]]] = None,
        request_timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Join one Pod list and one metrics list by exact Pod UID.

        Callers may provide the backend-owned topology snapshot to avoid a Node
        list on this UI hot path. A Node list is only a standalone-call fallback.
        """

        namespace = str(namespace or self.namespace)
        request_timeout = self._bounded_timeout(request_timeout_seconds)
        normalized_refs = {}
        for ref in pod_refs or ():
            ref_namespace = str(ref.get("namespace") or namespace)
            if ref_namespace != namespace:
                raise ValueError("runtime_metrics accepts Pod refs from exactly one namespace")
            name = str(ref.get("name") or "")
            uid = str(ref.get("uid") or "")
            if not name or not uid:
                raise ValueError("every Pod ref requires non-empty name and uid")
            if name in normalized_refs and normalized_refs[name] != uid:
                raise ValueError(f"conflicting UIDs supplied for Pod {name!r}")
            normalized_refs[name] = uid
        if not normalized_refs:
            return {}

        pods_response = self.core.list_namespaced_pod(
            namespace=namespace,
            label_selector=self.runtime_selector,
            _request_timeout=request_timeout,
        )
        try:
            metrics_response = self.custom.list_namespaced_custom_object(
                group="metrics.k8s.io",
                version="v1beta1",
                namespace=namespace,
                plural="pods",
                label_selector=self.runtime_selector,
                _request_timeout=request_timeout,
            )
        except Exception:
            # Metrics Server is optional. Pod readiness and exact UID identity
            # remain available even when live CPU/memory samples are not.
            metrics_response = {"items": []}
        nodes = {
            str(name): dict(record)
            for name, record in (node_inventory or {}).items()
        }
        if node_inventory is None:
            for node in _items(self.core.list_node(_request_timeout=request_timeout)):
                record = _node_record(node)
                if record["name"]:
                    nodes[record["name"]] = record
        metrics = {
            str(_get(item, "metadata", "name", default="") or ""): item
            for item in _items(metrics_response)
        }

        result = {}
        for pod in _items(pods_response):
            name = str(_get(pod, "metadata", "name", default="") or "")
            if name not in normalized_refs:
                continue
            uid = str(_get(pod, "metadata", "uid", default="") or "")
            if uid != normalized_refs[name]:
                continue
            node_name = str(_get(pod, "spec", "node_name", default="") or _get(
                pod, "spec", "nodeName", default="") or "")
            container_resources = {}
            for container in _get(pod, "spec", "containers", default=[]) or []:
                container_name = str(_get(container, "name", default="") or "")
                resources = _get(container, "resources", default={}) or {}
                container_resources[container_name] = {
                    "requests": dict(_get(resources, "requests", default={}) or {}),
                    "limits": dict(_get(resources, "limits", default={}) or {}),
                }
            metric = metrics.get(name) or {}
            usage = {
                str(_get(container, "name", default="") or ""): dict(_get(container, "usage", default={}) or {})
                for container in _get(metric, "containers", default=[]) or []
            }
            result[name] = {
                "name": name,
                "uid": uid,
                "namespace": namespace,
                "node": node_name,
                "phase": str(_get(pod, "status", "phase", default="") or ""),
                "ready": _pod_ready(pod),
                "pod_ip": str(_get(pod, "status", "pod_ip", default="") or _get(
                    pod, "status", "podIP", default="") or ""),
                "created_at": str(_get(pod, "metadata", "creation_timestamp", default="") or _get(
                    pod, "metadata", "creationTimestamp", default="") or ""),
                "resources": container_resources,
                "usage": usage,
                "node_info": nodes.get(node_name),
            }
        return result
