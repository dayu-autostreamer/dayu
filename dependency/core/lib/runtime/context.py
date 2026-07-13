"""Process-local runtime bootstrap without Kubernetes discovery."""

import json
import math
import os
import socket
import threading
from typing import Any, Dict, Iterable, List

from .model import RuntimeEndpoint


def _first(mapping, *names, default=None):
    if not isinstance(mapping, dict):
        return default
    for name in names:
        value = mapping.get(name)
        if value is not None:
            return value
    return default


class RuntimeContext:
    """Read-only topology and static endpoint context injected at pod start."""

    STATIC_COMPONENTS = frozenset({
        "backend", "datasource", "distributor", "monitor", "redis", "scheduler",
    })
    _default = None
    _default_lock = threading.Lock()

    def __init__(self, bootstrap=None):
        bootstrap = bootstrap or {}
        if not isinstance(bootstrap, dict):
            raise TypeError("DAYU_RUNTIME_BOOTSTRAP must decode to a JSON object")
        self.bootstrap = dict(bootstrap)
        self.mode = str(_first(
            bootstrap, "mode", default=os.getenv("DAYU_RUNTIME_MODE", "runtime-service")
        ) or "runtime-service")
        self.namespace = str(_first(
            bootstrap, "namespace", default=os.getenv("NAMESPACE", "dayu")
        ) or "dayu")
        self.install_id = str(_first(bootstrap, "install_id", "installID", "installId", default="") or "")
        self.directory_revision = self._as_int(_first(
            bootstrap, "runtime_directory_revision", "runtimeDirectoryRevision", "revision", default=0
        ))
        self.lease_ttl_seconds = self._positive_float(
            _first(
                bootstrap,
                "lease_ttl_seconds",
                "leaseTTLSeconds",
                default=os.getenv("DAYU_RUNTIME_LEASE_TTL_SECONDS", "3600"),
            ),
            field="lease_ttl_seconds",
        )
        self.nodes = self._normalize_nodes(_first(bootstrap, "nodes", "nodeDirectory", default={}))
        self.local_node = str(_first(
            bootstrap, "local_node", "localNode", "node_name", "nodeName",
            default=os.getenv("NODE_NAME") or os.getenv("DAYU_NODE_NAME") or socket.gethostname(),
        ) or "")
        self.cloud_node = str(_first(
            bootstrap, "cloud_node", "cloudNode", default=os.getenv("CLOUD_NODE_NAME", "")
        ) or self._node_for_role("cloud") or "")
        self._static_endpoints = self._normalize_endpoints(_first(
            bootstrap, "endpoints", "static_endpoints", "staticEndpoints", default={}
        ))
        self._static_endpoints.extend(self._environment_endpoints())

    @property
    def managed(self):
        return True

    @staticmethod
    def _as_int(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _positive_float(value, field):
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("{} must be numeric".format(field)) from exc
        if not math.isfinite(parsed) or parsed <= 0:
            raise ValueError("{} must be a finite positive number".format(field))
        return parsed

    @classmethod
    def from_env(cls):
        raw = os.getenv("DAYU_RUNTIME_BOOTSTRAP", "").strip()
        if not raw:
            return cls({})
        try:
            bootstrap = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("DAYU_RUNTIME_BOOTSTRAP is not valid JSON: {}".format(exc)) from exc
        return cls(bootstrap)

    @classmethod
    def get_default(cls, reload=False):
        if reload or cls._default is None:
            with cls._default_lock:
                if reload or cls._default is None:
                    cls._default = cls.from_env()
        return cls._default

    @classmethod
    def reset_default(cls):
        with cls._default_lock:
            cls._default = None

    @staticmethod
    def _normalize_nodes(value):
        result = {}
        if isinstance(value, dict):
            iterable = value.items()
        elif isinstance(value, list):
            iterable = ((
                _first(item, "name", "node", "node_name", "nodeName", default=""), item
            ) for item in value if isinstance(item, dict))
        else:
            iterable = []
        for name, details in iterable:
            if not name:
                continue
            if isinstance(details, str):
                details = {"address": details}
            result[str(name)] = dict(details or {})
        return result

    def _node_for_role(self, role):
        for name, details in self.nodes.items():
            if str(_first(details, "role", "node_role", "nodeRole", default="")).lower() == role:
                return name
        return ""

    def node_address(self, node):
        details = self.nodes.get(node, {})
        return str(_first(details, "address", "ip", "host", "fqdn", default=node) or node)

    def node_role(self, node):
        return str(_first(self.nodes.get(node, {}), "role", "node_role", "nodeRole", default="") or "")

    def edge_nodes(self):
        return sorted(name for name in self.nodes if self.node_role(name).lower() == "edge")

    @staticmethod
    def _looks_like_endpoint(value):
        if isinstance(value, str):
            return True
        if not isinstance(value, dict):
            return False
        endpoint_fields = {
            "url", "endpoint", "address", "fqdn", "host", "hostname", "ip", "port",
            "runtime_id", "runtimeID", "runtime_service_uid", "runtimeServiceUID",
        }
        return bool(endpoint_fields.intersection(value.keys()))

    @classmethod
    def _walk_endpoints(cls, value, hints=None):
        hints = dict(hints or {})
        if value is None:
            return
        if isinstance(value, list):
            for item in value:
                for endpoint in cls._walk_endpoints(item, hints):
                    yield endpoint
            return
        if cls._looks_like_endpoint(value):
            endpoint = RuntimeEndpoint.from_value(
                value,
                component=hints.get("component"),
                target_node=hints.get("target_node"),
                logical_service=hints.get("logical_service"),
            )
            if endpoint:
                yield endpoint
            return
        if not isinstance(value, dict):
            return

        for key, child in value.items():
            child_hints = dict(hints)
            parts = str(key).split("|")
            if len(parts) >= 1 and parts[0] in {
                "backend", "controller", "datasource", "distributor", "monitor", "processor", "redis", "scheduler"
            }:
                child_hints["component"] = parts[0]
                if len(parts) > 1 and parts[1]:
                    child_hints["logical_service"] = parts[1]
                if len(parts) > 2 and parts[2]:
                    child_hints["target_node"] = parts[2]
            elif "component" not in child_hints:
                child_hints["component"] = str(key)
            elif child_hints.get("component") == "processor" and "logical_service" not in child_hints:
                child_hints["logical_service"] = str(key)
            elif "target_node" not in child_hints:
                child_hints["target_node"] = str(key)
            for endpoint in cls._walk_endpoints(child, child_hints):
                yield endpoint

    @classmethod
    def _normalize_endpoints(cls, value):
        return list(cls._walk_endpoints(value))

    @classmethod
    def _environment_endpoints(cls):
        endpoints = []
        for component in cls.STATIC_COMPONENTS:
            value = os.getenv("DAYU_{}_ENDPOINT".format(component.upper())) or os.getenv(
                "{}_ENDPOINT".format(component.upper())
            )
            if value:
                endpoints.append(RuntimeEndpoint.from_value(value, component=component))
        return endpoints

    def list_static_endpoints(self, component=None, target_node=None, logical_service=None):
        return [
            endpoint for endpoint in self._static_endpoints
            if endpoint.matches(component, target_node, logical_service)
        ]

    def resolve_static_endpoint(self, component, target_node=None, logical_service=None, required=True):
        if component not in self.STATIC_COMPONENTS:
            raise ValueError(
                "component {!r} is task-routed and cannot be resolved from bootstrap".format(component)
            )
        matches = self.list_static_endpoints(component, target_node, logical_service)
        if not matches and logical_service:
            # Infrastructure endpoint maps sometimes omit logical_service.
            matches = self.list_static_endpoints(component, target_node, None)
        if not matches and target_node:
            # A singleton infrastructure service may omit its target node.
            matches = self.list_static_endpoints(component, None, logical_service)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                "ambiguous static runtime endpoint: component={!r}, service={!r}, node={!r}".format(
                    component, logical_service, target_node
                )
            )
        if required:
            raise LookupError(
                "runtime endpoint missing from DAYU_RUNTIME_BOOTSTRAP: component={!r}, service={!r}, node={!r}".format(
                    component, logical_service, target_node
                )
            )
        return None
