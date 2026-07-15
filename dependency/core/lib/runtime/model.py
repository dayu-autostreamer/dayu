"""Pure-Python runtime endpoint model.

The runtime data plane deliberately does not import the Kubernetes client.  An
endpoint is an immutable routing fact produced by the Dayu control plane and
carried either in ``DAYU_RUNTIME_BOOTSTRAP`` or in a task route snapshot.
"""

from dataclasses import dataclass, field
from typing import Any, Dict
from urllib.parse import urlsplit

from core.lib.network import connection_host


def _first(mapping: Dict[str, Any], *names: str, default=None):
    for name in names:
        value = mapping.get(name)
        if value is not None:
            return value
    return default


def _as_int(value, default=0):
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class RuntimeEndpoint:
    """One exact address plus the identities that make it safe to use."""

    component: str = ""
    target_node: str = ""
    logical_service: str = ""
    runtime_id: str = ""
    fqdn: str = ""
    port: int = 0
    protocol: str = "http"
    base_path: str = ""
    runtime_service_uid: str = ""
    service_uid: str = ""
    endpoint_pod_uid: str = ""
    deployment_revision: int = 0
    install_id: str = ""
    extra: Dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_value(cls, value, component=None, target_node=None, logical_service=None):
        """Normalize a URL or a route dictionary into a RuntimeEndpoint.

        Both Python-style snake_case and Kubernetes/Go-style camelCase fields
        are accepted at the boundary.  ``None`` means no endpoint; malformed
        endpoint types fail immediately instead of triggering discovery.
        """
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            value = {"url": value}
        if not isinstance(value, dict):
            raise TypeError("runtime endpoint must be a URL string or mapping")

        raw = dict(value)
        slot = raw.get("slot") if isinstance(raw.get("slot"), dict) else {}
        endpoint = raw.get("endpoint") if isinstance(raw.get("endpoint"), dict) else {}
        address = _first(endpoint, "url", "address", "base_url", "baseURL", default=_first(
            raw, "url", "endpoint", "address", "base_url", "baseURL"
        ))
        parsed = urlsplit(address if isinstance(address, str) and "://" in address else "")

        fqdn = _first(
            endpoint, "dns_name", "dnsName", "fqdn", "host", "hostname", "ip",
            default=_first(raw, "dns_name", "dnsName", "fqdn", "host", "hostname", "ip", default=""),
        ) or ""
        protocol = _first(endpoint, "protocol", "scheme", default=_first(raw, "protocol", "scheme", default="")) or ""
        port = _as_int(_first(
            endpoint, "port", "service_port", "servicePort",
            default=_first(raw, "port", "service_port", "servicePort"),
        ), default=0)
        base_path = _first(
            endpoint, "base_path", "basePath", "path",
            default=_first(raw, "base_path", "basePath", "path", default=""),
        ) or ""

        if parsed.scheme:
            protocol = protocol or parsed.scheme
            fqdn = fqdn or (parsed.hostname or "")
            port = port or (parsed.port or 0)
            base_path = base_path or parsed.path
        elif isinstance(address, str) and address:
            # A bare address may be either host[:port] or a DNS name.
            parsed_bare = urlsplit("//" + address)
            fqdn = fqdn or (parsed_bare.hostname or address)
            port = port or (parsed_bare.port or 0)

        known = {
            "slot", "component", "componentName", "target_node", "targetNode", "node", "node_name", "nodeName",
            "logical_service", "logicalService", "service_name", "serviceName", "runtime_id", "runtimeID",
            "runtimeId", "name", "fqdn", "host", "hostname", "ip", "port", "service_port", "servicePort",
            "protocol", "scheme", "base_path", "basePath", "path", "url", "endpoint", "address", "base_url",
            "baseURL", "runtime_service_uid", "runtimeServiceUID", "service_uid", "serviceUID",
            "endpoint_pod_uid", "endpointPodUID", "pod_uid", "podUID", "deployment_revision",
            "deploymentRevision", "runtime_revision", "runtimeRevision", "revision", "install_id", "installID",
            "installId",
        }
        extra = {key: item for key, item in raw.items() if key not in known}

        return cls(
            component=str(_first(
                raw, "component", "componentName", default=_first(slot, "component", "componentName", default=component or "")
            ) or ""),
            target_node=str(_first(
                raw, "target_node", "targetNode", "node", "node_name", "nodeName",
                default=_first(
                    slot, "target_node", "targetNode", "node", "node_name", "nodeName", default=target_node or ""
                )
            ) or ""),
            logical_service=str(_first(
                raw, "logical_service", "logicalService", "service_name", "serviceName",
                default=_first(
                    slot, "logical_service", "logicalService", "service_name", "serviceName",
                    default=logical_service or ""
                )
            ) or ""),
            runtime_id=str(_first(raw, "runtime_id", "runtimeID", "runtimeId", "name", default="") or ""),
            fqdn=str(fqdn),
            port=port,
            protocol=str(protocol or "http"),
            base_path=str(base_path),
            runtime_service_uid=str(_first(
                endpoint, "runtime_service_uid", "runtimeServiceUID",
                default=_first(raw, "runtime_service_uid", "runtimeServiceUID", default="")
            ) or ""),
            service_uid=str(_first(
                endpoint, "service_uid", "serviceUID", default=_first(raw, "service_uid", "serviceUID", default="")
            ) or ""),
            endpoint_pod_uid=str(_first(
                endpoint, "endpoint_pod_uid", "endpointPodUID", "pod_uid", "podUID",
                default=_first(raw, "endpoint_pod_uid", "endpointPodUID", "pod_uid", "podUID", default="")
            ) or ""),
            deployment_revision=_as_int(_first(
                raw, "deployment_revision", "deploymentRevision", "runtime_revision", "runtimeRevision", "revision"
            ), default=0),
            install_id=str(_first(raw, "install_id", "installID", "installId", default="") or ""),
            extra=extra,
        )

    @property
    def connection_host(self):
        return connection_host(self.fqdn)

    @property
    def base_url(self):
        if not self.fqdn:
            raise ValueError("runtime endpoint has no fqdn/host")
        port = ":{}".format(self.port) if self.port else ""
        path = "/{}".format(self.base_path.strip("/")) if self.base_path.strip("/") else ""
        return "{}://{}{}{}".format(self.protocol or "http", self.connection_host, port, path)

    def url(self, path=None):
        if not path:
            return self.base_url
        return "{}/{}".format(self.base_url.rstrip("/"), str(path).lstrip("/"))

    def to_dict(self):
        payload = dict(self.extra)
        payload.update({
            "component": self.component,
            "target_node": self.target_node,
            "logical_service": self.logical_service,
            "runtime_id": self.runtime_id,
            "fqdn": self.fqdn,
            "port": self.port,
            "protocol": self.protocol,
            "runtime_service_uid": self.runtime_service_uid,
            "service_uid": self.service_uid,
            "endpoint_pod_uid": self.endpoint_pod_uid,
            "deployment_revision": self.deployment_revision,
            "install_id": self.install_id,
        })
        if self.base_path:
            payload["base_path"] = self.base_path
        return payload

    def matches(self, component=None, target_node=None, logical_service=None):
        return (
            (not component or self.component == component)
            and (not target_node or self.target_node == target_node)
            and (not logical_service or self.logical_service == logical_service)
        )

    def validate_exact(self):
        missing = []
        for field_name in (
            "component", "target_node", "runtime_id", "fqdn", "port",
            "runtime_service_uid", "service_uid", "endpoint_pod_uid", "deployment_revision",
        ):
            if not getattr(self, field_name):
                missing.append(field_name)
        if self.component == "processor" and not self.logical_service:
            missing.append("logical_service")
        if missing:
            raise ValueError(
                "exact runtime route {!r} has incomplete identity: {}".format(
                    self.runtime_id or "<unknown>", ", ".join(missing)
                )
            )
        if not 1 <= self.port <= 65535:
            raise ValueError("exact runtime route port must be between 1 and 65535")
        return self
