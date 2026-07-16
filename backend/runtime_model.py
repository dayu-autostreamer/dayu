"""Canonical models shared by the Dayu managed-runtime control plane.

The objects in this module deliberately contain no Kubernetes clients.  They
describe Dayu's committed runtime directory and can therefore be serialized,
hashed and exchanged with runtime processes without exposing Kubernetes API
objects to those processes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from core.lib.network import connection_host

_DNS_1035_RE = re.compile(r"^[a-z]([-a-z0-9]*[a-z0-9])?$")
_LABEL_VALUE_RE = re.compile(r"^(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?$")
_VALID_POSITIONS = frozenset({"cloud", "edge"})


def canonical_json(value: Any) -> str:
    """Return the stable JSON representation used for hashes and CAS records."""

    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_text(name: str, value: Any, *, max_length: int = 253) -> str:
    value = str(value or "").strip()
    if not value:
        raise ValueError(f"{name} must be non-empty")
    if len(value) > max_length:
        raise ValueError(f"{name} must not exceed {max_length} characters")
    return value


def _optional_text(value: Any, *, max_length: int = 253) -> str:
    value = str(value or "").strip()
    if len(value) > max_length:
        raise ValueError(f"value must not exceed {max_length} characters")
    return value


def _require_timestamp(name: str, value: Any) -> str:
    value = _require_text(name, value, max_length=128)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return value


def _dns_fragment(value: str) -> str:
    value = re.sub(r"[^a-z0-9-]+", "-", str(value).lower())
    value = re.sub(r"-+", "-", value).strip("-")
    if not value or not value[0].isalpha():
        value = f"runtime-{value}" if value else "runtime"
    return value


@dataclass(frozen=True)
class RuntimeSlot:
    """Stable logical identity of one runtime worker, independent of revision."""

    component: str
    target_node: str
    position: str
    logical_service: str = ""
    source_id: str = ""

    def __post_init__(self):
        object.__setattr__(self, "component", _require_text("component", self.component, max_length=63))
        object.__setattr__(self, "target_node", _require_text("target_node", self.target_node))
        position = str(self.position or "").strip().lower()
        if position not in _VALID_POSITIONS:
            raise ValueError(f"position must be one of {sorted(_VALID_POSITIONS)}, got {self.position!r}")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "logical_service", _optional_text(self.logical_service))
        object.__setattr__(self, "source_id", _optional_text(self.source_id, max_length=63))

        if self.component == "processor" and not self.logical_service:
            raise ValueError("processor slots require logical_service")
        if self.component == "generator" and not self.source_id:
            raise ValueError("generator slots require source_id")

    @property
    def logical_key(self) -> str:
        return "/".join((
            self.component,
            self.logical_service,
            self.source_id,
            self.position,
            self.target_node,
        ))

    def to_dict(self) -> Dict[str, str]:
        result = {
            "component": self.component,
            "target_node": self.target_node,
            "position": self.position,
        }
        if self.logical_service:
            result["logical_service"] = self.logical_service
        if self.source_id:
            result["source_id"] = self.source_id
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeSlot":
        return cls(
            component=data.get("component", ""),
            target_node=data.get("target_node", data.get("targetNode", "")),
            position=data.get("position", ""),
            logical_service=data.get("logical_service", data.get("logicalService", "")),
            source_id=data.get("source_id", data.get("sourceID", "")),
        )

    def runtime_name(self, revision: int, install_id: str) -> str:
        """Build an install- and revision-scoped DNS-1035 resource name.

        A digest of the unsanitized logical key prevents aliases such as
        ``edge_x1`` and ``edge-x1`` from collapsing to the same Kubernetes name.
        A separate install digest prevents a new installation from colliding
        with dependents still being removed by Background garbage collection.
        """

        revision = int(revision)
        if revision < 1:
            raise ValueError("revision must be positive")
        install_id = _require_text("install_id", install_id, max_length=256)

        parts = [self.component]
        if self.logical_service:
            parts.append(self.logical_service)
        if self.source_id:
            parts.extend(("source", self.source_id))
        parts.append(self.target_node)
        readable = _dns_fragment("-".join(parts))
        slot_digest = canonical_hash(self.to_dict())[:10]
        install_digest = canonical_hash({"install_id": install_id})[:8]
        suffix = f"-{slot_digest}-{install_digest}-r{revision}"
        readable = readable[: 63 - len(suffix)].rstrip("-")
        if not readable:
            readable = "runtime"
        name = f"{readable}{suffix}"
        if len(name) > 63 or not _DNS_1035_RE.fullmatch(name):
            raise ValueError(f"generated RuntimeService name is not DNS-1035: {name!r}")
        return name


@dataclass(frozen=True)
class RuntimeEndpoint:
    dns_name: str
    port: int
    runtime_service_uid: str = ""
    service_uid: str = ""
    pod_uid: str = ""

    def __post_init__(self):
        object.__setattr__(self, "dns_name", _require_text("dns_name", self.dns_name))
        port = int(self.port)
        if port < 1 or port > 65535:
            raise ValueError("port must be in range 1..65535")
        object.__setattr__(self, "port", port)
        for name in ("runtime_service_uid", "service_uid", "pod_uid"):
            object.__setattr__(self, name, str(getattr(self, name) or ""))

    @property
    def connection_host(self) -> str:
        return connection_host(self.dns_name)

    @property
    def url_authority(self) -> str:
        return f"{self.connection_host}:{self.port}"

    def to_dict(self) -> Dict[str, Any]:
        result = {"dns_name": self.dns_name, "port": self.port}
        for key in ("runtime_service_uid", "service_uid", "pod_uid"):
            value = getattr(self, key)
            if value:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeEndpoint":
        return cls(
            dns_name=data.get("dns_name", data.get("dnsName", "")),
            port=data.get("port", 0),
            runtime_service_uid=data.get("runtime_service_uid", data.get("runtimeServiceUID", "")),
            service_uid=data.get("service_uid", data.get("serviceUID", "")),
            pod_uid=data.get("pod_uid", data.get("podUID", "")),
        )


@dataclass(frozen=True)
class RuntimeUnit:
    slot: RuntimeSlot
    runtime_id: str
    runtime_revision: int
    spec_hash: str
    endpoint: Optional[RuntimeEndpoint] = None
    rollout_hash: str = ""
    runtime_service_uid: str = ""
    pod_name: str = ""
    pod_uid: str = ""

    def __post_init__(self):
        runtime_id = _require_text("runtime_id", self.runtime_id, max_length=63)
        if not _DNS_1035_RE.fullmatch(runtime_id):
            raise ValueError(f"runtime_id is not a DNS-1035 label: {runtime_id!r}")
        object.__setattr__(self, "runtime_id", runtime_id)
        revision = int(self.runtime_revision)
        if revision < 1:
            raise ValueError("runtime_revision must be positive")
        object.__setattr__(self, "runtime_revision", revision)
        spec_hash = str(self.spec_hash or "").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", spec_hash):
            raise ValueError("spec_hash must be a lowercase SHA-256 hex digest")
        object.__setattr__(self, "spec_hash", spec_hash)
        rollout_hash = str(self.rollout_hash or "").lower()
        if rollout_hash and not re.fullmatch(r"[0-9a-f]{64}", rollout_hash):
            raise ValueError("rollout_hash must be empty or a lowercase SHA-256 hex digest")
        object.__setattr__(self, "rollout_hash", rollout_hash)
        object.__setattr__(self, "runtime_service_uid", str(self.runtime_service_uid or ""))
        object.__setattr__(self, "pod_name", str(self.pod_name or ""))
        object.__setattr__(self, "pod_uid", str(self.pod_uid or ""))

    @property
    def logical_key(self) -> str:
        return self.slot.logical_key

    def with_observed_spec_hash(self, observed_spec_hash: str) -> "RuntimeUnit":
        """Return the committed unit using Sedna's authoritative spec hash.

        A renderer may populate ``spec_hash`` with a deterministic Dayu-side
        transaction fingerprint before the object exists.  Once Sedna has
        reconciled the RuntimeService, only ``status.observedSpecHash`` is
        authoritative; callers must replace the provisional value through this
        method instead of attempting to reproduce Go's struct serialization.
        """

        return RuntimeUnit(
            slot=self.slot,
            runtime_id=self.runtime_id,
            runtime_revision=self.runtime_revision,
            spec_hash=observed_spec_hash,
            endpoint=self.endpoint,
            rollout_hash=self.rollout_hash,
            runtime_service_uid=self.runtime_service_uid,
            pod_name=self.pod_name,
            pod_uid=self.pod_uid,
        )

    def to_dict(self) -> Dict[str, Any]:
        result = {
            **self.slot.to_dict(),
            "runtime_id": self.runtime_id,
            "runtime_revision": self.runtime_revision,
            "spec_hash": self.spec_hash,
        }
        if self.endpoint is not None:
            result.update(self.endpoint.to_dict())
        return result

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize control-plane ownership fields without publishing them as routes."""
        result = self.to_dict()
        if self.rollout_hash:
            result["rollout_hash"] = self.rollout_hash
        resource_identity = {
            key: value for key, value in {
                "runtime_service_uid": self.runtime_service_uid,
                "pod_name": self.pod_name,
                "pod_uid": self.pod_uid,
            }.items() if value
        }
        if resource_identity:
            result["resource_identity"] = resource_identity
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeUnit":
        slot_data = data.get("slot") or data
        endpoint = data.get("endpoint")
        if endpoint is None and (data.get("dns_name") or data.get("dnsName")):
            endpoint = data
        resource_identity = data.get("resource_identity", data.get("resourceIdentity", {})) or {}
        return cls(
            slot=RuntimeSlot.from_dict(slot_data),
            runtime_id=data.get("runtime_id", data.get("runtimeID", "")),
            runtime_revision=data.get("runtime_revision", data.get("runtimeRevision", 0)),
            spec_hash=data.get("spec_hash", data.get("specHash", "")),
            endpoint=RuntimeEndpoint.from_dict(endpoint) if endpoint else None,
            rollout_hash=data.get("rollout_hash", data.get("rolloutHash", "")),
            runtime_service_uid=resource_identity.get(
                "runtime_service_uid", resource_identity.get("runtimeServiceUID", "")
            ),
            pod_name=resource_identity.get("pod_name", resource_identity.get("podName", "")),
            pod_uid=resource_identity.get("pod_uid", resource_identity.get("podUID", "")),
        )


@dataclass(frozen=True)
class RuntimeCleanupRef:
    """Compact exact ownership retained only for asynchronous deletion."""

    runtime_id: str
    runtime_service_uid: str = ""

    def __post_init__(self):
        runtime_id = _require_text("runtime_id", self.runtime_id, max_length=63)
        if not _DNS_1035_RE.fullmatch(runtime_id):
            raise ValueError(f"runtime_id is not a DNS-1035 label: {runtime_id!r}")
        object.__setattr__(self, "runtime_id", runtime_id)
        object.__setattr__(
            self,
            "runtime_service_uid",
            str(self.runtime_service_uid or ""),
        )

    @classmethod
    def from_unit(cls, unit: RuntimeUnit) -> "RuntimeCleanupRef":
        uid = unit.runtime_service_uid or (
            unit.endpoint.runtime_service_uid if unit.endpoint else ""
        )
        return cls(unit.runtime_id, uid)

    def to_dict(self) -> Dict[str, str]:
        value = {"runtime_id": self.runtime_id}
        if self.runtime_service_uid:
            value["runtime_service_uid"] = self.runtime_service_uid
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeCleanupRef":
        resource_identity = data.get(
            "resource_identity",
            data.get("resourceIdentity", {}),
        ) or {}
        return cls(
            runtime_id=data.get("runtime_id", data.get("runtimeID", "")),
            runtime_service_uid=(
                data.get("runtime_service_uid")
                or data.get("runtimeServiceUID")
                or resource_identity.get("runtime_service_uid")
                or resource_identity.get("runtimeServiceUID")
                or ""
            ),
        )


def _normalize_routes(routes: Iterable[RuntimeUnit]) -> Tuple[RuntimeUnit, ...]:
    by_key: Dict[str, RuntimeUnit] = {}
    runtime_ids = set()
    for unit in routes:
        if not isinstance(unit, RuntimeUnit):
            unit = RuntimeUnit.from_dict(unit)
        if unit.logical_key in by_key:
            raise ValueError(f"duplicate runtime slot {unit.logical_key!r}")
        if unit.runtime_id in runtime_ids:
            raise ValueError(f"duplicate runtime_id {unit.runtime_id!r}")
        by_key[unit.logical_key] = unit
        runtime_ids.add(unit.runtime_id)
    return tuple(by_key[key] for key in sorted(by_key))


def _normalize_ownership(
    units: Iterable[Any],
) -> Tuple[RuntimeCleanupRef, ...]:
    """Normalize garbage-collection ownership by immutable resource name.

    Unlike a RuntimeDirectory, cleanup may legitimately own several historical
    revisions of the same logical slot.  Only ``runtime_id`` is unique here;
    accepting two different descriptions for that immutable name would make an
    exact-UID delete ambiguous and is therefore rejected.
    """

    by_runtime_id: Dict[str, RuntimeCleanupRef] = {}
    for unit in units:
        if isinstance(unit, RuntimeUnit):
            unit = RuntimeCleanupRef.from_unit(unit)
        elif not isinstance(unit, RuntimeCleanupRef):
            unit = RuntimeCleanupRef.from_dict(unit)
        existing = by_runtime_id.get(unit.runtime_id)
        if (
            existing is not None
            and existing.runtime_service_uid
            and unit.runtime_service_uid
            and existing.runtime_service_uid != unit.runtime_service_uid
        ):
            raise ValueError(
                f"conflicting cleanup ownership for runtime_id {unit.runtime_id!r}"
            )
        if existing is None or (
            not existing.runtime_service_uid and unit.runtime_service_uid
        ):
            by_runtime_id[unit.runtime_id] = unit
    return tuple(by_runtime_id[name] for name in sorted(by_runtime_id))


@dataclass(frozen=True)
class RuntimeDirectory:
    install_id: str
    revision: int
    routes: Tuple[RuntimeUnit, ...] = field(default_factory=tuple)

    def __post_init__(self):
        install_id = _require_text("install_id", self.install_id, max_length=63)
        if not _LABEL_VALUE_RE.fullmatch(install_id):
            raise ValueError("install_id must be a valid Kubernetes label value")
        object.__setattr__(self, "install_id", install_id)
        revision = int(self.revision)
        if revision < 0:
            raise ValueError("directory revision must be non-negative")
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "routes", _normalize_routes(self.routes))

    @property
    def content_hash(self) -> str:
        return canonical_hash(self._content_dict())

    @property
    def directory_revision(self) -> int:
        return self.revision

    @property
    def nodes(self) -> Tuple[str, ...]:
        return tuple(sorted({unit.slot.target_node for unit in self.routes}))

    @property
    def deployment(self) -> Dict[str, list]:
        deployment: Dict[str, set] = {}
        for unit in self.routes:
            service = unit.slot.logical_service
            if not service:
                continue
            deployment.setdefault(service, set()).add(unit.slot.target_node)
        return {service: sorted(nodes) for service, nodes in sorted(deployment.items())}

    def get(self, slot_or_key: Any) -> Optional[RuntimeUnit]:
        key = slot_or_key.logical_key if isinstance(slot_or_key, RuntimeSlot) else str(slot_or_key)
        return next((unit for unit in self.routes if unit.logical_key == key), None)

    def _content_dict(self) -> Dict[str, Any]:
        return {
            "install_id": self.install_id,
            "directory_revision": self.revision,
            "nodes": list(self.nodes),
            "deployment": self.deployment,
            "routes": [unit.to_dict() for unit in self.routes],
        }

    def to_dict(self) -> Dict[str, Any]:
        result = self._content_dict()
        # ``revision`` is the compact runtime API spelling.  The explicit
        # ``directory_revision`` prevents it being confused with a unit's
        # RuntimeService revision in persisted control-plane records.
        result["revision"] = self.revision
        result["hash"] = self.content_hash
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeDirectory":
        raw_routes = data.get("routes")
        if raw_routes is None:
            entries = data.get("entries") or {}
            raw_routes = entries.values() if isinstance(entries, Mapping) else entries
        return cls(
            install_id=data.get("install_id", data.get("installID", "")),
            revision=data.get("directory_revision", data.get("revision", 0)),
            routes=tuple(RuntimeUnit.from_dict(item) for item in (raw_routes or ())),
        )


@dataclass(frozen=True)
class RuntimeRetirement:
    """Durable ownership of one superseded RuntimeDirectory revision."""

    revision: int
    units: Tuple[RuntimeUnit, ...]
    deadline: Optional[float]
    started_at: str = ""
    fenced: bool = False
    forced_count: int = 0

    def __post_init__(self):
        revision = int(self.revision)
        if revision < 1:
            raise ValueError("retirement revision must be positive")
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "units", _normalize_routes(self.units))
        deadline = self.deadline
        if deadline is not None:
            try:
                deadline = float(deadline)
            except (TypeError, ValueError) as exc:
                raise ValueError("retirement deadline must be numeric") from exc
            if not math.isfinite(deadline) or deadline <= 0:
                raise ValueError("retirement deadline must be finite and positive")
        object.__setattr__(self, "deadline", deadline)
        object.__setattr__(self, "started_at", str(self.started_at or ""))
        object.__setattr__(self, "fenced", bool(self.fenced))
        forced_count = int(self.forced_count)
        if forced_count < 0:
            raise ValueError("retirement forced_count must not be negative")
        object.__setattr__(self, "forced_count", forced_count)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "revision": self.revision,
            "units": [unit.to_state_dict() for unit in self.units],
            "deadline": self.deadline,
            "started_at": self.started_at,
            "fenced": self.fenced,
            "forced_count": self.forced_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeRetirement":
        return cls(
            revision=data.get("revision", 0),
            units=tuple(RuntimeUnit.from_dict(item) for item in (data.get("units") or ())),
            deadline=data.get("deadline", 0),
            started_at=data.get("started_at", data.get("startedAt", "")),
            fenced=data.get("fenced", False),
            forced_count=data.get("forced_count", data.get("forcedCount", 0)),
        )


@dataclass(frozen=True)
class RuntimeCleanupResource:
    """One Kubernetes object that still belongs to an uninstalling session."""

    kind: str
    name: str
    uid: str
    node: str = ""
    deletion_timestamp: str = ""
    finalizers: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        object.__setattr__(self, "kind", _require_text("kind", self.kind, max_length=63))
        object.__setattr__(self, "name", _require_text("name", self.name))
        object.__setattr__(self, "uid", _require_text("uid", self.uid, max_length=128))
        object.__setattr__(self, "node", _optional_text(self.node))
        object.__setattr__(
            self,
            "deletion_timestamp",
            _optional_text(self.deletion_timestamp, max_length=128),
        )
        object.__setattr__(
            self,
            "finalizers",
            tuple(sorted({
                _require_text("finalizer", value)
                for value in (self.finalizers or ())
            })),
        )

    @property
    def identity(self) -> Tuple[str, str, str]:
        return self.kind, self.name, self.uid

    def to_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "kind": self.kind,
            "name": self.name,
            "uid": self.uid,
        }
        if self.node:
            value["node"] = self.node
        if self.deletion_timestamp:
            value["deletion_timestamp"] = self.deletion_timestamp
        if self.finalizers:
            value["finalizers"] = list(self.finalizers)
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeCleanupResource":
        return cls(
            kind=data.get("kind", ""),
            name=data.get("name", ""),
            uid=data.get("uid", ""),
            node=data.get("node", ""),
            deletion_timestamp=data.get(
                "deletion_timestamp", data.get("deletionTimestamp", ""),
            ),
            finalizers=tuple(data.get("finalizers") or ()),
        )


@dataclass(frozen=True)
class RuntimeUninstallProgress:
    """Durable cleanup barrier and progress clock for one uninstall."""

    started_at: str
    last_progress_at: str
    deletion_submitted: bool = False
    remaining: Tuple[RuntimeCleanupResource, ...] = field(default_factory=tuple)

    def __post_init__(self):
        object.__setattr__(
            self,
            "started_at",
            _require_timestamp("uninstall started_at", self.started_at),
        )
        object.__setattr__(
            self,
            "last_progress_at",
            _require_timestamp(
                "uninstall last_progress_at",
                self.last_progress_at,
            ),
        )
        object.__setattr__(self, "deletion_submitted", bool(self.deletion_submitted))
        resources: Dict[Tuple[str, str, str], RuntimeCleanupResource] = {}
        for resource in self.remaining or ():
            if not isinstance(resource, RuntimeCleanupResource):
                resource = RuntimeCleanupResource.from_dict(resource)
            resources[resource.identity] = resource
        object.__setattr__(
            self,
            "remaining",
            tuple(resources[key] for key in sorted(resources)),
        )

    @property
    def identities(self) -> frozenset:
        return frozenset(resource.identity for resource in self.remaining)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "last_progress_at": self.last_progress_at,
            "deletion_submitted": self.deletion_submitted,
            "remaining": [resource.to_dict() for resource in self.remaining],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeUninstallProgress":
        return cls(
            started_at=data.get("started_at", data.get("startedAt", "")),
            last_progress_at=data.get(
                "last_progress_at", data.get("lastProgressAt", ""),
            ),
            deletion_submitted=data.get(
                "deletion_submitted", data.get("deletionSubmitted", False),
            ),
            remaining=tuple(
                RuntimeCleanupResource.from_dict(item)
                for item in (data.get("remaining") or ())
            ),
        )


@dataclass(frozen=True)
class RuntimeSession:
    install_id: str
    operation_id: str
    phase: str = "new"
    next_runtime_revision: int = 1
    active_directory_revision: int = 0
    active: Tuple[RuntimeUnit, ...] = field(default_factory=tuple)
    pending: Tuple[RuntimeUnit, ...] = field(default_factory=tuple)
    retirement: Optional[RuntimeRetirement] = None
    cleanup: Tuple[RuntimeCleanupRef, ...] = field(default_factory=tuple)
    uninstall: Optional[RuntimeUninstallProgress] = None
    source_label: str = ""
    policy_id: str = ""
    source_deploy: Any = field(default_factory=list)
    last_error: str = ""
    updated_at: str = ""

    def __post_init__(self):
        install_id = _require_text("install_id", self.install_id, max_length=63)
        if not _LABEL_VALUE_RE.fullmatch(install_id):
            raise ValueError("install_id must be a valid Kubernetes label value")
        object.__setattr__(self, "install_id", install_id)
        object.__setattr__(self, "operation_id", _require_text("operation_id", self.operation_id, max_length=128))
        next_revision = int(self.next_runtime_revision)
        if next_revision < 1:
            raise ValueError("next_runtime_revision must be positive")
        object.__setattr__(self, "next_runtime_revision", next_revision)
        phase = _require_text("phase", self.phase, max_length=63)
        object.__setattr__(self, "phase", phase)
        directory_revision = int(self.active_directory_revision)
        if directory_revision < 0:
            raise ValueError("active_directory_revision must be non-negative")
        object.__setattr__(self, "active_directory_revision", directory_revision)
        object.__setattr__(self, "active", _normalize_routes(self.active))
        object.__setattr__(self, "pending", _normalize_routes(self.pending))
        retirement = self.retirement
        if retirement is not None and not isinstance(retirement, RuntimeRetirement):
            retirement = RuntimeRetirement.from_dict(retirement)
        object.__setattr__(self, "retirement", retirement)
        cleanup = _normalize_ownership(self.cleanup)
        object.__setattr__(self, "cleanup", cleanup)
        uninstall = self.uninstall
        if uninstall is not None and not isinstance(uninstall, RuntimeUninstallProgress):
            uninstall = RuntimeUninstallProgress.from_dict(uninstall)
        if uninstall is not None and phase not in {"uninstalling", "finalizing-uninstall"}:
            raise ValueError("uninstall progress requires an uninstall lifecycle phase")
        object.__setattr__(self, "uninstall", uninstall)

        active_ids = {unit.runtime_id for unit in self.active}
        pending_ids = {unit.runtime_id for unit in self.pending}
        cleanup_ids = {unit.runtime_id for unit in cleanup}
        retirement_ids = {
            unit.runtime_id for unit in retirement.units
        } if retirement is not None else set()
        if active_ids & pending_ids:
            raise ValueError("active and pending RuntimeServices must not overlap")
        if cleanup_ids & (active_ids | pending_ids):
            raise ValueError("cleanup RuntimeServices must not overlap active or pending state")
        if cleanup_ids & retirement_ids:
            raise ValueError("cleanup RuntimeServices must not overlap retirement state")
        if pending_ids & retirement_ids:
            raise ValueError("pending RuntimeServices must not overlap retirement state")
        if phase == "active" and self.pending:
            raise ValueError("active runtime session must not contain pending RuntimeServices")
        if phase == "publishing-rollout":
            if retirement is None or retirement.revision != directory_revision:
                raise ValueError(
                    "publishing-rollout requires retirement of the active directory revision"
                )
            active_by_id = {unit.runtime_id: unit for unit in self.active}
            if any(active_by_id.get(unit.runtime_id) != unit for unit in retirement.units):
                raise ValueError(
                    "publishing-rollout retirement must be an exact subset of active state"
                )
        if phase == "active" and retirement is not None:
            if retirement.deadline is None:
                raise ValueError("active retirement requires an armed deadline")
            if retirement.revision != directory_revision - 1:
                raise ValueError(
                    "active retirement must immediately precede the active directory revision"
                )
            if active_ids & retirement_ids:
                raise ValueError("retired RuntimeServices must not overlap the active directory")
        object.__setattr__(self, "source_label", str(self.source_label or ""))
        object.__setattr__(self, "policy_id", str(self.policy_id or ""))
        object.__setattr__(self, "last_error", str(self.last_error or ""))
        object.__setattr__(self, "updated_at", str(self.updated_at or ""))
        # Reject non-JSON session context at the model boundary, and detach it
        # from caller-owned dictionaries before it is persisted.
        object.__setattr__(self, "source_deploy", json.loads(canonical_json(self.source_deploy)))

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def directory(self) -> RuntimeDirectory:
        return RuntimeDirectory(
            install_id=self.install_id,
            revision=self.active_directory_revision,
            routes=self.active,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "install_id": self.install_id,
            "operation_id": self.operation_id,
            "phase": self.phase,
            "next_runtime_revision": self.next_runtime_revision,
            "active_directory_revision": self.active_directory_revision,
            "active": [unit.to_state_dict() for unit in self.active],
            "pending": [unit.to_state_dict() for unit in self.pending],
            "retirement": self.retirement.to_dict() if self.retirement else None,
            "cleanup": [unit.to_dict() for unit in self.cleanup],
            "uninstall": self.uninstall.to_dict() if self.uninstall else None,
            "source_label": self.source_label,
            "policy_id": self.policy_id,
            "source_deploy": deepcopy(self.source_deploy),
            "last_error": self.last_error,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeSession":
        return cls(
            install_id=data.get("install_id", data.get("installID", "")),
            operation_id=data.get("operation_id", data.get("operationID", "")),
            phase=data.get("phase", "new"),
            next_runtime_revision=data.get("next_runtime_revision", data.get("nextRuntimeRevision", 1)),
            active_directory_revision=data.get("active_directory_revision", data.get("activeDirectoryRevision", 0)),
            active=tuple(RuntimeUnit.from_dict(item) for item in (data.get("active") or ())),
            pending=tuple(RuntimeUnit.from_dict(item) for item in (data.get("pending") or ())),
            retirement=(
                RuntimeRetirement.from_dict(data["retirement"])
                if data.get("retirement") else None
            ),
            cleanup=tuple(
                RuntimeCleanupRef.from_dict(item)
                for item in (data.get("cleanup") or ())
            ),
            uninstall=(
                RuntimeUninstallProgress.from_dict(data["uninstall"])
                if data.get("uninstall") else None
            ),
            source_label=data.get("source_label", data.get("sourceLabel", "")),
            policy_id=data.get("policy_id", data.get("policyID", "")),
            source_deploy=data.get("source_deploy", data.get("sourceDeploy", [])),
            last_error=data.get("last_error", data.get("lastError", "")),
            updated_at=data.get("updated_at", data.get("updatedAt", "")),
        )
