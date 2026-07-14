import copy
import hashlib
import json
import math
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Tuple


class RuntimeDirectoryError(ValueError):
    """Base error for invalid directory state transitions."""


class RuntimeDirectoryConflict(RuntimeDirectoryError):
    """Raised when a compare-and-swap or lease precondition fails."""


class RuntimeDirectoryNotFound(RuntimeDirectoryError):
    """Raised when a proposal or lease token does not exist."""


def _first(mapping, *names, default=None):
    if not isinstance(mapping, dict):
        return default
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _positive_int(value, field):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise RuntimeDirectoryError(f"{field} must be a positive integer")
    if parsed < 1:
        raise RuntimeDirectoryError(f"{field} must be a positive integer")
    return parsed


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_hash(value):
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RuntimeRoute:
    component: str
    target_node: str
    position: str
    logical_service: str
    source_id: str
    runtime_id: str
    runtime_revision: int
    spec_hash: str
    dns_name: str
    port: int
    runtime_service_uid: str
    service_uid: str
    pod_uid: str

    @classmethod
    def from_value(cls, value: Dict[str, Any]):
        if not isinstance(value, dict):
            raise RuntimeDirectoryError("runtime route must be an object")

        slot = value.get("slot") if isinstance(value.get("slot"), dict) else {}
        endpoint = value.get("endpoint") if isinstance(value.get("endpoint"), dict) else {}

        component = str(_first(value, "component", default=_first(slot, "component", default="")) or "")
        target_node = str(
            _first(value, "target_node", "targetNode", default=_first(slot, "target_node", "targetNode", default=""))
            or ""
        )
        position = str(_first(value, "position", default=_first(slot, "position", default="")) or "")
        logical_service = str(
            _first(
                value,
                "logical_service",
                "logicalService",
                default=_first(slot, "logical_service", "logicalService", default=""),
            )
            or ""
        )
        source_id_value = _first(value, "source_id", "sourceID", default=_first(slot, "source_id", "sourceID"))
        source_id = "" if source_id_value is None else str(source_id_value)
        runtime_id = str(_first(value, "runtime_id", "runtimeID", default="") or "")
        runtime_revision = _positive_int(
            _first(value, "runtime_revision", "runtimeRevision", "deployment_revision", "deploymentRevision"),
            "runtime_revision",
        )
        spec_hash = str(_first(value, "spec_hash", "specHash", default="") or "")

        dns_name = str(
            _first(
                endpoint,
                "dns_name",
                "dnsName",
                "fqdn",
                default=_first(value, "dns_name", "dnsName", "fqdn", default=""),
            )
            or ""
        )
        raw_port = _first(endpoint, "port", default=_first(value, "port"))
        runtime_service_uid = str(
            _first(
                endpoint,
                "runtime_service_uid",
                "runtimeServiceUID",
                default=_first(value, "runtime_service_uid", "runtimeServiceUID", default=""),
            )
            or ""
        )
        service_uid = str(
            _first(endpoint, "service_uid", "serviceUID", default=_first(value, "service_uid", "serviceUID", default=""))
            or ""
        )
        pod_uid = str(
            _first(
                endpoint,
                "pod_uid",
                "podUID",
                "endpoint_pod_uid",
                "endpointPodUID",
                default=_first(value, "pod_uid", "podUID", "endpoint_pod_uid", "endpointPodUID", default=""),
            )
            or ""
        )

        if not component:
            raise RuntimeDirectoryError("runtime route component is required")
        if not target_node:
            raise RuntimeDirectoryError("runtime route target_node is required")
        if position not in ("cloud", "edge"):
            raise RuntimeDirectoryError("runtime route position must be 'cloud' or 'edge'")
        if not runtime_id:
            raise RuntimeDirectoryError("runtime route runtime_id is required")
        if not re.fullmatch(r"[a-z]([-a-z0-9]*[a-z0-9])?", runtime_id) or len(runtime_id) > 63:
            raise RuntimeDirectoryError(f"runtime route {runtime_id!r} is not a DNS-1035 label")
        if not re.fullmatch(r"[0-9a-f]{64}", spec_hash):
            raise RuntimeDirectoryError(f"runtime route {runtime_id!r} has an invalid spec_hash")
        if _first(value, "ready", default=True) is False:
            raise RuntimeDirectoryError(f"runtime route {runtime_id!r} is not Ready")

        endpoint_values = (dns_name, raw_port, runtime_service_uid, service_uid, pod_uid)
        endpoint_present = any(item not in (None, "") for item in endpoint_values)
        port = 0
        if endpoint_present:
            if not all(item not in (None, "") for item in endpoint_values):
                raise RuntimeDirectoryError(f"runtime route {runtime_id!r} has an incomplete endpoint identity")
            port = _positive_int(raw_port, "endpoint.port")
            if port > 65535:
                raise RuntimeDirectoryError("endpoint.port must not exceed 65535")

        if component == "processor":
            if not logical_service:
                raise RuntimeDirectoryError(f"processor route {runtime_id!r} requires logical_service")
            if not endpoint_present:
                raise RuntimeDirectoryError(f"processor route {runtime_id!r} requires an endpoint")
        if component == "generator" and not source_id:
            raise RuntimeDirectoryError(f"generator route {runtime_id!r} requires source_id")

        return cls(
            component=component,
            target_node=target_node,
            position=position,
            logical_service=logical_service,
            source_id=source_id,
            runtime_id=runtime_id,
            runtime_revision=runtime_revision,
            spec_hash=spec_hash,
            dns_name=dns_name,
            port=port,
            runtime_service_uid=runtime_service_uid,
            service_uid=service_uid,
            pod_uid=pod_uid,
        )

    @property
    def logical_key(self):
        return "/".join((
            self.component,
            self.logical_service,
            self.source_id,
            self.position,
            self.target_node,
        ))

    @property
    def has_endpoint(self):
        return bool(self.dns_name and self.port)

    def to_dict(self):
        result = {
            "component": self.component,
            "target_node": self.target_node,
            "position": self.position,
            "runtime_id": self.runtime_id,
            "runtime_revision": self.runtime_revision,
            "spec_hash": self.spec_hash,
        }
        if self.logical_service:
            result["logical_service"] = self.logical_service
        if self.source_id:
            result["source_id"] = self.source_id
        if self.has_endpoint:
            result.update({
                "dns_name": self.dns_name,
                "port": self.port,
                "runtime_service_uid": self.runtime_service_uid,
                "service_uid": self.service_uid,
                "pod_uid": self.pod_uid,
            })
        return result


@dataclass(frozen=True)
class RuntimeDirectorySnapshot:
    install_id: str
    revision: int
    routes: Tuple[RuntimeRoute, ...]

    @classmethod
    def empty(cls):
        return cls(install_id="", revision=0, routes=())

    @classmethod
    def from_value(cls, value):
        if not isinstance(value, dict):
            raise RuntimeDirectoryError("runtime directory must be an object")
        explicit_revision = _first(value, "revision")
        explicit_directory_revision = _first(
            value, "directory_revision", "runtime_directory_revision", "runtimeDirectoryRevision"
        )
        if (
            explicit_revision is not None
            and explicit_directory_revision is not None
            and str(explicit_revision) != str(explicit_directory_revision)
        ):
            raise RuntimeDirectoryError("revision and directory_revision must match")
        revision_raw = explicit_directory_revision if explicit_directory_revision is not None else explicit_revision
        if revision_raw is None:
            revision_raw = 0
        try:
            revision = int(revision_raw)
        except (TypeError, ValueError):
            raise RuntimeDirectoryError("runtime directory revision must be an integer")
        if revision < 0:
            raise RuntimeDirectoryError("runtime directory revision must not be negative")

        install_id = str(_first(value, "install_id", "installID", default="") or "")
        route_values = _first(value, "routes", "entries", "runtime_routes", "runtimeRoutes", default=[])
        if isinstance(route_values, dict):
            route_values = list(route_values.values())
        if not isinstance(route_values, (list, tuple)):
            raise RuntimeDirectoryError("runtime directory routes must be a list or object")
        if route_values and revision < 1:
            raise RuntimeDirectoryError("a non-empty runtime directory requires revision >= 1")

        routes = []
        keys = set()
        runtime_ids = set()
        for route_value in route_values:
            route = RuntimeRoute.from_value(route_value)
            if route.logical_key in keys:
                raise RuntimeDirectoryError(f"duplicate runtime slot {route.logical_key!r}")
            if route.runtime_id in runtime_ids:
                raise RuntimeDirectoryError(f"duplicate runtime_id {route.runtime_id!r}")
            keys.add(route.logical_key)
            runtime_ids.add(route.runtime_id)
            routes.append(route)
        routes.sort(key=lambda item: item.logical_key)
        snapshot = cls(install_id=install_id, revision=revision, routes=tuple(routes))
        if routes and not install_id:
            raise RuntimeDirectoryError("a non-empty runtime directory requires install_id")

        supplied_nodes = value.get("nodes")
        if supplied_nodes is not None:
            if not isinstance(supplied_nodes, (list, tuple)):
                raise RuntimeDirectoryError("runtime directory nodes must be a list")
            if list(supplied_nodes) != list(snapshot.nodes):
                raise RuntimeDirectoryError("runtime directory nodes do not match routes")
        supplied_deployment = value.get("deployment")
        if supplied_deployment is not None and supplied_deployment != snapshot.deployment:
            raise RuntimeDirectoryError("runtime directory deployment does not match routes")
        supplied_hash = value.get("hash")
        if supplied_hash is not None and str(supplied_hash) != snapshot.content_hash:
            raise RuntimeDirectoryError("runtime directory hash does not match canonical content")
        return snapshot

    @property
    def nodes(self):
        return tuple(sorted({route.target_node for route in self.routes}))

    @property
    def deployment(self):
        result = {}
        for route in self.routes:
            if route.logical_service:
                result.setdefault(route.logical_service, set()).add(route.target_node)
        return {service: sorted(nodes) for service, nodes in sorted(result.items())}

    def _content_dict(self):
        return {
            "install_id": self.install_id,
            "directory_revision": self.revision,
            "nodes": list(self.nodes),
            "deployment": self.deployment,
            "routes": [route.to_dict() for route in self.routes],
        }

    @property
    def content_hash(self):
        return _canonical_hash(self._content_dict())

    def to_dict(self):
        result = self._content_dict()
        result["revision"] = self.revision
        result["hash"] = self.content_hash
        return result

    def find(self, component=None, target_node=None, logical_service=None):
        matches = []
        for route in self.routes:
            if component is not None and route.component != component:
                continue
            if target_node is not None and route.target_node != target_node:
                continue
            if logical_service is not None and route.logical_service != logical_service:
                continue
            matches.append(route)
        return matches

    def resolve(self, component, target_node=None, logical_service=None):
        matches = self.find(component=component, target_node=target_node, logical_service=logical_service)
        if len(matches) != 1:
            raise RuntimeDirectoryError(
                f"expected one route for component={component!r}, logical_service={logical_service!r}, "
                f"target_node={target_node!r}; found {len(matches)}"
            )
        route = matches[0]
        if not route.has_endpoint:
            raise RuntimeDirectoryError(f"runtime route {route.runtime_id!r} has no endpoint")
        return route

    def processor_deployment(self):
        result = {}
        for route in self.find(component="processor"):
            result.setdefault(route.logical_service, []).append(route.target_node)
        return {service: sorted(set(nodes)) for service, nodes in sorted(result.items())}

@dataclass(frozen=True)
class DirectoryProposal:
    proposal_id: str
    base_revision: int
    snapshot: RuntimeDirectorySnapshot
    created_at: float
    expires_at: float

    def to_dict(self):
        return {
            "proposal_id": self.proposal_id,
            "base_revision": self.base_revision,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "directory": self.snapshot.to_dict(),
        }


class RuntimeDirectoryStore:
    """Thread-safe owner of one immutable active runtime directory snapshot."""

    def __init__(self, initial=None, clock=None):
        self._lock = threading.RLock()
        self._clock = clock or time.time
        self._active = RuntimeDirectorySnapshot.empty() if initial is None else RuntimeDirectorySnapshot.from_value(initial)
        self._proposals = {}

    def _prune_locked(self):
        now = self._clock()
        self._proposals = {
            proposal_id: proposal
            for proposal_id, proposal in self._proposals.items()
            if proposal.expires_at > now
        }

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(self._active.to_dict())

    def snapshot_model(self):
        with self._lock:
            return self._active

    def replace(self, value, expected_revision):
        candidate = RuntimeDirectorySnapshot.from_value(value)
        with self._lock:
            self._prune_locked()
            if int(expected_revision) != self._active.revision:
                raise RuntimeDirectoryConflict(
                    f"runtime directory CAS expected revision {expected_revision}, current is {self._active.revision}"
                )
            if candidate.revision <= self._active.revision:
                raise RuntimeDirectoryConflict(
                    f"candidate revision {candidate.revision} must exceed current revision {self._active.revision}"
                )
            if self._active.install_id and candidate.install_id != self._active.install_id:
                raise RuntimeDirectoryConflict("runtime directory install_id is immutable")
            self._active = candidate
            return self.snapshot()

    def propose(self, value, base_revision, proposal_id=None, ttl_seconds=60.0):
        candidate = RuntimeDirectorySnapshot.from_value(value)
        ttl_seconds = max(1.0, float(ttl_seconds))
        with self._lock:
            self._prune_locked()
            if int(base_revision) != self._active.revision:
                raise RuntimeDirectoryConflict(
                    f"proposal base revision {base_revision} does not match current revision {self._active.revision}"
                )
            if candidate.revision <= self._active.revision:
                raise RuntimeDirectoryConflict(
                    f"candidate revision {candidate.revision} must exceed current revision {self._active.revision}"
                )
            if self._active.install_id and candidate.install_id != self._active.install_id:
                raise RuntimeDirectoryConflict("runtime directory install_id is immutable")
            proposal_id = str(proposal_id or uuid.uuid4())
            if proposal_id in self._proposals:
                existing = self._proposals[proposal_id]
                if existing.base_revision == int(base_revision) and existing.snapshot == candidate:
                    return copy.deepcopy(existing.to_dict())
                raise RuntimeDirectoryConflict(f"proposal {proposal_id!r} already exists with different content")
            now = self._clock()
            proposal = DirectoryProposal(
                proposal_id=proposal_id,
                base_revision=int(base_revision),
                snapshot=candidate,
                created_at=now,
                expires_at=now + ttl_seconds,
            )
            self._proposals[proposal_id] = proposal
            return copy.deepcopy(proposal.to_dict())

    def commit(self, proposal_id, expected_revision):
        with self._lock:
            self._prune_locked()
            proposal = self._proposals.get(str(proposal_id))
            if proposal is None:
                raise RuntimeDirectoryNotFound(f"proposal {proposal_id!r} does not exist or expired")
            if int(expected_revision) != self._active.revision or proposal.base_revision != self._active.revision:
                raise RuntimeDirectoryConflict(
                    f"proposal {proposal_id!r} is based on revision {proposal.base_revision}, "
                    f"current is {self._active.revision}"
                )
            self._active = proposal.snapshot
            del self._proposals[str(proposal_id)]
            return self.snapshot()

    def reject(self, proposal_id, reason=""):
        with self._lock:
            self._prune_locked()
            proposal = self._proposals.pop(str(proposal_id), None)
            if proposal is None:
                raise RuntimeDirectoryNotFound(f"proposal {proposal_id!r} does not exist or expired")
            return {
                "proposal_id": proposal.proposal_id,
                "rejected": True,
                "reason": str(reason or ""),
                "revision": proposal.snapshot.revision,
            }

    def clear(self, install_id):
        install_id = str(install_id or "").strip()
        if not install_id:
            raise RuntimeDirectoryError("runtime directory clear requires install_id")
        with self._lock:
            previous_revision = self._active.revision
            if self._active.install_id and self._active.install_id != install_id:
                raise RuntimeDirectoryConflict("runtime directory install_id does not match clear request")
            self._active = RuntimeDirectorySnapshot.empty()
            self._proposals.clear()
            return {
                "cleared": True,
                "install_id": install_id,
                "previous_revision": previous_revision,
            }

    def compact_routes_for_plan(self, plan, source_device="", cloud_node=""):
        with self._lock:
            snapshot = self._active
        return _compact_routes(snapshot, plan, source_device=source_device, cloud_node=cloud_node)


class RedisRuntimeDirectoryStore:
    """Durable RuntimeDirectory owner with Redis-side compare-and-swap.

    Scheduler Pods are disposable.  Keeping the active directory or a pending
    rollout proposal only in Python memory would make a harmless Pod restart
    invalidate every exact task route.  This store persists both objects in
    the Redis service already required by the managed runtime and performs all
    revision checks inside Lua, so a restart never needs Kubernetes discovery
    or a backend-driven cache refresh.
    """

    _REPLACE_SCRIPT = """
local current_raw = redis.call('GET', KEYS[1])
local current_revision = 0
local current_install_id = ''
if current_raw then
  local current = cjson.decode(current_raw)
  current_revision = tonumber(current.revision or current.directory_revision or 0)
  current_install_id = tostring(current.install_id or '')
end
local expected_revision = tonumber(ARGV[1])
local candidate_revision = tonumber(ARGV[2])
local candidate_install_id = tostring(ARGV[3])
if current_revision ~= expected_revision then
  return {0, current_revision}
end
if candidate_revision <= current_revision then
  return {1, current_revision}
end
if current_install_id ~= '' and current_install_id ~= candidate_install_id then
  return {2, current_revision}
end
redis.call('SET', KEYS[1], ARGV[4])
return {3, candidate_revision}
"""

    _PROPOSE_SCRIPT = """
local current_raw = redis.call('GET', KEYS[1])
local current_revision = 0
local current_install_id = ''
if current_raw then
  local current = cjson.decode(current_raw)
  current_revision = tonumber(current.revision or current.directory_revision or 0)
  current_install_id = tostring(current.install_id or '')
end
local base_revision = tonumber(ARGV[1])
local candidate_revision = tonumber(ARGV[2])
local candidate_install_id = tostring(ARGV[3])
if current_revision ~= base_revision then
  return {0, tostring(current_revision)}
end
if candidate_revision <= current_revision then
  return {1, tostring(current_revision)}
end
if current_install_id ~= '' and current_install_id ~= candidate_install_id then
  return {2, tostring(current_revision)}
end
local existing_raw = redis.call('GET', KEYS[2])
if existing_raw then
  local existing = cjson.decode(existing_raw)
  local existing_directory = existing.directory or {}
  local existing_hash = tostring(existing_directory.hash or '')
  if tonumber(existing.base_revision) == base_revision and existing_hash == tostring(ARGV[4]) then
    redis.call('SADD', KEYS[3], KEYS[2])
    local existing_index_ttl = redis.call('TTL', KEYS[3])
    if existing_index_ttl < tonumber(ARGV[6]) then
      redis.call('EXPIRE', KEYS[3], tonumber(ARGV[6]))
    end
    return {4, existing_raw}
  end
  return {3, existing_raw}
end
redis.call('SET', KEYS[2], ARGV[5], 'EX', tonumber(ARGV[6]))
redis.call('SADD', KEYS[3], KEYS[2])
local index_ttl = redis.call('TTL', KEYS[3])
if index_ttl < tonumber(ARGV[6]) then
  redis.call('EXPIRE', KEYS[3], tonumber(ARGV[6]))
end
return {4, ARGV[5]}
"""

    _COMMIT_SCRIPT = """
local proposal_raw = redis.call('GET', KEYS[2])
if not proposal_raw then
  redis.call('SREM', KEYS[3], KEYS[2])
  if redis.call('SCARD', KEYS[3]) == 0 then
    redis.call('DEL', KEYS[3])
  end
  return {0, ''}
end
local proposal = cjson.decode(proposal_raw)
local current_raw = redis.call('GET', KEYS[1])
local current_revision = 0
if current_raw then
  local current = cjson.decode(current_raw)
  current_revision = tonumber(current.revision or current.directory_revision or 0)
end
local expected_revision = tonumber(ARGV[1])
if current_revision ~= expected_revision or tonumber(proposal.base_revision) ~= current_revision then
  return {1, tostring(current_revision)}
end
local directory_raw = cjson.encode(proposal.directory)
redis.call('SET', KEYS[1], directory_raw)
redis.call('DEL', KEYS[2])
redis.call('SREM', KEYS[3], KEYS[2])
if redis.call('SCARD', KEYS[3]) == 0 then
  redis.call('DEL', KEYS[3])
end
return {2, directory_raw}
"""

    _CLEAR_SCRIPT = """
local current_raw = redis.call('GET', KEYS[1])
local current_revision = 0
if current_raw then
  local current = cjson.decode(current_raw)
  local current_install_id = tostring(current.install_id or '')
  current_revision = tonumber(current.revision or current.directory_revision or 0)
  if current_install_id ~= '' and current_install_id ~= tostring(ARGV[1]) then
    return {0, current_revision}
  end
end
local proposal_keys = redis.call('SMEMBERS', KEYS[2])
for _, proposal_key in ipairs(proposal_keys) do
  redis.call('DEL', proposal_key)
end
redis.call('DEL', KEYS[1])
redis.call('DEL', KEYS[2])
return {1, current_revision}
"""

    _REJECT_SCRIPT = """
local proposal_raw = redis.call('GET', KEYS[1])
if not proposal_raw then
  redis.call('SREM', KEYS[2], KEYS[1])
  if redis.call('SCARD', KEYS[2]) == 0 then
    redis.call('DEL', KEYS[2])
  end
  return {0, ''}
end
redis.call('DEL', KEYS[1])
redis.call('SREM', KEYS[2], KEYS[1])
if redis.call('SCARD', KEYS[2]) == 0 then
  redis.call('DEL', KEYS[2])
end
return {1, proposal_raw}
"""

    def __init__(
            self,
            redis_client,
            install_id,
            initial=None,
            clock=None,
            key_prefix="dayu:runtime-directory",
    ):
        self.redis = redis_client
        self.install_id = str(install_id or "").strip()
        if not self.install_id:
            raise RuntimeDirectoryError("Redis RuntimeDirectory requires a non-empty install_id")
        self._clock = clock or time.time
        self.key_prefix = str(key_prefix).rstrip(":")
        self._active_key = f"{self.key_prefix}:{self.install_id}:active"
        self._proposal_prefix = f"{self.key_prefix}:{self.install_id}:proposal"
        self._proposal_index_key = f"{self.key_prefix}:{self.install_id}:proposals"
        if initial is not None and self.redis.get(self._active_key) is None:
            candidate = RuntimeDirectorySnapshot.from_value(initial)
            if candidate.install_id and candidate.install_id != self.install_id:
                raise RuntimeDirectoryConflict("initial runtime directory install_id does not match bootstrap")
            self.redis.set(self._active_key, _canonical_json(candidate.to_dict()), nx=True)

    @classmethod
    def from_endpoint(cls, endpoint, install_id, initial=None, clock=None):
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("redis package is required for a durable RuntimeDirectory") from exc
        return cls(
            redis.Redis(host=endpoint.fqdn, port=endpoint.port or 6379, decode_responses=True),
            install_id=install_id,
            initial=initial,
            clock=clock,
        )

    def _proposal_key(self, proposal_id):
        return f"{self._proposal_prefix}:{proposal_id}"

    @staticmethod
    def _decode_snapshot(raw):
        if not raw:
            return RuntimeDirectorySnapshot.empty()
        try:
            return RuntimeDirectorySnapshot.from_value(json.loads(raw))
        except (TypeError, ValueError) as exc:
            raise RuntimeDirectoryError("persisted RuntimeDirectory is corrupt") from exc

    @staticmethod
    def _script_result(value):
        if not isinstance(value, (list, tuple)) or not value:
            raise RuntimeDirectoryError("Redis RuntimeDirectory transaction returned an invalid result")
        return int(value[0]), value[1] if len(value) > 1 else None

    def snapshot_model(self):
        return self._decode_snapshot(self.redis.get(self._active_key))

    def snapshot(self):
        return copy.deepcopy(self.snapshot_model().to_dict())

    def replace(self, value, expected_revision):
        candidate = RuntimeDirectorySnapshot.from_value(value)
        if candidate.install_id != self.install_id:
            raise RuntimeDirectoryConflict("runtime directory install_id does not match bootstrap")
        code, current = self._script_result(self.redis.eval(
            self._REPLACE_SCRIPT,
            1,
            self._active_key,
            int(expected_revision),
            candidate.revision,
            candidate.install_id,
            _canonical_json(candidate.to_dict()),
        ))
        if code == 0:
            raise RuntimeDirectoryConflict(
                f"runtime directory CAS expected revision {expected_revision}, current is {current}"
            )
        if code == 1:
            raise RuntimeDirectoryConflict(
                f"candidate revision {candidate.revision} must exceed current revision {current}"
            )
        if code == 2:
            raise RuntimeDirectoryConflict("runtime directory install_id is immutable")
        return self.snapshot()

    def propose(self, value, base_revision, proposal_id=None, ttl_seconds=60.0):
        candidate = RuntimeDirectorySnapshot.from_value(value)
        if candidate.install_id != self.install_id:
            raise RuntimeDirectoryConflict("runtime directory install_id does not match bootstrap")
        ttl_seconds = max(1.0, float(ttl_seconds))
        proposal_id = str(proposal_id or uuid.uuid4())
        now = self._clock()
        proposal = DirectoryProposal(
            proposal_id=proposal_id,
            base_revision=int(base_revision),
            snapshot=candidate,
            created_at=now,
            expires_at=now + ttl_seconds,
        )
        proposal_raw = _canonical_json(proposal.to_dict())
        code, existing_raw = self._script_result(self.redis.eval(
            self._PROPOSE_SCRIPT,
            3,
            self._active_key,
            self._proposal_key(proposal_id),
            self._proposal_index_key,
            int(base_revision),
            candidate.revision,
            candidate.install_id,
            candidate.content_hash,
            proposal_raw,
            max(1, int(math.ceil(ttl_seconds))),
        ))
        if code == 0:
            raise RuntimeDirectoryConflict(
                f"proposal base revision {base_revision} does not match current revision {existing_raw}"
            )
        if code == 1:
            raise RuntimeDirectoryConflict(
                f"candidate revision {candidate.revision} must exceed current revision {existing_raw}"
            )
        if code == 2:
            raise RuntimeDirectoryConflict("runtime directory install_id is immutable")
        if code == 3:
            raise RuntimeDirectoryConflict(f"proposal {proposal_id!r} already exists with different content")
        try:
            return copy.deepcopy(json.loads(existing_raw))
        except (TypeError, ValueError) as exc:
            raise RuntimeDirectoryError("persisted RuntimeDirectory proposal is corrupt") from exc

    def commit(self, proposal_id, expected_revision):
        proposal_id = str(proposal_id)
        code, value = self._script_result(self.redis.eval(
            self._COMMIT_SCRIPT,
            3,
            self._active_key,
            self._proposal_key(proposal_id),
            self._proposal_index_key,
            int(expected_revision),
        ))
        if code == 0:
            raise RuntimeDirectoryNotFound(f"proposal {proposal_id!r} does not exist or expired")
        if code == 1:
            raise RuntimeDirectoryConflict(
                f"proposal {proposal_id!r} cannot commit at current revision {value}"
            )
        return copy.deepcopy(self._decode_snapshot(value).to_dict())

    def reject(self, proposal_id, reason=""):
        proposal_id = str(proposal_id)
        code, raw = self._script_result(self.redis.eval(
            self._REJECT_SCRIPT,
            2,
            self._proposal_key(proposal_id),
            self._proposal_index_key,
        ))
        if code != 1:
            raise RuntimeDirectoryNotFound(f"proposal {proposal_id!r} does not exist or expired")
        try:
            proposal = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeDirectoryError("persisted RuntimeDirectory proposal is corrupt") from exc
        return {
            "proposal_id": proposal_id,
            "rejected": True,
            "reason": str(reason or ""),
            "revision": int((proposal.get("directory") or {}).get("revision") or 0),
        }

    def clear(self, install_id):
        install_id = str(install_id or "").strip()
        if not install_id:
            raise RuntimeDirectoryError("runtime directory clear requires install_id")
        if install_id != self.install_id:
            raise RuntimeDirectoryConflict(
                "runtime directory install_id does not match clear request"
            )
        code, previous_revision = self._script_result(self.redis.eval(
            self._CLEAR_SCRIPT,
            2,
            self._active_key,
            self._proposal_index_key,
            install_id,
        ))
        if code != 1:
            raise RuntimeDirectoryConflict(
                "runtime directory install_id does not match clear request"
            )
        return {
            "cleared": True,
            "install_id": install_id,
            "previous_revision": int(previous_revision or 0),
        }

    def compact_routes_for_plan(self, plan, source_device="", cloud_node=""):
        snapshot = self.snapshot_model()
        return _compact_routes(snapshot, plan, source_device=source_device, cloud_node=cloud_node)


def _compact_routes(snapshot, plan, source_device="", cloud_node=""):
    """Pure compact-route implementation shared by memory and Redis stores."""
    if snapshot.revision < 1:
        raise RuntimeDirectoryError("runtime directory is not initialized")
    if not isinstance(plan, dict) or not isinstance(plan.get("dag"), dict):
        raise RuntimeDirectoryError("schedule plan must contain a dag object")

    selected = {}
    devices = {str(source_device)} if source_device else set()
    for service_name, node in plan["dag"].items():
        if service_name in ("_start", "_end"):
            service = node.get("service", {}) if isinstance(node, dict) else {}
            device = service.get("execute_device") if isinstance(service, dict) else None
            if device:
                devices.add(str(device))
            continue
        if not isinstance(node, dict) or not isinstance(node.get("service"), dict):
            raise RuntimeDirectoryError(f"schedule plan service {service_name!r} is malformed")
        device = str(node["service"].get("execute_device") or "")
        if not device:
            raise RuntimeDirectoryError(f"schedule plan service {service_name!r} has no execute_device")
        route = snapshot.resolve("processor", target_node=device, logical_service=str(service_name))
        selected[route.logical_key] = route
        devices.add(device)

    for device in sorted(devices):
        route = snapshot.resolve("controller", target_node=device)
        selected[route.logical_key] = route

    distributor_routes = snapshot.find(component="distributor")
    if distributor_routes:
        route = snapshot.resolve("distributor", target_node=cloud_node or None)
        selected[route.logical_key] = route

    return [selected[key].to_dict() for key in sorted(selected)]


def create_runtime_directory_store(runtime_context, initial=None):
    """Create the production durable store; tests inject memory explicitly."""
    endpoint = runtime_context.resolve_static_endpoint("redis", required=False)
    if endpoint is None:
        raise RuntimeDirectoryError(
            "managed Scheduler requires a Redis endpoint for durable RuntimeDirectory state"
        )
    return RedisRuntimeDirectoryStore.from_endpoint(
        endpoint,
        install_id=runtime_context.install_id,
        initial=initial,
    )
