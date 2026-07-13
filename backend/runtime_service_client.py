"""Focused Kubernetes client for the fixed Sedna RuntimeService GVR."""

from __future__ import annotations

import copy
import math
import re
import time
from typing import Any, Dict, Mapping, Optional, Sequence

from kubernetes import watch
from kubernetes.client.rest import ApiException

from runtime_model import RuntimeEndpoint, RuntimeUnit

RUNTIME_GROUP = "sedna.io"
RUNTIME_VERSION = "v1alpha1"
RUNTIME_PLURAL = "runtimeservices"
RUNTIME_API_VERSION = f"{RUNTIME_GROUP}/{RUNTIME_VERSION}"
RUNTIME_KIND = "RuntimeService"


class RuntimeServiceError(RuntimeError):
    pass


class RuntimeServiceRejected(RuntimeServiceError):
    pass


class RuntimeServiceTimeout(RuntimeServiceError):
    pass


class RuntimeServiceInvalidStatus(RuntimeServiceError):
    pass


def _condition_map(obj: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(item.get("type")): item
        for item in ((obj.get("status") or {}).get("conditions") or ())
        if isinstance(item, Mapping) and item.get("type")
    }


def _expected_revision(value: Any) -> int:
    if isinstance(value, RuntimeUnit):
        return value.runtime_revision
    if isinstance(value, Mapping):
        return int(value.get("runtime_revision", value.get("deploymentRevision", value.get("revision", 0))))
    return int(value)


def _observed_spec_hash(obj: Mapping[str, Any]) -> str:
    value = str(((obj.get("status") or {}).get("observedSpecHash") or "")).strip()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        name = str((obj.get("metadata") or {}).get("name") or "<unknown>")
        raise RuntimeServiceInvalidStatus(
            f"{name}: status.observedSpecHash must be a non-empty lowercase SHA-256 digest"
        )
    return value


class RuntimeServiceClient:
    """Reuse one CustomObjectsApi and watch exact RuntimeService conditions."""

    def __init__(
        self,
        namespace: str,
        api,
        watch_factory=None,
        request_timeout_seconds: float = 30,
    ):
        self.namespace = str(namespace or "").strip()
        if not self.namespace:
            raise ValueError("namespace must be non-empty")
        if api is None:
            raise ValueError("api must be the shared ClusterClient CustomObjectsApi")
        self.api = api
        self.watch_factory = watch_factory or watch.Watch
        self.request_timeout = float(request_timeout_seconds)
        if self.request_timeout <= 0:
            raise ValueError("request_timeout_seconds must be positive")

    @staticmethod
    def _validate_manifest(manifest: Mapping[str, Any]) -> None:
        if manifest.get("apiVersion") != RUNTIME_API_VERSION:
            raise ValueError(f"RuntimeService apiVersion must be {RUNTIME_API_VERSION!r}")
        if manifest.get("kind") != RUNTIME_KIND:
            raise ValueError(f"resource kind must be {RUNTIME_KIND!r}")
        metadata = manifest.get("metadata") or {}
        if not metadata.get("name"):
            raise ValueError("RuntimeService metadata.name must be non-empty")

    def create(self, manifest: Mapping[str, Any]) -> Dict[str, Any]:
        self._validate_manifest(manifest)
        body = copy.deepcopy(dict(manifest))
        metadata = body.setdefault("metadata", {})
        manifest_namespace = metadata.get("namespace")
        if manifest_namespace and manifest_namespace != self.namespace:
            raise ValueError(
                f"RuntimeService namespace {manifest_namespace!r} does not match client namespace {self.namespace!r}"
            )
        metadata["namespace"] = self.namespace
        return self.api.create_namespaced_custom_object(
            group=RUNTIME_GROUP,
            version=RUNTIME_VERSION,
            namespace=self.namespace,
            plural=RUNTIME_PLURAL,
            body=body,
            _request_timeout=self.request_timeout,
        )

    def get(self, name: str) -> Dict[str, Any]:
        return self.api.get_namespaced_custom_object(
            group=RUNTIME_GROUP,
            version=RUNTIME_VERSION,
            namespace=self.namespace,
            plural=RUNTIME_PLURAL,
            name=name,
            _request_timeout=self.request_timeout,
        )

    def list(self, label_selector: Optional[str] = None) -> Dict[str, Any]:
        kwargs = {
            "group": RUNTIME_GROUP,
            "version": RUNTIME_VERSION,
            "namespace": self.namespace,
            "plural": RUNTIME_PLURAL,
            "_request_timeout": self.request_timeout,
        }
        if label_selector:
            kwargs["label_selector"] = label_selector
        return self.api.list_namespaced_custom_object(**kwargs)

    @staticmethod
    def _ready_for_expectation(
        obj: Mapping[str, Any],
        expected_revision: int,
        condition_types: Sequence[str],
    ) -> bool:
        metadata = obj.get("metadata") or {}
        spec = obj.get("spec") or {}
        status = obj.get("status") or {}
        generation = int(metadata.get("generation") or 0)
        if int(spec.get("deploymentRevision") or 0) != expected_revision:
            return False
        if int(status.get("observedGeneration") or 0) != generation:
            return False
        if int(status.get("observedRevision") or 0) != expected_revision:
            return False

        conditions = _condition_map(obj)
        rejected = conditions.get("SpecAccepted")
        if rejected and str(rejected.get("status")).lower() == "false":
            reason = rejected.get("reason") or "Rejected"
            message = rejected.get("message") or "RuntimeService spec was rejected"
            raise RuntimeServiceRejected(f"{metadata.get('name')}: {reason}: {message}")
        ready = all(
            condition_type in conditions and str(conditions[condition_type].get("status")).lower() == "true"
            for condition_type in condition_types
        )
        if ready:
            # Sedna hashes the typed Go RuntimeServiceSpec.  Dayu must consume
            # that controller-owned value rather than reproduce Go JSON field
            # ordering in Python.
            _observed_spec_hash(obj)
        return ready

    @classmethod
    def bind_observed_unit(
        cls,
        planned: RuntimeUnit,
        obj: Mapping[str, Any],
        condition_types: Sequence[str] = ("Ready", "Activated"),
    ) -> RuntimeUnit:
        """Bind a planned unit to an exact, successfully observed CR status.

        The returned unit contains Sedna's authoritative ``observedSpecHash``
        and, for routable workers, the exact RuntimeService, Service and Pod
        UIDs which EdgeMesh acknowledged.  This method is intended to be called
        on the object returned by :meth:`wait_for_conditions` before committing
        a RuntimeDirectory.
        """

        if not isinstance(planned, RuntimeUnit):
            raise TypeError("planned must be a RuntimeUnit")
        metadata = obj.get("metadata") or {}
        spec = obj.get("spec") or {}
        status = obj.get("status") or {}
        name = str(metadata.get("name") or "")
        if name != planned.runtime_id:
            raise RuntimeServiceInvalidStatus(
                f"observed RuntimeService name {name!r} does not match planned runtime {planned.runtime_id!r}"
            )
        if not cls._ready_for_expectation(obj, planned.runtime_revision, tuple(condition_types)):
            raise RuntimeServiceInvalidStatus(
                f"{name}: RuntimeService has not satisfied {tuple(condition_types)!r}"
            )

        observed_hash = _observed_spec_hash(obj)
        runtime_uid = str(metadata.get("uid") or "")
        pod_ref = status.get("podRef") or {}
        pod_name = str(pod_ref.get("name") or "")
        pod_uid = str(pod_ref.get("uid") or "")
        if not all((runtime_uid, pod_name, pod_uid)):
            raise RuntimeServiceInvalidStatus(
                f"{name}: ready workload status is missing RuntimeService or Pod identity"
            )
        endpoint = planned.endpoint
        if endpoint is not None:
            endpoint_status = status.get("endpoint") or {}
            service_ref = endpoint_status.get("serviceRef") or {}
            dns_name = str(endpoint_status.get("dnsName") or "")
            try:
                port = int(endpoint_status.get("port") or 0)
            except (TypeError, ValueError) as exc:
                raise RuntimeServiceInvalidStatus(f"{name}: status.endpoint.port is invalid") from exc
            service_uid = str(service_ref.get("uid") or "")
            pod_uid = str(pod_ref.get("uid") or "")
            if not all((runtime_uid, dns_name, service_uid, pod_uid)):
                raise RuntimeServiceInvalidStatus(
                    f"{name}: ready endpoint status is missing an acknowledged object identity"
                )
            if port != endpoint.port:
                raise RuntimeServiceInvalidStatus(
                    f"{name}: observed endpoint port {port} does not match planned port {endpoint.port}"
                )
            endpoint = RuntimeEndpoint(
                dns_name=dns_name,
                port=port,
                runtime_service_uid=runtime_uid,
                service_uid=service_uid,
                pod_uid=pod_uid,
            )
        elif status.get("endpoint") is not None:
            raise RuntimeServiceInvalidStatus(
                f"{name}: endpoint-less planned runtime unexpectedly reports status.endpoint"
            )

        return RuntimeUnit(
            slot=planned.slot,
            runtime_id=planned.runtime_id,
            runtime_revision=planned.runtime_revision,
            spec_hash=observed_hash,
            endpoint=endpoint,
            rollout_hash=planned.rollout_hash,
            runtime_service_uid=runtime_uid,
            pod_name=pod_name,
            pod_uid=pod_uid,
        )

    def wait_for_conditions(
        self,
        expectations: Mapping[str, Any],
        condition_types: Sequence[str] = ("Ready", "Activated"),
        timeout_seconds: float = 300,
        label_selector: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Wait for exact observed generation/revision and all requested gates.

        One initial list establishes a resourceVersion; the remaining objects are
        then followed through one namespaced watch instead of one poller per CR.
        """

        expected = {str(name): _expected_revision(value) for name, value in expectations.items()}
        if not expected:
            return {}
        timeout_seconds = float(timeout_seconds)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        conditions = tuple(condition_types)

        def ready_from_list(response: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
            """Build a fresh snapshot after every list/watch boundary.

            Retaining an object observed before a 410 Gone can publish a route
            for a RuntimeService that was deleted while another candidate was
            still activating.  A re-list is therefore a replacement snapshot,
            not an incremental merge.
            """

            snapshot: Dict[str, Dict[str, Any]] = {}
            for item in response.get("items") or ():
                name = str((item.get("metadata") or {}).get("name") or "")
                if name in expected and self._ready_for_expectation(
                    item, expected[name], conditions):
                    snapshot[name] = item
            return snapshot

        response = self.list(label_selector=label_selector)
        result = ready_from_list(response)
        if len(result) == len(expected):
            return result

        resource_version = str((response.get("metadata") or {}).get("resourceVersion") or "")
        deadline = time.monotonic() + timeout_seconds
        watcher = self.watch_factory()
        try:
            while len(result) < len(expected):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                stream_kwargs = {
                    "group": RUNTIME_GROUP,
                    "version": RUNTIME_VERSION,
                    "namespace": self.namespace,
                    "plural": RUNTIME_PLURAL,
                    "timeout_seconds": max(1, int(math.ceil(remaining))),
                    "_request_timeout": (
                        min(self.request_timeout, max(0.001, remaining)),
                        max(1, int(math.ceil(remaining))),
                    ),
                }
                if resource_version:
                    stream_kwargs["resource_version"] = resource_version
                if label_selector:
                    stream_kwargs["label_selector"] = label_selector

                saw_event = False
                try:
                    for event in watcher.stream(self.api.list_namespaced_custom_object, **stream_kwargs):
                        saw_event = True
                        obj = event.get("object") if isinstance(event, Mapping) else None
                        if not isinstance(obj, Mapping):
                            continue
                        metadata = obj.get("metadata") or {}
                        resource_version = str(metadata.get("resourceVersion") or resource_version)
                        name = str(metadata.get("name") or "")
                        if name not in expected:
                            continue
                        if str(event.get("type", "")).upper() == "DELETED":
                            result.pop(name, None)
                            continue
                        if self._ready_for_expectation(obj, expected[name], conditions):
                            result[name] = dict(obj)
                        else:
                            result.pop(name, None)
                        if len(result) == len(expected):
                            watcher.stop()
                            break
                except ApiException as exc:
                    if getattr(exc, "status", None) != 410:
                        raise
                    # The apiserver compacted this resourceVersion. Re-list to
                    # establish a new watch boundary instead of polling every CR.
                    watcher.stop()
                    watcher = self.watch_factory()
                    response = self.list(label_selector=label_selector)
                    resource_version = str(
                        (response.get("metadata") or {}).get("resourceVersion") or ""
                    )
                    result = ready_from_list(response)
                    continue
                if len(result) == len(expected):
                    break
                if not saw_event:
                    # A watch timeout is normal. Re-list to close any gap caused
                    # by server-side watch expiration before starting again.
                    response = self.list(label_selector=label_selector)
                    resource_version = str((response.get("metadata") or {}).get("resourceVersion") or resource_version)
                    result = ready_from_list(response)
        finally:
            watcher.stop()

        missing = sorted(set(expected) - set(result))
        if missing:
            raise RuntimeServiceTimeout(
                f"timed out waiting for RuntimeService conditions {tuple(condition_types)}: {missing}"
            )
        return result

    def delete(
        self,
        name: str,
        uid: Optional[str] = None,
        wait: bool = False,
        timeout_seconds: float = 120,
    ) -> bool:
        body: Dict[str, Any] = {
            "apiVersion": "v1",
            "kind": "DeleteOptions",
            "propagationPolicy": "Foreground",
        }
        if uid:
            body["preconditions"] = {"uid": str(uid)}
        try:
            self.api.delete_namespaced_custom_object(
                group=RUNTIME_GROUP,
                version=RUNTIME_VERSION,
                namespace=self.namespace,
                plural=RUNTIME_PLURAL,
                name=name,
                body=body,
                _request_timeout=self.request_timeout,
            )
        except ApiException as exc:
            if getattr(exc, "status", None) == 404:
                return True
            raise

        if not wait:
            return True
        deadline = time.monotonic() + float(timeout_seconds)
        while time.monotonic() < deadline:
            try:
                self.get(name)
            except ApiException as exc:
                if getattr(exc, "status", None) == 404:
                    return True
                raise
            time.sleep(0.2)
        raise RuntimeServiceTimeout(f"timed out waiting for RuntimeService {name!r} deletion")

    def delete_many(
        self,
        identities: Mapping[str, Optional[str]],
        timeout_seconds: float = 120,
        label_selector: Optional[str] = None,
    ) -> bool:
        """Delete exact RuntimeService UIDs and confirm absence with one watch.

        Kubernetes has no collection delete with per-object UID preconditions,
        so the DELETE requests remain one per immutable resource. Completion is
        nevertheless observed through one list/watch boundary instead of a GET
        polling loop per resource.
        """

        identities = {
            str(name): str(uid or "")
            for name, uid in (identities or {}).items()
            if str(name)
        }
        if not identities:
            return True
        missing_uids = sorted(name for name, uid in identities.items() if not uid)
        if missing_uids:
            raise ValueError(
                f"delete_many requires immutable UIDs for every RuntimeService: {missing_uids}"
            )
        timeout_seconds = float(timeout_seconds)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        for name in sorted(identities):
            self.delete(name, uid=identities[name], wait=False)

        expected = set(identities)

        def remaining_from_list(response: Mapping[str, Any]) -> set:
            present = {
                str((item.get("metadata") or {}).get("name") or "")
                for item in response.get("items") or ()
            }
            return expected & present

        response = self.list(label_selector=label_selector)
        remaining_names = remaining_from_list(response)
        if not remaining_names:
            return True
        resource_version = str((response.get("metadata") or {}).get("resourceVersion") or "")
        deadline = time.monotonic() + timeout_seconds
        watcher = self.watch_factory()
        try:
            while remaining_names:
                remaining_time = deadline - time.monotonic()
                if remaining_time <= 0:
                    break
                stream_kwargs = {
                    "group": RUNTIME_GROUP,
                    "version": RUNTIME_VERSION,
                    "namespace": self.namespace,
                    "plural": RUNTIME_PLURAL,
                    "timeout_seconds": max(1, int(math.ceil(remaining_time))),
                    "_request_timeout": (
                        min(self.request_timeout, max(0.001, remaining_time)),
                        max(1, int(math.ceil(remaining_time))),
                    ),
                }
                if resource_version:
                    stream_kwargs["resource_version"] = resource_version
                if label_selector:
                    stream_kwargs["label_selector"] = label_selector
                saw_event = False
                try:
                    for event in watcher.stream(
                        self.api.list_namespaced_custom_object, **stream_kwargs):
                        saw_event = True
                        obj = event.get("object") if isinstance(event, Mapping) else None
                        if not isinstance(obj, Mapping):
                            continue
                        metadata = obj.get("metadata") or {}
                        resource_version = str(
                            metadata.get("resourceVersion") or resource_version
                        )
                        name = str(metadata.get("name") or "")
                        if (
                            name in expected
                            and str(event.get("type", "")).upper() == "DELETED"):
                            remaining_names.discard(name)
                        if not remaining_names:
                            watcher.stop()
                            break
                except ApiException as exc:
                    if getattr(exc, "status", None) != 410:
                        raise
                    watcher.stop()
                    watcher = self.watch_factory()
                    response = self.list(label_selector=label_selector)
                    resource_version = str(
                        (response.get("metadata") or {}).get("resourceVersion") or ""
                    )
                    remaining_names = remaining_from_list(response)
                    continue
                if not remaining_names:
                    break
                if not saw_event:
                    response = self.list(label_selector=label_selector)
                    resource_version = str(
                        (response.get("metadata") or {}).get("resourceVersion")
                        or resource_version
                    )
                    remaining_names = remaining_from_list(response)
        finally:
            watcher.stop()
        if remaining_names:
            raise RuntimeServiceTimeout(
                f"timed out waiting for RuntimeService deletion: {sorted(remaining_names)}"
            )
        return True
