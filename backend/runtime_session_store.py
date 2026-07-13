"""Optimistic-concurrency persistence for managed-runtime transactions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from kubernetes.client.rest import ApiException

from runtime_model import RuntimeSession, canonical_hash, canonical_json

SESSION_DATA_KEY = "session.json"
SESSION_HASH_KEY = "session.sha256"


class RuntimeSessionStoreError(RuntimeError):
    pass


class RuntimeSessionConflict(RuntimeSessionStoreError):
    pass


class RuntimeSessionCorrupt(RuntimeSessionStoreError):
    pass


@dataclass(frozen=True)
class StoredRuntimeSession:
    session: RuntimeSession
    resource_version: str
    uid: str = ""


def _metadata_value(configmap: Any, key: str) -> str:
    metadata = configmap.get("metadata") if isinstance(configmap, Mapping) else getattr(configmap, "metadata", None)
    if isinstance(metadata, Mapping):
        return str(metadata.get(key, metadata.get("resource_version" if key == "resourceVersion" else key, "")) or "")
    attr = "resource_version" if key == "resourceVersion" else key
    return str(getattr(metadata, attr, "") or "")


def _data(configmap: Any) -> Mapping[str, str]:
    if isinstance(configmap, Mapping):
        return configmap.get("data") or {}
    return getattr(configmap, "data", None) or {}


class RuntimeSessionStore:
    """Persist compact RuntimeSession records in one ConfigMap with CAS writes."""

    def __init__(
        self,
        namespace: str,
        name: str = "dayu-runtime-session",
        api=None,
        request_timeout_seconds: float = 30,
    ):
        self.namespace = str(namespace or "").strip()
        self.name = str(name or "").strip()
        if not self.namespace or not self.name:
            raise ValueError("namespace and name must be non-empty")
        if api is None:
            raise ValueError("api must be the shared ClusterClient CoreV1Api")
        self.api = api
        self.request_timeout = float(request_timeout_seconds)
        if self.request_timeout <= 0:
            raise ValueError("request_timeout_seconds must be positive")

    def _decode(self, configmap: Any) -> StoredRuntimeSession:
        data = _data(configmap)
        raw = data.get(SESSION_DATA_KEY)
        if not raw:
            raise RuntimeSessionCorrupt(f"ConfigMap {self.name!r} is missing {SESSION_DATA_KEY!r}")
        expected_hash = str(data.get(SESSION_HASH_KEY) or "")
        try:
            value = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeSessionCorrupt(f"ConfigMap {self.name!r} contains invalid session JSON") from exc
        actual_hash = canonical_hash(value)
        if expected_hash and expected_hash != actual_hash:
            raise RuntimeSessionCorrupt(
                f"ConfigMap {self.name!r} session hash mismatch: expected {expected_hash}, got {actual_hash}"
            )
        try:
            session = RuntimeSession.from_dict(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeSessionCorrupt(f"ConfigMap {self.name!r} contains an invalid RuntimeSession") from exc
        resource_version = _metadata_value(configmap, "resourceVersion")
        uid = _metadata_value(configmap, "uid")
        if not resource_version or not uid:
            raise RuntimeSessionCorrupt(
                f"ConfigMap {self.name!r} is missing immutable metadata identity"
            )
        return StoredRuntimeSession(
            session=session,
            resource_version=resource_version,
            uid=uid,
        )

    def load(self) -> Optional[StoredRuntimeSession]:
        try:
            configmap = self.api.read_namespaced_config_map(
                name=self.name,
                namespace=self.namespace,
                _request_timeout=self.request_timeout,
            )
        except ApiException as exc:
            if getattr(exc, "status", None) == 404:
                return None
            raise
        return self._decode(configmap)

    @staticmethod
    def _encoded_data(session: RuntimeSession) -> Mapping[str, str]:
        value = session.to_dict()
        return {
            SESSION_DATA_KEY: canonical_json(value),
            SESSION_HASH_KEY: canonical_hash(value),
        }

    def compare_and_swap(
        self,
        session: RuntimeSession,
        expected_resource_version: Optional[str],
    ) -> StoredRuntimeSession:
        if not isinstance(session, RuntimeSession):
            raise TypeError("session must be a RuntimeSession")
        metadata = {
            "name": self.name,
            "namespace": self.namespace,
            "labels": {
                "app.kubernetes.io/part-of": "dayu",
                "app.kubernetes.io/managed-by": "dayu-backend",
                "dayu.io/install-id": session.install_id,
            },
        }
        body = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": metadata,
            "data": dict(self._encoded_data(session)),
        }
        try:
            if expected_resource_version is None:
                response = self.api.create_namespaced_config_map(
                    namespace=self.namespace,
                    body=body,
                    _request_timeout=self.request_timeout,
                )
            else:
                metadata["resourceVersion"] = str(expected_resource_version)
                response = self.api.replace_namespaced_config_map(
                    name=self.name,
                    namespace=self.namespace,
                    body=body,
                    _request_timeout=self.request_timeout,
                )
        except ApiException as exc:
            if getattr(exc, "status", None) in {409, 422}:
                raise RuntimeSessionConflict(
                    f"RuntimeSession CAS conflict for ConfigMap {self.name!r} at "
                    f"resourceVersion={expected_resource_version!r}"
                ) from exc
            raise
        return self._decode(response)

    def delete(self, expected_resource_version: Optional[str] = None) -> bool:
        stored = self.load()
        if stored is None:
            return True
        if expected_resource_version is not None and stored.resource_version != str(expected_resource_version):
            raise RuntimeSessionConflict(
                f"RuntimeSession delete conflict: expected resourceVersion={expected_resource_version!r}, "
                f"found {stored.resource_version!r}"
            )
        body = {
            "apiVersion": "v1",
            "kind": "DeleteOptions",
            "preconditions": {"uid": stored.uid},
        }
        try:
            self.api.delete_namespaced_config_map(
                name=self.name,
                namespace=self.namespace,
                body=body,
                _request_timeout=self.request_timeout,
            )
        except ApiException as exc:
            if getattr(exc, "status", None) == 404:
                return True
            if getattr(exc, "status", None) in {409, 422}:
                raise RuntimeSessionConflict(f"RuntimeSession delete conflict for {self.name!r}") from exc
            raise
        return True
