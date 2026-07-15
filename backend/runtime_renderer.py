"""Compile Dayu logical component templates into Sedna RuntimeServices."""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from runtime_model import (
    RuntimeEndpoint,
    RuntimeSlot,
    RuntimeUnit,
    canonical_hash,
)

RUNTIME_API_VERSION = "sedna.io/v1alpha1"
RUNTIME_KIND = "RuntimeService"
DATA_PATH_PREFIX = "/home/data"
FORBIDDEN_RUNTIME_ENV = frozenset({
    "KUBECONFIG",
    "KUBERNETES_SERVICE_HOST",
    "KUBERNETES_SERVICE_PORT",
    "KUBE_CACHE_TTL",
    "KUBE_CACHE_WARMUP_TIMEOUT",
    "KUBE_POD_LABEL_SELECTOR",
    "KUBE_POD_FIELD_SELECTOR",
    "KUBE_SERVICE_LABEL_SELECTOR",
})
_HOST_PATH_TYPES = frozenset({
    "Directory", "DirectoryOrCreate", "File", "FileOrCreate",
    "Socket", "CharDevice", "BlockDevice",
})


def _deep_merge(base: Mapping[str, Any], overlay: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(base or {}))
    for key, value in (overlay or {}).items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _dns_label(value: str, fallback: str = "runtime") -> str:
    value = re.sub(r"[^a-z0-9-]+", "-", str(value).lower())
    value = re.sub(r"-+", "-", value).strip("-")
    if not value or not value[0].isalpha():
        value = f"{fallback}-{value}" if value else fallback
    return value[:63].rstrip("-")


def _env_dict(items: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    result = {}
    for item in items or ():
        if not isinstance(item, Mapping) or not item.get("name"):
            continue
        result[str(item["name"])] = copy.deepcopy(dict(item))
    return result


def _upsert_env(container: Dict[str, Any], values: Mapping[str, Any]) -> None:
    env = _env_dict(container.get("env") or ())
    env_names = set(env) | {str(key) for key in values}
    forbidden_names = sorted(
        name for name in env_names
        if name in FORBIDDEN_RUNTIME_ENV
        or name.startswith("KUBERNETES_")
        or name.startswith("KUBE_")
    )
    if forbidden_names:
        raise ValueError(
            f"runtime templates must not define Kubernetes discovery/cache env: {forbidden_names}"
        )
    for key, value in values.items():
        if value is None:
            env.pop(str(key), None)
        else:
            env[str(key)] = {"name": str(key), "value": str(value)}
    container["env"] = [env[key] for key in sorted(env)]


def _position_matches(configured: str, actual: str) -> bool:
    return str(configured or "both").lower() in {"both", actual}


@dataclass(frozen=True)
class RenderedRuntimeService:
    manifest: Dict[str, Any]
    unit: RuntimeUnit


class RuntimeServiceRenderer:
    """Pure renderer for the additive Sedna RuntimeService API.

    The renderer never loads Kubernetes configuration.  All topology and
    placement decisions must already be present in ``RuntimeSlot`` or explicit
    environment values supplied by the backend control plane.
    """

    def __init__(
        self,
        namespace: str,
        install_id: str,
        log_level: str = "INFO",
        file_mount_prefix: str = "",
        image_resolver: Optional[Callable[[str], str]] = None,
        data_path_prefix: str = DATA_PATH_PREFIX,
    ):
        self.namespace = str(namespace or "").strip()
        self.install_id = str(install_id or "").strip()
        self.log_level = str(log_level or "INFO")
        self.file_mount_prefix = str(file_mount_prefix or "")
        self.image_resolver = image_resolver or (lambda value: value)
        self.data_path_prefix = os.path.normpath(str(data_path_prefix or DATA_PATH_PREFIX))
        if not self.namespace:
            raise ValueError("namespace must be non-empty")
        if not self.install_id:
            raise ValueError("install_id must be non-empty")

    def _container_template(self, logical_template: Mapping[str, Any], position: str) -> Dict[str, Any]:
        base = logical_template.get("pod-template") or {}
        overlay = logical_template.get(f"{position}-pod-template") or {}
        container = _deep_merge(base, overlay)
        if not container.get("image"):
            raise ValueError("logical template pod-template.image must be non-empty")
        return container

    @staticmethod
    def _endpoint_port(logical_template: Mapping[str, Any]) -> Optional[int]:
        port_open = logical_template.get("port-open")
        if not port_open:
            return None
        try:
            port = int(port_open["port"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("port-open.port must be an integer")
        if port < 1 or port > 65535:
            raise ValueError("port-open.port must be in range 1..65535")
        return port

    def _render_mounts(
        self,
        logical_template: Mapping[str, Any],
        position: str,
        containers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        mounts = [
            copy.deepcopy(item)
            for item in (logical_template.get("file-mount") or ())
            if _position_matches(item.get("pos", "both"), position)
        ]
        # Preserve the temporary shared directory contract previously injected
        # by TemplateHelper/Sedna JMES, but render it as native Pod fields.
        mounts.append({
            "name": "temporary-directory",
            "path": "temp/",
            "type": "DirectoryOrCreate",
            "target_path": "/temp",
            "env_name": "TEMP_PATH",
        })

        if mounts and not mounts[0].get("target_path") and not mounts[0].get("env_name"):
            mounts[0]["env_name"] = "DEFAULT_MOUNT_PATH"

        volumes: List[Dict[str, Any]] = []
        volume_names = set()
        container_indexes = {container["name"]: index for index, container in enumerate(containers)}
        for index, mount in enumerate(mounts):
            source_path = str(mount.get("path") or "").strip()
            if not source_path:
                raise ValueError(f"file-mount[{index}].path must be non-empty")
            path_type = str(mount.get("type") or "Directory")
            if path_type not in _HOST_PATH_TYPES:
                raise ValueError(f"file-mount[{index}].type {path_type!r} is not a valid hostPath type")

            volume_name = _dns_label(mount.get("name") or f"mount-{index}", fallback="mount")
            if volume_name in volume_names:
                raise ValueError(f"duplicate rendered volume name {volume_name!r}")
            volume_names.add(volume_name)

            if os.path.isabs(source_path):
                host_path = os.path.normpath(source_path)
            elif self.file_mount_prefix:
                host_path = os.path.normpath(os.path.join(self.file_mount_prefix, source_path))
            else:
                host_path = os.path.normpath(source_path)

            target_path = mount.get("target_path")
            if target_path:
                target_path = os.path.normpath(str(target_path))
                if not os.path.isabs(target_path):
                    raise ValueError(f"file-mount[{index}].target_path must be absolute")
            elif os.path.isabs(source_path):
                target_path = os.path.normpath(source_path)
            else:
                target_path = os.path.normpath(os.path.join(self.data_path_prefix, source_path))

            volumes.append({
                "name": volume_name,
                "hostPath": {"path": host_path, "type": path_type},
            })
            volume_mount = {
                "name": volume_name,
                "mountPath": target_path,
                "readOnly": bool(mount.get("read_only", False)),
            }
            if mount.get("sub_path"):
                volume_mount["subPath"] = str(mount["sub_path"])
            if mount.get("mount_propagation"):
                volume_mount["mountPropagation"] = str(mount["mount_propagation"])

            selected = mount.get("containers")
            if selected is None:
                indexes = range(len(containers))
            else:
                unknown = sorted(set(selected) - set(container_indexes))
                if unknown:
                    raise ValueError(f"file-mount[{index}] references unknown containers {unknown}")
                indexes = [container_indexes[name] for name in selected]

            for container_index in indexes:
                container = containers[container_index]
                existing_mounts = container.setdefault("volumeMounts", [])
                if any(item.get("name") == volume_name for item in existing_mounts):
                    raise ValueError(
                        f"container {container['name']!r} already has volumeMount {volume_name!r}"
                    )
                existing_mounts.append(copy.deepcopy(volume_mount))
                env_name = mount.get("env_name")
                if env_name:
                    _upsert_env(container, {str(env_name): target_path})

        return volumes

    def render(
        self,
        logical_template: Mapping[str, Any],
        slot: RuntimeSlot,
        revision: int,
        extra_env: Optional[Mapping[str, Any]] = None,
        container_overrides: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> RenderedRuntimeService:
        expected_position = str(logical_template.get("position") or "").lower()
        if expected_position not in {slot.position, "both"}:
            raise ValueError(
                f"logical template position {expected_position!r} cannot render a {slot.position!r} slot"
            )

        runtime_id = slot.runtime_name(revision, self.install_id)
        base_container = self._container_template(logical_template, slot.position)
        raw_containers = list(container_overrides or (base_container,))
        containers: List[Dict[str, Any]] = []
        for index, raw_container in enumerate(raw_containers):
            container = _deep_merge(base_container, raw_container)
            container["image"] = self.image_resolver(container["image"])
            container["name"] = _dns_label(
                container.get("name") or (slot.component if index == 0 else f"{slot.component}-{index}"),
                fallback="runtime",
            )
            _upsert_env(container, {
                "NAMESPACE": self.namespace,
                "SERVICE_NAME": runtime_id,
                "LOG_LEVEL": self.log_level,
                "NODE_NAME": slot.target_node,
                "NODE_ROLE": slot.position,
                "DATA_PATH_PREFIX": self.data_path_prefix,
                **dict(extra_env or {}),
            })
            containers.append(container)

        if len({container["name"] for container in containers}) != len(containers):
            raise ValueError("rendered container names must be unique")

        port_open = logical_template.get("port-open") or {}
        port = self._endpoint_port(logical_template) if _position_matches(
            port_open.get("pos", "both"), slot.position,
        ) else None
        if port is not None:
            for container in containers:
                _upsert_env(container, {"GUNICORN_PORT": port})
            declared = containers[0].setdefault("ports", [])
            if not any(int(item.get("containerPort", 0)) == port for item in declared):
                declared.append({"name": "runtime", "containerPort": port, "protocol": "TCP"})

        volumes = self._render_mounts(logical_template, slot.position, containers)
        pod_spec: Dict[str, Any] = {
            "automountServiceAccountToken": False,
            "dnsPolicy": "ClusterFirst",
            "enableServiceLinks": False,
            "restartPolicy": "Always",
            "containers": containers,
            "volumes": volumes,
        }
        # RuntimeService owns placement.  Deliberately omit nodeName and every
        # service-account field from the user-supplied PodTemplate.
        pod_template = {
            "metadata": {
                "labels": {
                    "app.kubernetes.io/part-of": "dayu",
                    "app.kubernetes.io/managed-by": "dayu-backend",
                }
            },
            "spec": pod_spec,
        }

        manifest: Dict[str, Any] = {
            "apiVersion": RUNTIME_API_VERSION,
            "kind": RUNTIME_KIND,
            "metadata": {
                "name": runtime_id,
                "namespace": self.namespace,
                "labels": {
                    "app.kubernetes.io/part-of": "dayu",
                    "app.kubernetes.io/managed-by": "dayu-backend",
                    "dayu.io/runtime-scope": "installation",
                    "dayu.io/install-id": self.install_id,
                    "dayu.io/component": slot.component,
                },
            },
            "spec": {
                "installID": self.install_id,
                "deploymentRevision": int(revision),
                "component": slot.component,
                "targetNode": slot.target_node,
                "podTemplate": pod_template,
            },
        }
        if slot.logical_service:
            manifest["spec"]["logicalService"] = slot.logical_service
        if port is not None:
            manifest["spec"]["endpoint"] = {"port": port}

        endpoint = None
        if "endpoint" in manifest["spec"]:
            endpoint = RuntimeEndpoint(
                dns_name=f"{runtime_id}.{self.namespace}.svc.cluster.local",
                port=port,
            )
        rollout_spec = copy.deepcopy(manifest["spec"])
        rollout_spec["deploymentRevision"] = 0
        for rollout_container in rollout_spec["podTemplate"]["spec"].get("containers", []):
            rollout_container["env"] = [
                item for item in (rollout_container.get("env") or [])
                if item.get("name") not in {"SERVICE_NAME", "DAYU_RUNTIME_BOOTSTRAP"}
            ]
        unit = RuntimeUnit(
            slot=slot,
            runtime_id=runtime_id,
            runtime_revision=int(revision),
            spec_hash=canonical_hash(manifest["spec"]),
            endpoint=endpoint,
            rollout_hash=canonical_hash(rollout_spec),
        )
        return RenderedRuntimeService(manifest=manifest, unit=unit)

    @staticmethod
    def _generator_dag(dag: Mapping[str, Any]) -> Dict[str, Any]:
        result = {}
        for key, value in (dag or {}).items():
            if str(key) == "start":
                continue
            result[str(key)] = {
                "service": {"service_name": str(key)},
                "prev_nodes": copy.deepcopy(value.get("prev", value.get("prev_nodes", []))),
                "next_nodes": copy.deepcopy(value.get("succ", value.get("next_nodes", []))),
            }
        return result

    def render_generator_sources(
        self,
        logical_template: Mapping[str, Any],
        source_deploy: Sequence[Mapping[str, Any]],
        revision: int,
        selected_nodes: Optional[Mapping[Any, str]] = None,
        common_env: Optional[Mapping[str, Any]] = None,
    ) -> List[RenderedRuntimeService]:
        """Render exactly one generator RuntimeService for each source."""

        rendered = []
        seen_source_ids = set()
        for source_info in source_deploy or ():
            source = source_info.get("source") or {}
            source_id = str(source.get("id", ""))
            if not source_id:
                raise ValueError("every generator source requires source.id")
            if source_id in seen_source_ids:
                raise ValueError(f"duplicate generator source.id {source_id!r}")
            seen_source_ids.add(source_id)

            selected = (selected_nodes or {}).get(source.get("id"))
            selected = selected or (selected_nodes or {}).get(source_id)
            selected = selected or source.get("source_device") or source_info.get("source_device")
            node_set = list(source_info.get("node_set") or ())
            selected = selected or (node_set[0] if node_set else None)
            if not selected:
                raise ValueError(f"source {source_id!r} has no selected target node")

            slot = RuntimeSlot(
                component="generator",
                target_node=selected,
                position="edge",
                source_id=source_id,
            )
            container = self._container_template(logical_template, "edge")
            container["name"] = f"generator-source-{source_id}"
            generator_env = {
                **dict(common_env or {}),
                "GEN_GETTER_NAME": source.get("source_mode", ""),
                "SOURCE_URL": source.get("url", ""),
                "SOURCE_TYPE": source.get("source_type", ""),
                "SOURCE_ID": source_id,
                "SOURCE_METADATA": str(source.get("metadata", {})),
                "ALL_EDGE_DEVICES": str(node_set),
                "DAG": str(self._generator_dag(source_info.get("dag") or {})),
            }
            rendered.append(self.render(
                logical_template=logical_template,
                slot=slot,
                revision=revision,
                extra_env=generator_env,
                container_overrides=[container],
            ))
        return rendered
