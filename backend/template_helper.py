"""Pure compiler for Dayu's logical template catalog.

This module deliberately has no Kubernetes client and performs no HTTP calls.
It only loads immutable catalog data, normalizes application DAGs, and delegates
RuntimeService manifest rendering to :mod:`runtime_renderer`.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Tuple

from core.lib.common import TaskConstant, YamlOps
from runtime_model import RuntimeSlot
from runtime_renderer import RuntimeServiceRenderer

_POLICY_COMPONENTS = ("generator", "controller", "distributor", "monitor")
_IMAGE_RE = re.compile(
    r"^(?:(?P<registry>[^/]+)/(?=.*/))?"
    r"(?:(?P<repository>[^/:]+)/)?"
    r"(?P<image>[^/:]+)"
    r"(?::(?P<tag>[^:]+))?$"
)


class TemplateHelper:
    """Load and compile trusted Dayu templates without runtime side effects."""

    def __init__(self, templates_dir: str):
        self.templates_dir = os.path.realpath(str(templates_dir))
        self._base_info = None

    def _read_template(self, *parts: str):
        for part in map(str, parts):
            if os.path.isabs(part) or part in {"", ".", ".."} or ".." in part.split(os.sep):
                raise ValueError(f"invalid template path segment: {part!r}")
        path = os.path.realpath(os.path.join(self.templates_dir, *map(str, parts)))
        if os.path.commonpath((self.templates_dir, path)) != self.templates_dir:
            raise ValueError(f"template path escapes catalog root: {os.path.join(*parts)!r}")
        if not os.path.isfile(path):
            raise ValueError(f"template file does not exist: {os.path.relpath(path, self.templates_dir)!r}")
        document = YamlOps.read_yaml(path)
        if not isinstance(document, Mapping):
            raise ValueError(f"template must contain one YAML object: {os.path.relpath(path, self.templates_dir)!r}")
        return copy.deepcopy(dict(document))

    def load_base_info(self) -> Dict[str, Any]:
        """Return an isolated copy of the base catalog.

        The catalog is read once because it is installation configuration, not
        mutable runtime state. Returning a deep copy prevents request handlers
        from changing the cached catalog accidentally.
        """

        if self._base_info is None:
            self._base_info = self._read_template("base.yaml")
        return copy.deepcopy(self._base_info)

    def load_policy_apply_yaml(self, policy: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        if not isinstance(policy, Mapping):
            raise ValueError("policy must be an object")
        scheduler_yaml = policy.get("yaml")
        if not scheduler_yaml:
            raise ValueError("policy.yaml must be non-empty")

        dependency = policy.get("dependency")
        if dependency is None:
            dependency = {name: policy[name] for name in _POLICY_COMPONENTS if policy.get(name)}
        if not isinstance(dependency, Mapping):
            raise ValueError("policy.dependency must be an object")

        missing = [name for name in _POLICY_COMPONENTS if not dependency.get(name)]
        if missing:
            raise ValueError(f"policy is missing component templates: {missing}")

        documents = {"scheduler": self._read_template("scheduler", scheduler_yaml)}
        for component in _POLICY_COMPONENTS:
            documents[component] = self._read_template(component, dependency[component])
        return documents

    def load_application_apply_yaml(
        self, service_dict: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Attach processor templates without mutating the compiled catalog."""

        if not isinstance(service_dict, Mapping):
            raise ValueError("service_dict must be an object")
        loaded = copy.deepcopy(dict(service_dict))
        for service_id, service_info in loaded.items():
            if not isinstance(service_info, Mapping):
                raise ValueError(f"service {service_id!r} must be an object")
            yaml_name = service_info.get("yaml")
            if not yaml_name:
                raise ValueError(f"service {service_id!r} is missing yaml")
            service_info["service"] = self._read_template("processor", yaml_name)
        return loaded

    def _service_catalog(self) -> Dict[str, Dict[str, Any]]:
        services = self.load_base_info().get("services")
        if not isinstance(services, list):
            raise ValueError("base.services must be a list")

        catalog = {}
        for index, raw_service in enumerate(services):
            if not isinstance(raw_service, Mapping):
                raise ValueError(f"base.services[{index}] must be an object")
            service = copy.deepcopy(dict(raw_service))
            catalog_id = str(service.get("id") or "").strip()
            service_name = str(service.get("service") or "").strip()
            yaml_name = str(service.get("yaml") or "").strip()
            if not catalog_id or not service_name or not yaml_name:
                raise ValueError(f"base.services[{index}] requires id, service, and yaml")
            if catalog_id in catalog:
                raise ValueError(f"duplicate service catalog id {catalog_id!r}")
            catalog[catalog_id] = service
        return catalog

    @staticmethod
    def _resolve_catalog_service(
        raw_key: str,
        raw_node: Mapping[str, Any],
        catalog: Mapping[str, Mapping[str, Any]],
        catalog_by_name: Mapping[str, list],
    ):
        embedded = raw_node.get("service")
        candidates = []
        if isinstance(embedded, Mapping):
            candidates.append(embedded.get("id"))
        candidates.extend((raw_node.get("service_id"), raw_node.get("id"), raw_key))
        candidates = [str(value) for value in candidates if value is not None and str(value)]

        for candidate in candidates:
            if candidate in catalog:
                return candidate, catalog[candidate]
        for candidate in candidates:
            matches = catalog_by_name.get(candidate) or []
            if len(matches) == 1:
                catalog_id = matches[0]
                return catalog_id, catalog[catalog_id]
            if len(matches) > 1:
                raise ValueError(
                    f"DAG node {raw_key!r} uses ambiguous logical service name {candidate!r}; "
                    "preserve the catalog service id"
                )
        requested = candidates[0] if candidates else raw_key
        raise ValueError(f"DAG node {raw_key!r} references unknown service id {requested!r}")

    @staticmethod
    def _reference_list(value: Any, field: str, node_id: str):
        if value is None:
            return []
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValueError(f"DAG node {node_id!r}.{field} must be a list")
        return [str(item) for item in value]

    def normalize_source_deploy(
        self, source_deploy: Sequence[Mapping[str, Any]],
    ) -> Tuple[list, Dict[str, Dict[str, Any]]]:
        """Compile DAG catalog IDs into unambiguous logical service names.

        Returns ``(normalized_source_deploy, service_dict)``.  The normalized
        DAG contains no synthetic ``_start`` node; every service key, node ID,
        and ``prev``/``succ`` reference uses the catalog's runtime ``service``
        name. ``service_dict`` is keyed by that same logical name and merges the
        candidate nodes from every source that uses it.

        Two catalog entries that map to the same runtime service name cannot be
        used in one installation because placement and routing are keyed by
        logical service name. Such ambiguity is rejected instead of silently
        selecting one processor image.
        """

        if isinstance(source_deploy, (str, bytes)) or not isinstance(source_deploy, Sequence):
            raise ValueError("source_deploy must be a list")

        catalog = self._service_catalog()
        catalog_by_name: Dict[str, list] = {}
        for catalog_id, service in catalog.items():
            catalog_by_name.setdefault(str(service["service"]), []).append(catalog_id)
        normalized_sources = copy.deepcopy(list(source_deploy))
        service_dict: Dict[str, Dict[str, Any]] = {}
        service_catalog_ids: Dict[str, str] = {}

        for source_index, source_info in enumerate(normalized_sources):
            if not isinstance(source_info, dict):
                raise ValueError(f"source_deploy[{source_index}] must be an object")
            dag = source_info.get("dag")
            if not isinstance(dag, Mapping):
                raise ValueError(f"source_deploy[{source_index}].dag must be an object")
            node_set = source_info.get("node_set") or []
            if isinstance(node_set, (str, bytes)) or not isinstance(node_set, Sequence):
                raise ValueError(f"source_deploy[{source_index}].node_set must be a list")
            normalized_nodes = []
            seen_nodes = set()
            for node in node_set:
                node = str(node).strip()
                if not node:
                    raise ValueError(f"source_deploy[{source_index}].node_set contains an empty node")
                if node not in seen_nodes:
                    normalized_nodes.append(node)
                    seen_nodes.add(node)
            source_info["node_set"] = normalized_nodes

            aliases: Dict[str, str] = {
                TaskConstant.START.value: TaskConstant.START.value,
                TaskConstant.END.value: TaskConstant.END.value,
            }
            compiled_nodes = []
            used_names = set()
            for raw_key, raw_node in dag.items():
                raw_key = str(raw_key)
                if raw_key in {TaskConstant.START.value, TaskConstant.END.value}:
                    continue
                if not isinstance(raw_node, Mapping):
                    raise ValueError(f"DAG node {raw_key!r} must be an object")
                catalog_id, service = self._resolve_catalog_service(
                    raw_key, raw_node, catalog, catalog_by_name,
                )
                service_name = str(service["service"])
                if service_name in used_names:
                    raise ValueError(
                        f"DAG maps multiple nodes to logical service name {service_name!r}; "
                        "logical service names must be unique within one DAG"
                    )
                used_names.add(service_name)
                for alias in (raw_key, catalog_id):
                    previous = aliases.get(alias)
                    if previous is not None and previous != service_name:
                        raise ValueError(f"DAG alias {alias!r} maps to multiple logical services")
                    aliases[alias] = service_name
                compiled_nodes.append((raw_key, catalog_id, service_name, raw_node, service))

            normalized_dag = {}
            for raw_key, catalog_id, service_name, raw_node, service in compiled_nodes:
                node = copy.deepcopy(dict(raw_node))
                node["id"] = service_name
                node["service"] = copy.deepcopy(service)
                for field in ("prev", "succ"):
                    references = self._reference_list(raw_node.get(field), field, raw_key)
                    unknown = [reference for reference in references if reference not in aliases]
                    if unknown:
                        raise ValueError(f"DAG node {raw_key!r}.{field} references unknown nodes {unknown}")
                    node[field] = [aliases[reference] for reference in references]
                normalized_dag[service_name] = node

                previous_catalog_id = service_catalog_ids.get(service_name)
                if previous_catalog_id is not None and previous_catalog_id != catalog_id:
                    raise ValueError(
                        f"logical service {service_name!r} is backed by both "
                        f"{previous_catalog_id!r} and {catalog_id!r}"
                    )
                service_catalog_ids[service_name] = catalog_id
                compiled = service_dict.setdefault(service_name, {
                    "catalog_id": catalog_id,
                    "service_name": service_name,
                    "yaml": service["yaml"],
                    "node": [],
                    "catalog": copy.deepcopy(service),
                })
                for node_name in normalized_nodes:
                    if node_name not in compiled["node"]:
                        compiled["node"].append(node_name)

            source_info["dag"] = normalized_dag

        return normalized_sources, service_dict

    def create_runtime_renderer(self, install_id: str) -> RuntimeServiceRenderer:
        base_info = self.load_base_info()
        return RuntimeServiceRenderer(
            namespace=base_info["namespace"],
            install_id=install_id,
            log_level=base_info.get("log-level", "INFO"),
            file_mount_prefix=base_info.get("default-file-mount-prefix", ""),
            image_resolver=self.process_image,
        )

    def render_runtime_service(
        self, logical_template, slot, revision, install_id, extra_env=None,
    ):
        if not isinstance(slot, RuntimeSlot):
            slot = RuntimeSlot.from_dict(slot)
        return self.create_runtime_renderer(install_id).render(
            logical_template=logical_template,
            slot=slot,
            revision=revision,
            extra_env=extra_env,
        )

    def render_generator_runtime_services(
        self, logical_template, source_deploy, revision, install_id,
        selected_nodes=None, common_env=None,
    ):
        return self.create_runtime_renderer(install_id).render_generator_sources(
            logical_template=logical_template,
            source_deploy=source_deploy,
            revision=revision,
            selected_nodes=selected_nodes,
            common_env=common_env,
        )

    @staticmethod
    def specify_jetpack_image(image: str, jetpack_major: int) -> str:
        if isinstance(jetpack_major, bool) or not isinstance(jetpack_major, int) or jetpack_major <= 0:
            return image
        return f"{image}-jp{jetpack_major}"

    def process_image(self, image: str) -> str:
        """Complete a catalog image reference with configured defaults."""

        if not isinstance(image, str):
            raise ValueError("image must be a string")
        match = _IMAGE_RE.fullmatch(image.strip())
        if not match:
            raise ValueError(f"Format of input image {image!r} is illegal")

        image_meta = self.load_base_info()["default-image-meta"]
        registry = match.group("registry") or image_meta["registry"]
        repository = match.group("repository") or image_meta["repository"]
        image_name = match.group("image")
        tag = match.group("tag") or image_meta["tag"]
        return f"{registry}/{repository}/{image_name}:{tag}"
