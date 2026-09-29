"""Authoritative source-node selection contract.

The Backend resolves the permitted source candidates from one Kubernetes
snapshot and persists them with the install session.  Scheduler policies only
consume that immutable permission set; they never expand it through runtime
discovery.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List


SELECTED_EDGE_NODES = "selected_edge_nodes"
ALL_EDGE_NODES = "all_edge_nodes"
VALID_SOURCE_SELECTION_SCOPES = frozenset({SELECTED_EDGE_NODES, ALL_EDGE_NODES})
SOURCE_CANDIDATE_NODES_FIELD = "source_candidate_nodes"
SOURCE_SELECTION_SCOPE_FIELD = "source_selection_scope"


def normalize_source_selection_scope(value: Any) -> str:
    """Return one exact supported scope or reject the configuration."""

    scope = str(value or SELECTED_EDGE_NODES).strip()
    if scope not in VALID_SOURCE_SELECTION_SCOPES:
        raise ValueError(
            f"source selection scope must be one of "
            f"{sorted(VALID_SOURCE_SELECTION_SCOPES)}, got {scope!r}"
        )
    return scope


def _node_list(value: Any, field: str) -> List[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a list")
    result = []
    seen = set()
    for raw_node in value:
        node = str(raw_node or "").strip()
        if not node:
            raise ValueError(f"{field} contains an empty node")
        if node not in seen:
            result.append(node)
            seen.add(node)
    return result


def source_selection_candidates(info: Mapping[str, Any], scope: Any) -> List[str]:
    """Return the Backend-authorized candidates for one source decision."""

    normalized_scope = normalize_source_selection_scope(scope)
    declared_scope = info.get(SOURCE_SELECTION_SCOPE_FIELD)
    if declared_scope is not None:
        declared_scope = normalize_source_selection_scope(declared_scope)
        if declared_scope != normalized_scope:
            raise ValueError(
                f"scheduler source scope {normalized_scope!r} does not match "
                f"Backend-authorized scope {declared_scope!r}"
            )
    field = (
        "node_set"
        if normalized_scope == SELECTED_EDGE_NODES
        else SOURCE_CANDIDATE_NODES_FIELD
    )
    if field not in info:
        raise ValueError(f"source selection input requires {field!r}")
    return _node_list(info.get(field), field)


def selection_scope_from_template(logical_template: Mapping[str, Any]) -> str:
    """Extract the source selection scope from a trusted scheduler template."""

    pod_template = logical_template.get("pod-template") or {}
    env_items = pod_template.get("env") or []
    if isinstance(env_items, (str, bytes)) or not isinstance(env_items, Sequence):
        raise ValueError("scheduler pod-template.env must be a list")

    env: Dict[str, Any] = {}
    for index, item in enumerate(env_items):
        if not isinstance(item, Mapping):
            raise ValueError(f"scheduler pod-template.env[{index}] must be an object")
        name = str(item.get("name") or "").strip()
        if name:
            env[name] = item.get("value")

    raw_params = env.get("SCH_SELECTION_POLICY_PARAMETERS")
    if raw_params in (None, ""):
        params = {}
    elif isinstance(raw_params, Mapping):
        params = dict(raw_params)
    elif isinstance(raw_params, str):
        try:
            params = ast.literal_eval(raw_params)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                "SCH_SELECTION_POLICY_PARAMETERS must be a Python/JSON object literal"
            ) from exc
    else:
        raise ValueError("SCH_SELECTION_POLICY_PARAMETERS must be an object literal")
    if not isinstance(params, Mapping):
        raise ValueError("SCH_SELECTION_POLICY_PARAMETERS must decode to an object")
    return normalize_source_selection_scope(params.get("scope"))
