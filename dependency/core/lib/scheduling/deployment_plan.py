"""Canonical deployment-plan contract shared by Scheduler and policy plugins."""

from copy import deepcopy


_SYNTHETIC_SERVICES = frozenset({"_start", "_end", "start", "end"})
CLOUD_NODE_TOKEN = "@cloud"


def dag_services(info):
    dag = info.get("dag") if isinstance(info, dict) else None
    if not isinstance(dag, dict):
        raise ValueError("deployment policy input requires a dag object")
    return tuple(
        str(service)
        for service in dag
        if str(service) not in _SYNTHETIC_SERVICES
    )


def allowed_nodes(info, cloud_node=""):
    raw_nodes = info.get("node_set") if isinstance(info, dict) else None
    if not isinstance(raw_nodes, (list, tuple)):
        raise ValueError("deployment policy input requires a node_set list")
    nodes = {str(node).strip() for node in raw_nodes if str(node).strip()}
    cloud_node = str(cloud_node or "").strip()
    if cloud_node:
        nodes.add(cloud_node)
    return nodes


def normalize_include_cloud(value):
    if not isinstance(value, bool):
        raise TypeError("include_cloud must be a boolean")
    return value


def require_cloud_node(cloud_node, policy_name="deployment policy"):
    cloud_node = str(cloud_node or "").strip()
    if not cloud_node:
        raise ValueError(f"{policy_name} requires system.cloud_device")
    return cloud_node


def validate_plan(plan, info, cloud_node=""):
    """Return the sole public shape: ``logical_service -> [node, ...]``.

    Extra services, omitted services, scalar node values, empty placements and
    nodes outside this source's immutable candidate set all fail at the policy
    boundary. Cloud placement is part of the policy result and must use the
    exact cloud identity injected into the Scheduler.
    """

    if not isinstance(plan, dict):
        raise ValueError("deployment policy must return an object")
    services = dag_services(info)
    service_set = set(services)
    unknown = sorted(str(service) for service in plan if str(service) not in service_set)
    if unknown:
        raise ValueError(f"deployment policy returned services outside the current DAG: {unknown}")
    missing = sorted(service for service in services if service not in plan)
    if missing:
        raise ValueError(f"deployment policy omitted current DAG services: {missing}")

    candidates = allowed_nodes(info, cloud_node)
    normalized = {}
    for service in services:
        raw_nodes = plan[service]
        if not isinstance(raw_nodes, list):
            raise ValueError(
                f"deployment policy for service {service!r} must return a JSON node list"
            )
        nodes = [str(node).strip() for node in raw_nodes]
        if any(not node for node in nodes):
            raise ValueError(
                f"deployment policy for service {service!r} returned an empty node name"
            )
        invalid = sorted(set(nodes) - candidates)
        if invalid:
            raise ValueError(
                f"deployment policy for service {service!r} selected non-candidate nodes: {invalid}; "
                f"allowed processor nodes: {sorted(candidates)}"
            )
        nodes = sorted(set(nodes))
        if not nodes:
            raise ValueError(f"deployment policy for service {service!r} returned no target nodes")
        normalized[service] = nodes
    return normalized


def cloud_replica_plan(plan, info, cloud_node, policy_name="deployment policy"):
    """Validate ``plan`` and add the exact cloud node to every service."""

    cloud_node = require_cloud_node(cloud_node, policy_name)
    normalized = validate_plan(plan, info, cloud_node=cloud_node)
    with_cloud = {
        service: [*nodes, cloud_node]
        for service, nodes in normalized.items()
    }
    return validate_plan(with_cloud, info, cloud_node=cloud_node)


def fixed_plan(policy, info, cloud_node="", include_cloud=False):
    if not isinstance(policy, dict):
        raise ValueError("fixed deployment policy must be an object")
    include_cloud = normalize_include_cloud(include_cloud)
    cloud_node = str(cloud_node or "").strip()
    services = dag_services(info)
    scoped = {}
    for service in services:
        if service not in policy:
            continue
        raw_nodes = deepcopy(policy[service])
        if isinstance(raw_nodes, list):
            resolved = []
            for raw_node in raw_nodes:
                node = str(raw_node).strip()
                if node == CLOUD_NODE_TOKEN:
                    node = require_cloud_node(
                        cloud_node,
                        f"fixed deployment {CLOUD_NODE_TOKEN}",
                    )
                resolved.append(node)
            if include_cloud:
                resolved.append(require_cloud_node(cloud_node, "fixed deployment include_cloud"))
            raw_nodes = resolved
        scoped[service] = raw_nodes
    return validate_plan(scoped, info, cloud_node=cloud_node)


def full_edge_plan(info, cloud_node=""):
    cloud_node = str(cloud_node or "").strip()
    nodes = sorted(
        node for node in allowed_nodes(info)
        if node != cloud_node
    )
    return validate_plan(
        {service: list(nodes) for service in dag_services(info)},
        info,
        cloud_node=cloud_node,
    )


def full_plan(info, cloud_node):
    cloud_node = require_cloud_node(cloud_node, "full deployment policy")
    nodes = sorted(allowed_nodes(info)) + [cloud_node]
    return validate_plan(
        {service: list(nodes) for service in dag_services(info)},
        info,
        cloud_node=cloud_node,
    )


def cloud_plan(system, info):
    cloud_node = require_cloud_node(
        getattr(system, "cloud_device", ""),
        "cloud deployment policy",
    )
    return validate_plan(
        {service: [cloud_node] for service in dag_services(info)},
        info,
        cloud_node=cloud_node,
    )
