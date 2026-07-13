"""Canonical deployment-plan contract shared by Scheduler and policy plugins."""

from copy import deepcopy


_SYNTHETIC_SERVICES = frozenset({"_start", "_end", "start", "end"})


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


def validate_plan(plan, info, cloud_node=""):
    """Return the sole public shape: ``logical_service -> [node, ...]``.

    Extra services, omitted services, scalar node values, empty placements and
    nodes outside this source's immutable candidate set all fail at the policy
    boundary. This policy contract never infers operational replicas; Backend
    may compose its configured cloud backup only after validation succeeds.
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
                f"deployment policy for service {service!r} selected non-candidate nodes: {invalid}"
            )
        nodes = sorted(set(nodes))
        if not nodes:
            raise ValueError(f"deployment policy for service {service!r} returned no target nodes")
        normalized[service] = nodes
    return normalized


def fixed_plan(policy, info, cloud_node=""):
    if not isinstance(policy, dict):
        raise ValueError("fixed deployment policy must be an object")
    services = dag_services(info)
    scoped = {
        service: deepcopy(policy[service])
        for service in services
        if service in policy
    }
    return validate_plan(scoped, info, cloud_node=cloud_node)


def cloud_plan(system, info):
    cloud_node = str(getattr(system, "cloud_device", "") or "").strip()
    if not cloud_node:
        raise ValueError("cloud deployment policy requires system.cloud_device")
    return validate_plan(
        {service: [cloud_node] for service in dag_services(info)},
        info,
        cloud_node=cloud_node,
    )
