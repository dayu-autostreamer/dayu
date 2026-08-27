"""Validated LIVE scheduling state for policy hooks."""

from copy import deepcopy

from .dag import service_names
from .snapshot import SchedulingSnapshotScope, deployment_from_snapshot

__all__ = (
    "active_deployment_for_dag",
    "active_targets",
    "get_live_snapshot",
    "live_resources",
    "require_active_plan",
)


def get_live_snapshot(system):
    """Read and validate one revision-consistent LIVE scheduling snapshot."""

    getter = getattr(system, "get_scheduling_snapshot", None)
    if not callable(getter):
        raise ValueError(
            "scheduling hooks require system.get_scheduling_snapshot()"
        )
    snapshot = getter(scope=SchedulingSnapshotScope.LIVE)
    if not isinstance(snapshot, dict):
        raise ValueError("LIVE scheduling snapshot must be an object")
    try:
        revision = int(snapshot.get("runtime_directory_revision") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("LIVE scheduling snapshot has an invalid revision") from exc
    if revision < 1:
        raise ValueError("RuntimeDirectory is not ready for scheduling")
    deployment_from_snapshot(snapshot)
    return deepcopy(snapshot)


def active_deployment_for_dag(system, dag):
    """Return ``(snapshot, deployment)`` covering every business DAG service."""

    services = service_names(dag)
    snapshot = get_live_snapshot(system)
    deployment = deployment_from_snapshot(snapshot)
    missing = sorted(
        service for service in services if not deployment.get(str(service))
    )
    if missing:
        raise ValueError(
            f"LIVE RuntimeDirectory has no active processor replicas for {missing}"
        )
    return snapshot, deployment


def active_targets(deployment, service, candidates=None):
    """Return active targets for one service, preserving candidate order."""

    if not isinstance(deployment, dict):
        raise TypeError("active deployment must be an object")
    active = {
        str(device)
        for device in (deployment.get(str(service)) or [])
        if str(device).strip()
    }
    if candidates is None:
        return sorted(active)
    return [str(device) for device in candidates if str(device) in active]


def live_resources(snapshot):
    """Return telemetry reported for the active RuntimeDirectory revision."""

    snapshot = snapshot if isinstance(snapshot, dict) else {}
    try:
        revision = int(snapshot.get("runtime_directory_revision") or 0)
    except (TypeError, ValueError):
        return {}
    if revision < 1:
        return {}

    revisions = snapshot.get("resource_runtime_revision") or {}
    resources = snapshot.get("resources") or {}
    if not isinstance(revisions, dict) or not isinstance(resources, dict):
        return {}

    result = {}
    for device, resource in resources.items():
        try:
            resource_revision = int(revisions.get(device))
        except (TypeError, ValueError):
            continue
        if resource_revision == revision and isinstance(resource, dict):
            result[str(device)] = deepcopy(resource)
    return result


def require_active_plan(plan, deployment):
    """Reject an offloading decision that is not served by the LIVE revision."""

    if not isinstance(plan, dict):
        raise TypeError("offloading plan must be an object")
    if not isinstance(deployment, dict):
        raise TypeError("active deployment must be an object")
    inactive = {
        str(service): str(device)
        for service, device in plan.items()
        if str(device) not in {
            str(active) for active in (deployment.get(str(service)) or [])
        }
    }
    if inactive:
        raise ValueError(
            f"offloading targets are not active in the LIVE RuntimeDirectory: {inactive}"
        )
    return deepcopy(plan)
