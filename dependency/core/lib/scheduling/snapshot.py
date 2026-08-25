from enum import Enum

__all__ = (
    "SchedulingSnapshotScope",
    "deployment_from_snapshot",
    "normalize_scheduling_snapshot_scope",
)


class SchedulingSnapshotScope(str, Enum):
    """Amount of runtime state requested by a scheduling extension."""

    LIVE = "live"
    COMMITTED = "committed"


def normalize_scheduling_snapshot_scope(value):
    """Return a validated :class:`SchedulingSnapshotScope` value."""

    if value is None:
        return SchedulingSnapshotScope.COMMITTED
    if isinstance(value, SchedulingSnapshotScope):
        return value
    try:
        return SchedulingSnapshotScope(str(value).strip().lower())
    except ValueError as exc:
        choices = ", ".join(scope.value for scope in SchedulingSnapshotScope)
        raise ValueError(
            f"unsupported scheduling snapshot scope {value!r}; "
            f"expected one of: {choices}"
        ) from exc


def deployment_from_snapshot(snapshot):
    """Return the normalized fixed deployment carried by a snapshot."""

    deployment = snapshot.get("deployment") if isinstance(snapshot, dict) else None
    if not isinstance(deployment, dict):
        raise ValueError("scheduling snapshot has no fixed deployment")
    normalized = {}
    for service, raw_devices in deployment.items():
        devices = [raw_devices] if isinstance(raw_devices, str) else raw_devices
        if not isinstance(devices, (list, tuple, set)):
            continue
        normalized[str(service)] = sorted({
            str(device).strip()
            for device in devices
            if str(device).strip()
        })
    return normalized
