"""Helpers for consuming the structured processor queue-state contract."""

import math
from copy import deepcopy

__all__ = (
    "queue_waiting_counts",
    "service_waiting_count",
    "snapshot_queue_states",
)


def queue_waiting_counts(resource):
    """Return non-negative waiting counts keyed by logical service."""
    if not isinstance(resource, dict):
        return {}
    queue_states = resource.get('queue_state')
    if not isinstance(queue_states, dict):
        return {}

    waiting_counts = {}
    for service_name, state in queue_states.items():
        if not isinstance(state, dict):
            continue
        value = state.get('waiting_count')
        if isinstance(value, bool):
            continue
        try:
            waiting_counts[str(service_name)] = max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    return waiting_counts


def service_waiting_count(resource, service_name):
    """Return one service's waiting count, or zero when it is unavailable."""
    return queue_waiting_counts(resource).get(str(service_name), 0.0)


def snapshot_queue_states(snapshot, max_age_s=None):
    """Normalize revision-consistent ``service@device`` queue observations.

    Processor-local ``observed_at`` timestamps are diagnostic values from
    different host clocks. Freshness is therefore measured exclusively from
    the Scheduler receive timestamps carried by the snapshot.
    """

    snapshot = snapshot if isinstance(snapshot, dict) else {}
    try:
        now = float(snapshot.get("captured_at") or 0.0)
    except (TypeError, ValueError):
        now = 0.0
    try:
        revision = int(snapshot.get("runtime_directory_revision") or 0)
    except (TypeError, ValueError):
        revision = 0
    if max_age_s is not None:
        max_age_s = max(0.0, float(max_age_s))

    received_at = snapshot.get("resource_received_at") or {}
    resource_revisions = snapshot.get("resource_runtime_revision") or {}
    states = {}
    for raw_device, resource in (snapshot.get("resources") or {}).items():
        if not isinstance(resource, dict):
            continue
        device = str(raw_device)
        try:
            resource_revision = int(resource_revisions.get(raw_device))
        except (TypeError, ValueError):
            resource_revision = None
        if revision and resource_revision != revision:
            continue
        try:
            observed_at = float(received_at.get(raw_device))
        except (TypeError, ValueError):
            observed_at = 0.0
        if not math.isfinite(observed_at) or observed_at <= 0.0:
            observed_at = now
        age = max(0.0, now - observed_at)
        if max_age_s is not None and age > max_age_s:
            continue
        for raw_service, state in (resource.get("queue_state") or {}).items():
            if not isinstance(state, dict):
                continue
            normalized = deepcopy(state)
            normalized["_age_s"] = age
            states[(str(raw_service), device)] = normalized
    return states
