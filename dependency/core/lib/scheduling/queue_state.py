"""Helpers for consuming the structured processor queue-state contract."""


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
