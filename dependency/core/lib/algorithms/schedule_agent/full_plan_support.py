import copy
import math

from core.lib.common import ConfigLoader, Context, TaskConstant

from .fragsplice import FragSpliceLatencyModel


START = TaskConstant.START.value
END = TaskConstant.END.value


def load_mapping(value, label):
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if isinstance(value, str):
        loaded = ConfigLoader.load(Context.get_file_path(value))
        if isinstance(loaded, dict):
            return loaded
    raise TypeError(f"{label} must be a mapping or mounted file path")


def load_profiled_latency_model(configuration, latency_profile):
    configuration = load_mapping(configuration, "configuration")
    profile = load_mapping(latency_profile, "latency_profile")
    profile_context = FragSpliceLatencyModel.validate_profile_context(
        profile,
        configuration,
    )
    model = FragSpliceLatencyModel(profile=profile)
    if profile_context is not None:
        model.ensure_profile_context(**profile_context, require_complete=True)
    else:
        model.ensure_profile_context(configuration=configuration)
    return configuration, model


def service_names(dag):
    return [name for name in dag if name not in (START, END)]


def topological_order(dag):
    nodes = list(dag)
    indegree = {
        name: len([
            predecessor
            for predecessor in dag[name].get("prev_nodes", [])
            if predecessor in dag
        ])
        for name in nodes
    }
    ready = sorted(name for name, degree in indegree.items() if degree == 0)
    result = []
    while ready:
        current = ready.pop(0)
        result.append(current)
        for successor in dag[current].get("next_nodes", []):
            if successor not in indegree:
                continue
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(result) != len(nodes):
        raise ValueError("full-DAG offloading requires an acyclic DAG")
    return result


def deployment_from_snapshot(system, snapshot):
    deployment = snapshot.get("deployment") if isinstance(snapshot, dict) else None
    if not isinstance(deployment, dict):
        deployment = system.runtime_service_nodes()
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


def validate_profile_coverage(
    model,
    configuration,
    dag,
    deployment,
    method_name,
):
    model.ensure_profile_context(
        configuration=configuration,
        deployment=deployment,
        dag=dag,
        require_complete=True,
    )
    missing = []
    for service in service_names(dag):
        devices = deployment.get(service, [])
        if not devices:
            raise ValueError(
                f"{method_name} fixed deployment has no replica for {service}"
            )
        for device in devices:
            if model.sample_count(service, device) == 0:
                missing.append(f"{service}@{device}")
    if missing:
        raise ValueError(
            f"{method_name} latency profile does not cover the active fixed "
            "deployment: " + ", ".join(sorted(missing))
        )


def live_queue_states(snapshot, max_age_s):
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    try:
        now = float(snapshot.get("captured_at") or 0.0)
    except (TypeError, ValueError):
        now = 0.0
    try:
        revision = int(snapshot.get("runtime_directory_revision") or 0)
    except (TypeError, ValueError):
        revision = 0
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
            resource_received_at = float(received_at.get(raw_device))
        except (TypeError, ValueError):
            resource_received_at = 0.0
        if not math.isfinite(resource_received_at):
            resource_received_at = 0.0
        for raw_service, state in (resource.get("queue_state") or {}).items():
            if not isinstance(state, dict):
                continue
            observed_at = resource_received_at
            if observed_at <= 0.0:
                try:
                    observed_at = float(state.get("observed_at") or 0.0)
                except (TypeError, ValueError):
                    observed_at = 0.0
            if not math.isfinite(observed_at) or observed_at <= 0.0:
                observed_at = now
            age = max(0.0, now - observed_at)
            if age > max(0.0, float(max_age_s)):
                continue
            normalized = copy.deepcopy(state)
            normalized["_age_s"] = age
            states[(str(raw_service), device)] = normalized
    return states


def replica_load(model, service, device, state, quantile):
    processing = model.estimate(service, device, quantile)
    handoff = model.estimate_handoff(service, device, quantile)
    demand = processing + handoff
    state = state if isinstance(state, dict) else {}
    try:
        waiting_count = max(0, int(state.get("waiting_count") or 0))
    except (TypeError, ValueError):
        waiting_count = 0
    busy = bool(state.get("busy"))
    remaining = 0.0
    if busy:
        phase = str(state.get("running_phase") or "processing").lower()
        try:
            elapsed = float(
                state.get("phase_elapsed_s", state.get("running_elapsed_s"))
                or 0.0
            ) + float(state.get("_age_s") or 0.0)
        except (TypeError, ValueError):
            elapsed = 0.0
        elapsed = max(0.0, elapsed)
        if phase == "processing":
            processing_remaining = processing - elapsed
            if processing_remaining <= 1e-6:
                processing_remaining = processing
            remaining = processing_remaining + handoff
        elif phase in ("handoff", "sending", "returning"):
            remaining = handoff - elapsed
            if remaining <= 1e-6:
                remaining = max(1e-3, handoff)
        else:
            remaining = demand
    return {
        "workload": max(0.0, remaining + waiting_count * demand),
        "demand": max(1e-6, demand),
        "waiting_count": waiting_count,
        "busy": busy,
    }


def visible_replica_loads(snapshot, model, dag, deployment, quantile, max_age_s):
    states = live_queue_states(snapshot, max_age_s)
    result = {}
    for service in service_names(dag):
        for device in deployment.get(service, []):
            result[(service, device)] = replica_load(
                model,
                service,
                device,
                states.get((service, device)),
                quantile,
            )
    return result


def apply_full_plan(configuration, dag, plan, source_device, cloud_device):
    services = set(service_names(dag))
    if set(plan) != services:
        missing = sorted(services - set(plan))
        extra = sorted(set(plan) - services)
        raise ValueError(
            f"full-DAG plan mismatch; missing={missing}, extra={extra}"
        )
    scheduled = copy.deepcopy(dag)
    for service, device in plan.items():
        scheduled[service]["service"]["execute_device"] = str(device)
    if START in scheduled:
        scheduled[START]["service"]["execute_device"] = str(
            source_device or ""
        )
    if END in scheduled:
        scheduled[END]["service"]["execute_device"] = str(
            cloud_device or ""
        )
    policy = copy.deepcopy(configuration)
    policy["dag"] = scheduled
    return policy


__all__ = (
    "END",
    "START",
    "apply_full_plan",
    "deployment_from_snapshot",
    "live_queue_states",
    "load_profiled_latency_model",
    "replica_load",
    "service_names",
    "topological_order",
    "validate_profile_coverage",
    "visible_replica_loads",
)
