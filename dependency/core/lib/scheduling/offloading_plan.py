"""Validation and materialization for per-task DAG offloading plans."""

from copy import deepcopy

from .dag import END, START, service_names

__all__ = ("materialize_offloading_plan",)


def materialize_offloading_plan(
    configuration,
    dag,
    plan,
    source_device,
    cloud_device,
):
    """Embed one complete service-to-device mapping in a schedule policy."""

    if not isinstance(configuration, dict):
        raise TypeError("schedule configuration must be a mapping")
    if not isinstance(plan, dict):
        raise TypeError("offloading plan must be a mapping")

    services = set(service_names(dag))
    assigned = set(plan)
    if assigned != services:
        missing = sorted(services - assigned)
        extra = sorted(assigned - services)
        raise ValueError(
            f"offloading plan mismatch; missing={missing}, extra={extra}"
        )

    scheduled = deepcopy(dag)
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

    policy = deepcopy(configuration)
    policy["dag"] = scheduled
    return policy
