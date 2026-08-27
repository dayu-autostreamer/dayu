from .base_redeployment_policy import BaseRedeploymentPolicy
from core.lib.scheduling import deployment_from_snapshot
from core.lib.scheduling.live_state import get_live_snapshot
from core.lib.scheduling.deployment_plan import allowed_nodes, dag_services, validate_plan
from core.lib.common import ClassFactory, ClassType, LOGGER

__all__ = ('NonRedeploymentPolicy',)

@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias='non')
class NonRedeploymentPolicy(BaseRedeploymentPolicy):
    """Preserve the active runtime deployment for the current source DAG."""

    def __init__(self, system, agent_id, policy=None):
        self.system = system
        self.cloud_device = str(getattr(system, "cloud_device", "") or "")

    def __call__(self, info):
        # Agents may be constructed before the initial Processor deployment is
        # published. Read the active runtime directory at request time instead
        # of caching an empty or obsolete deployment in __init__.
        snapshot = get_live_snapshot(self.system)
        service_deployment = deployment_from_snapshot(snapshot)

        # The runtime directory stores the deployment union for all sources.
        # Project it onto this source's DAG and immutable candidate set; the
        # Scheduler server merges the source-local projections back into the
        # same global deployment.
        candidates = allowed_nodes(info, self.cloud_device)
        scoped = {}
        for service in dag_services(info):
            if service not in service_deployment:
                continue
            raw_nodes = service_deployment[service]
            if not isinstance(raw_nodes, (list, tuple)):
                raise ValueError(
                    f"current runtime deployment for service {service!r} must be a node list"
                )
            scoped[service] = [
                str(node).strip()
                for node in raw_nodes
                if str(node).strip() in candidates
            ]

        plan = validate_plan(scoped, info, cloud_node=self.cloud_device)
        LOGGER.info(
            f"[Redeployment] Using NonRedeploymentPolicy, preserving current plan: {plan}"
        )
        return plan
