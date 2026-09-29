from .base_initial_deployment_policy import BaseInitialDeploymentPolicy

from core.lib.common import ClassFactory, ClassType, LOGGER
from core.lib.scheduling.deployment_plan import dag_services, validate_plan

__all__ = ("SourceEdgeCloudInitialDeploymentPolicy",)


@ClassFactory.register(
    ClassType.SCH_INITIAL_DEPLOYMENT_POLICY,
    alias="source-edge-cloud",
)
class SourceEdgeCloudInitialDeploymentPolicy(BaseInitialDeploymentPolicy):
    """Deploy each business service to the selected source edge and cloud node."""

    def __init__(self, system, agent_id):
        self.cloud_device = str(getattr(system, "cloud_device", "") or "").strip()

    def __call__(self, info):
        source_id = info["source"]["id"]
        source_device = str(
            info["source"].get("source_device") or ""
        ).strip()
        if not source_device or not self.cloud_device:
            raise ValueError(
                "source-edge-cloud deployment requires source and cloud devices"
            )
        nodes = list(dict.fromkeys([source_device, self.cloud_device]))
        plan = {
            service: list(nodes)
            for service in dag_services(info)
        }
        plan = validate_plan(plan, info, cloud_node=self.cloud_device)
        LOGGER.info(
            f"[Initial Deployment] (source {source_id}) "
            f"Source-edge/cloud pipeline policy: {plan}"
        )
        return plan
