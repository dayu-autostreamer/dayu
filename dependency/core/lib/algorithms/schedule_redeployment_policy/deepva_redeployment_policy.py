import abc
import copy
import threading
import time

from core.lib.common import ClassFactory, ClassType, LOGGER
from core.lib.scheduling import deployment_from_snapshot
from core.lib.scheduling.deployment_plan import (
    allowed_nodes,
    cloud_replica_plan,
    dag_services,
    normalize_include_cloud,
    validate_plan,
)
from core.lib.scheduling.live_state import get_live_snapshot

from .base_redeployment_policy import BaseRedeploymentPolicy

__all__ = ("DeepVARedeploymentPolicy",)


@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias="deepva")
class DeepVARedeploymentPolicy(BaseRedeploymentPolicy, abc.ABC):
    """Redeployment bridge for the DeepVA DRL baseline."""

    def __init__(
            self,
            system,
            agent_id,
            redeployment_interval=60,
            include_cloud=True,
    ):
        self.system = system
        self.agent_id = agent_id
        self.cloud_device = str(getattr(system, "cloud_device", "") or "").strip()
        self.include_cloud = normalize_include_cloud(include_cloud)
        self.redeployment_interval = float(redeployment_interval)
        self.last_redeployment_time = 0.0
        self.last_deploy_plan = None
        self.lock = threading.Lock()
        LOGGER.info(f"[DeepVA Redeployment] interval={self.redeployment_interval}s")

    def should_redeploy(self):
        with self.lock:
            now = time.time()
            if self.last_deployment_missing():
                self.last_redeployment_time = now
                return True
            if now - self.last_redeployment_time >= self.redeployment_interval:
                self.last_redeployment_time = now
                return True
            return False

    def last_deployment_missing(self):
        return self.last_deploy_plan is None

    def get_deployment_from_agent(self, source_id):
        agent = self.system.schedule_table.get(source_id)
        if agent is None or not hasattr(agent, "get_current_deployment"):
            return None
        return agent.get_current_deployment()

    @staticmethod
    def _normalize_devices(devices):
        if devices is None:
            return []
        if isinstance(devices, str):
            return [devices]
        if isinstance(devices, (list, tuple, set, frozenset)):
            return [str(device) for device in devices]
        return []

    def _sanitize_deployment(self, deployment, info):
        if not isinstance(deployment, dict):
            deployment = deployment_from_snapshot(get_live_snapshot(self.system))

        candidates = allowed_nodes(
            info,
            self.cloud_device,
        )
        plan = {}
        for service_name in dag_services(info):
            devices = [
                device for device in self._normalize_devices(deployment.get(service_name))
                if device in candidates
            ]
            if not devices:
                raise ValueError(
                    f"DeepVA deployment omitted an active candidate for "
                    f"service {service_name!r}"
                )
            plan[service_name] = devices
        plan = validate_plan(
            plan,
            info,
            cloud_node=self.cloud_device,
        )
        if self.include_cloud:
            plan = cloud_replica_plan(
                plan,
                info,
                self.cloud_device,
                policy_name="DeepVA deployment policy",
            )
        return plan

    def __call__(self, info):
        source_id = info["source"]["id"]
        if not self.should_redeploy():
            return validate_plan(
                copy.deepcopy(self.last_deploy_plan),
                info,
                cloud_node=self.cloud_device,
            )

        deployment = self.get_deployment_from_agent(source_id)
        deploy_plan = self._sanitize_deployment(deployment, info)
        self.last_deploy_plan = copy.deepcopy(deploy_plan)
        LOGGER.info(f"[DeepVA Redeployment] source={source_id}, deploy_plan={deploy_plan}")
        return deploy_plan
