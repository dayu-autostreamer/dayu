import abc
import copy
import threading
import time

from core.lib.common import ClassFactory, ClassType, LOGGER, TaskConstant

from .base_redeployment_policy import BaseRedeploymentPolicy

__all__ = ("DeepVARedeploymentPolicy",)


@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias="deepva")
class DeepVARedeploymentPolicy(BaseRedeploymentPolicy, abc.ABC):
    """Redeployment bridge for the DeepVA DRL baseline."""

    def __init__(self, system, agent_id, redeployment_interval=60):
        self.system = system
        self.agent_id = agent_id
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

    def _fallback_deployment(self, dag, node_set):
        devices = [device for device in getattr(self.system, "device_list", []) if device in node_set]
        if not devices:
            devices = list(node_set)
        if not devices:
            devices = [getattr(self.system, "cloud_device", "cloud")]

        service_names = [
            service_name for service_name in dag
            if service_name not in (TaskConstant.START.value, TaskConstant.END.value)
        ]
        plan = {}
        for idx, service_name in enumerate(service_names):
            plan[service_name] = [devices[idx % len(devices)]]
        return plan

    def _sanitize_deployment(self, deployment, dag, node_set):
        if not isinstance(deployment, dict):
            return self._fallback_deployment(dag, node_set)

        plan = {}
        fallback = self._fallback_deployment(dag, node_set)
        for service_name in dag:
            if service_name in (TaskConstant.START.value, TaskConstant.END.value):
                continue
            devices = [
                device for device in self._normalize_devices(deployment.get(service_name))
                if device in node_set
            ]
            plan[service_name] = devices or fallback.get(service_name, list(node_set)[:1])
        return plan

    def __call__(self, info):
        source_id = info["source"]["id"]
        dag = info["dag"]
        node_set = info["node_set"]

        if not self.should_redeploy():
            return copy.deepcopy(self.last_deploy_plan)

        deployment = self.get_deployment_from_agent(source_id)
        deploy_plan = self._sanitize_deployment(deployment, dag, node_set)
        self.last_deploy_plan = copy.deepcopy(deploy_plan)
        LOGGER.info(f"[DeepVA Redeployment] source={source_id}, deploy_plan={deploy_plan}")
        return deploy_plan
