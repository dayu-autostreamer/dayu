import copy
import threading
import time

from .base_redeployment_policy import BaseRedeploymentPolicy

from core.lib.scheduling.deployment_plan import (
    allowed_nodes,
    dag_services,
    fixed_plan,
    validate_plan,
)
from core.lib.common import ClassFactory, ClassType, ConfigLoader, Context, LOGGER

__all__ = ("DynamicRedeploymentPolicy",)


@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias="dynamic")
class DynamicRedeploymentPolicy(BaseRedeploymentPolicy):
    """Convert the latest exact offloading decision into a deployment plan.

    The configured policy is an explicit fallback, not an unscoped cache: it
    is projected onto the current DAG and validated against this source's
    candidate nodes on every call.
    """

    def __init__(
            self,
            system,
            agent_id,
            redeployment_interval_minutes=None,
            device_service_limits=None,
            default_service_limit=None,
            **kwargs,
    ):
        self.system = system
        self.agent_id = agent_id
        self.cloud_device = str(getattr(system, "cloud_device", "") or "")

        policy = kwargs.get("policy")
        if policy is None:
            self.policy = None
        elif isinstance(policy, dict):
            self.policy = copy.deepcopy(policy)
        elif isinstance(policy, str):
            self.policy = ConfigLoader.load(Context.get_file_path(policy))
        else:
            raise TypeError(
                f'Input "policy" must be of type str or dict, get type {type(policy)}'
            )

        interval = 5 if redeployment_interval_minutes is None else redeployment_interval_minutes
        if not isinstance(interval, (int, float)):
            raise TypeError(
                'Input "redeployment_interval_minutes" must be of type int or float, '
                f"get type {type(interval)}"
            )
        self.redeployment_interval_seconds = max(0.0, float(interval) * 60.0)

        if device_service_limits is None:
            self.device_service_limits = {}
        elif isinstance(device_service_limits, dict):
            self.device_service_limits = {
                str(device): int(limit) for device, limit in device_service_limits.items()
            }
        else:
            raise TypeError(
                'Input "device_service_limits" must be of type dict, '
                f"get type {type(device_service_limits)}"
            )
        limit = 2 if default_service_limit is None else default_service_limit
        if not isinstance(limit, (int, float)):
            raise TypeError(
                'Input "default_service_limit" must be of type int or float, '
                f"get type {type(limit)}"
            )
        self.default_service_limit = int(limit)
        if self.default_service_limit < 1:
            raise ValueError("default_service_limit must be positive")

        self.latest_offloading_policy = {}
        self.last_redeployment_time = time.time()
        self.lock = threading.Lock()

    def update_latest_offloading_policy(self, offloading_policy):
        if offloading_policy is not None and not isinstance(offloading_policy, dict):
            raise TypeError("offloading_policy must be an object")
        with self.lock:
            self.latest_offloading_policy = copy.deepcopy(offloading_policy or {})

    @staticmethod
    def count_services_per_device(deploy_plan):
        counts = {}
        for devices in deploy_plan.values():
            for device in devices:
                counts[device] = counts.get(device, 0) + 1
        return counts

    def get_device_service_limit(self, device_name):
        return self.device_service_limits.get(str(device_name), self.default_service_limit)

    def check_deployment_constraint(self, deploy_plan):
        return all(
            count <= self.get_device_service_limit(device)
            for device, count in self.count_services_per_device(deploy_plan).items()
        )

    def _configured_plan(self, info):
        if self.policy is None:
            raise RuntimeError(
                "[Dynamic Redeployment] No explicit fallback policy is configured."
            )
        return fixed_plan(self.policy, info, cloud_node=self.cloud_device)

    def convert_offloading_to_deployment_plan(self, offloading_policy, dag, node_set):
        info = {"dag": dag, "node_set": node_set}
        offloading_policy = offloading_policy if isinstance(offloading_policy, dict) else {}
        candidates = allowed_nodes(info, self.cloud_device)
        fallback = self._configured_plan(info) if self.policy is not None else {}
        plan = {}
        for service in dag_services(info):
            device = str(offloading_policy.get(service, "") or "").strip()
            if device in candidates:
                plan[service] = [device]
            elif service in fallback:
                plan[service] = fallback[service]
            else:
                raise ValueError(
                    f"dynamic offloading omitted a valid target for current DAG service {service!r}"
                )
        return validate_plan(plan, info, cloud_node=self.cloud_device)

    def should_redeploy(self):
        return time.time() - self.last_redeployment_time >= self.redeployment_interval_seconds

    def get_latest_offloading_from_agent(self, source_id):
        agent = getattr(self.system, "schedule_table", {}).get(source_id)
        getter = getattr(agent, "get_latest_offloading_policy", None)
        return getter() if callable(getter) else None

    def __call__(self, info):
        source_id = info["source"]["id"]
        if not self.should_redeploy():
            return self._configured_plan(info)

        latest = self.get_latest_offloading_from_agent(source_id)
        if not latest:
            with self.lock:
                latest = copy.deepcopy(self.latest_offloading_policy)

        plan = self.convert_offloading_to_deployment_plan(
            latest,
            info["dag"],
            info["node_set"],
        )
        if not self.check_deployment_constraint(plan):
            plan = self._configured_plan(info)
            if not self.check_deployment_constraint(plan):
                raise ValueError(
                    "dynamic fallback policy exceeds configured device service limits"
                )

        with self.lock:
            self.policy = copy.deepcopy(plan)
        self.last_redeployment_time = time.time()
        LOGGER.info(f"[Dynamic Redeployment] (source {source_id}) Deploy policy: {plan}")
        return plan
