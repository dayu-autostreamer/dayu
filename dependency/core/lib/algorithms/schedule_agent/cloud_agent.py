import copy

from core.lib.common import ClassFactory, ClassType, ConfigLoader, Context
from core.lib.scheduling import materialize_offloading_plan, service_names
from core.lib.scheduling.live_state import (
    active_deployment_for_dag,
    require_active_plan,
)

from .base_agent import BaseAgent

__all__ = ("CloudAgent",)


@ClassFactory.register(ClassType.SCH_AGENT, alias="cloud")
class CloudAgent(BaseAgent):
    """Schedule every non-source DAG stage on the injected cloud node."""

    def __init__(self, system, agent_id, configuration=None):
        super().__init__(system, agent_id)
        self.cloud_device = str(getattr(system, "cloud_device", "") or "").strip()
        if not self.cloud_device:
            raise ValueError("cloud schedule agent requires system.cloud_device")
        if configuration is None or isinstance(configuration, dict):
            self.configuration = copy.deepcopy(configuration or {})
        elif isinstance(configuration, str):
            self.configuration = ConfigLoader.load(Context.get_file_path(configuration))
        else:
            raise TypeError(
                f'Input "configuration" must be of type str or dict, get type {type(configuration)}'
            )

    def get_schedule_plan(self, info):
        dag = info["dag"]
        for service_name in service_names(dag):
            node = dag.get(service_name)
            service = node.get("service") if isinstance(node, dict) else None
            if not isinstance(service, dict):
                raise ValueError(
                    f"schedule DAG service {service_name!r} is malformed"
                )
        _, deployment = active_deployment_for_dag(self.system, dag)
        plan = require_active_plan(
            {service: self.cloud_device for service in service_names(dag)},
            deployment,
        )
        return materialize_offloading_plan(
            self.configuration,
            dag,
            plan,
            info.get("source_device"),
            self.cloud_device,
        )

    def run(self):
        return None

    def update_scenario(self, scenario):
        return None

    def update_resource(self, device, resource):
        return None

    def update_policy(self, policy):
        return None

    def update_task(self, task):
        return None

    def get_schedule_overhead(self):
        return 0
