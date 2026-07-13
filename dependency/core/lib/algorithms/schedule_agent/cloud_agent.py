import copy

from core.lib.common import ClassFactory, ClassType, ConfigLoader, Context, TaskConstant

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
        dag = copy.deepcopy(info["dag"])
        source_device = str(info.get("source_device") or "")
        for service_name, node in dag.items():
            service = node.get("service") if isinstance(node, dict) else None
            if not isinstance(service, dict):
                raise ValueError(f"schedule DAG service {service_name!r} is malformed")
            service["execute_device"] = (
                source_device
                if service_name == TaskConstant.START.value
                else self.cloud_device
            )
        return {**copy.deepcopy(self.configuration), "dag": dag}

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
