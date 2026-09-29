import abc
from core.lib.common import ClassFactory, ClassType, Context, ConfigLoader
from core.lib.estimation import OverheadEstimator
from core.lib.scheduling import materialize_offloading_plan, service_names
from core.lib.scheduling.live_state import (
    active_deployment_for_dag,
    require_active_plan,
)

from .base_agent import BaseAgent

__all__ = ('FixedAgent',)


@ClassFactory.register(ClassType.SCH_AGENT, alias='fixed')
class FixedAgent(BaseAgent, abc.ABC):

    def __init__(self, system, agent_id: int, configuration=None, offloading=None):
        super().__init__(system, agent_id)

        self.system = system
        self.agent_id = agent_id
        self.cloud_device = system.cloud_device

        if configuration is None or isinstance(configuration, dict):
            self.fixed_configuration = configuration
        elif isinstance(configuration, str):
            self.fixed_configuration = ConfigLoader.load(Context.get_file_path(configuration))
        else:
            raise TypeError(f'Input "configuration" must be of type str or dict, get type {type(configuration)}')

        if offloading is None or isinstance(offloading, dict):
            self.fixed_offloading = offloading
        elif isinstance(offloading, str):
            self.fixed_offloading = ConfigLoader.load(Context.get_file_path(offloading))
        else:
            raise TypeError(f'Input "offloading" must be of type str or dict, get type {type(offloading)}')

        self.overhead_estimator = OverheadEstimator('Fixed', 'scheduler/fixed', agent_id=self.agent_id)

    def get_schedule_plan(self, info):
        if self.fixed_configuration is None or self.fixed_offloading is None:
            raise ValueError(
                'FixedAgent requires both configuration and offloading mappings'
            )

        with self.overhead_estimator:
            dag = info['dag']
            services = service_names(dag)
            missing = sorted(
                service for service in services
                if not str(self.fixed_offloading.get(service) or '').strip()
            )
            unknown = sorted(set(self.fixed_offloading) - set(services))
            if missing or unknown:
                raise ValueError(
                    f'fixed offloading must match the current DAG; '
                    f'missing={missing}, extra={unknown}'
                )
            offloading = {
                service: str(self.fixed_offloading[service])
                for service in services
            }
            _, deployment = active_deployment_for_dag(self.system, dag)
            require_active_plan(offloading, deployment)
            policy = materialize_offloading_plan(
                self.fixed_configuration,
                dag,
                offloading,
                source_device=info['source_device'],
                cloud_device=self.cloud_device,
            )
        return policy

    def run(self):
        pass

    def update_scenario(self, scenario):
        pass

    def update_resource(self, device, resource):
        pass

    def update_policy(self, policy):
        pass

    def update_task(self, task):
        pass

    def get_schedule_overhead(self):
        return self.overhead_estimator.get_latest_overhead()
