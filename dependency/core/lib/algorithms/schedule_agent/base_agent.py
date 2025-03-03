import abc

from core.lib.common import Context


class BaseAgent(metaclass=abc.ABCMeta):
    def __init__(self):
        self.source_selection_policy = Context.get_algorithm('SCH_SELECTION_POLICY')
        self.service_deployment_policy = Context.get_algorithm('SCH_DEPLOYMENT_POLICY')

    def __call__(self):
        raise NotImplementedError

    def update_scenario(self, scenario):
        raise NotImplementedError

    def update_resource(self, device, resource):
        raise NotImplementedError

    def update_policy(self, policy):
        raise NotImplementedError

    def update_task(self, task):
        raise NotImplementedError

    def get_schedule_plan(self, info):
        raise NotImplementedError

    def get_source_selection_plan(self, info):
        return self.source_selection_policy(info)

    def get_service_deployment_plan(self, info):
        return self.service_deployment_policy(info)

    def run(self):
        raise NotImplementedError
