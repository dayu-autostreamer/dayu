import abc

from core.lib.common import Context


class BaseAgent(metaclass=abc.ABCMeta):
    def __init__(self, system, agent_id):
        self.source_selection_policy = Context.get_algorithm('SCH_SELECTION_POLICY',
                                                             system=system, agent_id=agent_id)
        self.initial_deployment_policy = Context.get_algorithm('SCH_INITIAL_DEPLOYMENT_POLICY',
                                                               system=system, agent_id=agent_id)
        self.redeployment_policy = Context.get_algorithm('SCH_REDEPLOYMENT_POLICY',
                                                         system=system, agent_id=agent_id)

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

    def get_initial_deployment_plan(self, info):
        return self.initial_deployment_policy(info)

    def get_redeployment_plan(self, info):
        return self.redeployment_policy(info)

    def get_schedule_overhead(self):
        return 0

    def run(self):
        raise NotImplementedError
