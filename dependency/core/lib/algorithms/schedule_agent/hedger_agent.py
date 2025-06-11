import abc

from core.lib.common import ClassFactory, ClassType, GlobalInstanceManager
from core.lib.algorithms.shared.hedger import Hedger

from .base_agent import BaseAgent

__all__ = ('HedgerAgent',)


@ClassFactory.register(ClassType.SCH_AGENT, alias='hei')
class HedgerAgent(BaseAgent, abc.ABC):
    def __init__(self, system,
                 agent_id: int,
                 ):
        super().__init__()

        self.hedger = GlobalInstanceManager.get_instance(Hedger, 'hedger')

    def get_schedule_plan(self, info):
        return None

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
        return 0
