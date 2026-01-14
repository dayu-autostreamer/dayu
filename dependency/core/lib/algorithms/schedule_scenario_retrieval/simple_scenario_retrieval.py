import abc

from core.lib.common import ClassFactory, ClassType
from .base_scenario_retrieval import BaseScenarioRetrieval

__all__ = ('SimpleScenarioRetrieval',)


@ClassFactory.register(ClassType.SCH_SCENARIO_RETRIEVAL, alias='simple')
class SimpleScenarioRetrieval(BaseScenarioRetrieval, abc.ABC):
    def __call__(self, task):
        scenario = task.get_first_scenario_data()
        delay = task.calculate_total_time()
        meta_data = task.get_metadata()
        scenario['delay'] = delay / meta_data['buffer_size']
        return scenario
