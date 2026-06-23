import abc

from core.lib.common import ClassFactory, ClassType
from .base_scenario_retrieval import BaseScenarioRetrieval

__all__ = ('SteadyScenarioRetrieval',)


@ClassFactory.register(ClassType.SCH_SCENARIO_RETRIEVAL, alias='steady')
class SteadyScenarioRetrieval(BaseScenarioRetrieval, abc.ABC):
    def __call__(self, task):
        scenario = task.get_first_scenario_data()
        delay = task.calculate_total_time()
        meta_data = task.get_metadata()
        tmp_data = task.get_tmp_data()
        scenario['delay'] = delay / meta_data['buffer_size']
        scenario['file_size'] = tmp_data['file_size']
        return scenario
