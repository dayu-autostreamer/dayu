import abc

from core.lib.common import ClassFactory, ClassType
from .base_scenario_extraction import BaseScenarioExtraction

__all__ = ('SimpleScenarioExtraction',)


@ClassFactory.register(ClassType.SCH_SCENARIO_EXTRACTION, alias='simple')
class SimpleScenarioExtraction(BaseScenarioExtraction, abc.ABC):
    def __call__(self, task):
        scenario = task.get_scenario_data()
        delay = task.calculate_total_time()
        meta_data = task.get_metadata()
        scenario['delay'] = delay / meta_data['buffer_size']
        return scenario
