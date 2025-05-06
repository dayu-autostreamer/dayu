import abc

from core.lib.common import ClassFactory, ClassType
from .base_scenario_extraction import BaseScenarioExtraction

__all__ = ('CASVAScenarioExtraction',)


@ClassFactory.register(ClassType.SCH_SCENARIO_EXTRACTION, alias='casva')
class CASVAScenarioExtraction(BaseScenarioExtraction, abc.ABC):
    def __call__(self, task):
        scenario = task.get_scenario_data()
        delay = task.calculate_cloud_edge_transmit_time()
        tmp_data = task.get_tmp_data()
        meta_data = task.get_metadata()
        scenario['delay'] = delay
        scenario['segment_size'] = tmp_data['file_size']
        scenario['content_dynamics'] = tmp_data['file_dynamics']
        scenario['buffer_size'] = meta_data['buffer_size']
        return scenario
