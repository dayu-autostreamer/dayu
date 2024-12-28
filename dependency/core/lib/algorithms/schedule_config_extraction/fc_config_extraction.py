import abc
import os

from core.lib.common import ClassFactory, ClassType, YamlOps, Context,FileNameConstant
from .base_config_extraction import BaseConfigExtraction

__all__ = ('FCConfigExtraction',)


@ClassFactory.register(ClassType.SCH_CONFIG, alias='fc')
class FCConfigExtraction(BaseConfigExtraction, abc.ABC):
    def __call__(self, scheduler):
        config_path = Context.get_file_path(os.path.join('scheduler/fc', FileNameConstant.SCHEDULE_CONFIG.value))
        configs = YamlOps.read_yaml(config_path)
        scheduler.resolution_list = configs['resolution']

        scheduler.schedule_knobs = ['resolution']
