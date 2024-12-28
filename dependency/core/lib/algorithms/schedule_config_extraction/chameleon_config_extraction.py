import abc
import os

from core.lib.common import ClassFactory, ClassType, YamlOps, Context,FileNameConstant
from .base_config_extraction import BaseConfigExtraction

__all__ = ('ChameleonConfigExtraction',)


@ClassFactory.register(ClassType.SCH_CONFIG, alias='chameleon')
class ChameleonConfigExtraction(BaseConfigExtraction, abc.ABC):
    def __call__(self, scheduler):
        config_path = Context.get_file_path(os.path.join('scheduler/chameleon', FileNameConstant.SCHEDULE_CONFIG.value))
        configs = YamlOps.read_yaml(config_path)
        scheduler.fps_list = configs['fps']
        scheduler.resolution_list = configs['resolution']

        scheduler.schedule_knobs = ['resolution', 'fps']
