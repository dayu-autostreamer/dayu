import abc
import os

from core.lib.common import ClassFactory, ClassType, YamlOps, Context, FileNameConstant
from .base_config_extraction import BaseConfigExtraction

__all__ = ('HEIConfigExtraction',)


@ClassFactory.register(ClassType.SCH_CONFIG, alias='hei')
class HEIConfigExtraction(BaseConfigExtraction, abc.ABC):

    def __init__(self, hei_drl_config: str, hei_hyper_config: str):
        self.HEI_DRL_CONFIG = hei_drl_config
        self.HEI_HYPER_CONFIG = hei_hyper_config

    def __call__(self, scheduler):
        config_path = Context.get_file_path(os.path.join('scheduler/hei', FileNameConstant.SCHEDULE_CONFIG.value))
        configs = YamlOps.read_yaml(config_path)
        scheduler.fps_list = configs['fps']
        scheduler.resolution_list = configs['resolution']
        scheduler.buffer_size_list = configs['buffer_size']
        scheduler.monotonic_schedule_knobs = ['resolution', 'fps', 'buffer_size']
        scheduler.non_monotonic_schedule_knobs = ['pipeline']

        drl_parameters_config_path = Context.get_file_path(os.path.join('scheduler/hei',self.HEI_DRL_CONFIG))
        scheduler.drl_params = YamlOps.read_yaml(drl_parameters_config_path)

        hyper_parameters_config_path = Context.get_file_path(os.path.join('scheduler/hei',self.HEI_HYPER_CONFIG))
        scheduler.hyper_params = YamlOps.read_yaml(hyper_parameters_config_path)
