import abc

from core.lib.common import ClassFactory, ClassType, YamlOps, Context, FileNameConstant
from .base_config_extraction import BaseConfigExtraction

__all__ = ('CASVAConfigExtraction',)


@ClassFactory.register(ClassType.SCH_CONFIG, alias='casva')
class CASVAConfigExtraction(BaseConfigExtraction, abc.ABC):

    def __init__(self, casva_drl_config: str, casva_hyper_config: str) -> None:
        self.CASVA_DRL_CONFIG = casva_drl_config
        self.CASVA_HYPER_CONFIG = casva_hyper_config

    def __call__(self, scheduler):
        config_path = Context.get_file_path(FileNameConstant.SCHEDULE_CONFIG.value)
        configs = YamlOps.read_yaml(config_path)
        scheduler.fps_list = configs['fps']
        scheduler.resolution_list = configs['resolution']
        scheduler.buffer_size_list = configs['buffer_size']
        scheduler.monotonic_schedule_knobs = ['resolution', 'fps', 'buffer_size']
        scheduler.non_monotonic_schedule_knobs = ['pipeline']

        drl_parameters_config_path = Context.get_file_path(self.CASVA_DRL_CONFIG)
        scheduler.drl_params = YamlOps.read_yaml(drl_parameters_config_path)

        hyper_parameters_config_path = Context.get_file_path(self.CASVA_HYPER_CONFIG)
        scheduler.hyper_params = YamlOps.read_yaml(hyper_parameters_config_path)