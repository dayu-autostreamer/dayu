import abc
import os

from core.lib.common import ClassFactory, ClassType, YamlOps, Context
from .base_config_extraction import BaseConfigExtraction

__all__ = ('HedgerConfigExtraction',)


@ClassFactory.register(ClassType.SCH_CONFIG_EXTRACTION, alias='hedger')
class HedgerConfigExtraction(BaseConfigExtraction, abc.ABC):

    def __init__(self, hedger_network_config: str):
        self.HEDGER_NETWORK_CONFIG = hedger_network_config

    def __call__(self, scheduler):
        hedger_network_config_path = Context.get_file_path(os.path.join('scheduler/hedger',self.HEDGER_NETWORK_CONFIG))
        scheduler.network_params = YamlOps.read_yaml(hedger_network_config_path)

        hedger_hyper_config_path = Context.get_file_path(os.path.join('scheduler/hedger','hedger_hyper_config.yaml'))
        scheduler.hyper_params = YamlOps.read_yaml(hedger_hyper_config_path)

