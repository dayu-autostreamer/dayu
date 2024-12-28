import abc
import time

from .base_getter_filter import BaseDataGetterFilter

from core.lib.common import ClassFactory, ClassType, LOGGER

__all__ = ('CASVADataGetterFilter',)


@ClassFactory.register(ClassType.GEN_GETTER_FILTER, alias='casva')
class CASVADataGetterFilter(BaseDataGetterFilter, abc.ABC):

    def __init__(self, data_coming_interval: float = 4):
        self.skip_count = 0
        self.latest_time = 0
        self.data_coming_interval = data_coming_interval

    def reset_filter(self):
        self.skip_count = 0

    def __call__(self, system):
        record_time = time.time()
        if self.latest_time > 0 and record_time - self.latest_time > self.data_coming_interval:
            LOGGER.debug(f'[Getter Filter] delay time: {record_time - self.latest_time}')
            self.skip_count += 1
            self.latest_time = record_time
            return False
        else:
            self.latest_time = record_time
            return True
