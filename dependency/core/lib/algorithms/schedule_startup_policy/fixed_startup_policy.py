import abc
import copy

from core.lib.common import ClassFactory, ClassType, TaskConstant
from .base_startup_policy import BaseStartupPolicy

__all__ = ('FixedStartupPolicy',)


@ClassFactory.register(ClassType.SCH_STARTUP_POLICY, alias='fixed')
class FixedStartupPolicy(BaseStartupPolicy, abc.ABC):
    def __call__(self, info):
        dag = copy.deepcopy(info['dag'])
        source_device = str(info.get('source_device') or '')
        cloud_device = str(info.get('cloud_device') or '')
        if not source_device or not cloud_device:
            raise ValueError(
                'fixed startup policy requires source_device and cloud_device'
            )
        for service_name, node in dag.items():
            service = node.get('service') if isinstance(node, dict) else None
            if not isinstance(service, dict):
                raise ValueError(
                    f'startup DAG service {service_name!r} is malformed'
                )
            service['execute_device'] = (
                source_device
                if service_name == TaskConstant.START.value
                else cloud_device
            )
        return {'dag': dag}
