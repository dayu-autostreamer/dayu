import abc

from core.lib.common import ClassFactory, ClassType
from .base_policy_extraction import BasePolicyExtraction

__all__ = ('SimplePolicyExtraction',)


@ClassFactory.register(ClassType.SCH_POLICY_EXTRACTION, alias='simple')
class SimplePolicyExtraction(BasePolicyExtraction, abc.ABC):
    def __call__(self, task):
        policy = {}

        meta_data = task.get_metadata()
        policy.update(meta_data)

        dag = task.get_dag_deployment_info()
        policy['dag'] = dag
        policy['edge_device'] = task.get_source_device()

        return policy
