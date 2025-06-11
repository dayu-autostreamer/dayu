import abc

from .base_initial_deployment_policy import BaseInitialDeploymentPolicy

from core.lib.common import ClassFactory, ClassType, GlobalInstanceManager
from core.lib.algorithms.shared.hedger import Hedger

__all__ = ('HedgerInitialDeploymentPolicy',)


@ClassFactory.register(ClassType.SCH_INITIAL_DEPLOYMENT_POLICY, alias='hedger')
class HedgerInitialDeploymentPolicy(BaseInitialDeploymentPolicy, abc.ABC):
    def __init__(self):
        self.hedger = GlobalInstanceManager.get_instance(Hedger, 'hedger')

    def __call__(self, info):
        return None
