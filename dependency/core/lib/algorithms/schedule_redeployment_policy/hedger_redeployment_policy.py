import abc

from .base_redeployment_policy import BaseRedeploymentPolicy

from core.lib.common import ClassFactory, ClassType, GlobalInstanceManager
from core.lib.algorithms.shared.hedger import Hedger

__all__ = ('HedgerRedeploymentPolicy',)


@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias='hedger')
class HedgerRedeploymentPolicy(BaseRedeploymentPolicy, abc.ABC):
    def __init__(self):
        self.hedger = GlobalInstanceManager.get_instance(Hedger, 'hedger')

    def __call__(self, info):
        return None
