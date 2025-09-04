import abc

from .base_redeployment_policy import BaseRedeploymentPolicy

from core.lib.common import ClassFactory, ClassType, GlobalInstanceManager
from core.lib.algorithms.shared.hedger import Hedger

__all__ = ('HedgerRedeploymentPolicy',)


@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias='hedger')
class HedgerRedeploymentPolicy(BaseRedeploymentPolicy, abc.ABC):
    def __init__(self):
        self.hedger = None

    def register_hedger(self, hedger_id='hedger'):
        if self.hedger is None:
            self.hedger = GlobalInstanceManager.get_instance(Hedger, hedger_id)
        self.hedger

    def __call__(self, info):
        source_id = info['source']['id']
        dag = info['dag']
        node_set = info['node_set']

        self.register_hedger(f'hedger_{source_id}')
