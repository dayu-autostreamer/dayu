import abc

from core.lib.common import ClassFactory, ClassType
from core.lib.algorithms.shared.hedger_ablation import HedgerFlat

from .ablation import HedgerAblationRedeploymentPolicyBase

__all__ = ("HedgerFlatRedeploymentPolicy",)


@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias='hedger-flat')
class HedgerFlatRedeploymentPolicy(HedgerAblationRedeploymentPolicyBase, abc.ABC):
    controller_cls = HedgerFlat
    controller_alias = "hedger_flat"
