import abc

from core.lib.common import ClassFactory, ClassType
from core.lib.algorithms.shared.hedger_ablation import HedgerFlat

from .hedger_ablation_support import HedgerAblationInitialDeploymentPolicyBase

__all__ = ("HedgerFlatInitialDeploymentPolicy",)


@ClassFactory.register(ClassType.SCH_INITIAL_DEPLOYMENT_POLICY, alias='hedger-flat')
class HedgerFlatInitialDeploymentPolicy(HedgerAblationInitialDeploymentPolicyBase, abc.ABC):
    controller_cls = HedgerFlat
    controller_alias = "hedger_flat"
