import abc

from core.lib.common import ClassFactory, ClassType
from core.lib.algorithms.shared.hedger_ablation import HedgerOffloadingOnly

from .ablation import HedgerAblationRedeploymentPolicyBase

__all__ = ("HedgerOffloadingOnlyRedeploymentPolicy",)


@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias='hedger-offloading-only')
class HedgerOffloadingOnlyRedeploymentPolicy(HedgerAblationRedeploymentPolicyBase, abc.ABC):
    controller_cls = HedgerOffloadingOnly
    controller_alias = "hedger_offloading_only"
    use_heuristic_deployment = True
