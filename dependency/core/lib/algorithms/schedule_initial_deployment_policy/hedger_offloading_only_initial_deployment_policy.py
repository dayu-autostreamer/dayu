import abc

from core.lib.common import ClassFactory, ClassType
from core.lib.algorithms.shared.hedger_ablation import HedgerOffloadingOnly

from .hedger_ablation_support import HedgerAblationInitialDeploymentPolicyBase

__all__ = ("HedgerOffloadingOnlyInitialDeploymentPolicy",)


@ClassFactory.register(ClassType.SCH_INITIAL_DEPLOYMENT_POLICY, alias='hedger-offloading-only')
class HedgerOffloadingOnlyInitialDeploymentPolicy(HedgerAblationInitialDeploymentPolicyBase, abc.ABC):
    controller_cls = HedgerOffloadingOnly
    controller_alias = "hedger_offloading_only"
    use_heuristic_deployment = True
