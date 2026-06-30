import abc

from core.lib.common import ClassFactory, ClassType
from core.lib.algorithms.shared.hedger_ablation import HedgerDeploymentOnly

from .hedger_ablation_support import HedgerAblationInitialDeploymentPolicyBase

__all__ = ("HedgerDeploymentOnlyInitialDeploymentPolicy",)


@ClassFactory.register(ClassType.SCH_INITIAL_DEPLOYMENT_POLICY, alias='hedger-deployment-only')
class HedgerDeploymentOnlyInitialDeploymentPolicy(HedgerAblationInitialDeploymentPolicyBase, abc.ABC):
    controller_cls = HedgerDeploymentOnly
    controller_alias = "hedger_deployment_only"
