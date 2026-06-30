import abc

from core.lib.common import ClassFactory, ClassType
from core.lib.algorithms.shared.hedger_ablation import HedgerDeploymentOnly

from .hedger_ablation_support import HedgerAblationRedeploymentPolicyBase

__all__ = ("HedgerDeploymentOnlyRedeploymentPolicy",)


@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias='hedger-deployment-only')
class HedgerDeploymentOnlyRedeploymentPolicy(HedgerAblationRedeploymentPolicyBase, abc.ABC):
    controller_cls = HedgerDeploymentOnly
    controller_alias = "hedger_deployment_only"
