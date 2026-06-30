import abc

from core.lib.common import ClassFactory, ClassType
from core.lib.algorithms.shared.hedger_ablation import HedgerNoGraphEncoder

from .hedger_ablation_support import HedgerAblationInitialDeploymentPolicyBase

__all__ = ("HedgerNoGraphEncoderInitialDeploymentPolicy",)


@ClassFactory.register(ClassType.SCH_INITIAL_DEPLOYMENT_POLICY, alias='hedger-no-graph-encoder')
class HedgerNoGraphEncoderInitialDeploymentPolicy(HedgerAblationInitialDeploymentPolicyBase, abc.ABC):
    controller_cls = HedgerNoGraphEncoder
    controller_alias = "hedger_no_graph_encoder"
