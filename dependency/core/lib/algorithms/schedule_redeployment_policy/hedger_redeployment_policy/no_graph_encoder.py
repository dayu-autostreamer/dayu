import abc

from core.lib.common import ClassFactory, ClassType
from core.lib.algorithms.shared.hedger_ablation import HedgerNoGraphEncoder

from .ablation import HedgerAblationRedeploymentPolicyBase

__all__ = ("HedgerNoGraphEncoderRedeploymentPolicy",)


@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias='hedger-no-graph-encoder')
class HedgerNoGraphEncoderRedeploymentPolicy(HedgerAblationRedeploymentPolicyBase, abc.ABC):
    controller_cls = HedgerNoGraphEncoder
    controller_alias = "hedger_no_graph_encoder"
