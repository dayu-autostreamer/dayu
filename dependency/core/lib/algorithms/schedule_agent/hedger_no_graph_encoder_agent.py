import abc

from core.lib.common import ClassFactory, ClassType
from core.lib.algorithms.shared.hedger_ablation import HedgerNoGraphEncoder

from .hedger_ablation_support import HedgerAblationAgentBase

__all__ = ("HedgerNoGraphEncoderAgent",)


@ClassFactory.register(ClassType.SCH_AGENT, alias='hedger-no-graph-encoder')
class HedgerNoGraphEncoderAgent(HedgerAblationAgentBase, abc.ABC):
    controller_cls = HedgerNoGraphEncoder
    controller_alias = "hedger_no_graph_encoder"
